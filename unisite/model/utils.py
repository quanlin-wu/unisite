import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import pandas as pd
import torch
import torch.nn.functional as F


def hull_center(hull):
    """Compute the center of a convex hull.

    Parameters
    ----------
    hull: scipy.spatial.ConvexHull
        Convex hull to compute the center of.

    Returns
    -------
    numpy.ndarray
        Convex hull center of mass.
    """

    hull_com = np.zeros(3)
    tetras = Delaunay(hull.points[hull.vertices])

    for i in range(len(tetras.simplices)):
        tetra_verts = tetras.points[tetras.simplices][i]

        a, b, c, d = tetra_verts
        a, b, c = a - d, b - d, c - d
        tetra_vol = np.abs(np.linalg.det([a, b, c])) / 6

        tetra_com = np.mean(tetra_verts, axis=0)

        hull_com += tetra_com * tetra_vol

    hull_com = hull_com / hull.volume
    return hull_com


def calc_center(protein, masks, methods=["hull", "geo"]):
    """Calculate the center of a protein pocket.

    Parameters
    ----------
    protein: torchdrug.data.Protein
        Protein to calculate the center of.
    masks: numpy.ndarray
        Pocket masks to calculate the center of.
    methods: list of str
        Methods to use for calculating the center. Options are "hull" and "geo".

    Returns
    -------
    numpy.ndarray
        Center of the protein pocket.
    """
    hull_centers = []
    geo_centers = []
    for mask in masks:
        if mask.sum() == 0:
            if "hull" in methods:
                hull_centers.append(np.zeros((1, 3)))
            if "geo" in methods:
                geo_centers.append(np.zeros((1, 3)))
            continue
        atom_positions = []
        for i in range(len(mask)):
            if mask[i] == 1:
                atom_positions.append(protein[i].node_position)
        atom_positions = np.concatenate(atom_positions, axis=0)
        if "hull" in methods:
            if len(atom_positions) < 4:
                hull_centers.append(np.mean(atom_positions, axis=0, keepdims=True))
            else:
                hull = ConvexHull(atom_positions)
                hull_centers.append(hull_center(hull)[None, :])
        if "geo" in methods:
            geo_centers.append(np.mean(atom_positions, axis=0, keepdims=True))
    ret = {}
    if "hull" in methods:
        hull_centers = np.concatenate(hull_centers, axis=0)
        ret["hull_centers"] = hull_centers
    if "geo" in methods:
        geo_centers = np.concatenate(geo_centers, axis=0)
        ret["geo_centers"] = geo_centers
    return ret


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    len = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / len


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def get_df_from_dict(data):
    """
    Convert a dictionary to a pandas DataFrame.
    Args:
        data: Dict
    Returns:
        pd.DataFrame
    """
    num_pred = len(data["scores"])
    length = data["pocket_masks"].shape[1]
    if "map_dict" in data:
        map_dict = data["map_dict"]
        inv_map_dict = {v: k for k, v in map_dict.items()}
    else:
        inv_map_dict = None
    if "pdb_file_path" in data:
        with open(data["pdb_file_path"], "r") as f:
            residue_numbers = []
            for line in f.readlines():
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    residue_number = line[22:27].strip()
                    if residue_number in residue_numbers:
                        continue
                    residue_numbers.append(residue_number)
    else:
        residue_numbers = None
    df_rows = []
    for i in range(num_pred):
        position_id = []
        residue_id = []
        for j in range(length):
            if data["pocket_masks"][i][j] == 1:
                position_id.append(str(j+1))
                if residue_numbers is not None:
                    residue_id.append(residue_numbers[j])
                elif inv_map_dict is not None and str(j+1) in inv_map_dict:
                    residue_id.append(inv_map_dict[str(j+1)])
        if float(data["scores"][i]) > 0 and len(position_id) > 3:
            df_rows.append({
                "score": data["scores"][i],
                "center": None,
                "residue_id": "+".join(residue_id),
                "position_id": "+".join(position_id),       
            })
    return pd.DataFrame(df_rows, columns=["score", "center", "residue_id", "position_id"])
 