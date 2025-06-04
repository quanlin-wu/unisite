import os
import torch
import tree
import pandas as pd
import pickle
from rdkit import Chem
import esm
import torchdrug
from torch.utils import data
import warnings

from unisite.data import residue_constants 
from unisite.data import utils as du


class Pockets:
    def __init__(self, labels, pocket_masks, res_mask):
        self._labels = labels
        self._pocket_masks = pocket_masks
        self._res_mask = res_mask
    
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        if key == "labels":
            return self._labels
        elif key == "pocket_masks":
            return self._pocket_masks
        elif key == "res_mask":
            return self._res_mask
        elif isinstance(key, int):
            idx = key
            return {
                "labels": self._labels[idx],
                "pocket_masks": self._pocket_masks[idx],
                "res_mask": self._res_mask,
            }
        else:
            raise KeyError(f"Invalid key: {key}")

    @property
    def device(self):
        return self._labels.device

    def cuda(self):
        return Pockets(self._labels.cuda(), self._pocket_masks.cuda(), self._res_mask.cuda())
    
    def numpy(self):
        return {
            "labels": self._labels.cpu().numpy(),
            "pocket_masks": self._pocket_masks.cpu().numpy(),
            "res_mask": self._res_mask.cpu().numpy(),
        }
    
    def to(self, device):
        return Pockets(self._labels.to(device), self._pocket_masks.to(device), self._res_mask.to(device))


def length_collate(batch):
    max_len = max([x['length'] for x in batch])
    names = [x.pop('name') for x in batch]
    targets = [x.pop('target') for x in batch] if "target" in batch[0] else None
    graphs = [{'graph': x.pop('graph')} for x in batch] if "graph" in batch[0] else None
    map_dicts = [x.pop('map_dict') for x in batch] if "map_dict" in batch[0] else None
    pdb_file_paths = [x.pop('pdb_file_path') for x in batch] if "pdb_file_path" in batch[0] else None
    pad_example = lambda x: du.pad_feats(x, max_len)
    padded_batch = [pad_example(x) for x in batch]
    batch = torch.utils.data.default_collate(padded_batch)
    batch["name"] = names
    if targets is not None:
        batch["target"] = targets 
    if graphs is not None:
        batch["graph"] = torchdrug.data.graph_collate(graphs)["graph"]
    if map_dicts is not None:
        batch["map_dict"] = map_dicts
    if pdb_file_paths is not None:
        batch["pdb_file_path"] = pdb_file_paths
    return batch


def load_pdb(pdb_file, sanitize=False, removeHs=True):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=removeHs, sanitize=sanitize)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein = torchdrug.data.Protein.from_molecule(mol)
            protein.view = "residue"
    except Exception as e:
        assert False, f"Error loading PDB file {pdb_file}: {e}"
    return protein


def map_to(feats, device):
    return tree.map_structure(
        lambda x: x.to(device) 
        if (isinstance(x, torch.Tensor) or isinstance(x, Pockets) or isinstance(x, torchdrug.data.Protein)) else x,
        feats
    )


def get_esm_embedding(seq, esm_model, batch_converter, device):
    # get the ESM embedding
    labels, strs, tokens = batch_converter([("", seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        outputs = esm_model(tokens, repr_layers=[33])
    token_representations = outputs["representations"][33][0][1:-1]
    assert token_representations.shape[0] == len(seq)
    token_representations = token_representations.cpu().numpy()
    return token_representations 


class InferDataset(data.Dataset):
    def __init__(self, dataset_df: pd.DataFrame, esm_func, seq_only=True):
        self._dataset_df = dataset_df
        self._seq_only = seq_only
        self._esm_func = esm_func

    def __len__(self):
        return len(self._dataset_df)

    def __getitem__(self, idx):
        row = self._dataset_df.iloc[idx]
        
        seq = None
        graph = None
        map_dict = None
        
        if "pkl_path" in row.keys():
            data = du.read_pkl(row["pkl_path"])
            seq = data["sequence"]
            map_dict = data.get("map_dict", None)
            if not self._seq_only and "pdb_file_path" in data.keys():
                graph = load_pdb(data["pdb_file_path"])
                pdb_file_path = data["pdb_file_path"]    
        if seq is None and "sequence" in row.keys():
            seq = row["sequence"]
        if (not self._seq_only) and graph is None and "pdb_file_path" in row.keys():
            pdb_file_path = row["pdb_file_path"]
            graph = load_pdb(row["pdb_file_path"])
            # if only PDB file is provided, we need to get the sequence and map_dict from the PDB file
            if seq is None:
                seq = graph.to_sequence()
                seq = seq.replace(".", "")
                assert len(seq) == graph.num_residue
            if map_dict is None:
                residue_number = graph.residue_number.tolist()
                map_dict = {str(r): str(i + 1) for i, r in enumerate(residue_number)}

        length = len(seq)
        # convert unknown residues to 'X'
        restype = residue_constants.restypes
        seq_with_x = [r if r in restype else 'X' for r in seq]
        # get aatype
        aatype = torch.tensor([residue_constants.restype_order_with_x[r] for r in seq_with_x]).long() 
        feats = {
            "name": row["name"],
            "length": length,
            "aatype": aatype,
            "seq_idx": torch.arange(length) + 1,
            "res_mask": torch.ones(length, dtype=torch.bool),   # no residue should be masked
            "esm_embed": self._esm_func(seq),
        }
        if not self._seq_only:
            feats.update({
                "pdb_file_path": pdb_file_path,
                "graph": graph,
                "map_dict": map_dict,
            })
        return feats
