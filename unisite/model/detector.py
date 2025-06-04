"""Binding site/pocket detector network."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from . import transformer
from .gearnet import GearNetTransformer
from .utils import (
    batch_sigmoid_ce_loss_jit,
    batch_dice_loss_jit,
    sigmoid_ce_loss_jit,
    dice_loss_jit,
)
import functools as fn
from scipy.optimize import linear_sum_assignment


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class DetectorEmbedder(nn.Module):
    """
    Embedder for the pocket detector network. Only generates the node embeddings.
    Input:
        seq_idx, amino acid type, esm
    """
    def __init__(self, model_conf):
        super(DetectorEmbedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Sequence index embedding
        index_embed_size = self._embed_conf.index_embed_size
        node_embed_dims = index_embed_size
        
        # Amino acid embedding
        aa_embed_size = self._embed_conf.aatype_embed_size
        node_embed_dims += aa_embed_size
        
        # esm
        node_embed_dims += 1280

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )
        self.aa_embedder = nn.Embedding(23, aa_embed_size)

    def forward(
            self,
            *,
            seq_idx,
            aatype,
            esm_embed,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            aatype: [..., N] Amino acid type for each residue.
            esm_embed: [..., N, 1280]

        Returns:
            node_embed: [B, N, D_node]
        """
        # num_batch, num_res = seq_idx.shape
        node_feats = []
        
        # esm_embed
        node_feats.append(esm_embed)

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))

        # Amino acid features.
        aa_embed = self.aa_embedder(aatype)
        node_feats.append(aa_embed)

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())

        return node_embed


class DetectorNetwork(nn.Module):
    """DETR-like network for pocket detection."""
    def __init__(self, model_conf):
        super(DetectorNetwork, self).__init__()
        self._model_conf = model_conf
        self.embedding_layer = DetectorEmbedder(model_conf)
        self.num_queries = model_conf.num_queries
        hidden_dim = model_conf.node_embed_size
        self.aux_loss = model_conf.aux_loss
        if self._model_conf.model_type == "seq_only":
            self.transformer = transformer.Transformer(
                d_model=model_conf.node_embed_size,
                nhead=model_conf.no_heads,
                num_encoder_layers=model_conf.enc_layers,
                num_decoder_layers=model_conf.dec_layers,
                dim_feedforward=model_conf.dim_feedforward,
                dropout=model_conf.dropout,
                return_intermediate_dec=True,
            )
        else:
            self.transformer = GearNetTransformer(model_conf)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.cls_embed = MLP(hidden_dim, hidden_dim, model_conf.num_classes + 1, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, input_feats):

        init_node_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            aatype=input_feats['aatype'],
            esm_embed=input_feats['esm_embed'],
        )

        # 1 means ignore
        attn_mask =  ~input_feats["res_mask"].type(torch.bool)

        assert attn_mask is not None
        pos_embed = get_index_embedding(input_feats["seq_idx"], embed_size=init_node_embed.shape[-1])
        # hs: [dec_layers, B, num_queries, hidden_dim], node_embed: [B, num_res, hidden_dim]
        if self._model_conf.model_type == "seq_only":
            hs, node_embed = self.transformer(init_node_embed, attn_mask, self.query_embed.weight, pos_embed)
        else:
            hs, node_embed = self.transformer(init_node_embed, input_feats, attn_mask, self.query_embed.weight, pos_embed)

        if self.aux_loss:
            output_class = self.cls_embed(hs) # [dec_layers, B, num_queries, num_classes + 1]
            mask_embed = self.mask_embed(hs) # [dec_layers, B, num_queries, hidden_dim]
            output_mask = torch.einsum("dbqc,blc->dbql", mask_embed, node_embed)
            out = {"pred_logits": output_class[-1], "pred_masks": output_mask[-1]}
            out["aux_outputs"] = [{"pred_logits": a, "pred_masks": b} for a, b in zip(output_class[:-1], output_mask[:-1])]
        else:
            hs = hs[-1]
            output_class = self.cls_embed(hs) # [B, num_queries, num_classes + 1]
            mask_embed = self.mask_embed(hs) # [B, num_queries, hidden_dim]
            output_mask = torch.einsum("bqc,blc->bql", mask_embed, node_embed) # [B, num_queries, num_res]
            out = {"pred_logits": output_class, "pred_masks": output_mask}
        return out

    def post_process(self, outputs, lengths, mask_thresh=0.5, return_numpy=False):
        """Post-process the outputs to get the final predictions."""
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        final_outputs = []
        for i, num_res in enumerate(lengths):
            pred_prob = pred_logits[i].softmax(-1) # [num_queries, num_classes + 1]
            scores, labels = pred_prob[:, :-1].max(-1) # delete the background/last class
            mask_probs = pred_masks[i][:, :num_res].sigmoid() # [num_queries, num_res]
            masks = (mask_probs > mask_thresh).float() # [num_queries, num_res]
            # multiply average mask prob
            scores *= (masks * mask_probs).sum(-1) / (masks.sum(-1) + 1e-6)
            # sort by scores
            _, indices = scores.sort(descending=True)
            scores = scores[indices]
            labels = labels[indices]
            masks = masks[indices]
            if return_numpy:
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                masks = masks.cpu().numpy()
            final_outputs.append({"scores": scores, "labels": labels, "pocket_masks": masks})
        return final_outputs

 
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the pocket mask error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, num_res] with the predicted pocket masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_pockets] (where num_target_pockets is the number of ground-truth
                           pockets in the protein) containing the class labels
                 "pocket_masks": Tensor of dim [num_target_pockets, num_res] containing the target pocket masks
                 "res_mask": Tensor of dim [num_res] containing the mask for each residue

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_pockets)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_probs = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_masks = outputs["pred_masks"]  # [batch_size, num_queries, num_res]

        ret = []
        for i, target_per_protein in enumerate(targets):
            tgt_ids = target_per_protein["labels"]
            tgt_masks = target_per_protein["pocket_masks"]  # [num_target_pockets, num_res]
            tgt_res_mask = target_per_protein["res_mask"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_probs[i, :, tgt_ids] # [num_queries, num_target_pockets]

            # align the length of prediciton and target
            length = len(tgt_res_mask)
            pred_masks = out_masks[i][:, : length]  # [num_queries, num_res]

            # apply the residue mask
            tgt_masks = tgt_masks[:, tgt_res_mask]
            pred_masks = pred_masks[:, tgt_res_mask]

            cost_mask = batch_sigmoid_ce_loss_jit(pred_masks, tgt_masks)  # [num_queries, num_target_pockets]
            cost_dice = batch_dice_loss_jit(pred_masks, tgt_masks)  # [num_queries, num_target_pockets]

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = C.cpu()

            row_ids, col_ids = linear_sum_assignment(C)
            ret.append((torch.as_tensor(row_ids, dtype=torch.int64), torch.as_tensor(col_ids, dtype=torch.int64)))

        return ret


class SetCriterion(nn.Module):
    """ 
    This class computes the loss for Pocket Detection.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth pockets and predicted pockets
        2) we supervise the model based on this assignment
    """
    def __init__(self, model_conf):
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=model_conf.matcher.cost_class, 
            cost_mask=model_conf.matcher.cost_mask,
            cost_dice=model_conf.matcher.cost_dice
        )
        self.criterion_class = nn.CrossEntropyLoss()
        self.model_conf = model_conf
        self.num_classes = model_conf.num_classes

        self.eos_coef = 0.1 #eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))

        return loss_class, target_classes
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """only calculate the loss for the matched pocket masks"""
        loss_mask = 0
        loss_dice = 0
        for out_masks, t, ind in zip(outputs["pred_masks"], targets, indices):
            target_masks = t["pocket_masks"]    # [num_target_pockets, num_res]
            res_mask = t["res_mask"]
            row_ids, col_ids = ind
            out_masks = out_masks[row_ids]
            out_masks = out_masks[: , : len(res_mask)]   # align the length of prediciton and target
            out_masks = out_masks[:, res_mask]
            target_masks = target_masks[col_ids]
            target_masks = target_masks[:, res_mask]
            loss_mask += sigmoid_ce_loss_jit(out_masks, target_masks, num_masks)
            loss_dice += dice_loss_jit(out_masks, target_masks, num_masks)
        return loss_mask, loss_dice
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / dist.get_world_size(), min=1).item()
        else:
            num_masks = num_masks.item()

        loss_class, target_classes = self.loss_labels(outputs, targets, indices)
        loss_mask, loss_dice = self.loss_masks(outputs, targets, indices, num_masks)
        loss_dict = {"loss_cls": loss_class, "loss_mask": loss_mask, "loss_dice": loss_dice}
        loss_dict["target_classes"] = target_classes
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                loss_class_i, _ = self.loss_labels(aux_outputs, targets, indices)
                loss_mask_i, loss_dice_i = self.loss_masks(aux_outputs, targets, indices, num_masks)
                loss_dict.update({
                    f"loss_cls_{i}": loss_class_i,
                    f"loss_mask_{i}": loss_mask_i,
                    f"loss_dice_{i}": loss_dice_i,
                })
        return loss_dict


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
