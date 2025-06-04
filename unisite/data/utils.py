import io
import os
from typing import Any
import numpy as np
import pickle
import torch


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)


UNPADDED_FEATS = [
    'length', 'target',
]


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats


def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f'Invalid pad amount {pad_amt}')
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)


def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        logger=None,
        use_torch=True,
        save_interval=50000,
        metrics=None,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
        save_interval: Interval of steps to save checkpoints.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Delete non-interval .pth files (retain only the ones at each save_interval step)
    for fname in os.listdir(ckpt_dir):
        if "best" in fname:
            continue
        if fname.endswith('.pth'):
            step_in_fname = ''.join(filter(str.isdigit, fname))  # Extract step number from filename
            if step_in_fname and int(step_in_fname) % save_interval != 0:
                os.remove(os.path.join(ckpt_dir, fname))

    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
        },
        use_torch=use_torch)

move_to_np = lambda x: x.cpu().detach().numpy()
