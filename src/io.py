import os
import glob
import numpy as np
import torch

CHANNEL_LO = 1   
CHANNEL_HI = 0   
N_SAMPLES = 7    

def load_shards(data_dir, split, max_shards=None):
    """Load and concatenate all shards for a given split.

    Parameters
    ----------
    data_dir : str
        Root data directory containing train/, val/, test/ subdirectories.
    split : str
        One of 'train', 'val', 'test'.
    max_shards : int or None
        If set, load at most this many shards (useful for quick debugging).

    Returns
    -------
    dict with keys:
        X      : np.ndarray [M, 7]   -- low-gain sample windows
        y      : np.ndarray [M]      -- normalized low-gain target energies
        y_OF   : np.ndarray [M]      -- OF baseline predictions (unnormalized)
        X_hi   : np.ndarray [M, 7]   -- high-gain sample windows (kept for optional use)
        y_hi   : np.ndarray [M]      -- normalized high-gain target energies
    """
    shard_dir = os.path.join(data_dir, split)
    pattern = os.path.join(shard_dir, f"{split}_*.pt")
    shard_paths = sorted(glob.glob(pattern))

    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found matching {pattern}. "
            f"Check that data_dir='{data_dir}' contains a '{split}/' subdirectory."
        )

    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    all_X, all_y, all_y_OF = [], [], []
    all_X_hi, all_y_hi = [], []

    for path in shard_paths:
        shard = torch.load(path, weights_only=False)
        # X: [N, 2, 7], y: [N, 2], y_OF: [N, 2]
        X_tensor = shard["X"]
        y_tensor = shard["y"]
        y_OF_tensor = shard["y_OF"]
        all_X.append(X_tensor[:, CHANNEL_LO, :].numpy())
        all_y.append(y_tensor[:, CHANNEL_LO].numpy())
        all_y_OF.append(y_OF_tensor[:, CHANNEL_LO].numpy())
        all_X_hi.append(X_tensor[:, CHANNEL_HI, :].numpy())
        all_y_hi.append(y_tensor[:, CHANNEL_HI].numpy())

    return {
        "X": np.concatenate(all_X, axis=0),
        "y": np.concatenate(all_y, axis=0),
        "y_OF": np.concatenate(all_y_OF, axis=0),
        "X_hi": np.concatenate(all_X_hi, axis=0),
        "y_hi": np.concatenate(all_y_hi, axis=0),
    }


def load_y_stats(data_dir):
    """Load target normalization statistics.

    Returns
    -------
    dict with keys:
        mean_lo : float
        std_lo  : float
        mean_hi : float
        std_hi  : float
    """
    path = os.path.join(data_dir, "y_stats.npz")
    stats = np.load(path)
    # stats['mean'] shape [1, 2], stats['std'] shape [1, 2]

    return {
        "mean_lo": float(stats["mean"][0, CHANNEL_LO]),
        "std_lo": float(stats["std"][0, CHANNEL_LO]),
        "mean_hi": float(stats["mean"][0, CHANNEL_HI]),
        "std_hi": float(stats["std"][0, CHANNEL_HI]),
    }


def denormalize(y_norm, mean, std):
    """Convert normalized targets back to physical energy units.

    Parameters
    ----------
    y_norm : np.ndarray
        Normalized target values.
    mean : float
        Mean used for normalization.
    std : float
        Std used for normalization.

    Returns
    -------
    np.ndarray
        Physical energy values.
    """
    
    return y_norm * std + mean
