# src/adjacencies.py
import numpy as np
import torch

def corr_adjacency_from_series(values_train: np.ndarray, eps: float = 1e-8,
                               keep_abs: bool = True, zero_diag: bool = True,
                               clip_negatives: bool = True) -> torch.Tensor:
    """
    values_train: array (T_train, N) raw (un-windowed) node series from TRAIN RANGE ONLY.
    Returns A_static: (N, N) torch.FloatTensor.

    Steps:
      1) Pearson corr over time per node pair
      2) |corr| if keep_abs else corr
      3) zero diagonal
      4) clip negatives to 0 (optional)
      5) symmetrize
      6) add small eps + row-normalize (not GCN norm; we’ll GCN-normalize in the layer)
    """
    # 1) correlation
    C = np.corrcoef(values_train.T)  # (N, N)
    if keep_abs:
        C = np.abs(C)
    if zero_diag:
        np.fill_diagonal(C, 0.0)
    if clip_negatives:
        C = np.maximum(C, 0.0)

    # 5) symmetrize
    C = 0.5 * (C + C.T)

    # 6) small eps to avoid zero rows; row normalize
    C = C + eps
    row_sums = C.sum(axis=1, keepdims=True)
    C = C / row_sums

    return torch.from_numpy(C.astype(np.float32))  # (N, N)
