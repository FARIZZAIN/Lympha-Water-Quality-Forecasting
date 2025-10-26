import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import train_val_test_split_idx

class TimeGraphDataset(Dataset):
    def __init__(self, X_windows: np.ndarray, Y_next: np.ndarray):
        """
        X_windows: (S, N, T)
        Y_next:    (S, N)  # next-step targets per node
        """
        self.X = X_windows.astype(np.float32)
        self.Y = Y_next.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

def _pick_columns(df: pd.DataFrame, target_nodes: tuple[str, ...] | None):
    if target_nodes and all(c in df.columns for c in target_nodes):
        return list(target_nodes)
    # else: pick numeric columns and drop obvious time cols
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    drop_like = {"time","date","stamp"}
    cols = [c for c in cols if not any(k in c.lower() for k in drop_like)]
    if len(cols) < 2:
        raise ValueError("Could not find enough numeric columns for nodes.")
    return cols

def build_windows(values: np.ndarray, window: int, horizon: int):
    # values: (T_total, N)
    T_total, N = values.shape
    X, Y = [], []
    for t in range(window, T_total - horizon + 1):
        X.append(values[t-window:t, :].T)        # (N, T)
        Y.append(values[t + horizon - 1, :])     # (N,)
    X = np.stack(X, axis=0)                      # (S, N, T)
    Y = np.stack(Y, axis=0)                      # (S, N)
    return X, Y

def make_dataloaders(csv_path: str, window: int, horizon: int,
                     target_nodes: tuple[str, ...] | None,
                     batch_size=64, train_ratio=0.7, val_ratio=0.15):
    df = pd.read_csv(csv_path)
    cols = _pick_columns(df, target_nodes)
    values = df[cols].to_numpy()                 # (T_total, N)
    X, Y = build_windows(values, window, horizon)
    S = X.shape[0]
    idx = train_val_test_split_idx(S, train_ratio, val_ratio)
    (a,b), (c,d), (e,f) = idx["train"], idx["val"], idx["test"]

    # Standardize using train split only (per-node)
    mean = X[a:b].mean(axis=(0,2), keepdims=True)   # (1, N, 1)
    std  = X[a:b].std(axis=(0,2), keepdims=True) + 1e-8
    X = (X - mean) / std
    # apply same transform to Y (per-node)
    Y = (Y - mean.squeeze(axis=(0,2))) / std.squeeze(axis=(0,2))

    ds_tr = TimeGraphDataset(X[a:b], Y[a:b])
    ds_va = TimeGraphDataset(X[c:d], Y[c:d])
    ds_te = TimeGraphDataset(X[e:f], Y[e:f])

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False)

    meta = {
        "num_nodes": X.shape[1],
        "window": X.shape[2],
        "cols": cols,
        "train_samples": b-a,
        "val_samples": d-c,
        "test_samples": f-e,
        "mean": mean.squeeze((0,2)).tolist(),
        "std": std.squeeze((0,2)).tolist(),
    }
    return dl_tr, dl_va, dl_te, meta

def load_raw_numeric(csv_path: str, target_nodes: tuple[str, ...] | None):
    """
    Return raw numeric matrix (T_total, N) and chosen column names.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if target_nodes and all(c in df.columns for c in target_nodes):
        cols = list(target_nodes)
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        drop_like = {"time","date","stamp"}
        cols = [c for c in cols if not any(k in c.lower() for k in drop_like)]
    values = df[cols].to_numpy()
    return values, cols
