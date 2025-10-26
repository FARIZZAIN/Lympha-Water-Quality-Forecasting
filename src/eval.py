# src/eval.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.metrics import mae, rmse, rse, corr
from src.data import make_dataloaders
from src.config import Config
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.stae import STAE
from src.clustering import ClusteringLayer, target_distribution
from src.ar_head import ARHead
from src.adjacencies import corr_adjacency_from_series

@torch.no_grad()
def inverse_standardize(Y_std: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Y_std: (B, N) tensor in z-score space
    mean, std: arrays of length N (from meta)
    returns: (B, N) numpy in original units
    """
    mean = torch.tensor(mean, device=Y_std.device).view(1, -1)
    std  = torch.tensor(std,  device=Y_std.device).view(1, -1)
    Y = Y_std * std + mean
    return Y.cpu().numpy()

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, cfg: Config, use_rl: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (we need meta for inverse scaling)
    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    mean, std = np.array(meta["mean"]), np.array(meta["std"])
    N = meta["num_nodes"]

    # Build A_static (same way as in training scripts)
    full = None  # we won’t need it unless you want to recompute static
    A_static = torch.eye(N, dtype=torch.float32, device=device)

    # Models
    temp = TemporalEncoder1D(
    d_out=cfg.d_temporal,
    k=cfg.temporal_kernel,
    n_layers=cfg.temporal_layers
    ).to(device)
    tcgc = TCGCBlock(
    d_in=cfg.d_temporal,
    d_gcn=cfg.d_gcn,
    d_fused=cfg.d_fused,
    use_rl_channel=True
    ).to(device)

    stae = STAE(d_fused=cfg.d_fused, d_z=cfg.d_z, recon_channels=3).to(device)
    cluster = ClusteringLayer(cfg.num_clusters, cfg.d_z).to(device)
    ar = ARHead(d_in=cfg.d_fused).to(device)


    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    stae.load_state_dict(ckpt["stae"])
    cluster.load_state_dict(ckpt["cluster"])
    ar.load_state_dict(ckpt["ar"])
    A_static = ckpt.get("A_static", A_static).to(device)

    temp.eval(); tcgc.eval(); stae.eval(); cluster.eval(); ar.eval()

    # Collect predictions on test
    Yhat_list, Y_list = [], []
    for X, Y in dl_te:
        X = X.to(device)  # (B,N,T)
        Y = Y.to(device)  # (B,N)

        O = temp(X)
        B = X.size(0)
        A_static_b = A_static.unsqueeze(0).expand(B, N, N)

        H, A_dict = tcgc(O, A_static_b)  # RL adjacency in tcgc was placeholder during eval
        Yhat = ar(H)  # (B,N)

        # store standardized
        Yhat_list.append(Yhat)
        Y_list.append(Y)

    Yhat_std = torch.cat(Yhat_list, dim=0)  # (S_te, N)
    Y_std    = torch.cat(Y_list, dim=0)     # (S_te, N)

    # inverse-standardize to original units
    Yhat = inverse_standardize(Yhat_std, mean, std)
    Ytrue = inverse_standardize(Y_std, mean, std)

    with torch.no_grad():
        # after inverse_standardize(...)
        # After you have Yhat_std and Y_std (both torch tensors)
        # after inverse_standardize(...)
        Yhat_t = torch.from_numpy(Yhat).float()
        Ytrue_t = torch.from_numpy(Ytrue).float()

# then pass tensors to metrics
        pred = Yhat_t
        true = Ytrue_t


        from src.metrics import mae, rmse, rse, corr

        mae_val = mae(pred, true)
        rmse_val = rmse(pred, true)
        rse_val  = rse(pred, true)
        corr_val = corr(pred, true)

    print(f"[TEST]  MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  RSE={rse_val:.4f}  CORR={corr_val:.4f}")

    node_names = ["chlorine", "phosphorus", "nitrate", "sulphate"]

    for i, name in enumerate(node_names):
        p = pred[:, i]
        t = true[:, i]

        node_mae = mae(p, t)
        node_rmse = rmse(p, t)
        node_rse = rse(p, t)
        node_corr = corr(p, t)

        print(f"{name:>12}: MAE={node_mae:.4f}, RMSE={node_rmse:.4f}, RSE={node_rse:.4f}, CORR={node_corr:.4f}")


    # metrics per node
    mae_per = np.mean(np.abs(Yhat - Ytrue), axis=0)
    rmse_per = np.sqrt(np.mean((Yhat - Ytrue)**2, axis=0))
    mae = mae_per.mean(); rmse = rmse_per.mean()

    return {
        "cols": meta["cols"],
        "Yhat": Yhat, "Ytrue": Ytrue,
        "mae_per": mae_per, "rmse_per": rmse_per,
        "mae": mae, "rmse": rmse
    }

def plot_series(y_true: np.ndarray, y_pred: np.ndarray, title: str, savepath: str, last: int | None = 200):
    """
    y_true, y_pred: (T, ) arrays
    """
    if last is not None and last < len(y_true):
        y_true = y_true[-last:]
        y_pred = y_pred[-last:]

    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Test time index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
