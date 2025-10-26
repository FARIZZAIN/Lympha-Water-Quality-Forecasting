# src/train_min.py
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import Config
from src.utils import set_seed, train_val_test_split_idx
from src.data import make_dataloaders, load_raw_numeric
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.ar_head import ARHead
from src.adjacencies import corr_adjacency_from_series

def run_train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Data loaders (already standardized per train split)
    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    N = meta["num_nodes"]

    # Build static adjacency from RAW TRAIN series (no windows)
    full_values, cols = load_raw_numeric(cfg.data_path, cfg.target_nodes)  # (T_total, N)
    # compute window sample count --> S
    S = full_values.shape[0] - cfg.window - cfg.horizon + 1
    idx = train_val_test_split_idx(S, cfg.train_ratio, cfg.val_ratio)
    (a,b) = idx["train"]  # these are in windowed-sample space

    # Convert window-sample indices back to row ranges in raw series for A_static:
    # windows cover t in [window ... window+S-1], so train uses raw rows [0 ... window+(b-1)]
    t_train_end = cfg.window + (b - 1)
    raw_train = full_values[:t_train_end+1]  # inclusive
    A_static = corr_adjacency_from_series(
    raw_train, keep_abs=False, clip_negatives=False
    )
  # (N, N)
    A_static = A_static.to(device)

    # Model
    temp = TemporalEncoder1D(
    d_out=cfg.d_temporal,
    k=cfg.temporal_kernel,
    n_layers=cfg.temporal_layers
)
    tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(device)
    ar   = ARHead(d_in=cfg.d_fused).to(device)

    params = list(temp.parameters()) + list(tcgc.parameters()) + list(ar.parameters())
    opt = Adam(params, lr=cfg.lr)
    crit = nn.MSELoss()

    def epoch_loop(loader, train=True):
        if train: temp.train(); tcgc.train(); ar.train()
        else:     temp.eval();  tcgc.eval();  ar.eval()

        total_loss, total_mae, total_cnt = 0.0, 0.0, 0
        for X, Y in loader:
            X = X.to(device)   # (B,N,T)
            Y = Y.to(device)   # (B,N)

            with torch.set_grad_enabled(train):
                O = temp(X)                                 # (B,N,d_t)
                B, Nn, _ = O.shape
                A_static_batch = A_static.unsqueeze(0).expand(B, N, N)  # (B,N,N)
                H, _ = tcgc(O, A_static_batch)              # (B,N,d_fused)
                Yhat = ar(H)                                # (B,N)

                loss = crit(Yhat, Y)
                mae = (Yhat - Y).abs().mean()

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss += loss.item() * X.size(0)
            total_mae  += mae.item()  * X.size(0)
            total_cnt  += X.size(0)

        return total_loss/total_cnt, total_mae/total_cnt

    best_val = float("inf")
    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_mae = epoch_loop(dl_tr, train=True)
        va_loss, va_mae = epoch_loop(dl_va, train=False)
        print(f"[epoch {epoch:03d}] train MSE={tr_loss:.4f} MAE={tr_mae:.4f} | val MSE={va_loss:.4f} MAE={va_mae:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "temp": temp.state_dict(),
                "tcgc": tcgc.state_dict(),
                "ar": ar.state_dict(),
                "A_static": A_static.detach().cpu(),
                "meta": meta
            }, "checkpoint_min.pt")

    # Test with the best checkpoint
    ckpt = torch.load("checkpoint_min.pt", map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    ar.load_state_dict(ckpt["ar"])
    test_mse, test_mae = epoch_loop(dl_te, train=False)
    print(f"[TEST]  MSE={test_mse:.4f}  MAE={test_mae:.4f}")

if __name__ == "__main__":
    run_train(Config())
