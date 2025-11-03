from dataclasses import asdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import Config
from src.utils import set_seed, train_val_test_split_idx
from src.data import make_dataloaders, read_long_and_pivot
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.ar_head import ARHead
from src.adjacencies import corr_adjacency_from_series

def _build_A_static_from_train(cfg: Config, meta) -> torch.Tensor:
    """
    Build static adjacency using TRAIN RANGE ONLY, from raw (unstandardized) values.
    """
    # read raw wide matrix (T, N) like dataloader did
    if cfg.is_long_format:
        values, cols = read_long_and_pivot(
            cfg.data_path,
            option=cfg.long_option,
            node_id_filter=cfg.node_id_filter,
            resample_rule=cfg.resample_rule,
            impute=cfg.impute
        )
    else:
        df = pd.read_csv(cfg.data_path)
        cols = list(cfg.target_nodes) if cfg.target_nodes else \
            [c for c in df.columns if df[c].dtype != object]
        values = df[cols].to_numpy()

    # windowed samples S and train slice end in raw index
    S = values.shape[0] - cfg.window - cfg.horizon + 1
    (a, b) = train_val_test_split_idx(S, cfg.train_ratio, cfg.val_ratio)["train"]
    t_train_end = cfg.window + (b - 1)
    raw_train = values[: t_train_end + 1]  # inclusive

    A_static = corr_adjacency_from_series(
        raw_train, keep_abs=False, clip_negatives=False
    )
    return A_static

def run_train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # dataloaders (standardized)
    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio,
        is_long_format=cfg.is_long_format,
        long_option=cfg.long_option,
        node_id_filter=cfg.node_id_filter,
        resample_rule=cfg.resample_rule,
        impute=cfg.impute
    )
    N = meta["num_nodes"]

    # A_static from TRAIN
    A_static = _build_A_static_from_train(cfg, meta).to(device)

    # model
    temp = TemporalEncoder1D(
        d_out=cfg.d_temporal,
        k=cfg.temporal_kernel,
        n_layers=cfg.temporal_layers
    ).to(device)
    tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(device)
    ar   = ARHead(d_in=cfg.d_fused).to(device)

    params = list(temp.parameters()) + list(tcgc.parameters()) + list(ar.parameters())
    opt = Adam(params, lr=cfg.lr)
    crit = nn.MSELoss()

    def epoch_loop(loader, train=True):
        if train: temp.train(); tcgc.train(); ar.train()
        else:     temp.eval();  tcgc.eval();  ar.eval()

        total_loss = total_mae = total_cnt = 0.0
        for X, Y in loader:
            X = X.to(device); Y = Y.to(device)
            with torch.set_grad_enabled(train):
                O = temp(X)
                B = X.size(0)
                A_b = A_static.unsqueeze(0).expand(B, N, N)
                H, _ = tcgc(O, A_b)
                Yhat = ar(H)
                loss = crit(Yhat, Y)
                mae  = (Yhat - Y).abs().mean()
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * B
            total_mae  += mae.item()  * B
            total_cnt  += B
        return total_loss/total_cnt, total_mae/total_cnt

    best_val = float("inf")
    for epoch in range(1, cfg.epochs+1):
        tr_mse, tr_mae = epoch_loop(dl_tr, train=True)
        va_mse, va_mae = epoch_loop(dl_va, train=False)
        print(f"[epoch {epoch:03d}] train MSE={tr_mse:.4f} MAE={tr_mae:.4f} | val MSE={va_mse:.4f} MAE={va_mae:.4f}")
        if va_mse < best_val:
            best_val = va_mse
            torch.save({
                "temp": temp.state_dict(),
                "tcgc": tcgc.state_dict(),
                "ar": ar.state_dict(),
                "A_static": A_static.detach().cpu(),
                "meta": meta
            }, "checkpoint_min.pt")

    # test
    ckpt = torch.load("checkpoint_min.pt", map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    ar.load_state_dict(ckpt["ar"])
    te_mse, te_mae = epoch_loop(dl_te, train=False)
    print(f"[TEST]  MSE={te_mse:.4f}  MAE={te_mae:.4f}")

if __name__ == "__main__":
    run_train(Config())
