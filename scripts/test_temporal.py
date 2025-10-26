# scripts/test_temporal.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.config import Config
from src.data import make_dataloaders
from src.encoders import TemporalEncoder1D

def main():
    cfg = Config()
    dl_tr, _, _, _ = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    xb, yb = next(iter(dl_tr))  # xb:(B,N,T) yb:(B,N)

    enc = TemporalEncoder1D(
        d_out=cfg.d_temporal,
        k=cfg.temporal_kernel,
        n_layers=cfg.temporal_layers
    )
    with torch.no_grad():
        O = enc(xb)  # (B,N,d_temporal)

    print("Input X shape  (B,N,T):", tuple(xb.shape))
    print("Target Y shape (B,N):  ", tuple(yb.shape))
    print("Encoded O shape(B,N,d):", tuple(O.shape))
    print("OK.")

if __name__ == "__main__":
    main()
