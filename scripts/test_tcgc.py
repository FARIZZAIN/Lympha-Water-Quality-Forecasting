# scripts/test_tcgc.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.config import Config
from src.data import make_dataloaders
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.ar_head import ARHead

def main():
    cfg = Config()
    dl_tr, _, _, _ = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    xb, yb = next(iter(dl_tr))  # xb: (B, N, T), yb: (B, N)

    # Temporal encoder
    temp = TemporalEncoder1D(d_out=cfg.d_temporal, k=3, n_layers=1)
    O = temp(xb)  # (B, N, d_temporal)

    # Static adjacency: identity for now (we'll add correlation-based soon)
    B, N, _ = O.shape
    A_static = torch.eye(N).unsqueeze(0).expand(B, N, N)  # (B, N, N)

    # TCGC fusion
    tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True)
    H_fused, A_dict = tcgc(O, A_static)  # (B, N, d_fused)

    # AR head
    ar = ARHead(d_in=cfg.d_fused)
    yhat = ar(H_fused)   # (B, N)

    print("O (temporal) shape:   ", tuple(O.shape))
    print("H_fused (TCGC) shape: ", tuple(H_fused.shape))
    print("yhat (AR) shape:      ", tuple(yhat.shape))
    print("target Y shape:       ", tuple(yb.shape))
    print("Attn A shape:         ", tuple(A_dict['A_attn'].shape))
    print("RL A (placeholder) shape:", tuple(A_dict['A_rl'].shape))
    print("OK (forward wired).")

if __name__ == "__main__":
    main()
