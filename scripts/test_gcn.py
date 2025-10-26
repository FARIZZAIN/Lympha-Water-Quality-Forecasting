# scripts/test_gcn.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.config import Config
from src.data import make_dataloaders
from src.encoders import TemporalEncoder1D
from src.gcn import GraphConv, AttentionAdjacency

def main():
    cfg = Config()
    dl_tr, _, _, _ = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    xb, _ = next(iter(dl_tr))  # (B,N,T)

    # Temporal encoder
    temp = TemporalEncoder1D(
        d_out=cfg.d_temporal,
        k=cfg.temporal_kernel,
        n_layers=cfg.temporal_layers
    )
    with torch.no_grad():
        O = temp(xb)  # (B,N,d_temporal)

    # Attention adjacency
    attn_adj = AttentionAdjacency(cfg.d_temporal)
    with torch.no_grad():
        A = attn_adj(O)  # (B,N,N)

    # Static adjacency (identity for quick check)
    N = O.shape[1]
    A_static = torch.eye(N).unsqueeze(0).expand(O.shape[0], N, N)

    # GCN on both
    gcn = GraphConv(cfg.d_temporal, cfg.d_gcn)
    with torch.no_grad():
        O_static = gcn(O, A_static)
        O_attn   = gcn(O, A)

    print("O (B,N,d_in):            ", tuple(O.shape))
    print("A_attn (B,N,N):          ", tuple(A.shape))
    print("GCN static out (B,N,d):  ", tuple(O_static.shape))
    print("GCN attn out   (B,N,d):  ", tuple(O_attn.shape))
    print("OK.")

if __name__ == "__main__":
    main()
