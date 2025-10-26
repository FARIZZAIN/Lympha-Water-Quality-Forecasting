# src/tcgc.py
import torch
import torch.nn as nn
from src.gcn import GraphConv, AttentionAdjacency

class TCGCBlock(nn.Module):
    """
    Triple-Channel Graph Convolution with 1x1 fusion.
      - Channel 1: Static adjacency (e.g., identity/correlation)
      - Channel 2: Attention adjacency (learned from features)
      - Channel 3: Placeholder for RL-learned adjacency (we'll add RL later)
    """
    def __init__(self, d_in: int, d_gcn: int = 32, d_fused: int = 64, use_rl_channel: bool = True):
        super().__init__()
        self.use_rl = use_rl_channel

        # One GCN per channel
        self.gcn_static = GraphConv(d_in, d_gcn)
        self.gcn_attn   = GraphConv(d_in, d_gcn)
        if self.use_rl:
            self.gcn_rl = GraphConv(d_in, d_gcn)

        # Attention adjacency builder
        self.attn_adj = AttentionAdjacency(d_in)

        # 1x1 fusion (implemented as a per-node linear mix across channels)
        in_fuse = d_gcn * (3 if self.use_rl else 2)
        self.fuse = nn.Linear(in_fuse, d_fused)

    def forward(self, O: torch.Tensor, A_static: torch.Tensor, A_rl: torch.Tensor | None = None):
        """
        O:        (B, N, d_in)   node features from temporal encoder
        A_static: (N, N) or (B, N, N)  prior/static adjacency
        A_rl:     (N, N) or (B, N, N)  RL adjacency (optional, can be None for now)
        returns H_fused: (B, N, d_fused)
        """
        B, N, _ = O.shape

        # Channel 2: attention adjacency is built from O
        A_attn = self.attn_adj(O)             # (B, N, N)

        # Run GCNs
        H_static = self.gcn_static(O, A_static)  # (B, N, d_gcn)
        H_attn   = self.gcn_attn(O, A_attn)      # (B, N, d_gcn)

        feats = [H_static, H_attn]

        if self.use_rl:
            if A_rl is None:
                # Placeholder: if RL not ready yet, reuse attention adjacency
                A_rl = A_attn.detach()  # no grad through attn for the rl channel placeholder
            H_rl = self.gcn_rl(O, A_rl)     # (B, N, d_gcn)
            feats.append(H_rl)

        # Concatenate along feature dim and fuse via 1x1 (Linear)
        H_concat = torch.cat(feats, dim=-1)      # (B, N, d_gcn * C)
        H_fused  = self.fuse(H_concat)           # (B, N, d_fused)
        return H_fused, {"A_attn": A_attn, "A_rl": A_rl}
