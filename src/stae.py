# src/stae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class STAE(nn.Module):
    """
    Spatio-Temporal Autoencoder:
      Encoder: (B, N, d_fused) -> (B, N, d_z)
      Decoder: reconstruct 3 adjacencies via Z W Z^T (with sigmoid)
    """
    def __init__(self, d_fused: int, d_z: int, recon_channels: int = 3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_fused, d_fused),
            nn.ReLU(inplace=True),
            nn.Linear(d_fused, d_z)
        )
        self.recon_channels = recon_channels
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(d_z, d_z) * (1.0 / (d_z ** 0.5)))
            for _ in range(recon_channels)
        ])

    def encode(self, H_fused: torch.Tensor) -> torch.Tensor:
        # H_fused: (B, N, d_fused) -> Z: (B, N, d_z)
        return self.enc(H_fused)

    def decode_all(self, Z: torch.Tensor):
        """
        Z: (B, N, d_z)
        returns list of reconstructed adjacencies [(B,N,N), ...]
        """
        B, N, d = Z.shape
        Zt = Z.transpose(1, 2)  # (B, d, N)
        outs = []
        for k in range(self.recon_channels):
            Wk = self.W[k]                       # (d, d)
            # (B,N,d) @ (d,d) -> (B,N,d); then @ (B,d,N) -> (B,N,N)
            Zw = torch.matmul(Z, Wk)             # (B,N,d)
            logits = torch.matmul(Zw, Zt)        # (B,N,N)
            A_hat = torch.sigmoid(logits)        # keep in (0,1)
            outs.append(A_hat)
        return outs
