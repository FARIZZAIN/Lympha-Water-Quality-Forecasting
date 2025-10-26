# src/clustering.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def student_t_distribution(z, centers, alpha=1.0):
    """
    z: (M, d) flattened embeddings
    centers: (K, d) cluster centers
    returns q: (M, K)
    """
    M, d = z.shape
    K, _ = centers.shape
    # pairwise squared distances
    # (M,1,d) - (1,K,d) -> (M,K,d) -> sum d
    dist = torch.cdist(z, centers, p=2.0) ** 2   # (M,K)
    num = (1.0 + dist / alpha) ** (-(alpha + 1) / 2)
    q = num / (num.sum(dim=1, keepdim=True) + 1e-12)
    return q

def target_distribution(q):
    """
    DEC target p from soft assignments q.
    """
    f = q.sum(dim=0, keepdim=True)           # (1,K)
    p = (q ** 2) / (f + 1e-12)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
    return p

class ClusteringLayer(nn.Module):
    """
    Learnable cluster centers trained via KL loss.
    """
    def __init__(self, n_clusters: int, d_z: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_clusters, d_z) * (1.0 / (d_z ** 0.5)))

    def forward(self, Z: torch.Tensor):
        """
        Z: (B, N, d_z)
        returns q: (B*N, K)
        """
        B, N, d = Z.shape
        z_flat = Z.reshape(B * N, d)     # (M, d)
        q = student_t_distribution(z_flat, self.centers)  # (M, K)
        return q, z_flat
