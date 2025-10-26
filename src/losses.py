# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

mse = nn.MSELoss(reduction="mean")

def pred_loss(yhat, y):
    return mse(yhat, y)

def recon_loss_adj(targets, preds):
    """
    targets: list of A_true tensors [(B,N,N), ...]
    preds:   list of A_hat tensors  [(B,N,N), ...]
    """
    loss = 0.0
    for At, Ah in zip(targets, preds):
        loss = loss + mse(Ah, At)
    return loss

def kl_divergence(p, q):
    """
    KL(P || Q), averaged over samples
    p, q: (M, K)
    """
    p = torch.clamp(p, 1e-8, 1.0)
    q = torch.clamp(q, 1e-8, 1.0)
    kl = (p * (p.log() - q.log())).sum(dim=1).mean()
    return kl
