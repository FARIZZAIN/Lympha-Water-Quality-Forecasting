# src/train_stae.py
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
from src.stae import STAE
from src.clustering import ClusteringLayer, target_distribution
from src.losses import pred_loss, recon_loss_adj, kl_divergence
from src.data import load_split_data, corr_adjacency_from_series

class ScalarWarmup:
    def __init__(self, start_val: float, end_val: float, warmup_epochs: int):
        self.start = start_val
        self.end = end_val
        self.warm = warmup_epochs
    def value(self, epoch: int) -> float:
        if epoch >= self.warm:
            return self.end
        t = epoch / max(1, self.warm)
        return self.start + t * (self.end - self.start)


def kl_weight_scheduler(epoch, start_epoch=10, max_weight=0.05):
    """Linearly ramp up KL loss weight after a few epochs."""
    if epoch < start_epoch:
        return 0.0
    # Gradually ramp up after start_epoch
    progress = (epoch - start_epoch) / (50 - start_epoch)  # 50 is total epochs (adjust if different)
    return min(progress * max_weight, max_weight)



def build_A_static(cfg: Config, meta, full_values):
    # full_values: (T_total, N)
    S = full_values.shape[0] - cfg.window - cfg.horizon + 1
    idx = train_val_test_split_idx(S, cfg.train_ratio, cfg.val_ratio)
    (a,b) = idx["train"]
    t_train_end = cfg.window + (b - 1)
    raw_train = full_values[:t_train_end+1]
    A_static = corr_adjacency_from_series(raw_train)  # (N, N)
    return A_static

def run_train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Data
    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )
    full_values, cols = load_raw_numeric(cfg.data_path, cfg.target_nodes)
    N = meta["num_nodes"]

    # Static adjacency prior (from TRAIN only)
    raw_train, raw_val, raw_test, meta = load_split_data(
    cfg.data_path,
    cfg.window,
    cfg.horizon,
    cfg.target_nodes,
    cfg.train_ratio,
    cfg.val_ratio
    )

    A_static = corr_adjacency_from_series(
    raw_train,
    keep_abs=False,
    clip_negatives=False
    ).to(device)


    # Models
    temp = TemporalEncoder1D(
    d_out=cfg.d_temporal,
    k=cfg.temporal_kernel,
    n_layers=cfg.temporal_layers
    ).to(device)
    tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(device)
    stae = STAE(d_fused=cfg.d_fused, d_z=cfg.d_z, recon_channels=3).to(device)
    cluster = ClusteringLayer(cfg.num_clusters, cfg.d_z).to(device)
    ar = ARHead(d_in=cfg.d_fused).to(device)

    params = list(temp.parameters()) + list(tcgc.parameters()) + \
             list(stae.parameters()) + list(cluster.parameters()) + list(ar.parameters())
    opt = Adam(params, lr=cfg.lr)
    kl_sched = ScalarWarmup(start_val=0.0, end_val=0.05, warmup_epochs=10)
    from torch.optim.lr_scheduler import StepLR
    lr_sched = StepLR(opt, step_size=20, gamma=0.9)


    def step(loader, train=True, w_kl_now: float = 0.0):
        if train:
            temp.train(); tcgc.train(); stae.train(); cluster.train(); ar.train()
        else:
            temp.eval();  tcgc.eval();  stae.eval();  cluster.eval();  ar.eval()

        tot_pred, tot_recon, tot_kl = 0.0, 0.0, 0.0
        tot_mae, tot_cnt = 0.0, 0

        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            B = X.size(0)
            A_static_batch = A_static.unsqueeze(0).expand(B, N, N)

            with torch.set_grad_enabled(train):
                # Forward backbone
                O = temp(X)                          # (B,N,d_t)
                H_fused, A_dict = tcgc(O, A_static_batch)   # (B,N,d_fused), attn & rl adj
                # AR prediction
                Yhat = ar(H_fused)                   # (B,N)
                loss_pred = pred_loss(Yhat, Y)

                # ST-AE encode + decode
                Z = stae.encode(H_fused)             # (B,N,d_z)
                A_hat_static, A_hat_attn, A_hat_rl = stae.decode_all(Z)
                # Targets for recon: current batch A_static, A_attn, A_rl
                A_targets = [A_static_batch, A_dict["A_attn"].detach(), A_dict["A_rl"].detach()]
                loss_recon = recon_loss_adj(A_targets, [A_hat_static, A_hat_attn, A_hat_rl])

                # Clustering q/p and KL
                q, z_flat = cluster(Z)               # q: (B*N, K)
                p = target_distribution(q).detach()  # stop-grad on p
                loss_kl = kl_divergence(p, q)

                # Total
                loss = cfg.w_pred * loss_pred + cfg.w_recon * loss_recon + w_kl_now * loss_kl
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            mae = (Yhat - Y).abs().mean()
            tot_pred += loss_pred.item() * B
            tot_recon += loss_recon.item() * B
            tot_kl += loss_kl.item() * B
            tot_mae += mae.item() * B
            tot_cnt += B

        return {
            "pred": tot_pred / tot_cnt,
            "recon": tot_recon / tot_cnt,
            "kl": tot_kl / tot_cnt,
            "mae": tot_mae / tot_cnt
        }

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        w_kl_now = kl_sched.value(epoch - 1)

        tr = step(dl_tr, train=True,  w_kl_now=w_kl_now)
        va = step(dl_va, train=False, w_kl_now=w_kl_now)

        lr_sched.step()  # optional

        print(
            f"[epoch {epoch:03d}] "
            f"train: MSE={tr['pred']:.4f} Recon={tr['recon']:.4f} KL={tr['kl']:.4f} MAE={tr['mae']:.4f} | "
            f"val: MSE={va['pred']:.4f} Recon={va['recon']:.4f} KL={va['kl']:.4f} MAE={va['mae']:.4f} "
            f"| w_kl={w_kl_now:.3f}"
        )
        val_score = cfg.w_pred * va["pred"] + cfg.w_recon * va["recon"] + w_kl_now * va["kl"]


        if val_score < best_val:
            best_val = val_score
            torch.save({
                "temp": temp.state_dict(),
                "tcgc": tcgc.state_dict(),
                "stae": stae.state_dict(),
                "cluster": cluster.state_dict(),
                "ar": ar.state_dict(),
                "A_static": A_static.detach().cpu(),
                "meta": meta
            }, "checkpoint_stae.pt")

    # Test best
    ckpt = torch.load("checkpoint_stae.pt", map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    stae.load_state_dict(ckpt["stae"])
    cluster.load_state_dict(ckpt["cluster"])
    ar.load_state_dict(ckpt["ar"])

    te = step(dl_te, train=False)
    print(f"[TEST]  MSE={te['pred']:.4f}  Recon={te['recon']:.4f}  KL={te['kl']:.4f}  MAE={te['mae']:.4f}")
