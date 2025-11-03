import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import Config
from src.utils import set_seed, train_val_test_split_idx
from src.data import make_dataloaders, read_long_and_pivot
from src.adjacencies import corr_adjacency_from_series
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.stae import STAE
from src.clustering import ClusteringLayer, target_distribution
from src.ar_head import ARHead
from src.losses import pred_loss, recon_loss_adj, kl_divergence
from src.rl_dqn import DQNAgent, DQNConfig, build_adj_from_pairs

class ScalarWarmup:
    def __init__(self, start_val: float, end_val: float, warmup_epochs: int):
        self.start = start_val; self.end = end_val; self.warm = warmup_epochs
    def value(self, epoch: int) -> float:
        if epoch >= self.warm: return self.end
        t = epoch / max(1, self.warm)
        return self.start + t * (self.end - self.start)

def _build_A_static_from_train(cfg: Config, meta) -> torch.Tensor:
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

    S = values.shape[0] - cfg.window - cfg.horizon + 1
    (a, b) = train_val_test_split_idx(S, cfg.train_ratio, cfg.val_ratio)["train"]
    t_train_end = cfg.window + (b - 1)
    raw_train = values[: t_train_end + 1]
    return corr_adjacency_from_series(raw_train, keep_abs=False, clip_negatives=False)

def run_train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

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
    A_static = _build_A_static_from_train(cfg, meta).to(device)

    temp = TemporalEncoder1D(d_out=cfg.d_temporal, k=cfg.temporal_kernel, n_layers=cfg.temporal_layers).to(device)
    tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(device)
    stae = STAE(d_fused=cfg.d_fused, d_z=cfg.d_z, recon_channels=3).to(device)
    cluster = ClusteringLayer(cfg.num_clusters, cfg.d_z).to(device)
    ar = ARHead(d_in=cfg.d_fused).to(device)

    dqn_cfg = DQNConfig(
        eps_start=cfg.dqn_eps_start,
        eps_end=cfg.dqn_eps_end,
        eps_decay_steps=cfg.dqn_eps_decay_steps,
        top_k=cfg.dqn_top_k
    )
    agent = DQNAgent(d_in=cfg.d_temporal, N_nodes=N, cfg=dqn_cfg).to(device)

    params = list(temp.parameters()) + list(tcgc.parameters()) + list(stae.parameters()) + list(cluster.parameters()) + list(ar.parameters())
    opt = Adam(params, lr=cfg.lr)
    kl_sched = ScalarWarmup(start_val=cfg.w_kl, end_val=0.05, warmup_epochs=10)
    from torch.optim.lr_scheduler import StepLR
    lr_sched = StepLR(opt, step_size=20, gamma=0.9)

    def step(loader, train=True, w_kl_now: float = 0.0):
        if train: temp.train(); tcgc.train(); stae.train(); cluster.train(); ar.train()
        else:     temp.eval();  tcgc.eval();  stae.eval();  cluster.eval();  ar.eval()

        tot_pred = tot_recon = tot_kl = tot_mae = tot_rl = 0.0
        tot_cnt = 0

        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            B = X.size(0)
            A_b = A_static.unsqueeze(0).expand(B, N, N)

            O = temp(X)

            Q_all, picks, mask_idx = agent.select_pairs(O, device)
            A_rl_b = torch.stack([build_adj_from_pairs(N, picks[b], device) for b in range(B)], dim=0)

            H, A_dict = tcgc(O, A_b, A_rl=A_rl_b)
            Yhat = ar(H)
            loss_pred = pred_loss(Yhat, Y)
            mae = (Yhat - Y).abs().mean()

            Z = stae.encode(H)
            A_hat_s, A_hat_at, A_hat_rl = stae.decode_all(Z)
            A_targets = [A_b, A_dict["A_attn"].detach(), A_rl_b.detach()]
            loss_recon = recon_loss_adj(A_targets, [A_hat_s, A_hat_at, A_hat_rl])

            q, _ = cluster(Z)
            p = target_distribution(q).detach()
            loss_kl = kl_divergence(p, q)

            loss_backbone = cfg.w_pred * loss_pred + cfg.w_recon * loss_recon + w_kl_now * loss_kl
            if train:
                opt.zero_grad(); loss_backbone.backward(); opt.step()

                r = (-loss_pred.detach()).repeat(B)
                rl_loss = agent.learn(O.detach(), mask_idx, r)
            else:
                rl_loss = 0.0

            tot_pred += float(loss_pred.item()) * B
            tot_recon += float(loss_recon.item()) * B
            tot_kl += float(loss_kl.item()) * B
            tot_mae += float(mae.item()) * B
            tot_rl += float(rl_loss) * B
            tot_cnt += B

        return {"pred": tot_pred/tot_cnt, "recon": tot_recon/tot_cnt, "kl": tot_kl/tot_cnt,
                "mae": tot_mae/tot_cnt, "rl": tot_rl/max(1, tot_cnt)}

    best_val = float("inf")
    for epoch in range(1, cfg.epochs+1):
        w_kl_now = kl_sched.value(epoch - 1)
        tr = step(dl_tr, train=True,  w_kl_now=w_kl_now)
        va = step(dl_va, train=False, w_kl_now=w_kl_now)
        lr_sched.step()

        print(f"[epoch {epoch:03d}] train: MSE={tr['pred']:.4f} Recon={tr['recon']:.4f} KL={tr['kl']:.4f} RL={tr['rl']:.4f} MAE={tr['mae']:.4f} | "
              f"val: MSE={va['pred']:.4f} Recon={va['recon']:.4f} KL={va['kl']:.4f} MAE={va['mae']:.4f} "
              f"| w_kl={w_kl_now:.3f} lr={lr_sched.get_last_lr()[0]:.2e}")

        if va["pred"] < best_val:
            best_val = va["pred"]
            torch.save({
                "temp": temp.state_dict(),
                "tcgc": tcgc.state_dict(),
                "stae": stae.state_dict(),
                "cluster": cluster.state_dict(),
                "ar": ar.state_dict(),
                "agent": agent.state_dict(),
                "A_static": A_static.detach().cpu(),
                "meta": meta
            }, "checkpoint_rl.pt")

    ckpt = torch.load("checkpoint_rl.pt", map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    stae.load_state_dict(ckpt["stae"])
    cluster.load_state_dict(ckpt["cluster"])
    ar.load_state_dict(ckpt["ar"])
    agent.load_state_dict(ckpt["agent"])

    te = step(dl_te, train=False)
    print(f"[TEST]  MSE={te['pred']:.4f} Recon={te['recon']:.4f} KL={te['kl']:.4f} MAE={te['mae']:.4f}")

if __name__ == "__main__":
    run_train(Config())
