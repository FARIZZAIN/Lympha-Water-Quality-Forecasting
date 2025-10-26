from dataclasses import dataclass

@dataclass
class Config:
    # data
    data_path: str = "data/raw/synthetic_water_quality.csv"
    target_nodes: tuple[str, ...] = ("chlorine", "phosphorus", "sulphate", "nitrate")
    window: int = 14          # try 14 later
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # model dims
    d_temporal: int = 64      # set to 64 for your experiment
    temporal_kernel: int = 5  # set to 5
    temporal_layers: int = 2  # set to 2
    d_gcn: int = 32
    d_fused: int = 96         # set to 96
    d_z: int = 8
    num_clusters: int = 3

    # training
    batch_size: int = 64
    epochs: int = 150          # bump to 100/150 if you like
    lr: float = 1e-3
    seed: int = 42

    # loss weights (initial values; KL will be overridden by scheduler in train)
    w_pred: float = 1.5       # keep
    w_recon: float = 0.02     # try 0.02
    w_kl: float = 0.0         # start at 0.0, warm up to 0.05

    # RL/DQN hyperparams
    dqn_top_k: int = 2            # try 2 or 3
    dqn_eps_start: float = 0.2
    dqn_eps_end: float = 0.05
    dqn_eps_decay_steps: int = 1000
