from dataclasses import dataclass

@dataclass
class Config:
    # =============================================================
    # DATA SETTINGS
    # =============================================================
    # Path to your dataset (update this to your machine’s path)
    data_path: str = "data/raw/Static_Sensors_Data/data10min.csv"

    # Leave empty for auto detection in long format
    target_nodes: tuple[str, ...] = ()

    # Sliding window and prediction horizon
    window: int = 48         # past steps (≈ 4 hours @ 10 min)
    horizon: int = 1         # predict next step (10 min ahead)

    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64

    # =============================================================
    # LONG-FORMAT CONTROLS
    # =============================================================
    # Your new dataset is long-format, so enable this
    is_long_format: bool = True
    # single_node → variables become graph nodes for one station
    # all_pairs → (Node Id × variable) pairs become nodes
    long_option: str = "single_node"
    node_id_filter: str | None = "1"      # Node Id to use in single_node mode
    resample_rule: str = "10min"            # exact 10-minute grid
    impute: str = "ffill_bfill"           # fill small gaps

    # =============================================================
    # MODEL HYPERPARAMETERS
    # =============================================================
    d_temporal: int = 64
    temporal_kernel: int = 5
    temporal_layers: int = 2
    d_gcn: int = 32
    d_fused: int = 96
    d_z: int = 8
    num_clusters: int = 3

    # =============================================================
    # TRAINING HYPERPARAMETERS
    # =============================================================
    epochs: int = 150
    lr: float = 1e-3
    seed: int = 42

    # =============================================================
    # LOSS WEIGHTS
    # =============================================================
    w_pred: float = 1.5
    w_recon: float = 0.02
    w_kl: float = 0.0

    # =============================================================
    # DQN / RL CHANNEL CONFIG
    # =============================================================
    dqn_top_k: int = 2
    dqn_eps_start: float = 0.2
    dqn_eps_end: float = 0.05
    dqn_eps_decay_steps: int = 1000
