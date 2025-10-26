# scripts/prepare_data.py
import os
import sys
from pathlib import Path

print("[prepare_data] starting...")  # early print so you see output

# Ensure project root (folder containing 'src' and 'scripts') is on sys.path when run directly.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import Config
    from src.data import make_dataloaders
except Exception as e:
    print("[prepare_data] import error:", e)
    print("[prepare_data] sys.path is:", sys.path)
    raise

def main():
    cfg = Config()
    if len(sys.argv) > 1:
        cfg.data_path = sys.argv[1]

    print(f"[prepare_data] using data: {cfg.data_path}")
    print(f"[prepare_data] target nodes: {cfg.target_nodes}")
    print(f"[prepare_data] window={cfg.window}, horizon={cfg.horizon}")

    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio
    )

    print("[prepare_data] Columns (nodes):", meta["cols"])
    print("[prepare_data] num_nodes:", meta["num_nodes"], "window:", meta["window"])
    print("[prepare_data] splits  (train/val/test):",
          meta["train_samples"], meta["val_samples"], meta["test_samples"])

    xb, yb = next(iter(dl_tr))
    print("[prepare_data] Batch X shape (B, N, T):", tuple(xb.shape))
    print("[prepare_data] Batch Y shape (B, N):   ", tuple(yb.shape))
    print("[prepare_data] OK.")

if __name__ == "__main__":
    main()
