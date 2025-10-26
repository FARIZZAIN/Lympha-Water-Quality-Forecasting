# scripts/eval_and_plot.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.eval import evaluate_checkpoint, plot_series
import numpy as np
import csv

def main():
    cfg = Config()
    # Pick checkpoint: "checkpoint_rl.pt" (RL) or "checkpoint_stae.pt" (no-RL)
    ckpt_path = "checkpoint_rl.pt"

    out = evaluate_checkpoint(ckpt_path, cfg, use_rl=True)
    cols = out["cols"]
    Yhat, Ytrue = out["Yhat"], out["Ytrue"]

    print("Per-node MAE:")
    for c, m in zip(cols, out["mae_per"]):
        print(f"  {c:>12s}: {m:.4f}")
    print("Per-node RMSE:")
    for c, r in zip(cols, out["rmse_per"]):
        print(f"  {c:>12s}: {r:.4f}")
    print(f"Overall  MAE: {out['mae']:.4f}")
    print(f"Overall RMSE: {out['rmse']:.4f}")

    # Save CSV with predictions & truth (test set order)
    test_csv = Path("predictions_test.csv")
    with test_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = []
        for c in cols:
            header += [f"{c}_true", f"{c}_pred"]
        w.writerow(header)
        for i in range(Yhat.shape[0]):
            row = []
            for j in range(len(cols)):
                row += [Ytrue[i, j], Yhat[i, j]]
            w.writerow(row)
    print(f"Saved predictions to {test_csv.resolve()}")

    # Plots per node (last 200 test points)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    for j, c in enumerate(cols):
        plot_series(Ytrue[:, j], Yhat[:, j], f"{c} — test (last 200)", str(plots_dir / f"{c}_test.png"))
    print(f"Saved plots to {plots_dir.resolve()}")

if __name__ == "__main__":
    main()
