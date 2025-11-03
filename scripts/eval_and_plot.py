import os
import argparse
import pandas as pd
import numpy as np

from src.config import Config
from src.eval import evaluate_checkpoint, plot_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoint_rl.pt", help="Path to checkpoint (RL or STAE).")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save test plots.")
    parser.add_argument("--preds_csv", type=str, default="predictions_test.csv", help="CSV path for test predictions.")
    parser.add_argument("--last", type=int, default=200, help="How many tail samples to plot per series.")
    args = parser.parse_args()

    cfg = Config()  # uses your current project config

    # Run evaluation (prints concise overall + per-variable metrics)
    out = evaluate_checkpoint(args.ckpt, cfg, use_rl=True)

    cols: list[str] = out["cols"]
    Yhat: np.ndarray = out["Yhat"]   # (S, N)
    Ytrue: np.ndarray = out["Ytrue"] # (S, N)

    # Save predictions CSV (both true & pred, side-by-side)
    os.makedirs(os.path.dirname(args.preds_csv) or ".", exist_ok=True)
    data = {}
    for i, c in enumerate(cols):
        data[f"{c}_true"] = Ytrue[:, i]
        data[f"{c}_pred"] = Yhat[:, i]
    pd.DataFrame(data).to_csv(args.preds_csv, index=False)
    print(f"Saved predictions to {os.path.abspath(args.preds_csv)}")

    # Save per-variable plots
    os.makedirs(args.plots_dir, exist_ok=True)
    for i, c in enumerate(cols):
        savepath = os.path.join(args.plots_dir, f"{c}_test.png")
        plot_series(Ytrue[:, i], Yhat[:, i], title=f"{c} — test predictions", savepath=savepath, last=args.last)

    print(f"Saved plots to {os.path.abspath(args.plots_dir)}")


if __name__ == "__main__":
    main()
