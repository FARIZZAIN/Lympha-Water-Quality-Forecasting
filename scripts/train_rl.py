# scripts/train_rl.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.train_rl import run_train

if __name__ == "__main__":
    run_train(Config())
