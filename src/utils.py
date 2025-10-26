import torch, random, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_val_test_split_idx(S, train_ratio=0.7, val_ratio=0.15):
    n_train = int(S * train_ratio)
    n_val = int(S * val_ratio)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, S)
    }
