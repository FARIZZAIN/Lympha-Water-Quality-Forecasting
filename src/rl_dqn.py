# src/rl_dqn.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def pair_indices(N: int) -> List[Tuple[int,int]]:
    """All undirected pairs i<j."""
    return [(i,j) for i in range(N) for j in range(i+1, N)]

def build_adj_from_pairs(N: int, pairs: List[Tuple[int,int]], device, sym=True):
    A = torch.zeros(N, N, device=device)
    for (i,j) in pairs:
        A[i,j] = 1.0
        if sym:
            A[j,i] = 1.0
    # add small self-loop to avoid empty rows; scaled lightly
    A = A + 1e-4 * torch.eye(N, device=device)
    return A

class PairwiseFeaturizer(nn.Module):
    """
    Turn per-node embeddings O (B,N,d) into per-pair features Φ (B,M,f).
    Φ_ij = [o_i, o_j, |o_i-o_j|, o_i * o_j]  -> linear projection
    """
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4*d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1)  # output single Q per pair (before squeeze)
        )

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        # O: (B,N,d)
        B, N, d = O.shape
        pairs = pair_indices(N)
        M = len(pairs)
        # build pair features
        oi = []
        oj = []
        for (i,j) in pairs:
            oi.append(O[:, i, :])  # (B,d)
            oj.append(O[:, j, :])  # (B,d)
        Oi = torch.stack(oi, dim=1)      # (B,M,d)
        Oj = torch.stack(oj, dim=1)      # (B,M,d)
        feats = torch.cat([Oi, Oj, (Oi - Oj).abs(), Oi * Oj], dim=-1)  # (B,M,4d)
        q = self.proj(feats).squeeze(-1)  # (B,M)
        return q  # Q-values per pair

@dataclass
class DQNConfig:
    eps_start: float = 0.2
    eps_end: float = 0.05
    eps_decay_steps: int = 1500      # linear decay steps
    top_k: int = 2                   # how many edges to pick per batch sample
    gamma: float = 0.0               # 1-step bandit; keep 0.0
    lr: float = 1e-3
    target_update: int = 200         # copy online -> target every N steps
    buffer_size: int = 5000
    batch_size: int = 64

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.idx = 0

    def push(self, s, a_idx, r, s2):
        # s, s2: (M) Q-input context vectors not needed here; we store pairwise Q predictions lazily
        self.data.append((s, a_idx, r, s2))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size: int):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)

class DQNAgent(nn.Module):
    """
    DQN over pair selections:
      - network: PairwiseFeaturizer to produce Q per pair
      - ε-greedy selection of top-K pairs (per batch item)
      - TD(0) loss with gamma=0: target = reward (bandit setting)
    """
    def __init__(self, d_in: int, N_nodes: int, cfg: DQNConfig):
        super().__init__()
        self.N = N_nodes
        self.cfg = cfg
        self.online = PairwiseFeaturizer(d_in)
        self.target = PairwiseFeaturizer(d_in)
        self.update_target()
        self.buf = ReplayBuffer(cfg.buffer_size)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.steps = 0
        self.pairs = pair_indices(N_nodes)  # fixed mapping
        self.M = len(self.pairs)

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def epsilon(self):
        e0, e1, K = self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_decay_steps
        t = min(self.steps, K)
        return e0 + (e1 - e0) * (t / K)

    def select_pairs(self, O: torch.Tensor, device) -> Tuple[torch.Tensor, list[list[Tuple[int,int]]], torch.Tensor]:
        """
        O: (B,N,d) -> returns:
          Q: (B,M) full Q per pair,
          picks: list of list of (i,j) pairs per batch,
          mask_idx: (B,M) boolean mask of selected indices
        """
        with torch.no_grad():
            Q = self.online(O)  # (B,M)

        B = O.size(0)
        eps = self.epsilon()
        picks = []
        mask_idx = torch.zeros(B, self.M, dtype=torch.bool, device=device)

        for b in range(B):
            if random.random() < eps:
                # random top-K
                idx = torch.randperm(self.M, device=device)[:self.cfg.top_k]
            else:
                idx = torch.topk(Q[b], k=self.cfg.top_k, dim=-1).indices
            chosen = [self.pairs[int(i)] for i in idx]
            picks.append(chosen)
            mask_idx[b, idx] = True

        self.steps += 1
        return Q, picks, mask_idx

    def learn(self, O: torch.Tensor, mask_idx: torch.Tensor, reward: torch.Tensor):
        """
        Update DQN via TD(0): target = reward
        O: (B,N,d), mask_idx: (B,M) bool, reward: (B,)
        """
        # online Q for all pairs
        Q_all = self.online(O)                # (B,M)
        # chosen Qs
        Q_chosen = Q_all[mask_idx]            # (B*top_k,)
        # Targets: reward replicated for each chosen pair
        r_rep = reward.unsqueeze(1).repeat(1, self.cfg.top_k).reshape(-1)  # (B*top_k,)

        loss = F.mse_loss(Q_chosen, r_rep.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.steps % self.cfg.target_update == 0:
            self.update_target()

        return float(loss.item())
