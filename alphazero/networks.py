"""Solver and stacker (policy, value) networks.

Small MLPs (~30k params): a 2-layer trunk with two heads — categorical policy
logits and a tanh value scalar. The policy head outputs raw logits; masking of
illegal actions is the caller's responsibility (applied as -inf before softmax
in MCTS / training).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Trunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SolverNet(nn.Module):
    """Input: flat solver-state vector. Outputs: (policy logits, value)."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.trunk = _Trunk(in_dim, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


class StackerNet(nn.Module):
    """Input: flattened [4, Gx, Gy] tensor. Outputs: (policy logits, value)."""

    def __init__(self, grid_h: int, grid_w: int, n_actions: int,
                 in_channels: int = 4, hidden: int = 128):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.in_channels = in_channels
        self.n_actions = n_actions
        in_dim = in_channels * grid_h * grid_w
        self.trunk = _Trunk(in_dim, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() == 3:
            x = x.flatten(0)
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax with illegal-action mask (True = legal). Stable across rows."""
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    return F.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_log_softmax(logits, mask).exp()
