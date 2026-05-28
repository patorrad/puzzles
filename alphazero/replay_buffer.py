"""Tensor replay buffer for AlphaZero-style training.

Stores (encoded_state, policy_target, value_target, legal_mask) tuples. One
buffer per player; both use the same class.
"""

from __future__ import annotations

from collections import deque
import random

import torch


class ReplayBuffer:
    def __init__(self, maxlen: int = 50_000):
        self.buf: deque = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, x: torch.Tensor, pi: torch.Tensor, z: float,
             legal_mask: torch.Tensor):
        self.buf.append((
            x.detach().cpu().float(),
            pi.detach().cpu().float(),
            float(z),
            legal_mask.detach().cpu().bool(),
        ))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self.buf))
        items = random.sample(self.buf, batch_size)
        xs = torch.stack([it[0] for it in items])
        pis = torch.stack([it[1] for it in items])
        zs = torch.tensor([it[2] for it in items], dtype=torch.float32)
        masks = torch.stack([it[3] for it in items])
        return xs, pis, zs, masks
