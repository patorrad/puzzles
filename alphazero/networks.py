"""Solver and stacker (policy, value) networks.

Three architectures, selected via the ``net_arch`` config knob (see
``build_solver_net`` / ``build_stacker_net``):

  - mlp:         2-layer trunk (~30k params), the original baseline.
  - resnet:      AlphaZero-style residual CNN. The solver variant runs the
                 per-object grid one-hots through conv blocks and the
                 continuous xyz poses through an MLP branch.
  - transformer: solver only — each object (target + N obstacles) becomes a
                 token; self-attention captures relational structure.

All variants share the head contract: forward(x) -> (policy logits, tanh
value). The policy head outputs raw logits; masking of illegal actions is the
caller's responsibility (applied as -inf before softmax in MCTS / training).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import OBJ_POSE_DIM


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


class ResBlock(nn.Module):
    """Conv3x3 → norm → ReLU → Conv3x3 → norm, with skip connection.

    GroupNorm instead of BatchNorm: MCTS evaluates leaves with batch size 1
    while the net is in train mode, which makes BatchNorm running stats
    unreliable.
    """

    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


def _conv_stem(in_channels: int, channels: int, n_blocks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, 3, padding=1),
        nn.GroupNorm(min(8, channels), channels),
        nn.ReLU(),
        *[ResBlock(channels) for _ in range(n_blocks)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


class SolverResNet(nn.Module):
    """Residual CNN over per-object grid one-hots + MLP branch over poses.

    The flat solver encoding is split back into its two parts (see
    encoders.encode_solver_state): the leading 7·(1+N) continuous pose features
    (xyz + quaternion per object) go through a linear branch, the trailing
    (1+N)·Gx·Gy one-hots are reshaped to a (1+N, Gx, Gy) image — one channel
    per object — for the conv tower.
    """

    def __init__(self, in_dim: int, n_actions: int, n_obstacles: int,
                 grid_h: int, grid_w: int, n_blocks: int = 4, channels: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.pose_dim = OBJ_POSE_DIM * (1 + n_obstacles)
        self.grid_channels = 1 + n_obstacles
        self.grid_h = grid_h
        self.grid_w = grid_w
        assert in_dim == self.pose_dim + self.grid_channels * grid_h * grid_w, \
            f'in_dim={in_dim} inconsistent with N={n_obstacles}, grid {grid_h}x{grid_w}'
        self.conv = _conv_stem(self.grid_channels, channels, n_blocks)
        self.pose_fc = nn.Sequential(nn.Linear(self.pose_dim, channels), nn.ReLU())
        self.policy = nn.Linear(2 * channels, n_actions)
        self.value = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        pose = x[:, :self.pose_dim]
        grid = x[:, self.pose_dim:].view(-1, self.grid_channels,
                                         self.grid_h, self.grid_w)
        h = torch.cat([self.conv(grid), self.pose_fc(pose)], dim=-1)
        logits = self.policy(h)
        value = torch.tanh(self.value(h)).squeeze(-1)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


class StackerResNet(nn.Module):
    """Residual CNN over the [4, Gx, Gy] stacker image."""

    def __init__(self, grid_h: int, grid_w: int, n_actions: int,
                 in_channels: int = 4, n_blocks: int = 4, channels: int = 32):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.conv = _conv_stem(in_channels, channels, n_blocks)
        self.policy = nn.Linear(channels, n_actions)
        self.value = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = x.dim() == 3
        if unbatched:
            x = x.unsqueeze(0)
        h = self.conv(x)
        logits = self.policy(h)
        value = torch.tanh(self.value(h)).squeeze(-1)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


class SolverTransformer(nn.Module):
    """Self-attention over object tokens (target + N obstacles).

    Each object's token is built from its continuous xyz pose and its grid
    cell one-hot. A learned per-slot embedding distinguishes the target
    (slot 0) from each obstacle slot — necessary because the action space
    addresses obstacles by slot index, so the policy head must too.
    """

    def __init__(self, in_dim: int, n_actions: int, n_obstacles: int,
                 grid_h: int, grid_w: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.n_objects = 1 + n_obstacles
        self.n_cells = grid_h * grid_w
        self.pose_dim = OBJ_POSE_DIM * self.n_objects
        assert in_dim == self.pose_dim + self.n_objects * self.n_cells, \
            f'in_dim={in_dim} inconsistent with N={n_obstacles}, grid {grid_h}x{grid_w}'
        self.embed = nn.Linear(OBJ_POSE_DIM + self.n_cells, d_model)
        self.slot_emb = nn.Parameter(torch.randn(self.n_objects, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.policy = nn.Linear(d_model, n_actions)
        self.value = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        B = x.shape[0]
        # Encoder layout is section-contiguous ([all xyz, all quats, all one-hots],
        # target first within each section), so each section reshapes directly
        # to per-object rows.
        n_obj = self.n_objects
        xyz = x[:, :3 * n_obj].view(B, n_obj, 3)
        quat = x[:, 3 * n_obj:self.pose_dim].view(B, n_obj, 4)
        cells = x[:, self.pose_dim:].view(B, n_obj, self.n_cells)
        tokens = self.embed(torch.cat([xyz, quat, cells], dim=-1)) + self.slot_emb
        h = self.encoder(tokens).mean(dim=1)
        logits = self.policy(h)
        value = torch.tanh(self.value(h)).squeeze(-1)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def build_solver_net(arch: str, in_dim: int, n_actions: int,
                     n_obstacles: int, grid_h: int, grid_w: int) -> nn.Module:
    if arch == 'mlp':
        return SolverNet(in_dim=in_dim, n_actions=n_actions)
    if arch == 'resnet':
        return SolverResNet(in_dim=in_dim, n_actions=n_actions,
                            n_obstacles=n_obstacles,
                            grid_h=grid_h, grid_w=grid_w)
    if arch == 'transformer':
        return SolverTransformer(in_dim=in_dim, n_actions=n_actions,
                                 n_obstacles=n_obstacles,
                                 grid_h=grid_h, grid_w=grid_w)
    raise ValueError(f"unknown net_arch '{arch}' (expected mlp | resnet | transformer)")


def build_stacker_net(arch: str, grid_h: int, grid_w: int, n_actions: int,
                      in_channels: int = 4) -> nn.Module:
    if arch == 'mlp':
        return StackerNet(grid_h=grid_h, grid_w=grid_w, n_actions=n_actions,
                          in_channels=in_channels)
    if arch in ('resnet', 'transformer'):
        # The stacker state is a grid image with no token decomposition, so
        # the transformer arch falls back to the CNN here.
        return StackerResNet(grid_h=grid_h, grid_w=grid_w, n_actions=n_actions,
                             in_channels=in_channels)
    raise ValueError(f"unknown net_arch '{arch}' (expected mlp | resnet | transformer)")


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax with illegal-action mask (True = legal). Stable across rows."""
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    return F.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_log_softmax(logits, mask).exp()
