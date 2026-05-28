"""State encoders for the solver and stacker networks.

Both encoders are pure functions of (state, GridSpec) — no env dependency
beyond the grid geometry — so they can be JIT-batched in the trainer.
"""

from __future__ import annotations

import torch

from .grid import GridSpec


def solver_state_dim(spec: GridSpec, n_obstacles: int) -> int:
    """Length of the flat vector returned by encode_solver_state."""
    return 3 + 3 * n_obstacles + spec.Gx * spec.Gy * (1 + n_obstacles)


def encode_solver_state(state: dict, spec: GridSpec, n_obstacles: int) -> torch.Tensor:
    """Flat encoding: continuous poses + per-cell one-hots of each object.

    [ target_xyz (3),
      obstacle_xyz (3*N),
      target_cell_one_hot (Gx*Gy),
      per-obstacle_cell_one_hot (Gx*Gy * N) ]
    """
    target_pos = state['target_pos'].detach().float().cpu()[:3]
    obstacle_pos = state['obstacle_pos'].detach().float().cpu().reshape(-1, 3)

    parts = [target_pos.flatten()]
    if n_obstacles > 0:
        flat_obs = obstacle_pos.flatten()
        if flat_obs.numel() < 3 * n_obstacles:
            pad = torch.zeros(3 * n_obstacles - flat_obs.numel())
            flat_obs = torch.cat([flat_obs, pad])
        parts.append(flat_obs[:3 * n_obstacles])

    n_cells = spec.Gx * spec.Gy
    target_oh = torch.zeros(n_cells)
    ti, tj, _ = spec.nearest_cell(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
    target_oh[ti * spec.Gy + tj] = 1.0
    parts.append(target_oh)

    for n in range(n_obstacles):
        oh = torch.zeros(n_cells)
        if n < obstacle_pos.shape[0]:
            x, y, z = obstacle_pos[n].tolist()
            i, j, _ = spec.nearest_cell(x, y, z)
            oh[i * spec.Gy + j] = 1.0
        parts.append(oh)

    return torch.cat(parts)


def solver_action_dim(n_obstacles: int, n_z_levels: int) -> int:
    """A_solver = 4 action_types * (N+1) object_indices * Z z-levels."""
    return 4 * (n_obstacles + 1) * n_z_levels


def solver_action_to_index(action_type_idx: int, obj_idx: int, z_idx: int,
                           n_obstacles: int, n_z_levels: int) -> int:
    """Pack (action_type, obj_idx, z_idx) into a flat policy index."""
    return ((action_type_idx * (n_obstacles + 1)) + obj_idx) * n_z_levels + z_idx


def solver_index_to_action(a: int, n_obstacles: int, n_z_levels: int) -> tuple[int, int, int]:
    z_idx = a % n_z_levels
    a //= n_z_levels
    obj_idx = a % (n_obstacles + 1)
    at_idx = a // (n_obstacles + 1)
    return at_idx, obj_idx, z_idx


def encode_stacker_state(occupied: torch.Tensor, target_cell: tuple[int, int],
                         blocks_remaining: int, n_obstacles: int,
                         spec: GridSpec) -> torch.Tensor:
    """4-channel grid encoding (channels collapsed across z for now).

    Channel 0: occupancy heatmap (number of stacked blocks per (i,j), normalized).
    Channel 1: target cell mask.
    Channel 2: highest occupied z-level per (i,j), normalized to [0,1].
    Channel 3: constant plane = blocks_remaining / max(n_obstacles, 1).
    """
    Gx, Gy, Z = spec.Gx, spec.Gy, spec.Z
    out = torch.zeros(4, Gx, Gy, dtype=torch.float32)

    counts = occupied.sum(dim=-1).float()
    out[0] = counts / max(Z, 1)

    ti, tj = target_cell
    out[1, ti, tj] = 1.0

    if Z > 1:
        z_idx = torch.arange(1, Z + 1).float().view(1, 1, Z) * occupied.float()
        highest = z_idx.max(dim=-1).values
        out[2] = highest / Z
    out[3] = blocks_remaining / max(n_obstacles, 1)
    return out
