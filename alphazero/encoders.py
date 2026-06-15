"""State encoders for the solver and stacker networks.

Both encoders are pure functions of (state, GridSpec) — no env dependency
beyond the grid geometry — so they can be JIT-batched in the trainer.
"""

from __future__ import annotations

import torch

from .grid import GridSpec


# Per-object continuous features: xyz (3) + canonicalized quaternion (4).
OBJ_POSE_DIM = 7


def solver_state_dim(spec: GridSpec, n_obstacles: int) -> int:
    """Length of the flat vector returned by encode_solver_state."""
    n_objects = 1 + n_obstacles
    return OBJ_POSE_DIM * n_objects + spec.Gx * spec.Gy * n_objects


def _canonical_quat(q: torch.Tensor) -> torch.Tensor:
    """Fix the q ≡ -q sign ambiguity (convention-agnostic: works for wxyz/xyzw).

    Flips the quaternion so its largest-magnitude component is positive, making
    physically identical orientations encode identically.
    """
    q = q.reshape(-1, 4)
    lead = q.gather(1, q.abs().argmax(dim=1, keepdim=True))
    return torch.where(lead < 0, -q, q)


def _pad_rows(x: torch.Tensor, n_rows: int) -> torch.Tensor:
    if x.shape[0] >= n_rows:
        return x[:n_rows]
    return torch.cat([x, torch.zeros(n_rows - x.shape[0], x.shape[1])])


def encode_solver_state(state: dict, spec: GridSpec, n_obstacles: int) -> torch.Tensor:
    """Flat encoding: continuous poses + orientations + per-cell one-hots.

    Section-contiguous layout (lets SolverTransformer reshape each section to
    per-object rows without copying):
    [ target_xyz (3), obstacle_xyz (3*N),
      target_quat (4), obstacle_quat (4*N),
      target_cell_one_hot (Gx*Gy),
      per-obstacle_cell_one_hot (Gx*Gy * N) ]
    """
    target_pos = state['target_pos'].detach().float().cpu()[:3]
    obstacle_pos = state['obstacle_pos'].detach().float().cpu().reshape(-1, 3)
    target_quat = _canonical_quat(state['target_quat'].detach().float().cpu())
    obstacle_quat = _canonical_quat(state['obstacle_quat'].detach().float().cpu())

    parts = [target_pos.flatten()]
    if n_obstacles > 0:
        parts.append(_pad_rows(obstacle_pos, n_obstacles).flatten())
    parts.append(target_quat.flatten()[:4])
    if n_obstacles > 0:
        parts.append(_pad_rows(obstacle_quat, n_obstacles).flatten())

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
