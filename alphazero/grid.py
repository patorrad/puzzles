"""Grid utilities for the stacker player.

The bin occupies xy ∈ [0, bin_w] × [0, bin_d]. We tile it into a Gx × Gy grid
of cells (cell width chosen so at least n_obstacles+1 cells fit per axis). Each
cell stacks up to Z levels (Z = env.n_z_levels). A stacker action picks one
empty cell (i, j, k) to place its current block.

Index conventions:
  - cell index: (i, j, k) with i ∈ [0,Gx), j ∈ [0,Gy), k ∈ [0,Z)
  - flat index: a = ((i * Gy) + j) * Z + k
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class GridSpec:
    """Geometry of the stacker's discrete placement grid."""
    Gx: int
    Gy: int
    Z: int
    cell_w: float
    cell_d: float
    z_levels: list[float]

    @property
    def n_actions(self) -> int:
        return self.Gx * self.Gy * self.Z

    def flat_index(self, i: int, j: int, k: int) -> int:
        return ((i * self.Gy) + j) * self.Z + k

    def unflatten(self, a: int) -> tuple[int, int, int]:
        k = a % self.Z
        a //= self.Z
        j = a % self.Gy
        i = a // self.Gy
        return i, j, k

    def cell_xyz(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        return ((i + 0.5) * self.cell_w,
                (j + 0.5) * self.cell_d,
                self.z_levels[k])

    def nearest_cell(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        i = int(min(max(x / self.cell_w - 0.5, 0), self.Gx - 1) + 0.5)
        j = int(min(max(y / self.cell_d - 0.5, 0), self.Gy - 1) + 0.5)
        k = min(range(self.Z), key=lambda kk: abs(self.z_levels[kk] - z))
        return i, j, k


def build_grid_spec(env) -> GridSpec:
    """Construct a GridSpec from the simulator env.

    Default Gx = Gy = n_obstacles + 1 (always at least 2 — gives the stacker
    minimal freedom to leave a non-target cell for each obstacle). cell_w/d are
    fitted to bin_w/d so the grid spans the bin exactly.
    """
    g = max(2, env.n_obstacles + 1)
    cell_w = env.bin_w / g
    cell_d = env.bin_d / g
    Z = max(1, env.n_z_levels)
    z_levels = list(env.z_levels) if env.z_levels else [env._OBJ_SIZE / 2]
    return GridSpec(Gx=g, Gy=g, Z=Z,
                    cell_w=cell_w, cell_d=cell_d, z_levels=z_levels)


def legal_mask(spec: GridSpec, occupied: torch.Tensor, target_cell: tuple[int, int]) -> torch.Tensor:
    """Boolean (n_actions,) mask: True for cells the stacker may place into.

    Rules:
      - cell (i,j,k) is full if occupied[i,j,k] == 1
      - cell at z>0 requires cell directly below to be occupied (only when stackable)
      - target's (i,j) at k=0 is blocked (target sits there) but stacking ON the target is allowed
    """
    mask = torch.zeros(spec.n_actions, dtype=torch.bool)
    ti, tj = target_cell
    for i in range(spec.Gx):
        for j in range(spec.Gy):
            for k in range(spec.Z):
                if occupied[i, j, k]:
                    continue
                if k > 0 and not occupied[i, j, k - 1]:
                    continue
                if k == 0 and (i, j) == (ti, tj):
                    continue
                mask[spec.flat_index(i, j, k)] = True
    return mask


def realize_state(spec: GridSpec, placed: list[tuple[int, int, int]],
                  target_cell: tuple[int, int], obj_size: float) -> dict:
    """Convert grid placements into a state dict for env.set_state.

    target sits at (target_cell, k=0); obstacles at the placed cells.
    Identity quaternions; z is the spec's z_level (block bottom + half-height).
    """
    obj_h = obj_size / 2
    ti, tj = target_cell
    tx, ty, _ = spec.cell_xyz(ti, tj, 0)
    target_pos = torch.tensor([tx, ty, obj_h], dtype=torch.float32)
    target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    obs_pos = []
    for i, j, k in placed:
        x, y, _ = spec.cell_xyz(i, j, k)
        z = obj_h + k * obj_size
        obs_pos.append([x, y, z])
    obstacle_pos = torch.tensor(obs_pos, dtype=torch.float32) if obs_pos else torch.empty((0, 3))
    obstacle_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * len(placed),
                                 dtype=torch.float32) if placed else torch.empty((0, 4))
    return {
        'target_pos': target_pos,
        'target_quat': target_quat,
        'obstacle_pos': obstacle_pos,
        'obstacle_quat': obstacle_quat,
    }
