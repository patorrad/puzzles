"""
Pure-PyTorch object placement utilities.

Provides random_initial_state() (no simulator dependency) so placement logic
is not duplicated across simulator envs. Also provides make_state() for
constructing test states from simple (x, y) coordinates.
"""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

BIN_W    = 1.0
BIN_D    = 1.0


def _place_objects_once(
    n_obstacles: int,
    stackable: bool,
    difficult_spawn: bool,
    bin_w: float,
    bin_d: float,
    obj_size: float,
    n_z_levels: int = 1,
    target_z_level: int | None = None,
    force_obstacle_on_target: bool = False,
    obj_height: float | None = None,
) -> dict:
    """Single placement attempt. Caller is responsible for seeding."""
    obj_h    = obj_height if obj_height is not None else obj_size / 2
    margin   = obj_size * 0.7
    x_lo, x_hi = margin, bin_w - margin
    y_lo, y_hi = margin, bin_d - margin

    columns:   list[tuple[float, float, int]] = []
    positions: list[list[float]] = []

    # Resolve None → random level after obstacles are placed (done inside the loop).
    _target_z = target_z_level  # None means randomise when target turn arrives

    n_objects = n_obstacles + 1  # obstacles first, target last
    for obj_i in range(n_objects):
        is_target = (obj_i == n_obstacles)
        if is_target and _target_z is None:
            # Pick randomly from levels that have eligible columns, falling back to 0.
            eligible_levels = sorted({c[2] for c in columns if c[2] < n_z_levels})
            if eligible_levels and n_z_levels > 1:
                _target_z = int(eligible_levels[int(torch.randint(len(eligible_levels), (1,)).item())])
            else:
                _target_z = 0
        obj_y_lo  = bin_d / 2 if (is_target and difficult_spawn) else y_lo

        placed = False

        if is_target and _target_z > 0:
            # Find a column that already has exactly target_z_level objects so
            # the target sits at that height with full support beneath it.
            eligible = [(i, c) for i, c in enumerate(columns)
                        if c[2] == _target_z and (not difficult_spawn or c[1] >= bin_d / 2)]
            if eligible:
                idx = int(torch.randint(len(eligible), (1,)).item())
                choice_i, (x, y, count) = eligible[idx]
                positions.append([x, y, obj_h + obj_size * count])
                columns[choice_i] = (x, y, count + 1)
                placed = True
            # Fall through to floor placement if no suitable column exists.

        if not placed:
            sep = obj_size * 1.05  # minimum column separation
            can_stack = (stackable or n_z_levels > 1) and not is_target
            for _ in range(500):
                x = torch.empty(1).uniform_(x_lo, x_hi).item()
                y = torch.empty(1).uniform_(obj_y_lo, y_hi).item()

                if can_stack and columns:
                    # If the sample lands within the rejection radius of a column,
                    # stack on the closest eligible one instead of retrying.
                    nearby = [
                        (i, c) for i, c in enumerate(columns)
                        if ((x - c[0]) ** 2 + (y - c[1]) ** 2) ** 0.5 < sep
                        and c[2] < n_z_levels
                    ]
                    if nearby:
                        choice_i, (cx, cy, count) = min(
                            nearby, key=lambda ic: (x - ic[1][0]) ** 2 + (y - ic[1][1]) ** 2
                        )
                        positions.append([cx, cy, obj_h + obj_size * count])
                        columns[choice_i] = (cx, cy, count + 1)
                        placed = True
                        break

                if all(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 > sep
                       for cx, cy, _ in columns):
                    positions.append([x, y, obj_h])
                    columns.append((x, y, 1))
                    placed = True
                    break

            if not placed:
                positions.append([bin_w / 2, bin_d / 2, obj_h])
                columns.append((bin_w / 2, bin_d / 2, 1))

    # If the target didn't land at the requested level, swap with an obstacle at that level.
    if n_obstacles > 0 and _target_z is not None and _target_z > 0:
        target_z_height = obj_h + obj_size * _target_z
        if abs(positions[n_obstacles][2] - target_z_height) > 1e-4:
            for obs_idx in range(n_obstacles):
                if abs(positions[obs_idx][2] - target_z_height) < 1e-4:
                    if difficult_spawn and positions[obs_idx][1] < bin_d / 2:
                        continue
                    positions[n_obstacles], positions[obs_idx] = positions[obs_idx], positions[n_obstacles]
                    break

    # Force one obstacle directly on top of the target if requested and there is room above it.
    if force_obstacle_on_target and n_obstacles > 0 and _target_z is not None and _target_z + 1 < n_z_levels:
        target_x, target_y = positions[n_obstacles][0], positions[n_obstacles][1]
        above_z = obj_h + obj_size * (_target_z + 1)
        sep = obj_size * 1.05
        for obs_idx in range(n_obstacles):
            ox, oy = positions[obs_idx][0], positions[obs_idx][1]
            if ((ox - target_x) ** 2 + (oy - target_y) ** 2) ** 0.5 >= sep:
                positions[obs_idx] = [target_x, target_y, above_z]
                break

    identity = [1.0, 0.0, 0.0, 0.0]
    # Sort obstacles by ascending z so lower objects are placed first in
    # sequential simulators (e.g. IsaacLab), ensuring support before stacking.
    obstacle_positions = sorted(positions[:n_obstacles], key=lambda p: p[2])
    return {
        'target_pos':    torch.tensor(positions[n_obstacles], dtype=torch.float32),
        'target_quat':   torch.tensor(identity,               dtype=torch.float32),
        'obstacle_pos':  (torch.tensor(obstacle_positions, dtype=torch.float32)
                          if n_obstacles > 0 else torch.zeros(0, 3, dtype=torch.float32)),
        'obstacle_quat': (torch.tensor([identity] * n_obstacles, dtype=torch.float32)
                          if n_obstacles > 0 else torch.zeros(0, 4, dtype=torch.float32)),
    }


def _path_blocker_value(state: dict, n_obstacles: int, scale: float = 0.16, weight: float = 0.5) -> float:
    """Compute path_blocker reward component (<=0; more negative = more blocked)."""
    tx = state['target_pos'][0].item()
    ty = state['target_pos'][1].item()
    value = 0.0
    for i in range(n_obstacles):
        oy = state['obstacle_pos'][i][1].item()
        if 0.0 < oy < ty:
            x_dist = abs(state['obstacle_pos'][i][0].item() - tx)
            value -= weight * max(0.0, 1.0 - x_dist / scale)
    return value



def random_initial_state(
    n_obstacles: int,
    obj_size: float,
    stackable: bool = False,
    difficult_spawn: bool = False,
    seed: Optional[int] = None,
    bin_w: Optional[float] = None,
    bin_d: Optional[float] = None,
    max_attempts: int = 200,
    debug: bool = False,
    n_z_levels: int = 1,
    target_z_level: Optional[int] = None,
    force_obstacle_on_target: bool = False,
    obj_height: Optional[float] = None,
) -> dict:
    """
    Generate a random non-overlapping initial state dict (pure PyTorch, no simulator).

    Replicates the column-based placement algorithm used by the simulator envs:
    - Objects placed randomly in [margin, bin_w-margin] × [margin, bin_d-margin]
    - Minimum column separation of obj_size × 1.05
    - Optional stacking (50% chance per object after the first column exists)
    - difficult_spawn=True: target restricted to y ∈ [bin_d/2, bin_d-margin] AND
      at least one obstacle must be south of the target within the path-blocker
      x-range (retries up to max_attempts times).

    Obstacles are placed first, target last.

    Parameters
    ----------
    n_obstacles : int
    obj_size : float
        Object cube side length (metres).
    stackable : bool
    difficult_spawn : bool
    seed : int | None
    bin_w : float | None
        Bin width (x). Defaults to BIN_W module constant if not given.
    bin_d : float | None
        Bin depth (y). Defaults to BIN_D module constant if not given.
    max_attempts : int
        Max placement retries when difficult_spawn=True (default 200).

    Returns
    -------
    dict with torch tensors: target_pos (3,), target_quat (4,),
                              obstacle_pos (n,3), obstacle_quat (n,4)
    """
    if bin_w is None:
        bin_w = BIN_W
    if bin_d is None:
        bin_d = BIN_D

    if seed is not None:
        torch.manual_seed(seed)

    attempts = max_attempts if difficult_spawn else 1
    state = None
    for attempt in range(attempts):
        state = _place_objects_once(n_obstacles, stackable, difficult_spawn, bin_w, bin_d,
                                    obj_size, n_z_levels,
                                    target_z_level=target_z_level,
                                    force_obstacle_on_target=force_obstacle_on_target,
                                    obj_height=obj_height)
        if not difficult_spawn:
            return state
        pb = _path_blocker_value(state, n_obstacles)
        if debug:
            logger.debug('attempt %d/%d: path_blocker=%.4f', attempt + 1, attempts, pb)
        if pb < 0.0:
            if debug:
                logger.debug('success: found blocking obstacle on attempt %d', attempt + 1)
            return state

    if debug:
        logger.warning('no blocking obstacle found after %d attempts, using last state', attempts)
    return state


def make_state(target_xy: tuple, obstacle_xys: list, obj_size: float) -> dict:
    """
    Build a state dict from (x, y) tuples; z is set to obj_size/2 for all objects.

    Parameters
    ----------
    target_xy : (x, y)
    obstacle_xys : list of (x, y)
    obj_size : float

    Returns
    -------
    State dict compatible with env.set_state() and env._compute_reward()
    """
    obj_h = obj_size / 2
    tx, ty = target_xy
    n = len(obstacle_xys)
    identity = [1.0, 0.0, 0.0, 0.0]
    return {
        'target_pos':    torch.tensor([tx, ty, obj_h], dtype=torch.float32),
        'target_quat':   torch.tensor(identity, dtype=torch.float32),
        'obstacle_pos':  (torch.tensor([[ox, oy, obj_h] for ox, oy in obstacle_xys], dtype=torch.float32)
                          if n > 0 else torch.zeros(0, 3, dtype=torch.float32)),
        'obstacle_quat': (torch.tensor([identity] * n, dtype=torch.float32)
                          if n > 0 else torch.zeros(0, 4, dtype=torch.float32)),
    }
