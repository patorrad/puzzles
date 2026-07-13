"""
Object-contour push sampler for MORE (Huang et al., ICRA 2022).

Generates candidate push actions for a given scene state.  Each action is a
'push_dir' dict (compatible with SimulatorEnv.batch_evaluate) whose start point
lies just outside an object's bounding contour and whose end point passes through
the object's XY centroid, mirroring MORE's "contour-following" push sampling.

Because our state is pose-based (not image-based), each object's 2D footprint is
approximated as a square with side = obj_half_extent * 2, rotated by the object's
yaw angle.  This is conservative and consistent with the axis-aligned meshes used
in this project.

Usage::

    sampler = ContourSampler(env)
    actions = sampler.sample(state, k_per_object=8)
    # actions is a list of dicts, each with action_type='push_dir'
"""

from __future__ import annotations

import math
import torch


class ContourSampler:
    """
    Sample K contour-push actions per object from a pose-based scene state.

    Parameters
    ----------
    obj_half_extent : float
        Half-side of the square footprint approximation (metres).  Defaults to
        env._OBJ_SIZE / 2 when an env is supplied.
    approach_dist : float
        Gap between the start point and the object boundary so the pusher
        approaches from outside.
    z_levels : list[float] | None
        Push heights to sample; inherits env.z_levels when an env is supplied.
    include_target : bool
        If False, only generate pushes for obstacle objects.
    """

    def __init__(self,
                 obj_half_extent: float,
                 approach_dist: float = 0.02,
                 z_levels: list[float] | None = None,
                 include_target: bool = True):
        self.obj_half_extent = obj_half_extent
        self.approach_dist = approach_dist
        self.z_levels = z_levels if z_levels else [0.025]
        self.include_target = include_target

    @classmethod
    def from_env(cls, env,
                 approach_dist: float = 0.02,
                 include_target: bool = True) -> 'ContourSampler':
        """Build a ContourSampler from a SimulatorEnv instance."""
        return cls(
            obj_half_extent=env._OBJ_SIZE / 2.0,
            approach_dist=approach_dist,
            z_levels=list(env.z_levels),
            include_target=include_target,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, state: dict, k_per_object: int = 8) -> list[dict]:
        """
        Return up to len(objects) * k_per_object * len(z_levels) push_dir actions.

        Parameters
        ----------
        state : dict
            Solver state with 'target_pos', 'obstacle_pos', 'obstacle_quat'.
        k_per_object : int
            Number of start-point samples uniformly distributed around each
            object's bounding contour.
        """
        actions: list[dict] = []
        obj_centers = self._get_centers(state)

        for obj_idx, center_xy in enumerate(obj_centers):
            if obj_idx == 0 and not self.include_target:
                continue
            for z in self.z_levels:
                starts = self._contour_samples(center_xy, k_per_object)
                for start_xy in starts:
                    actions.append({
                        'action_type':   'push_dir',
                        'push_start_xy': start_xy,
                        'push_end_xy':   center_xy.clone(),
                        'push_z':        z,
                        'obj_idx':       obj_idx,
                    })
        return actions

    def sample_for_object(self, center_xy: torch.Tensor, obj_idx: int,
                          k: int = 8) -> list[dict]:
        """Sample k pushes toward a single object centroid (all z levels)."""
        actions: list[dict] = []
        for z in self.z_levels:
            for start_xy in self._contour_samples(center_xy, k):
                actions.append({
                    'action_type':   'push_dir',
                    'push_start_xy': start_xy,
                    'push_end_xy':   center_xy.clone(),
                    'push_z':        z,
                    'obj_idx':       obj_idx,
                })
        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_centers(self, state: dict) -> list[torch.Tensor]:
        """Return [target_xy, obs0_xy, obs1_xy, ...] as a list of (2,) tensors."""
        centers = [state['target_pos'][:2].float()]
        for i in range(len(state['obstacle_pos'])):
            centers.append(state['obstacle_pos'][i, :2].float())
        return centers

    def _contour_samples(self, center_xy: torch.Tensor,
                         k: int) -> list[torch.Tensor]:
        """
        K start points uniformly distributed around the bounding square,
        offset by approach_dist so the pusher starts just outside.

        Points are sampled along the perimeter of the object's bounding
        square at angles 0, 2π/k, 4π/k, ... from the centroid.  The start
        point is projected onto the square boundary and then moved outward
        by approach_dist along the radial direction.
        """
        r = self.obj_half_extent + self.approach_dist
        starts = []
        for i in range(k):
            theta = 2.0 * math.pi * i / k
            dx = math.cos(theta)
            dy = math.sin(theta)
            # Clamp to square boundary (infinity-norm radius = r)
            scale = r / max(abs(dx), abs(dy))
            sx = float(center_xy[0]) + dx * scale
            sy = float(center_xy[1]) + dy * scale
            starts.append(torch.tensor([sx, sy], dtype=torch.float32))
        return starts
