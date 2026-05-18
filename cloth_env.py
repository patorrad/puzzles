"""
ClothEnv: Genesis simulation of a cloth folding puzzle using PBD.

A flat cloth is discretized into a checkerboard grid. Each fold action
selects a grid cell and a direction ('N', 'S', 'E', 'W'), which defines
the fold crease. All particles on the far side of that crease are swept
through a semicircular arc — kinematically driven, then released so the
PBD solver handles the rest of the cloth.
"""

import os
import tempfile
import time

import numpy as np
import torch

import genesis as gs

gs.init(logging_level='warning')

CLOTH_W = 0.4   # default cloth width  (x direction), metres
CLOTH_H = 0.4   # default cloth height (y direction), metres
CLOTH_Z = 0.01  # initial height above the ground plane


class ClothEnv:
    """
    Genesis PBD cloth folding puzzle.

    The cloth is a CLOTH_W × CLOTH_H flat sheet divided into a
    grid_n × grid_n checkerboard. A fold action selects a cell (row, col)
    and a boundary direction; every cloth particle on the far side of that
    boundary is swept through a semicircular arc, then released.

    Parameters
    ----------
    grid_n : int
        Number of rows/columns in the checkerboard (default 4 → 4×4 grid).
    cloth_w, cloth_h : float
        Cloth dimensions in metres.
    cloth_z : float
        Initial cloth height; keeps particles above the ground plane.
    show_viewer : bool
    dt : float
    substeps : int
    particle_size : float
        PBD particle spacing.  Smaller → denser simulation.
    fold_steps : int
        Arc sweep steps per fold action.
    settle_steps : int
        Free-simulation steps after each fold, letting cloth come to rest.
    """

    def __init__(
        self,
        grid_n: int = 4,
        cloth_w: float = CLOTH_W,
        cloth_h: float = CLOTH_H,
        cloth_z: float = CLOTH_Z,
        show_viewer: bool = False,
        dt: float = 4e-3,
        substeps: int = 10,
        particle_size: float = 0.025,
        fold_steps: int = 40,
        settle_steps: int = 80,
    ):
        self.grid_n = grid_n
        self.cloth_w = cloth_w
        self.cloth_h = cloth_h
        self.cloth_z = cloth_z
        self.show_viewer = show_viewer
        self.fold_steps = fold_steps
        self.settle_steps = settle_steps

        self._tmp_dir = tempfile.mkdtemp()
        self._mesh_path = os.path.join(self._tmp_dir, 'cloth.obj')
        self._ckpt_path = os.path.join(self._tmp_dir, 'initial')

        # Dense enough that each cell has several particles
        self._write_cloth_obj(cloth_w, cloth_h, grid_n * 6, self._mesh_path)

        self._build_scene(dt, substeps, particle_size)
        self._init_grid()

    # ------------------------------------------------------------------
    # Cloth mesh generation
    # ------------------------------------------------------------------

    @staticmethod
    def _write_cloth_obj(width, height, n_divs, path):
        """Write a flat n_divs × n_divs quad mesh centred at the XY origin."""
        with open(path, 'w') as f:
            for j in range(n_divs + 1):
                for i in range(n_divs + 1):
                    x = -width  / 2 + i * width  / n_divs
                    y = -height / 2 + j * height / n_divs
                    f.write(f'v {x:.6f} {y:.6f} 0.000000\n')
            for j in range(n_divs):
                for i in range(n_divs):
                    v00 = j * (n_divs + 1) + i + 1  # OBJ is 1-indexed
                    v10 = v00 + 1
                    v01 = v00 + n_divs + 1
                    v11 = v01 + 1
                    f.write(f'f {v00} {v10} {v11}\n')
                    f.write(f'f {v00} {v11} {v01}\n')

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, dt, substeps, particle_size):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            pbd_options=gs.options.PBDOptions(particle_size=particle_size),
            viewer_options=gs.options.ViewerOptions(
                camera_fov=40,
                camera_pos=(0.0, -0.70, 0.55),
                camera_lookat=(0.0, 0.0, 0.05),
            ),
            show_viewer=self.show_viewer,
        )

        self.scene.add_entity(morph=gs.morphs.Plane())

        self.cloth = self.scene.add_entity(
            material=gs.materials.PBD.Cloth(),
            morph=gs.morphs.Mesh(
                file=self._mesh_path,
                pos=(0.0, 0.0, self.cloth_z),
                scale=1.0,
            ),
            surface=gs.surfaces.Default(
                color=(0.80, 0.65, 0.45, 1.0),
                vis_mode='visual',
            ),
        )

        self.scene.build()

        # Let cloth settle flat on the ground plane
        for _ in range(40):
            self.scene.step()

        self.scene.save_checkpoint(self._ckpt_path)

    # ------------------------------------------------------------------
    # Grid mapping (runs after scene.build)
    # ------------------------------------------------------------------

    def _init_grid(self):
        """Assign each particle to a checkerboard cell from its XY position."""
        pos = self.cloth.get_particles_pos().cpu().numpy()  # (N, 3)
        cell_w = self.cloth_w / self.grid_n
        cell_h = self.cloth_h / self.grid_n

        self._cell_particles: list[list[list[int]]] = [
            [[] for _ in range(self.grid_n)]
            for _ in range(self.grid_n)
        ]

        for p_idx, (px, py, _) in enumerate(pos):
            col = int((px + self.cloth_w / 2) / cell_w)
            row = int((py + self.cloth_h / 2) / cell_h)
            col = int(np.clip(col, 0, self.grid_n - 1))
            row = int(np.clip(row, 0, self.grid_n - 1))
            self._cell_particles[row][col].append(p_idx)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_particle_positions(self) -> np.ndarray:
        """Return (N, 3) array of current cloth particle positions."""
        return self.cloth.get_particles_pos().cpu().numpy()

    def reset(self) -> np.ndarray:
        """Restore cloth to its initial flat configuration."""
        self.scene.load_checkpoint(self._ckpt_path)
        if self.show_viewer:
            self.scene.visualizer.update()
        return self.get_particle_positions()

    # ------------------------------------------------------------------
    # Fold action
    # ------------------------------------------------------------------

    def _fold_axis(self, cell_row: int, cell_col: int, direction: str):
        """
        Return (axis_val, axis_dim, flap_gt, rot_sign) for a fold action.

        axis_val  – coordinate of the fold crease
        axis_dim  – 0 for E/W folds (crease is a y-parallel line),
                    1 for N/S folds (crease is an x-parallel line)
        flap_gt   – True when the flap is on the side where coord > axis_val
        rot_sign  – +1 or -1; ensures the arc sweeps through positive z
                    (i.e. the flap always lifts upward before landing)
        """
        cell_w = self.cloth_w / self.grid_n
        cell_h = self.cloth_h / self.grid_n
        x_lo = -self.cloth_w / 2 + cell_col * cell_w
        y_lo = -self.cloth_h / 2 + cell_row * cell_h

        if direction == 'N':   # crease at top of cell; fold everything north of it down
            return y_lo + cell_h, 1, True,  +1
        if direction == 'S':   # crease at bottom of cell; fold everything south of it up
            return y_lo,         1, False, -1
        if direction == 'E':   # crease at right of cell; fold everything east of it left
            return x_lo + cell_w, 0, True,  +1
        # 'W'                  # crease at left of cell; fold everything west of it right
        return x_lo, 0, False, -1

    def execute_fold(
        self,
        cell_row: int,
        cell_col: int,
        direction: str = 'N',
        step_delay: float = 0.0,
    ) -> np.ndarray:
        """
        Fold the cloth along the specified boundary of cell (cell_row, cell_col).

        Every particle on the far side of the fold crease is pinned and driven
        through a π-radian arc in the plane perpendicular to the crease, then
        released to settle under PBD physics.

        Arc geometry
        ------------
        Let d = signed distance of a flap particle from the crease (in the
        axis_dim direction), and z = its height above the ground.
        At arc angle θ ∈ [0, π]:

            new_d = d·cos θ − rot_sign·z·sin θ
            new_z = rot_sign·d·sin θ + z·cos θ

        rot_sign ensures new_z > 0 during the sweep regardless of which side
        the flap starts on.

        Parameters
        ----------
        cell_row, cell_col : int  (0-indexed from bottom-left)
        direction : 'N' | 'S' | 'E' | 'W'
        step_delay : float
            Sleep time per step; useful for slowing down visual replay.

        Returns
        -------
        np.ndarray (N, 3) – particle positions after settling.
        """
        axis_val, axis_dim, flap_gt, rot_sign = self._fold_axis(
            cell_row, cell_col, direction)

        pos0 = self.cloth.get_particles_pos().cpu().numpy()  # (N, 3)

        margin = 1e-4
        if flap_gt:
            flap_mask = pos0[:, axis_dim] > axis_val + margin
        else:
            flap_mask = pos0[:, axis_dim] < axis_val - margin
        flap_idx = np.where(flap_mask)[0].astype(np.int32)

        if len(flap_idx) == 0:
            return pos0

        # Pin flap particles so PBD integration doesn't override our positions
        self.cloth.fix_particles(flap_idx)

        d = pos0[flap_idx, axis_dim] - axis_val  # (F,) signed crease distances
        z = pos0[flap_idx, 2]                     # (F,) initial heights

        new_pos = pos0.copy()
        for step in range(self.fold_steps + 1):
            theta = np.pi * step / self.fold_steps
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            new_pos[flap_idx, axis_dim] = axis_val + d * cos_t - rot_sign * z * sin_t
            new_pos[flap_idx, 2]        = rot_sign * d * sin_t + z * cos_t

            arc_pos = torch.tensor(new_pos[flap_idx], dtype=torch.float32)
            self.cloth.set_particles_pos(arc_pos, particles_idx_local=flap_idx)
            self.scene.step()

            if step_delay > 0:
                time.sleep(step_delay)

        # Release kinematic control and let cloth settle
        self.cloth.release_particle(flap_idx)

        for _ in range(self.settle_steps):
            self.scene.step()
            if step_delay > 0:
                time.sleep(step_delay * 0.25)

        return self.get_particle_positions()

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def get_valid_actions(self) -> list[dict]:
        """Return all (cell_row, cell_col, direction) fold combinations."""
        return [
            {'cell_row': r, 'cell_col': c, 'direction': d}
            for r in range(self.grid_n)
            for c in range(self.grid_n)
            for d in ('N', 'S', 'E', 'W')
        ]