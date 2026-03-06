"""
BinEnv: Genesis simulation of a bin with objects.
A pusher (kinematically controlled box) can push rigid objects.
Goal: move the target object out of the bin through the open south side.

Bin layout (top-down, z-up):
  - Floor at z=0
  - North wall: +y side
  - West wall:  -x side
  - East wall:  +x side
  - OPEN south side: -y exit
  - Objects start inside the bin
"""

import os
import tempfile
import time
import numpy as np
import genesis as gs

gs.init(logging_level='warning')

# Bin dimensions
BIN_W = 0.3   # x extent
BIN_D = 0.3   # y extent (depth, from 0 to BIN_D)
BIN_H = 0.15  # wall height
WALL_T = 0.02 # wall thickness

OBJ_SIZE = 0.08   # object cube half-size * 2
PUSHER_SIZE = 0.07  # pusher cube size
OBJ_H = OBJ_SIZE / 2  # resting z

EXIT_Y = -0.05  # target exits when its y < EXIT_Y


class BinEnv:
    """
    Wraps a Genesis scene with a bin, N obstacle objects, and 1 target object.
    The pusher is a kinematically moved box.

    Parameters
    ----------
    n_obstacles : int
        Number of obstacle objects in the bin.
    show_viewer : bool
        Whether to open the interactive viewer.
    dt : float
        Simulation timestep.
    seed : int | None
        Random seed for object placement.
    """

    def __init__(self, n_obstacles: int = 2, show_viewer: bool = False,
                 dt: float = 0.01, seed: int | None = None, stackable: bool = False,
                 friction: float = 1.0):
        self.n_obstacles = n_obstacles
        self.show_viewer = show_viewer
        self.friction = friction
        self.dt = dt
        self.stackable = stackable
        self.rng = np.random.default_rng(seed)

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_fov = 30,
                camera_pos = (0.48511935, -0.76658447, 0.6780057), 
                camera_lookat = (0.30222618, 0.05807024, 0.14275379),
            ),
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, -9.81),
            ),
        )

        # --- floor ---
        self.floor = self.scene.add_entity(
            gs.morphs.Box(
                size=(BIN_W + 2 * WALL_T, BIN_D + 2 * WALL_T, WALL_T),
                pos=(BIN_W / 2, BIN_D / 2, -WALL_T / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.6, 0.5, 1.0)),
        )

        # --- north wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(BIN_W + 2 * WALL_T, WALL_T, BIN_H),
                pos=(BIN_W / 2, BIN_D + WALL_T / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- west wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(WALL_T, BIN_D, BIN_H),
                pos=(-WALL_T / 2, BIN_D / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- east wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(WALL_T, BIN_D, BIN_H),
                pos=(BIN_W + WALL_T / 2, BIN_D / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- pusher (kinematic) ---
        self.pusher = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.5*PUSHER_SIZE, 0.5*PUSHER_SIZE, 0.5*PUSHER_SIZE),
                pos=(BIN_W / 2, BIN_D / 2, OBJ_H),
            ),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2), opacity=0.35),
        )

        # --- target object ---
        self.target = self.scene.add_entity(
            gs.morphs.Box(
                size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
                pos=(BIN_W / 2, BIN_D / 2, OBJ_H),
            ),
            material=gs.materials.Rigid(rho=500, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), opacity=0.35),
        )

        # --- obstacle objects ---
        self.obstacles = []
        for _ in range(self.n_obstacles):
            obs = self.scene.add_entity(
                gs.morphs.Box(
                    size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
                    pos=(BIN_W / 2, BIN_D / 2, OBJ_H),
                ),
                material=gs.materials.Rigid(rho=500, friction=self.friction),
                surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9), opacity=0.35),
            )
            self.obstacles.append(obs)

        self.scene.build()
        self._place_objects()

    # ------------------------------------------------------------------
    # Object placement
    # ------------------------------------------------------------------

    def _place_objects(self):
        """Randomly place objects inside the bin, optionally stacking them."""
        margin = OBJ_SIZE * 0.7
        x_lo, x_hi = margin, BIN_W - margin
        y_lo, y_hi = margin, BIN_D - margin

        # positions list stores (x, y, stack_height) where stack_height is
        # how many objects are already at that (x, y) column.
        columns: list[tuple[float, float, int]] = []  # (x, y, count)
        all_objs = [self.target] + self.obstacles

        for obj in all_objs:
            placed = False
            # If stackable, first try placing on top of an existing column.
            if self.stackable and columns and self.rng.random() < 0.5:
                idx = self.rng.integers(len(columns))
                x, y, count = columns[idx]
                z = OBJ_H + OBJ_SIZE * count
                obj.set_pos([x, y, z])
                columns[idx] = (x, y, count + 1)
                placed = True

            if not placed:
                for _ in range(200):
                    x = self.rng.uniform(x_lo, x_hi)
                    y = self.rng.uniform(y_lo, y_hi)
                    if all(np.hypot(x - cx, y - cy) > OBJ_SIZE * 1.5
                           for cx, cy, _ in columns):
                        obj.set_pos([x, y, OBJ_H])
                        columns.append((x, y, 1))
                        break

        # Park pusher at the bin exit edge (y=0, still on the floor)
        self.pusher.set_pos([BIN_W / 2, 0.0, OBJ_H])
        self._zero_all_velocities()

        # Settle objects
        for _ in range(60):
            self.scene.step()

        self._initial_state = self._get_state()
        self._ckpt_path = os.path.join(tempfile.mkdtemp(), 'initial')
        self.scene.save_checkpoint(self._ckpt_path)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _zero_all_velocities(self):
        for obj in [self.pusher, self.target] + self.obstacles:
            obj.zero_all_dofs_velocity()

    def _get_state(self) -> dict:
        """Return a lightweight state dict (numpy arrays, copyable)."""
        state = {
            'target_pos': self.target.get_pos().cpu().numpy().copy(),
            'target_quat': self.target.get_quat().cpu().numpy().copy(),
            'obstacle_pos': np.array([o.get_pos().cpu().numpy() for o in self.obstacles]),
            'obstacle_quat': np.array([o.get_quat().cpu().numpy() for o in self.obstacles]),
            'pusher_pos': self.pusher.get_pos().cpu().numpy().copy(),
        }
        return state

    def _set_state(self, state: dict):
        """Restore simulation state from a dict (without disk I/O)."""
        self.target.set_pos(state['target_pos'].tolist())
        self.target.set_quat(state['target_quat'].tolist())
        for i, obs in enumerate(self.obstacles):
            obs.set_pos(state['obstacle_pos'][i].tolist())
            obs.set_quat(state['obstacle_quat'][i].tolist())
        self.pusher.set_pos(state['pusher_pos'].tolist())
        self._zero_all_velocities()

    def get_state(self) -> dict:
        return self._get_state()

    def set_state(self, state: dict):
        self._set_state(state)

    def reset(self, seed: int | None = None) -> dict:
        """Full physics reset to initial placement via checkpoint."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._place_objects()
            return self._get_state()
        self.scene.load_checkpoint(self._ckpt_path)
        if self.show_viewer:
            self.scene.visualizer.update()
        return self._get_state()

    # ------------------------------------------------------------------
    # Pushing actions
    # ------------------------------------------------------------------

    def execute_push(self, push_pos_2d: np.ndarray, push_dir_2d: np.ndarray,
                     push_z: float | None = None,
                     push_dist: float = 0.25, approach_dist: float = 0.12,
                     push_steps: int = 80, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """
        Move pusher behind `push_pos_2d` and push in `push_dir_2d`.

        Parameters
        ----------
        push_pos_2d : (2,) array - (x, y) point to push at
        push_dir_2d : (2,) array - unit push direction in xy
        push_z      : z height of pusher center during push; defaults to OBJ_H (floor level)
        push_dist   : total distance to travel forward
        approach_dist : how far behind the push point to start
        push_steps  : simulation steps during push stroke

        Returns
        -------
        state : dict
        reward : float
        done : bool
        """
        d = push_dir_2d / (np.linalg.norm(push_dir_2d) + 1e-9)
        z = OBJ_H if push_z is None else float(push_z)

        # Start position: behind the push point
        start_xy = push_pos_2d - d * approach_dist
        end_xy = push_pos_2d + d * push_dist

        # Move pusher to start (teleport, no collision)
        start_3d = [start_xy[0], start_xy[1], z]
        self.pusher.set_pos(start_3d)
        self._zero_all_velocities()
        # settle briefly
        for _ in range(5):
            self.pusher.set_pos(start_3d)
            self.scene.step()

        # Execute push stroke
        for i in range(push_steps):
            t = (i + 1) / push_steps
            cur_xy = start_xy + t * (end_xy - start_xy)
            self.pusher.set_pos([cur_xy[0], cur_xy[1], z])
            self.scene.step()
            if step_delay > 0:
                time.sleep(step_delay)

        # Retract pusher to exit edge
        self.pusher.set_pos([BIN_W / 2, 10.0, OBJ_H])
        self._zero_all_velocities()
        for _ in range(20):
            self.scene.step()

        state = self._get_state()
        reward = self._compute_reward(state)
        done = self._is_goal(state)
        return state, reward, done

    def execute_pull(self, pull_pos_2d: np.ndarray,
                     pull_z: float | None = None,
                     approach_dist: float = 0.12,
                     pull_steps: int = 80, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """
        Pull an object toward the exit by entering from the south, teleporting
        to the north side of the object, then sweeping southward to the exit.

        Parameters
        ----------
        pull_pos_2d : (2,) array - (x, y) position of the object to pull
        pull_z      : z height of pusher; defaults to OBJ_H
        approach_dist : how far north of the object to start the pull stroke
        pull_steps  : simulation steps during pull stroke
        """
        z = OBJ_H if pull_z is None else float(pull_z)

        # Start north of the object (arm enters from south, hooks behind)
        start_xy = np.array([pull_pos_2d[0], pull_pos_2d[1] + approach_dist])
        # Pull all the way to the exit edge
        end_xy = np.array([pull_pos_2d[0], EXIT_Y - approach_dist])

        start_3d = [start_xy[0], start_xy[1], z]
        self.pusher.set_pos(start_3d)
        self._zero_all_velocities()
        for _ in range(5):
            self.pusher.set_pos(start_3d)
            self.scene.step()

        # Sweep southward
        for i in range(pull_steps):
            t = (i + 1) / pull_steps
            cur_xy = start_xy + t * (end_xy - start_xy)
            self.pusher.set_pos([cur_xy[0], cur_xy[1], z])
            self.scene.step()
            if step_delay > 0:
                time.sleep(step_delay)

        # Retract pusher to exit edge
        self.pusher.set_pos([BIN_W / 2, 0.0, OBJ_H])
        self._zero_all_velocities()
        for _ in range(20):
            self.scene.step()

        state = self._get_state()
        reward = self._compute_reward(state)
        done = self._is_goal(state)
        return state, reward, done

    # ------------------------------------------------------------------
    # Reward / goal
    # ------------------------------------------------------------------

    def _obstacles_dropped(self, state: dict) -> bool:
        """Return True if any obstacle has left the bin."""
        return any(
            state['obstacle_pos'][i][1] < EXIT_Y
            for i in range(len(self.obstacles))
        )

    def _compute_reward(self, state: dict) -> float:
        """Reward = progress of target toward exit, minus penalty for dropped obstacles."""
        y = state['target_pos'][1]
        # Normalize: 0 when y=BIN_D/2 (center), 1 when y=EXIT_Y
        r = (BIN_D / 2 - y) / (BIN_D / 2 - EXIT_Y)
        r = float(np.clip(r, 0, 1))
        # Penalty for each obstacle that has left the bin
        self.n_dropped = sum(
            1 for i in range(len(self.obstacles))
            if state['obstacle_pos'][i][1] < EXIT_Y
        )
        r -= 0.5 * self.n_dropped
        return r

    def _is_goal(self, state: dict) -> bool:
        if self.n_dropped > 0:
            return False
        return float(state['target_pos'][1]) < EXIT_Y

    def is_goal(self, state: dict) -> bool:
        return self._is_goal(state)

    # ------------------------------------------------------------------
    # Convenience: query object positions
    # ------------------------------------------------------------------

    def get_target_pos_2d(self, state: dict | None = None) -> np.ndarray:
        if state is None:
            state = self._get_state()
        return state['target_pos'][:2]

    def get_all_obj_positions_2d(self, state: dict | None = None) -> np.ndarray:
        """Returns (N+1, 2) array: [target, obs0, obs1, ...]"""
        if state is None:
            state = self._get_state()
        pos = [state['target_pos'][:2]]
        for i in range(len(self.obstacles)):
            pos.append(state['obstacle_pos'][i][:2])
        return np.array(pos)
