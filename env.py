"""
BinEnv: Genesis simulation of a bin with objects.
Two thin pushers (N/S-oriented and E/W-oriented) act kinematically.
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

OBJ_SIZE = 0.08        # object cube side length
OBJ_H    = OBJ_SIZE / 2  # object center z when resting on floor

PUSHER_T = 0.012              # thin dimension of each pusher blade
PUSHER_W = OBJ_SIZE * 0.88   # wide dimension (slightly smaller than objects)

EXIT_Y = -0.05  # target exits when its y < EXIT_Y

_PARK = [BIN_W / 2, -2.0, OBJ_H]  # safe parking position outside the bin


class BinEnv:
    """
    Wraps a Genesis scene with a bin, N obstacle objects, and 1 target object.

    Two thin pusher blades are kinematically controlled:
      pusher_ns  – thin in y, wide in x; used for north/south strokes
      pusher_ew  – thin in x, wide in y; used for east/west strokes

    Z heights for pushing are discretized into `n_z_levels` evenly spaced
    levels from floor height up through the stacking range.

    Parameters
    ----------
    n_obstacles : int
    show_viewer : bool
    dt : float
    seed : int | None
    stackable : bool
    friction : float
    n_z_levels : int
        Number of discrete push heights (1 = floor only).
    """

    def __init__(self, n_obstacles: int = 2, show_viewer: bool = False,
                 dt: float = 0.01, seed: int | None = None,
                 stackable: bool = False, friction: float = 1.0,
                 n_z_levels: int = 1,
                 push_steps: int = 20, substeps: int = 4):
        self.n_obstacles = n_obstacles
        self.show_viewer = show_viewer
        self.friction = friction
        self.dt = dt
        self.stackable = stackable
        self.n_z_levels = n_z_levels
        self.push_steps = push_steps
        self.substeps = substeps
        self.rng = np.random.default_rng(seed)

        # Discrete z levels: floor height, one-box up, two-boxes up, …
        self.z_levels = [OBJ_H + i * OBJ_SIZE for i in range(n_z_levels)]

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_fov=30,
                camera_pos=(0.48511935, -0.76658447, 0.6780057),
                camera_lookat=(0.30222618, 0.05807024, 0.14275379),
            ),
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, -9.81),
                box_box_detection=False,      # accurate box-box contacts (all objects are boxes)
                enable_self_collision=False,  # boxes can't self-collide
                iterations=15,               # constraint solver iters (default 50)
                ls_iterations=10,            # line-search iters (default 50)
                use_hibernation=True,        # sleep resting objects
            ),
        )

        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
        )

        # --- floor ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(BIN_W + 2 * WALL_T, BIN_D + 2 * WALL_T, WALL_T),
                pos=(BIN_W / 2, BIN_D / 2, -WALL_T / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.6, 0.5)),
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

        # --- N/S pusher blade: wide in x, thin in y ---
        self.pusher_ns = self.scene.add_entity(
            gs.morphs.Box(
                size=(PUSHER_W, PUSHER_T, PUSHER_W),
                pos=_PARK,
            ),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), opacity=0.8),
        )

        # --- E/W pusher blade: thin in x, wide in y ---
        self.pusher_ew = self.scene.add_entity(
            gs.morphs.Box(
                size=(PUSHER_T, PUSHER_W, PUSHER_W),
                pos=_PARK,
            ),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.6, 0.1), opacity=0.8),
        )

        # --- target object ---
        self.target = self.scene.add_entity(
            gs.morphs.Box(
                size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
                pos=(BIN_W / 2, BIN_D / 2, OBJ_H),
            ),
            material=gs.materials.Rigid(rho=50, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), opacity=0.6),
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
                surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9), opacity=0.6),
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

        columns: list[tuple[float, float, int]] = []
        all_objs = self.obstacles + [self.target]

        for obj in all_objs:
            placed = False
            if self.stackable and columns and self.rng.random() < 0.5:
                idx = self.rng.integers(len(columns))
                x, y, count = columns[idx]
                obj.set_pos([x, y, OBJ_H + OBJ_SIZE * count])
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

        self._park_pushers()
        self._zero_all_velocities()

        for _ in range(60):
            self.scene.step()

        self._ckpt_path = os.path.join(tempfile.mkdtemp(), 'initial')
        self.scene.save_checkpoint(self._ckpt_path)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _park_pushers(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        self.pusher_ns.set_pos(_PARK)
        self.pusher_ns.set_quat(identity)
        self.pusher_ew.set_pos(_PARK)
        self.pusher_ew.set_quat(identity)

    def _zero_all_velocities(self):
        for obj in [self.pusher_ns, self.pusher_ew, self.target] + self.obstacles:
            obj.zero_all_dofs_velocity()

    def _get_state(self) -> dict:
        return {
            'target_pos':    self.target.get_pos().cpu().numpy().copy(),
            'target_quat':   self.target.get_quat().cpu().numpy().copy(),
            'obstacle_pos':  np.array([o.get_pos().cpu().numpy() for o in self.obstacles]),
            'obstacle_quat': np.array([o.get_quat().cpu().numpy() for o in self.obstacles]),
        }

    def _set_state(self, state: dict):
        self.target.set_pos(state['target_pos'].tolist())
        self.target.set_quat(state['target_quat'].tolist())
        for i, obs in enumerate(self.obstacles):
            obs.set_pos(state['obstacle_pos'][i].tolist())
            obs.set_quat(state['obstacle_quat'][i].tolist())
        self._park_pushers()
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
    # Action primitives
    # ------------------------------------------------------------------

    def _stroke_and_eval(self, pusher, start_3d: list, end_3d: list,
                         steps: int, step_delay: float) -> tuple[dict, float, bool]:
        """Teleport pusher to start, sweep to end, park, settle, return result."""
        start = np.array(start_3d, dtype=float)
        end   = np.array(end_3d,   dtype=float)
        identity = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z

        pusher.set_pos(start.tolist())
        pusher.set_quat(identity)
        self._zero_all_velocities()
        for _ in range(2):
            pusher.set_pos(start.tolist())
            pusher.set_quat(identity)
            self.scene.step()

        for i in range(steps):
            t = (i + 1) / steps
            pusher.set_pos((start + t * (end - start)).tolist())
            pusher.set_quat(identity)
            self.scene.step()
            if step_delay > 0:
                time.sleep(step_delay)

        self._park_pushers()
        self._zero_all_velocities()
        for _ in range(2):
            self.scene.step()

        state  = self._get_state()
        reward = self._compute_reward(state)
        done   = self._is_goal(state)
        return state, reward, done

    def execute_ns_push(self, pos_2d: np.ndarray, z: float,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push northward: pusher_ns enters from south, sweeps north."""
        start = [pos_2d[0], pos_2d[1] - approach_dist, z]
        end   = [pos_2d[0], pos_2d[1] + push_dist,     z]
        return self._stroke_and_eval(self.pusher_ns, start, end,
                                     push_steps or self.push_steps, step_delay)

    def execute_ns_pull(self, pos_2d: np.ndarray, z: float,
                        approach_dist: float = 0.12,
                        pull_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Pull southward: pusher_ns hooks north of object, sweeps south to exit."""
        start = [pos_2d[0], pos_2d[1] + approach_dist, z]
        end   = [pos_2d[0], EXIT_Y - approach_dist,     z]
        return self._stroke_and_eval(self.pusher_ns, start, end,
                                     pull_steps or self.push_steps, step_delay)

    def execute_ew_push(self, pos_2d: np.ndarray, z: float, direction: int,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push east (direction=+1) or west (direction=-1): pusher_ew sweeps laterally."""
        start = [pos_2d[0] - direction * approach_dist, pos_2d[1], z]
        end   = [pos_2d[0] + direction * push_dist,     pos_2d[1], z]
        return self._stroke_and_eval(self.pusher_ew, start, end,
                                     push_steps or self.push_steps, step_delay)

    # ------------------------------------------------------------------
    # Reward / goal
    # ------------------------------------------------------------------

    def _obstacles_dropped(self, state: dict) -> bool:
        return any(state['obstacle_pos'][i][1] < EXIT_Y
                   for i in range(len(self.obstacles)))

    def _compute_reward(self, state: dict) -> float:
        y = state['target_pos'][1]
        r = (BIN_D / 2 - y) / (BIN_D / 2 - EXIT_Y)
        r = float(np.clip(r, 0, 1))
        n_dropped = sum(1 for i in range(len(self.obstacles))
                        if state['obstacle_pos'][i][1] < EXIT_Y)
        r -= 0.5 * n_dropped
        return r

    def _is_goal(self, state: dict) -> bool:
        if self._obstacles_dropped(state):
            return False
        return float(state['target_pos'][1]) < EXIT_Y

    def is_goal(self, state: dict) -> bool:
        return self._is_goal(state)

    # ------------------------------------------------------------------
    # Convenience
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
