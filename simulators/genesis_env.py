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

Can operate in single-env mode (n_envs=1) or parallel mode (n_envs>1).
In parallel mode, multiple independent physics worlds are evaluated simultaneously on GPU.
"""

import colorsys
import os
import tempfile
import time
import torch
import genesis as gs
from .base_env import SimulatorEnv


def _obstacle_color(i: int, n: int) -> tuple:
    """Blue-family color for obstacle i of n, spread from cyan-blue to indigo-blue."""
    t = i / max(n - 1, 1)
    hue = 0.55 + t * 0.17
    sat = 0.65
    val = 0.95 - t * 0.20
    return colorsys.hsv_to_rgb(hue, sat, val)

# Bin dimensions
BIN_W = 1.0   # x extent
BIN_D = 1.0   # y extent (depth, from 0 to BIN_D)
BIN_H = 0.5  # wall height
WALL_T = 0.05 # wall thickness

PUSHER_T = 0.012              # thin dimension of each pusher blade

EXIT_Y = -0.05  # target exits when its y < EXIT_Y


class BinEnv(SimulatorEnv):
    """
    Wraps a Genesis scene with a bin, N obstacle objects, and 1 target object.

    Two thin pusher blades are kinematically controlled:
      pusher_ns  – thin in y, wide in x; used for north/south strokes
      pusher_ew  – thin in x, wide in y; used for east/west strokes

    Z heights for pushing are discretized into `n_z_levels` evenly spaced
    levels from floor height up through the stacking range.

    Can operate in single-env mode (n_envs=1) or parallel mode (n_envs>1).
    In parallel mode, multiple independent physics worlds are evaluated simultaneously on GPU.

    Parameters
    ----------
    n_obstacles : int
    n_envs : int
        Number of environments. n_envs=1 is single mode with RNG and checkpoint support.
        n_envs>1 is parallel mode with multiple independent worlds.
    show_viewer : bool
        Whether to show viewer.
    dt : float
    seed : int | None
    stackable : bool
    friction : float
    max_stack_height : int
        Number of discrete push heights (1 = floor only).
    push_steps : int
    substeps : int
    """

    def __init__(self, n_obstacles: int = 2, n_envs: int = 1,
                 dt: float = 0.01, seed: int | None = None,
                 stackable: bool = False, friction: float = 1.0,
                 max_stack_height: int = 1,
                 push_steps: int = 20, substeps: int = 4,
                 wall_thickness: float = WALL_T,
                 difficult_spawn: bool = False,
                 reward_cfg=None,
                 bin_size: float | None = None,
                 bin_size_factor: float = 0.9,
                 obj_size: float = 0.05,
                 debug: bool = False):
        # Initialize Genesis once per process
        try:
            gs.init(logging_level='info', backend=gs.gpu, performance_mode=True)
        except gs.GenesisException:
            # Already initialized
            pass

        super().__init__(n_obstacles=n_obstacles, n_envs=n_envs,
                         dt=dt, seed=seed, stackable=stackable, friction=friction,
                         max_stack_height=max_stack_height, push_steps=push_steps, substeps=substeps,
                         wall_thickness=wall_thickness, difficult_spawn=difficult_spawn,
                         reward_cfg=reward_cfg, bin_size=bin_size,
                         bin_size_factor=bin_size_factor, obj_size=obj_size, debug=debug)

        self._OBJ_H    = self._OBJ_SIZE / 2
        self._pusher_w = self._OBJ_SIZE * 0.88

        self._park = [-0.3, self.bin_d / 2, self._OBJ_H]
        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self):
        """Build Genesis scene with obstacles and pushers."""
        wt = self.wall_thickness
        bw, bd = self.bin_w, self.bin_d
        # Camera scales with bin size: lookat at front-center of bin, pulled back 1.5x bin size
        _view_dist = max(bw, bd) * 1.5
        _lookat = (bw / 2, bd * 0.1, self._OBJ_H)
        _cam_dir = (0.183, -0.825, 0.535)  # unit vector: right, back, up
        _cam_pos = tuple(_lookat[i] + _cam_dir[i] * _view_dist for i in range(3))
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_fov=30,
                camera_pos=_cam_pos,
                camera_lookat=_lookat,
            ),
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, -9.81),
                box_box_detection=False,
                enable_self_collision=False,
                iterations=8,
                ls_iterations=5,
                use_hibernation=True,
                use_contact_island=True
            ),
        )
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane())

        # --- floor ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(bw + 2 * wt, bd + 2 * wt, wt),
                pos=(bw / 2, bd / 2, -wt / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.6, 0.5)),
        )

        # --- north wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(bw + 2 * wt, wt, BIN_H),
                pos=(bw / 2, bd + wt / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- west wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(wt, bd, BIN_H),
                pos=(-wt / 2, bd / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- east wall ---
        self.scene.add_entity(
            gs.morphs.Box(
                size=(wt, bd, BIN_H),
                pos=(bw + wt / 2, bd / 2, BIN_H / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.8), opacity=0.35),
        )

        # --- N/S pusher blade: wide in x, thin in y ---
        self.pusher_ns = self.scene.add_entity(
            gs.morphs.Box(
                size=(self._pusher_w, PUSHER_T, self._pusher_w),
                pos=self._park,
            ),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2), opacity=0.8),
        )

        # --- E/W pusher blade: thin in x, wide in y ---
        self.pusher_ew = self.scene.add_entity(
            gs.morphs.Box(
                size=(PUSHER_T, self._pusher_w, self._pusher_w),
                pos=self._park,
            ),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.6, 0.1), opacity=0.8),
        )

        # --- target object ---
        self.target = self.scene.add_entity(
            gs.morphs.Box(
                size=(self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE),
                pos=(bw / 2, bd / 2, self._OBJ_H),
            ),
            material=gs.materials.Rigid(rho=50, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), opacity=0.6),
        )

        # --- obstacle objects ---
        self.obstacles = []
        for oi in range(self.n_obstacles):
            obs = self.scene.add_entity(
                gs.morphs.Box(
                    size=(self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE),
                    pos=(bw / 2, bd / 2, self._OBJ_H),
                ),
                material=gs.materials.Rigid(rho=500, friction=self.friction),
                surface=gs.surfaces.Default(color=_obstacle_color(oi, self.n_obstacles), opacity=0.6),
            )
            self.obstacles.append(obs)

        wt = self.wall_thickness
        self.scene.build(n_envs=self.n_envs,
                         env_spacing=((bw + 2*wt) * 2, (bd + 2*wt) * 2))
        self._place_objects()

    # ------------------------------------------------------------------
    # Object placement
    # ------------------------------------------------------------------

    def _place_objects(self, state: dict | None = None):
        """Place objects in the bin. If state is given, use it directly; otherwise generate randomly."""
        from simulators.placement import random_initial_state
        if state is None:
            state = random_initial_state(
                self.n_obstacles,
                obj_size=self._OBJ_SIZE,
                stackable=self.stackable,
                difficult_spawn=self.difficult_spawn,
                bin_w=self.bin_w,
                bin_d=self.bin_d,
                debug=self.debug,
            )
        self._set_state(state)

        for _ in range(60):
            self.scene.step()

        self._ckpt_path = os.path.join(tempfile.mkdtemp(), 'initial')
        self.scene.save_checkpoint(self._ckpt_path)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _park_pushers(self, env_idx: int | None = None):
        """Park pushers at safe position. If env_idx set, only park in that env slot."""
        identity = [1.0, 0.0, 0.0, 0.0]
        kwargs = {'envs_idx': [env_idx]} if env_idx is not None else {}
        self.pusher_ns.set_pos(self._park, **kwargs)
        self.pusher_ns.set_quat(identity, **kwargs)
        self.pusher_ew.set_pos(self._park, **kwargs)
        self.pusher_ew.set_quat(identity, **kwargs)

    def _zero_all_velocities(self, env_idx: int | None = None):
        """Zero all object velocities. If env_idx set, only zero in that env slot."""
        kwargs = {'envs_idx': [env_idx]} if env_idx is not None else {}
        for obj in [self.pusher_ns, self.pusher_ew, self.target] + self.obstacles:
            obj.zero_all_dofs_velocity(**kwargs)

    def _get_state(self, env_idx: int) -> dict:
        """Get state (pos, quat of all objects) from the given env slot.
        Returns torch tensors (squeezed to remove batch dimension).
        """
        ei = [env_idx]
        return {
            'target_pos':    self.target.get_pos(envs_idx=ei)[0].squeeze(),
            'target_quat':   self.target.get_quat(envs_idx=ei)[0].squeeze(),
            'obstacle_pos':  torch.stack([o.get_pos(envs_idx=ei)[0].squeeze() for o in self.obstacles]),
            'obstacle_quat': torch.stack([o.get_quat(envs_idx=ei)[0].squeeze() for o in self.obstacles]),
        }

    def _set_state(self, state: dict, env_idx: int | None = None):
        """Set state (teleport all objects). If env_idx set, set in that env slot.
        Accepts torch tensors (will convert to list for Genesis API).
        """
        identity = [1.0, 0.0, 0.0, 0.0]
        kwargs = {'envs_idx': [env_idx]} if env_idx is not None else {}

        self.target.set_pos(state['target_pos'].tolist(), **kwargs)
        self.target.set_quat(state['target_quat'].tolist(), **kwargs)
        self.target.zero_all_dofs_velocity(**kwargs)

        for i, obs in enumerate(self.obstacles):
            obs.set_pos(state['obstacle_pos'][i].tolist(), **kwargs)
            obs.set_quat(state['obstacle_quat'][i].tolist(), **kwargs)
            obs.zero_all_dofs_velocity(**kwargs)

        for pusher in (self.pusher_ns, self.pusher_ew):
            pusher.set_pos(self._park, **kwargs)
            pusher.set_quat(identity, **kwargs)
            pusher.zero_all_dofs_velocity(**kwargs)

    def get_state(self, env_idx: int) -> dict:
        """Public interface: get state from the given env slot."""
        return self._get_state(env_idx)

    def set_state(self, state: dict, env_idx: int | None = None):
        """Public interface: set state in env_idx (parallel) or current (single)."""
        self._set_state(state, env_idx)

    def reset(self, seed: int | None = None, env_idx: int | None = None) -> dict:
        """
        Reset environment.
        - If env_idx is given: teleport that env slot to a neutral initial state.
        - Otherwise: full physics reset via checkpoint (or re-place if seed given).
        """
        if env_idx is not None:
            initial_state = {
                'target_pos': torch.tensor([self.bin_w/2, self.bin_d/2, self._OBJ_H]),
                'target_quat': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                'obstacle_pos': torch.tile(
                    torch.tensor([self.bin_w/2, self.bin_d/2, self._OBJ_H]), (self.n_obstacles, 1)
                ),
                'obstacle_quat': torch.tile(
                    torch.tensor([1.0, 0.0, 0.0, 0.0]), (self.n_obstacles, 1)
                ),
            }
            self._set_state(initial_state, env_idx)
            return self._get_state(env_idx)

        if seed is not None:
            torch.manual_seed(seed)
            self._place_objects()
            return self._get_state(0)
        self.scene.load_checkpoint(self._ckpt_path)
        if self.show_viewer:
            self.scene.visualizer.update()
        return self._get_state(0)

    # ------------------------------------------------------------------
    # Action primitives (single env and parallel modes)
    # ------------------------------------------------------------------



    def execute_ns_push(self, pos_2d: torch.Tensor, z: float,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push northward: pusher_ns enters from south, sweeps north."""
        state = self._get_state(0)
        action = {
            'action_type': 'push_n',
            'push_pos': pos_2d,
            'push_z': z,
        }
        results = self.batch_evaluate([(state, action)])
        return results[0]

    def execute_ns_pull(self, pos_2d: torch.Tensor, z: float,
                        approach_dist: float = 0.12,
                        pull_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Pull southward: pusher_ns hooks north of object, sweeps south to exit."""
        state = self._get_state(0)
        action = {
            'action_type': 'pull_s',
            'push_pos': pos_2d,
            'push_z': z,
        }
        results = self.batch_evaluate([(state, action)])
        return results[0]

    def execute_ew_push(self, pos_2d: torch.Tensor, z: float, direction: int,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push east (direction=+1) or west (direction=-1): pusher_ew sweeps laterally."""
        state = self._get_state(0)
        action = {
            'action_type': 'push_e' if direction > 0 else 'push_w',
            'push_pos': pos_2d,
            'push_z': z,
        }
        results = self.batch_evaluate([(state, action)])
        return results[0]

    # ------------------------------------------------------------------
    # Batch evaluation (parallel mode only)
    # ------------------------------------------------------------------

    def _batch_evaluate_impl(self, pairs: list[tuple[dict, dict]]) -> list[tuple[dict, float, bool]]:
        """
        Evaluate up to n_envs (state, action) pairs in parallel. (Parallel mode only)

        Each pair gets its own env slot. All slots advance together via
        a single scene.step() call per physics tick — regardless of how
        many pairs are active, the wall-clock cost is ~constant (GPU).

        Parameters
        ----------
        pairs : list of (state_dict, action_dict)
            len(pairs) must be <= self.n_envs

        Returns
        -------
        list of (new_state, reward, done) — one per input pair
        """
        k = len(pairs)
        if k == 0:
            return []
        if k > self.n_envs:
            # Chunk: split into batches and concatenate results
            results = []
            for i in range(0, k, self.n_envs):
                results.extend(self._batch_evaluate_impl(pairs[i:i + self.n_envs]))
            return results

        identity = [1.0, 0.0, 0.0, 0.0]

        # 1. Teleport each env to its starting state and compute stroke geometry
        strokes: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for env_idx, (state, action) in enumerate(pairs):
            self._set_state(state, env_idx)
            pusher_type, start, end = self._action_to_stroke(action)
            strokes.append((pusher_type, start, end))

        # 2. Warm-up: place pushers at stroke start, settle 2 ticks
        for env_idx, (ptype, start, _) in enumerate(strokes):
            pusher = self.pusher_ns if ptype == 'ns' else self.pusher_ew
            pusher.set_pos(start, envs_idx=[env_idx])
            pusher.set_quat(identity, envs_idx=[env_idx])
        self.scene.step()
        self.scene.step()

        # 3. Sweep — one scene.step() advances ALL envs simultaneously
        for step_i in range(self.push_steps):
            t = (step_i + 1) / self.push_steps
            for env_idx, (ptype, start, end) in enumerate(strokes):
                pos    = [s + t * (e - s) for s, e in zip(start, end)]
                pusher = self.pusher_ns if ptype == 'ns' else self.pusher_ew
                pusher.set_pos(pos, envs_idx=[env_idx])
                pusher.set_quat(identity, envs_idx=[env_idx])
            self.scene.step()

        # 4. Park all pushers and settle
        for env_idx in range(k):
            for pusher in (self.pusher_ns, self.pusher_ew):
                pusher.set_pos(self._park, envs_idx=[env_idx])
                pusher.set_quat(identity, envs_idx=[env_idx])
                pusher.zero_all_dofs_velocity(envs_idx=[env_idx])
        self.scene.step()
        self.scene.step()

        # 5. Read back results
        results = []
        for env_idx in range(k):
            state  = self._get_state(env_idx)
            reward = self._compute_reward(state)
            done   = self._is_goal(state)
            results.append((state, reward, done))
        return results

    # ------------------------------------------------------------------
    # Reward / goal
    # ------------------------------------------------------------------

    def _obstacles_dropped(self, state: dict) -> bool:
        return any(state['obstacle_pos'][i][1] < EXIT_Y
                   for i in range(len(self.obstacles)))

    def _is_goal(self, state: dict) -> bool:
        if self._obstacles_dropped(state):
            return False
        return bool(state['target_pos'][1] <= EXIT_Y)

    def step_physics(self) -> None:
        self.scene.step()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_target_pos_2d(self, state: dict | None = None, env_idx: int = 0) -> torch.Tensor:
        if state is None:
            state = self._get_state(env_idx)
        return state['target_pos'][:2]

    def get_all_obj_positions_2d(self, state: dict | None = None, env_idx: int = 0) -> torch.Tensor:
        """Returns (N+1, 2) tensor: [target, obs0, obs1, ...]"""
        if state is None:
            state = self._get_state(env_idx)
        pos = [state['target_pos'][:2]]
        for i in range(len(self.obstacles)):
            pos.append(state['obstacle_pos'][i][:2])
        return torch.stack(pos)
