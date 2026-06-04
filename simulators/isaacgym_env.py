"""
IsaacGym implementation of the bin environment.

Mirrors genesis_env.py structure: same bin layout, same action primitives,
same reward/goal logic — only the simulator API differs.

Bin layout (top-down, z-up):
  - Floor at z=0
  - North wall: +y side
  - West wall:  -x side
  - East wall:  +x side
  - OPEN south side: -y exit
  - Objects start inside the bin

Supports single-env mode (n_envs=1) and parallel mode (n_envs>1).
"""

import numpy as np
import torch
from typing import List
from .base_env import SimulatorEnv

try:
    from isaacgym import gymapi, gymtorch
    ISAACGYM_AVAILABLE = True
except ImportError:
    ISAACGYM_AVAILABLE = False
    gymapi = None
    gymtorch = None

# Bin dimensions (same as Genesis)
BIN_W = 1.0   # x extent
BIN_D = 1.0   # y extent (depth, from 0 to BIN_D)
BIN_H = 0.5   # wall height
WALL_T = 0.05  # wall thickness

PUSHER_T = 0.012              # thin dimension of each pusher blade

EXIT_Y = -0.05  # target exits when its y < EXIT_Y

# Actor ordering within each env (must match create_actor call order in _build_scene)
_IDX_FLOOR    = 0
_IDX_NORTH    = 1
_IDX_WEST     = 2
_IDX_EAST     = 3
_IDX_NS       = 4  # N/S pusher
_IDX_EW       = 5  # E/W pusher
_IDX_TARGET   = 6
_IDX_OBS_BASE = 7  # obstacles start here


class BinEnvIsaacGym(SimulatorEnv):
    """
    IsaacGym implementation of the bin environment with pushers and objects.

    Two thin pusher blades are kinematically controlled each step by directly
    setting their root-state positions in the actor root-state tensor.

    Parameters
    ----------
    n_obstacles : int
    n_envs : int
        Number of environments. n_envs=1 is single mode with checkpoint support.
        n_envs>1 is parallel mode.
    show_viewer : bool
    dt : float
    seed : int | None
    stackable : bool
    friction : float
    max_stack_height : int
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
                 obj_size: float = 0.05):
        if not ISAACGYM_AVAILABLE:
            raise ImportError("isaacgym is not installed. Install it before using BinEnvIsaacGym.")

        super().__init__(n_obstacles=n_obstacles, n_envs=n_envs,
                         dt=dt, seed=seed,
                         stackable=stackable, friction=friction,
                         max_stack_height=max_stack_height, push_steps=push_steps,
                         substeps=substeps, wall_thickness=wall_thickness,
                         difficult_spawn=difficult_spawn, reward_cfg=reward_cfg,
                         bin_size=bin_size, bin_size_factor=bin_size_factor,
                         obj_size=obj_size)

        self._OBJ_H    = self._OBJ_SIZE / 2
        self._pusher_w = self._OBJ_SIZE * 0.88

        _park_y = -(max(self.bin_w, self.bin_d) * 1.5 + 0.1)
        self._park = [self.bin_w / 2, _park_y, self._OBJ_H]
        # Number of actors per env: floor + 3 walls + 2 pushers + 1 target + n_obstacles
        self.n_actors_per_env = 7 + n_obstacles

        # Checkpoint: saved root-state tensor for single-mode reset
        self._initial_root_states: torch.Tensor | None = None

        self._init_sim()
        self._build_scene()
        self._acquire_tensors()

        if self.n_envs == 1:
            self._place_objects()

    # ------------------------------------------------------------------
    # Simulator initialisation
    # ------------------------------------------------------------------

    def _init_sim(self):
        """Acquire gym handle and create the IsaacGym simulation."""
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = self.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True

        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1          # TGS
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(
            compute_device=0, graphics_device=0,
            type=gymapi.SIM_PHYSX, params=sim_params
        )
        if self.sim is None:
            raise RuntimeError("Failed to create IsaacGym simulation.")

        self.viewer = None
        if self.show_viewer:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            if self.viewer is None:
                raise RuntimeError("Failed to create IsaacGym viewer.")
            # Match Genesis camera view
            cam_pos    = gymapi.Vec3(0.485, -0.767, 0.678)
            cam_target = gymapi.Vec3(0.302,  0.058, 0.143)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _make_asset_options(self, fixed: bool = False):
        opts = gymapi.AssetOptions()
        opts.fix_base_link = fixed
        opts.disable_gravity = False
        opts.armature = 0.0
        return opts

    def _make_pose(self, pos: list, quat_xyzw: list | None = None):
        """Create a gymapi.Transform from pos [x,y,z] and optional quat [x,y,z,w]."""
        t = gymapi.Transform()
        t.p = gymapi.Vec3(*pos)
        if quat_xyzw is None:
            t.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # identity
        else:
            t.r = gymapi.Quat(*quat_xyzw)
        return t

    def _build_scene(self):
        """Create all IsaacGym assets and populate each env with actors."""
        gym, sim = self.gym, self.sim

        # ---- create assets ----

        fixed_opts   = self._make_asset_options(fixed=True)
        dynamic_opts = self._make_asset_options(fixed=False)
        pusher_opts  = self._make_asset_options(fixed=False)
        pusher_opts.disable_gravity = True  # pushers float at chosen z

        bw, bd = self.bin_w, self.bin_d
        wt = self.wall_thickness
        floor_asset      = gym.create_box_asset(sim, bw + 2*wt, bd + 2*wt, wt, fixed_opts)
        north_wall_asset = gym.create_box_asset(sim, bw + 2*wt, wt, BIN_H, fixed_opts)
        west_wall_asset  = gym.create_box_asset(sim, wt, bd, BIN_H, fixed_opts)
        east_wall_asset  = gym.create_box_asset(sim, wt, bd, BIN_H, fixed_opts)
        pusher_ns_asset  = gym.create_box_asset(sim, self._pusher_w, PUSHER_T, self._pusher_w, pusher_opts)
        pusher_ew_asset  = gym.create_box_asset(sim, PUSHER_T, self._pusher_w, self._pusher_w, pusher_opts)
        target_asset     = gym.create_box_asset(sim, self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE, dynamic_opts)
        obstacle_asset   = gym.create_box_asset(sim, self._OBJ_SIZE, self._OBJ_SIZE, self._OBJ_SIZE, dynamic_opts)

        n_cols = max(1, int(np.ceil(np.sqrt(self.n_envs))))
        park_y = self._park[1]
        env_lower = gymapi.Vec3(-(wt + 0.1), park_y - 0.2, 0.0)
        env_upper = gymapi.Vec3(bw + wt + 0.1, bd + wt + 0.1, 2.0)

        self.envs: list = []

        for env_idx in range(self.n_envs):
            env = gym.create_env(sim, env_lower, env_upper, n_cols)
            self.envs.append(env)

            # Collision group = env_idx so parallel envs don't interact;
            # filter = 0 means all actors within an env collide with each other.
            cg = env_idx

            # 0: floor
            gym.create_actor(env, floor_asset,
                             self._make_pose([bw/2, bd/2, -wt/2]),
                             "floor", cg, 0)
            # Set floor friction
            self._set_friction(env, 0, self.friction)

            # 1: north wall
            gym.create_actor(env, north_wall_asset,
                             self._make_pose([bw/2, bd + wt/2, BIN_H/2]),
                             "north_wall", cg, 0)

            # 2: west wall
            gym.create_actor(env, west_wall_asset,
                             self._make_pose([-wt/2, bd/2, BIN_H/2]),
                             "west_wall", cg, 0)

            # 3: east wall
            gym.create_actor(env, east_wall_asset,
                             self._make_pose([bw + wt/2, bd/2, BIN_H/2]),
                             "east_wall", cg, 0)

            # 4: N/S pusher
            gym.create_actor(env, pusher_ns_asset,
                             self._make_pose(self._park),
                             "pusher_ns", cg, 0)
            self._set_friction(env, _IDX_NS, self.friction)

            # 5: E/W pusher
            gym.create_actor(env, pusher_ew_asset,
                             self._make_pose(self._park),
                             "pusher_ew", cg, 0)
            self._set_friction(env, _IDX_EW, self.friction)

            # 6: target
            gym.create_actor(env, target_asset,
                             self._make_pose([bw/2, bd/2, self._OBJ_H]),
                             "target", cg, 0)
            self._set_friction(env, _IDX_TARGET, self.friction)

            # 7...: obstacles
            for obs_i in range(self.n_obstacles):
                gym.create_actor(env, obstacle_asset,
                                 self._make_pose([bw/2, bd/2, self._OBJ_H]),
                                 f"obstacle_{obs_i}", cg, 0)
                self._set_friction(env, _IDX_OBS_BASE + obs_i, self.friction)

        gym.prepare_sim(sim)

    def _set_friction(self, env, actor_idx: int, friction: float):
        """Apply friction to all shapes of an actor."""
        actor = self.gym.get_actor_handle(env, actor_idx)
        props = self.gym.get_actor_rigid_shape_properties(env, actor)
        for p in props:
            p.friction = friction
            p.rolling_friction = 0.0
            p.torsion_friction = 0.0
        self.gym.set_actor_rigid_shape_properties(env, actor, props)

    # ------------------------------------------------------------------
    # Tensor acquisition
    # ------------------------------------------------------------------

    def _acquire_tensors(self):
        """Wrap the actor root-state tensor after prepare_sim."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        _root = self.gym.acquire_actor_root_state_tensor(self.sim)
        # shape: (n_envs * n_actors_per_env, 13)
        # each row: [px, py, pz, rx, ry, rz, rw, lvx, lvy, lvz, avx, avy, avz]
        self.root_states: torch.Tensor = gymtorch.wrap_tensor(_root)

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _global_idx(self, env_idx: int, local_actor_idx: int) -> int:
        return env_idx * self.n_actors_per_env + local_actor_idx

    def _set_root_states_indexed(self, global_indices: List[int]):
        """Push updated root_states rows back to the simulator for the given global actor indices."""
        idx_tensor = torch.tensor(global_indices, dtype=torch.int32, device=self.root_states.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(idx_tensor),
            len(global_indices),
        )

    # ------------------------------------------------------------------
    # Object placement (single mode)
    # ------------------------------------------------------------------

    def _place_objects(self):
        """Randomly place objects inside the bin, optionally stacking them."""
        self._refresh()
        margin = self._OBJ_SIZE * 0.7
        x_lo, x_hi = margin, self.bin_w - margin
        y_lo, y_hi = margin, self.bin_d - margin

        columns: list[tuple[float, float, int]] = []
        all_local_idxs = [_IDX_OBS_BASE + i for i in range(self.n_obstacles)] + [_IDX_TARGET]

        for local_idx in all_local_idxs:
            is_target = (local_idx == _IDX_TARGET)
            obj_y_lo = self.bin_d / 2 if (is_target and self.difficult_spawn) else y_lo

            placed = False
            if self.stackable and columns and torch.rand(1).item() < 0.5:
                col_i = torch.randint(len(columns), (1,)).item()
                x, y, count = columns[col_i]
                self._set_actor_pos(0, local_idx, [x, y, self._OBJ_H + self._OBJ_SIZE * count])
                columns[col_i] = (x, y, count + 1)
                placed = True

            if not placed:
                for _ in range(200):
                    x = torch.empty(1).uniform_(x_lo, x_hi).item()
                    y = torch.empty(1).uniform_(obj_y_lo, y_hi).item()
                    if all(((x - cx)**2 + (y - cy)**2)**0.5 > self._OBJ_SIZE * 1.05
                           for cx, cy, _ in columns):
                        self._set_actor_pos(0, local_idx, [x, y, self._OBJ_H])
                        columns.append((x, y, 1))
                        break

        self._park_pushers(0)
        self._zero_velocities(0)

        # Commit all changes at once
        all_idxs = [self._global_idx(0, i) for i in range(self.n_actors_per_env)]
        self._set_root_states_indexed(all_idxs)

        # Settle
        for _ in range(60):
            self._step_sim()

        # Save initial state as checkpoint
        self._refresh()
        self._initial_root_states = self.root_states.clone()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_actor_pos(self, env_idx: int, local_idx: int,
                       pos: list, quat_xyzw: list | None = None):
        """Write pos (and optionally orientation) into root_states for one actor.
        Does NOT call set_actor_root_state_tensor_indexed — caller must do that.
        """
        gi = self._global_idx(env_idx, local_idx)
        self.root_states[gi, 0] = pos[0]
        self.root_states[gi, 1] = pos[1]
        self.root_states[gi, 2] = pos[2]
        if quat_xyzw is not None:
            self.root_states[gi, 3] = quat_xyzw[0]
            self.root_states[gi, 4] = quat_xyzw[1]
            self.root_states[gi, 5] = quat_xyzw[2]
            self.root_states[gi, 6] = quat_xyzw[3]
        else:
            # identity quaternion (x,y,z,w)
            self.root_states[gi, 3:7] = torch.tensor([0., 0., 0., 1.],
                                                      device=self.root_states.device)

    def _zero_vel_actor(self, env_idx: int, local_idx: int):
        gi = self._global_idx(env_idx, local_idx)
        self.root_states[gi, 7:13] = 0.0

    def _park_pushers(self, env_idx: int):
        """Write park position into root_states for both pushers (no commit)."""
        for local_idx in (_IDX_NS, _IDX_EW):
            self._set_actor_pos(env_idx, local_idx, self._park)
            self._zero_vel_actor(env_idx, local_idx)

    def _zero_velocities(self, env_idx: int):
        """Zero velocities for all dynamic actors in env_idx (no commit)."""
        for local_idx in ([_IDX_NS, _IDX_EW, _IDX_TARGET] +
                          [_IDX_OBS_BASE + i for i in range(self.n_obstacles)]):
            self._zero_vel_actor(env_idx, local_idx)

    def _step_sim(self):
        """Advance one physics step (all envs simultaneously)."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _get_state(self, env_idx: int) -> dict:
        """Read pos/quat of all objects from root_states for the given env slot."""
        self._refresh()

        def _pos(local_idx):
            gi = self._global_idx(env_idx, local_idx)
            return self.root_states[gi, :3].clone()

        def _quat_wxyz(local_idx):
            """Return quaternion as [w,x,y,z] (matching Genesis convention)."""
            gi = self._global_idx(env_idx, local_idx)
            xyzw = self.root_states[gi, 3:7].clone()
            return torch.stack([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        return {
            'target_pos':    _pos(_IDX_TARGET),
            'target_quat':   _quat_wxyz(_IDX_TARGET),
            'obstacle_pos':  torch.stack([_pos(_IDX_OBS_BASE + i) for i in range(self.n_obstacles)]),
            'obstacle_quat': torch.stack([_quat_wxyz(_IDX_OBS_BASE + i) for i in range(self.n_obstacles)]),
        }

    def _set_state(self, state: dict, env_idx: int | None = None):
        """Teleport all objects to the given state, park pushers, zero velocities."""
        if env_idx is None:
            env_idx = 0

        def _to_list(t):
            return t.tolist() if torch.is_tensor(t) else list(t)

        def _wxyz_to_xyzw(q):
            """Convert [w,x,y,z] → [x,y,z,w] for IsaacGym tensor storage."""
            q = _to_list(q)
            return [q[1], q[2], q[3], q[0]]

        # Target
        self._set_actor_pos(env_idx, _IDX_TARGET,
                            _to_list(state['target_pos']),
                            _wxyz_to_xyzw(state['target_quat']))
        self._zero_vel_actor(env_idx, _IDX_TARGET)

        # Obstacles
        for i in range(self.n_obstacles):
            self._set_actor_pos(env_idx, _IDX_OBS_BASE + i,
                                _to_list(state['obstacle_pos'][i]),
                                _wxyz_to_xyzw(state['obstacle_quat'][i]))
            self._zero_vel_actor(env_idx, _IDX_OBS_BASE + i)

        # Pushers
        self._park_pushers(env_idx)

        # Commit all dynamic actors
        dynamic_local = ([_IDX_NS, _IDX_EW, _IDX_TARGET] +
                         [_IDX_OBS_BASE + i for i in range(self.n_obstacles)])
        self._set_root_states_indexed([self._global_idx(env_idx, l) for l in dynamic_local])

    def get_state(self, env_idx: int = 0) -> dict:
        return self._get_state(env_idx)

    def set_state(self, state: dict, env_idx: int | None = None):
        self._set_state(state, env_idx)

    def reset(self, seed: int | None = None, env_idx: int | None = None) -> dict:
        """
        Reset environment.
        - Single mode (n_envs=1): restore saved initial root-state tensor (or re-place if seed given)
        - Parallel mode: teleport one env slot to bin-center initial state
        """
        if self.n_envs > 1:
            if env_idx is not None:
                initial_state = {
                    'target_pos':    torch.tensor([self.bin_w/2, self.bin_d/2, self._OBJ_H]),
                    'target_quat':   torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    'obstacle_pos':  torch.tile(
                        torch.tensor([self.bin_w/2, self.bin_d/2, self._OBJ_H]).unsqueeze(0),
                        (self.n_obstacles, 1)
                    ),
                    'obstacle_quat': torch.tile(
                        torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0),
                        (self.n_obstacles, 1)
                    ),
                }
                self._set_state(initial_state, env_idx)
                return self._get_state(env_idx)
            return {}
        else:
            if seed is not None:
                torch.manual_seed(seed)
                self._place_objects()
                return self._get_state(0)
            # Restore from saved checkpoint
            if self._initial_root_states is not None:
                self.root_states.copy_(self._initial_root_states)
                all_idxs = list(range(self.n_envs * self.n_actors_per_env))
                self._set_root_states_indexed(all_idxs)
            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            return self._get_state(0)

    # ------------------------------------------------------------------
    # Action primitives (single env)
    # ------------------------------------------------------------------

    def execute_ns_push(self, pos_2d: torch.Tensor, z: float,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push northward: pusher_ns enters from south, sweeps north."""
        state = self._get_state(0)
        action = {'action_type': 'push_n', 'push_pos': pos_2d, 'push_z': z}
        return self.batch_evaluate([(state, action)])[0]

    def execute_ns_pull(self, pos_2d: torch.Tensor, z: float,
                        approach_dist: float = 0.12,
                        pull_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Pull southward: pusher_ns hooks north of object, sweeps south to exit."""
        state = self._get_state(0)
        action = {'action_type': 'pull_s', 'push_pos': pos_2d, 'push_z': z}
        return self.batch_evaluate([(state, action)])[0]

    def execute_ew_push(self, pos_2d: torch.Tensor, z: float, direction: int,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: int | None = None, step_delay: float = 0.0) -> tuple[dict, float, bool]:
        """Push east (direction=+1) or west (direction=-1)."""
        state = self._get_state(0)
        action = {
            'action_type': 'push_e' if direction > 0 else 'push_w',
            'push_pos': pos_2d,
            'push_z': z,
        }
        return self.batch_evaluate([(state, action)])[0]

    # ------------------------------------------------------------------
    # Batch evaluation (parallel mode)
    # ------------------------------------------------------------------

    def _batch_evaluate_impl(self, pairs: list[tuple[dict, dict]]) -> list[tuple[dict, float, bool]]:
        """
        Evaluate up to n_envs (state, action) pairs in parallel.

        All envs advance together via a single gym.simulate() call per tick,
        so wall-clock cost is roughly constant in k (GPU-bound).

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
            results = []
            for i in range(0, k, self.n_envs):
                results.extend(self._batch_evaluate_impl(pairs[i:i + self.n_envs]))
            return results

        # 1. Teleport each env to its starting state; compute stroke geometry
        strokes: list[tuple[str, list, list]] = []
        for env_idx, (state, action) in enumerate(pairs):
            self._set_state(state, env_idx)
            pusher_type, start, end = self._action_to_stroke(action)
            strokes.append((pusher_type, start, end))

        # 2. Warm-up: place pushers at stroke start, settle 2 ticks
        changed_idxs = []
        for env_idx, (ptype, start, _) in enumerate(strokes):
            local_idx = _IDX_NS if ptype == 'ns' else _IDX_EW
            self._set_actor_pos(env_idx, local_idx, start)
            changed_idxs.append(self._global_idx(env_idx, local_idx))
        self._set_root_states_indexed(changed_idxs)
        self._step_sim()
        self._step_sim()

        # 3. Sweep — one gym.simulate() advances ALL envs simultaneously
        for step_i in range(self.push_steps):
            t = (step_i + 1) / self.push_steps
            changed_idxs = []
            for env_idx, (ptype, start, end) in enumerate(strokes):
                pos = [s + t * (e - s) for s, e in zip(start, end)]
                local_idx = _IDX_NS if ptype == 'ns' else _IDX_EW
                self._set_actor_pos(env_idx, local_idx, pos)
                self._zero_vel_actor(env_idx, local_idx)
                changed_idxs.append(self._global_idx(env_idx, local_idx))
            self._set_root_states_indexed(changed_idxs)
            self._step_sim()

        # 4. Park all pushers and settle
        changed_idxs = []
        for env_idx in range(k):
            for local_idx in (_IDX_NS, _IDX_EW):
                self._set_actor_pos(env_idx, local_idx, self._park)
                self._zero_vel_actor(env_idx, local_idx)
                changed_idxs.append(self._global_idx(env_idx, local_idx))
        self._set_root_states_indexed(changed_idxs)
        self._step_sim()
        self._step_sim()

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
                   for i in range(self.n_obstacles))

    def _is_goal(self, state: dict) -> bool:
        if self._obstacles_dropped(state):
            return False
        return bool(float(state['target_pos'][1]) <= EXIT_Y)

    def step_physics(self) -> None:
        self._step_sim()

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
        for i in range(self.n_obstacles):
            pos.append(state['obstacle_pos'][i][:2])
        return torch.stack(pos)
