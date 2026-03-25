"""
parallel_env.py – Genesis scene with n_envs parallel environments.

All environments share one scene and advance together via scene.step().
This means N (state, action) pairs cost the same as 1 in wall-clock time
(GPU runs all envs in the same CUDA kernel).

Usage
-----
    from parallel_env import ParallelBinEnv
    penv = ParallelBinEnv(n_envs=8, n_obstacles=2)
    results = penv.batch_evaluate([(state0, action0), (state1, action1), ...])
    # results: list of (new_state, reward, done)
"""

import copy
import numpy as np
import genesis as gs

from env import (BIN_W, BIN_D, BIN_H, WALL_T, OBJ_SIZE, OBJ_H,
                 PUSHER_T, PUSHER_W, EXIT_Y, _PARK)


class ParallelBinEnv:
    """
    Wraps a single Genesis scene with n_envs parallel environments.

    Each env is an independent physics world; scene.step() advances all
    of them simultaneously on the GPU.

    Parameters
    ----------
    n_envs      : int   – how many parallel worlds
    n_obstacles : int
    friction    : float
    n_z_levels  : int   – discrete push-height levels
    push_steps  : int   – stroke steps (same for all envs per batch)
    substeps    : int   – physics substeps per scene.step()
    dt          : float
    """

    def __init__(self, n_envs: int = 8, n_obstacles: int = 2,
                 friction: float = 1.0, n_z_levels: int = 1,
                 push_steps: int = 20, substeps: int = 4,
                 dt: float = 0.01):
        self.n_envs      = n_envs
        self.n_obstacles = n_obstacles
        self.friction    = friction
        self.push_steps  = push_steps
        self.substeps    = substeps
        self.z_levels    = [OBJ_H + i * OBJ_SIZE for i in range(n_z_levels)]
        self._build_scene(dt)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, dt: float):
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, -9.81),
                box_box_detection=False,
                enable_self_collision=False,
                # iterations=15,               # constraint solver iters (default 50)
                # ls_iterations=10,            # line-search iters (default 50)
                iterations=8,
                ls_iterations=5,
                use_hibernation=True,
                use_contact_island=True
            ),
        )

        def _box(size, pos, fixed=False, rho=500, color=(0.7, 0.6, 0.5)):
            return self.scene.add_entity(
                gs.morphs.Box(size=size, pos=pos, fixed=fixed),
                **(dict(material=gs.materials.Rigid(rho=rho, friction=self.friction))
                   if not fixed else {}),
                surface=gs.surfaces.Default(color=color),
            )

        # Static geometry (identical across all envs)
        _box((BIN_W + 2*WALL_T, BIN_D + 2*WALL_T, WALL_T),
             (BIN_W/2, BIN_D/2, -WALL_T/2), fixed=True)                       # floor
        _box((BIN_W + 2*WALL_T, WALL_T, BIN_H),
             (BIN_W/2, BIN_D + WALL_T/2, BIN_H/2), fixed=True,
             color=(0.5, 0.5, 0.8))                                             # north
        _box((WALL_T, BIN_D, BIN_H),
             (-WALL_T/2, BIN_D/2, BIN_H/2), fixed=True,
             color=(0.5, 0.5, 0.8))                                             # west
        _box((WALL_T, BIN_D, BIN_H),
             (BIN_W + WALL_T/2, BIN_D/2, BIN_H/2), fixed=True,
             color=(0.5, 0.5, 0.8))                                             # east

        # Pushers
        self.pusher_ns = self.scene.add_entity(
            gs.morphs.Box(size=(PUSHER_W, PUSHER_T, PUSHER_W), pos=_PARK),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.2, 0.9, 0.2)),
        )
        self.pusher_ew = self.scene.add_entity(
            gs.morphs.Box(size=(PUSHER_T, PUSHER_W, PUSHER_W), pos=_PARK),
            material=gs.materials.Rigid(rho=10000, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.6, 0.1)),
        )

        # Target
        self.target = self.scene.add_entity(
            gs.morphs.Box(size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
                          pos=(BIN_W/2, BIN_D/2, OBJ_H)),
            material=gs.materials.Rigid(rho=50, friction=self.friction),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2), opacity=0.6),
        )

        # Obstacles
        self.obstacles = []
        for _ in range(self.n_obstacles):
            obs = self.scene.add_entity(
                gs.morphs.Box(size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
                              pos=(BIN_W/2, BIN_D/2, OBJ_H)),
                material=gs.materials.Rigid(rho=500, friction=self.friction),
                surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9), opacity=0.6),
            )
            self.obstacles.append(obs)

        self.scene.build(n_envs=self.n_envs)

    # ------------------------------------------------------------------
    # Per-env state get / set
    # ------------------------------------------------------------------

    def set_state(self, state: dict, env_idx: int):
        """Teleport all objects and park pushers in one env slot."""
        ei = [env_idx]
        identity = [1.0, 0.0, 0.0, 0.0]

        self.target.set_pos(state['target_pos'].tolist(),    envs_idx=ei)
        self.target.set_quat(state['target_quat'].tolist(),  envs_idx=ei)
        self.target.zero_all_dofs_velocity(envs_idx=ei)

        for i, obs in enumerate(self.obstacles):
            obs.set_pos(state['obstacle_pos'][i].tolist(),   envs_idx=ei)
            obs.set_quat(state['obstacle_quat'][i].tolist(), envs_idx=ei)
            obs.zero_all_dofs_velocity(envs_idx=ei)

        for pusher in (self.pusher_ns, self.pusher_ew):
            pusher.set_pos(_PARK,     envs_idx=ei)
            pusher.set_quat(identity, envs_idx=ei)
            pusher.zero_all_dofs_velocity(envs_idx=ei)

    def get_state(self, env_idx: int) -> dict:
        """Read pos/quat tensors from one env slot into numpy arrays."""
        ei = [env_idx]
        return {
            'target_pos':    self.target.get_pos(envs_idx=ei)[0].cpu().numpy().copy(),
            'target_quat':   self.target.get_quat(envs_idx=ei)[0].cpu().numpy().copy(),
            'obstacle_pos':  np.array([
                o.get_pos(envs_idx=ei)[0].cpu().numpy() for o in self.obstacles]),
            'obstacle_quat': np.array([
                o.get_quat(envs_idx=ei)[0].cpu().numpy() for o in self.obstacles]),
        }

    # ------------------------------------------------------------------
    # Stroke helpers
    # ------------------------------------------------------------------

    def _action_to_stroke(self, action: dict,
                          approach_dist: float = 0.12,
                          push_dist:     float = 0.25
                          ) -> tuple[str, np.ndarray, np.ndarray]:
        """Return (pusher_type, start_3d, end_3d) for an action dict."""
        atype = action['action_type']
        pos   = action['push_pos']
        z     = action['push_z']
        if atype == 'push_n':
            return ('ns',
                    np.array([pos[0], pos[1] - approach_dist, z]),
                    np.array([pos[0], pos[1] + push_dist,     z]))
        if atype == 'pull_s':
            return ('ns',
                    np.array([pos[0], pos[1] + approach_dist,  z]),
                    np.array([pos[0], EXIT_Y  - approach_dist, z]))
        if atype == 'push_e':
            return ('ew',
                    np.array([pos[0] - approach_dist, pos[1], z]),
                    np.array([pos[0] + push_dist,     pos[1], z]))
        # push_w
        return ('ew',
                np.array([pos[0] + approach_dist, pos[1], z]),
                np.array([pos[0] - push_dist,     pos[1], z]))

    # ------------------------------------------------------------------
    # Batch evaluation  ← the core of this file
    # ------------------------------------------------------------------

    def batch_evaluate(
        self,
        pairs: list[tuple[dict, dict]],
    ) -> list[tuple[dict, float, bool]]:
        """
        Evaluate up to n_envs (state, action) pairs in parallel.

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
                results.extend(self.batch_evaluate(pairs[i:i + self.n_envs]))
            return results

        identity = [1.0, 0.0, 0.0, 0.0]

        # 1. Teleport each env to its starting state and compute stroke geometry
        strokes: list[tuple[str, np.ndarray, np.ndarray]] = []
        for env_idx, (state, action) in enumerate(pairs):
            self.set_state(state, env_idx)
            pusher_type, start, end = self._action_to_stroke(action)
            strokes.append((pusher_type, start, end))

        # 2. Warm-up: place pushers at stroke start, settle 2 ticks
        for env_idx, (ptype, start, _) in enumerate(strokes):
            pusher = self.pusher_ns if ptype == 'ns' else self.pusher_ew
            pusher.set_pos(start.tolist(), envs_idx=[env_idx])
            pusher.set_quat(identity,      envs_idx=[env_idx])
        self.scene.step()
        self.scene.step()

        # 3. Sweep — one scene.step() advances ALL envs simultaneously
        for step_i in range(self.push_steps):
            t = (step_i + 1) / self.push_steps
            for env_idx, (ptype, start, end) in enumerate(strokes):
                pos    = start + t * (end - start)
                pusher = self.pusher_ns if ptype == 'ns' else self.pusher_ew
                pusher.set_pos(pos.tolist(), envs_idx=[env_idx])
                pusher.set_quat(identity,    envs_idx=[env_idx])
            self.scene.step()

        # 4. Park all pushers and settle
        for env_idx in range(k):
            for pusher in (self.pusher_ns, self.pusher_ew):
                pusher.set_pos(_PARK,     envs_idx=[env_idx])
                pusher.set_quat(identity, envs_idx=[env_idx])
                pusher.zero_all_dofs_velocity(envs_idx=[env_idx])
        self.scene.step()
        self.scene.step()

        # 5. Read back results
        results = []
        for env_idx in range(k):
            state  = self.get_state(env_idx)
            reward = self._compute_reward(state)
            done   = self._is_goal(state)
            results.append((state, reward, done))
        return results

    # ------------------------------------------------------------------
    # Reward / goal (mirrors env.py)
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict) -> float:
        y = state['target_pos'][1]
        r = float(np.clip((BIN_D / 2 - y) / (BIN_D / 2 - EXIT_Y), 0, 2))
        n_dropped = sum(1 for i in range(len(self.obstacles))
                        if state['obstacle_pos'][i][1] < EXIT_Y)
        return r - 0.5 * n_dropped

    def _is_goal(self, state: dict) -> bool:
        if any(state['obstacle_pos'][i][1] < EXIT_Y
               for i in range(len(self.obstacles))):
            return False
        return float(state['target_pos'][1]) < EXIT_Y

    def _obstacles_dropped(self, state: dict) -> bool:
        return any(state['obstacle_pos'][i][1] < EXIT_Y
                   for i in range(len(self.obstacles)))
