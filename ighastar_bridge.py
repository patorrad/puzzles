"""
Bridge between the puzzles simulator API (jack branch) and the IGHA* generic
environment.

IGHA*'s generic environment is configured from Python callbacks
(sample_controls / dynamics / cost / validity / heuristic / goal_test) that
operate on flat float32 state/control vectors. This module translates between
those vectors and the puzzles dict-state + discrete push actions, driving the
forward model through ``SimulatorEnv.batch_evaluate`` (any backend: Isaac Lab,
Genesis, IsaacGym). It also records the (parent -> child) push that produced
every successor so the planned state path can be turned back into the
``list[dict]`` action format the rest of the repo expects.

State vector (N_DIMS = 2 * (1 + n_obstacles)), bin-local xy:
    [target_x, target_y, obs0_x, obs0_y, obs1_x, obs1_y, ...]

Control vector (N_CONT = 3), discrete macro-action encoded as floats:
    [action_type_idx (0..3), obj_idx (0=target, 1..N=obstacle), z_level_idx]

Goal (jack branch): target exits the open -y face, i.e. target_y <= EXIT_Y,
with no obstacle dropped (obstacle_y >= EXIT_Y).
"""

from __future__ import annotations

import numpy as np
import torch

from simulators import SimulatorEnv

# Mirrors planner.py
_ACTION_TYPES = ['push_n', 'pull_s', 'push_e', 'push_w']
_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


class BinIGHAStarBridge:
    """Adapts a SimulatorEnv as an IGHA* generic environment."""

    def __init__(self, env: SimulatorEnv, grid_z: bool = True):
        self.env = env
        self.n_obstacles = env.n_obstacles
        self.exit_y = float(env._EXIT_Y)
        self.z_levels = list(env.z_levels)
        self.n_z_levels = max(1, len(self.z_levels))
        self.n_cont = 3
        self.batch_size = max(1, int(getattr(env, 'n_envs', 1)))

        # State layout: the FULL per-object pose is carried losslessly so the
        # search never modifies a node's true state. Only the leading gridded
        # block is hashed/dedup'd (HASH_DIMS); the rest rides along for the
        # dynamics (set_state) without being discretised or reset.
        #   grid_z=False: gridded = [x,y] per obj;  carried = [z, quat] per obj
        #   grid_z=True : gridded = [x,y,z] per obj; carried = [quat] per obj
        self.grid_z = bool(grid_z)
        self.n_obj = 1 + self.n_obstacles
        self._g = 3 if self.grid_z else 2       # gridded coords per object
        self._carry_per = 7 - self._g           # carried coords per object
        self.hash_dims = self._g * self.n_obj   # leading gridded block
        self._carry = self.hash_dims            # offset of the carried block
        self.n_dims = 7 * self.n_obj            # xyz(3) + quat(4) per object

        # Fixed enumeration of every discrete macro-action template -> [K, 3].
        controls = []
        for atype_idx in range(len(_ACTION_TYPES)):
            for obj_idx in range(self.n_obstacles + 1):
                for z_idx in range(self.n_z_levels):
                    controls.append([float(atype_idx), float(obj_idx), float(z_idx)])
        self._controls = torch.tensor(controls, dtype=torch.float32)
        self.num_controls = self._controls.shape[0]

        # Every (parent_vec, child_vec, action) seen in dynamics(), used to map
        # the planned state path back to push actions.
        self.edges: list[tuple[np.ndarray, np.ndarray, dict]] = []

    # ------------------------------------------------------------------
    # State <-> vector
    # ------------------------------------------------------------------

    @staticmethod
    def _to_np(v, n: int) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()[:n].astype(np.float64)
        return np.asarray(v, dtype=np.float64)[:n]

    def encode_state(self, state: dict) -> np.ndarray:
        """dict -> full-pose vector (gridded block first, then carried block).

        Lossless: the entire pose is preserved so the search never has to
        fabricate z / orientation."""
        vec = np.zeros(self.n_dims, dtype=np.float32)
        pos = [state['target_pos']] + [state['obstacle_pos'][i]
                                       for i in range(self.n_obstacles)]
        quat = [state['target_quat']] + [state['obstacle_quat'][i]
                                         for i in range(self.n_obstacles)]
        g = self._g
        for j in range(self.n_obj):
            p = self._to_np(pos[j], 3)
            q = self._to_np(quat[j], 4)
            vec[g * j] = p[0]
            vec[g * j + 1] = p[1]
            base = self._carry + self._carry_per * j
            if self.grid_z:
                vec[g * j + 2] = p[2]
                vec[base:base + 4] = q
            else:
                vec[base] = p[2]
                vec[base + 1:base + 5] = q
        return vec

    def decode_state(self, vec: np.ndarray) -> dict:
        """full-pose vector -> dict. Inverse of encode_state; nothing reset."""
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        g = self._g

        def obj_pose(j):
            base = self._carry + self._carry_per * j
            if self.grid_z:
                p = torch.tensor([float(vec[g * j]), float(vec[g * j + 1]),
                                  float(vec[g * j + 2])], dtype=torch.float32)
                q = torch.tensor([float(vec[base + k]) for k in range(4)],
                                 dtype=torch.float32)
            else:
                p = torch.tensor([float(vec[g * j]), float(vec[g * j + 1]),
                                  float(vec[base])], dtype=torch.float32)
                q = torch.tensor([float(vec[base + 1 + k]) for k in range(4)],
                                 dtype=torch.float32)
            return p, q

        tp, tq = obj_pose(0)
        obstacle_pos = torch.empty((self.n_obstacles, 3), dtype=torch.float32)
        obstacle_quat = torch.empty((self.n_obstacles, 4), dtype=torch.float32)
        for i in range(self.n_obstacles):
            p, q = obj_pose(i + 1)
            obstacle_pos[i] = p
            obstacle_quat[i] = q
        return {
            'target_pos': tp,
            'target_quat': tq,
            'obstacle_pos': obstacle_pos,
            'obstacle_quat': obstacle_quat,
        }

    def control_to_action(self, state_vec: np.ndarray, control_vec: np.ndarray) -> dict:
        atype_idx = int(round(float(control_vec[0])))
        obj_idx = int(round(float(control_vec[1])))
        z_idx = int(round(float(control_vec[2])))
        atype = _ACTION_TYPES[atype_idx]
        off = self._g * obj_idx
        push_pos = torch.tensor([float(state_vec[off]), float(state_vec[off + 1])],
                                dtype=torch.float32)
        push_z = float(self.z_levels[z_idx]) if self.z_levels else 0.025
        return {'action_type': atype, 'obj_idx': obj_idx,
                'push_pos': push_pos, 'push_z': push_z}

    # ------------------------------------------------------------------
    # Goal / validity helpers (mirror SimulatorEnv._is_goal / _obstacles_dropped)
    # ------------------------------------------------------------------

    def _obj_y_index(self, j: int) -> int:
        # y of object j is the 2nd coord of its gridded slot (x at g*j, y at g*j+1)
        return self._g * j + 1

    def _obstacle_dropped_vec(self, vec: np.ndarray) -> bool:
        for i in range(self.n_obstacles):
            if vec[self._obj_y_index(i + 1)] < self.exit_y:
                return True
        return False

    def _is_goal_vec(self, vec: np.ndarray) -> bool:
        if self._obstacle_dropped_vec(vec):
            return False
        return float(vec[self._obj_y_index(0)]) <= self.exit_y

    # ------------------------------------------------------------------
    # Generic-env callbacks
    # ------------------------------------------------------------------

    def make_callbacks(self) -> dict:
        return {
            'sample_controls_fn': self._sample_controls,
            'dynamics_fn': self._dynamics,
            'cost_fn': self._cost,
            'validity_fn': self._validity,
            'heuristic_fn': self._heuristic,
            'goal_test_fn': self._goal_test,
        }

    def _sample_controls(self) -> torch.Tensor:
        return self._controls

    def _dynamics(self, states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        states_np = states.detach().cpu().numpy().astype(np.float64)
        controls_np = controls.detach().cpu().numpy().astype(np.float64)
        B = states_np.shape[0]

        actions = [self.control_to_action(states_np[i], controls_np[i])
                   for i in range(B)]
        pairs = [(self.decode_state(states_np[i]), actions[i]) for i in range(B)]

        # Chunk to the simulator's parallel width.
        results = []
        for s in range(0, B, self.batch_size):
            results.extend(self.env.batch_evaluate(pairs[s:s + self.batch_size]))

        out = np.empty((B, self.n_dims), dtype=np.float32)
        for i in range(B):
            child_vec = self.encode_state(results[i][0])
            out[i] = child_vec
            self.edges.append((states_np[i].astype(np.float32).copy(),
                               child_vec.copy(), actions[i]))
        return torch.from_numpy(out)

    def _cost(self, states: torch.Tensor, controls: torch.Tensor,
              next_states: torch.Tensor) -> torch.Tensor:
        # Minimise number of pushes: every edge costs 1.
        return torch.ones(states.shape[0], dtype=torch.float32)

    def _validity(self, states: torch.Tensor) -> torch.Tensor:
        states_np = states.detach().cpu().numpy()
        B = states_np.shape[0]
        valid = np.ones(B, dtype=np.float32)
        for i in range(B):
            if self._obstacle_dropped_vec(states_np[i]):
                valid[i] = 0.0
        return torch.from_numpy(valid)

    def _heuristic(self, states: torch.Tensor) -> torch.Tensor:
        states_np = states.detach().cpu().numpy()
        # Admissible: remaining +y distance the target must travel to clear the
        # exit (<= true push count, which is >= 1 when not yet at the goal).
        h = np.maximum(0.0, states_np[:, 1] - self.exit_y).astype(np.float32)
        return torch.from_numpy(h)

    def _goal_test(self, state: torch.Tensor) -> bool:
        vec = state.detach().cpu().numpy().reshape(-1)
        return bool(self._is_goal_vec(vec))

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def actions_from_state_path(self, state_path: np.ndarray) -> list[dict]:
        """Map a start->goal ordered [P, n_dims] state path back to the action
        dicts that produced each transition, using nearest-edge lookup to
        tolerate physics non-determinism and grid rounding."""
        if len(self.edges) == 0 or len(state_path) < 2:
            return []
        parents = np.stack([e[0] for e in self.edges]).astype(np.float64)
        children = np.stack([e[1] for e in self.edges]).astype(np.float64)
        actions = []
        for i in range(len(state_path) - 1):
            s_i = state_path[i].astype(np.float64)
            s_next = state_path[i + 1].astype(np.float64)
            dist = (np.linalg.norm(parents - s_i, axis=1)
                    + np.linalg.norm(children - s_next, axis=1))
            actions.append(self.edges[int(np.argmin(dist))][2])
        return actions

    # ------------------------------------------------------------------
    # IGHA* config
    # ------------------------------------------------------------------

    def build_config(self, max_expansions: int = 5000, hysteresis: int = 500,
                     resolution: float = 0.03, tolerance: float = 0.015,
                     z_resolution: float = 0.025, z_tolerance: float = 0.0125,
                     max_level: int = 4, division_factor: float = 2.0,
                     preemptive_enabled: bool = False,
                     min_preemptive: int = 8, max_preemptive: int = 32) -> dict:
        n = self.n_dims
        h = self.hash_dims
        g = self._g
        bin_w = float(getattr(self.env, 'bin_w', 0.5))
        bin_d = float(getattr(self.env, 'bin_d', 0.5))
        z_lo = (min(self.z_levels) - 0.1) if self.z_levels else -0.1
        z_hi = (max(self.z_levels) + 0.2) if self.z_levels else 1.0

        # Only the leading gridded block (hash_dims) is hashed; carried dims get
        # benign placeholders (unused by calc_hash, which iterates only over
        # hash_dims) but must be present for length == n_dims. Per object the
        # gridded slot is [x, y] (+ [z] when grid_z).
        resolution_vec, tolerance_vec = [], []
        bounds_lower, bounds_upper = [], []
        for _ in range(self.n_obj):
            resolution_vec += [resolution, resolution]
            tolerance_vec += [tolerance, tolerance]
            bounds_lower += [-0.1, self.exit_y - 0.1]
            bounds_upper += [bin_w + 0.1, bin_d + 0.1]
            if self.grid_z:
                resolution_vec += [z_resolution]
                tolerance_vec += [z_tolerance]
                bounds_lower += [z_lo]
                bounds_upper += [z_hi]
        # carried block (z?+quat): unused by hashing
        resolution_vec += [1.0] * (n - h)
        tolerance_vec += [1.0] * (n - h)
        for _ in range(n - h):
            bounds_lower.append(-10.0); bounds_upper.append(10.0)

        config = {
            'experiment_info_default': {
                'state_dim': n,
                'control_dim': self.n_cont,
                'hash_dims': h,
                'num_controls': self.num_controls,
                'resolution': resolution_vec,
                'tolerance': tolerance_vec,
                'bounds_lower': bounds_lower,
                'bounds_upper': bounds_upper,
                'max_level': max_level,
                'division_factor': division_factor,
                'max_expansions': max_expansions,
                'hysteresis': hysteresis,
                'preemptive_expansion': {
                    'enabled': bool(preemptive_enabled),
                    'min_preemptive': int(min_preemptive),
                    'max_preemptive': int(max_preemptive),
                },
                'node_info': {'node_type': 'generic', 'timesteps': 1},
            },
        }
        config.update(self.make_callbacks())
        return config
