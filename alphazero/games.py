"""Per-player Game adapters for AZMCTS.

SolverGame:  world = env state dict. Transition = env.batch_evaluate.
StackerGame: world = (occupancy tensor, target_cell, blocks_remaining).
             Transition = symbolic (mark cell as occupied).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch

from .encoders import (encode_solver_state, encode_stacker_state,
                       solver_action_dim, solver_index_to_action)
from .grid import GridSpec, legal_mask as grid_legal_mask


_SOLVER_ACTION_TYPES = ['push_n', 'pull_s', 'push_e', 'push_w']


@dataclass(frozen=True)
class StackerState:
    """Immutable stacker world. The occupied tensor has shape (Gx, Gy, Z).

    A new state is constructed each turn via dataclasses.replace so MCTS nodes
    keep stable handles.
    """
    occupied: torch.Tensor       # (Gx, Gy, Z) int8
    target_cell: tuple[int, int]
    blocks_remaining: int
    n_obstacles_total: int

    def with_placement(self, i: int, j: int, k: int) -> 'StackerState':
        new_occ = self.occupied.clone()
        new_occ[i, j, k] = 1
        return replace(self,
                       occupied=new_occ,
                       blocks_remaining=self.blocks_remaining - 1)


class StackerGame:
    """Pure-symbolic stacker game (no env)."""

    def __init__(self, spec: GridSpec, n_obstacles: int):
        self.spec = spec
        self.n_obstacles = n_obstacles
        self.n_actions = spec.n_actions

    def initial_state(self, target_cell: tuple[int, int]) -> StackerState:
        return StackerState(
            occupied=torch.zeros(self.spec.Gx, self.spec.Gy, self.spec.Z,
                                 dtype=torch.int8),
            target_cell=target_cell,
            blocks_remaining=self.n_obstacles,
            n_obstacles_total=self.n_obstacles,
        )

    def encode(self, state: StackerState) -> torch.Tensor:
        return encode_stacker_state(state.occupied, state.target_cell,
                                    state.blocks_remaining,
                                    self.n_obstacles, self.spec)

    def legal_mask(self, state: StackerState) -> torch.Tensor:
        if state.blocks_remaining <= 0:
            return torch.zeros(self.n_actions, dtype=torch.bool)
        return grid_legal_mask(self.spec, state.occupied, state.target_cell)

    def is_terminal(self, state: StackerState) -> bool:
        return state.blocks_remaining <= 0

    def terminal_value(self, state: StackerState) -> float:
        # Outcome is determined by the solver's run, not within stacker MCTS.
        # The value head learns it; terminal_value is unused during search
        # because the stacker tree only reaches terminal after running out of
        # placements (final leaf falls into the network's prediction in run()).
        return 0.0

    def transition(self, state: StackerState, action_idx: int) -> tuple[StackerState, bool]:
        i, j, k = self.spec.unflatten(action_idx)
        next_s = state.with_placement(i, j, k)
        return next_s, self.is_terminal(next_s)

    def batched_transition(self, states, action_indices):
        """Symbolic transitions are cheap — just loop."""
        return [self.transition(s, a) for s, a in zip(states, action_indices)]


def action_idx_to_solver_dict(action_idx: int, state: dict, env,
                              n_obstacles: int, n_z_levels: int) -> dict:
    """Convert a packed (atype, obj, z) index to the env's action dict.

    push_pos snaps to the chosen object's current xy (mirrors planner._sample_action).
    """
    at_idx, obj_idx, z_idx = solver_index_to_action(action_idx, n_obstacles, n_z_levels)
    if obj_idx == 0:
        push_pos = state['target_pos'][:2].detach().cpu().clone()
    else:
        push_pos = state['obstacle_pos'][obj_idx - 1][:2].detach().cpu().clone()
    push_z = env.z_levels[z_idx] if env.z_levels else float(state['target_pos'][2])
    return {
        'action_type': _SOLVER_ACTION_TYPES[at_idx],
        'obj_idx': obj_idx,
        'push_pos': push_pos,
        'push_z': float(push_z),
    }


class SolverGame:
    """Solver-side adapter. Wraps env for transitions and goal/dropout checks."""

    def __init__(self, env, spec: GridSpec, max_depth: int = 10):
        self.env = env
        self.spec = spec
        self.n_obstacles = env.n_obstacles
        self.n_z_levels = max(1, env.n_z_levels)
        self.max_depth = max_depth
        self.n_actions = solver_action_dim(self.n_obstacles, self.n_z_levels)

    def encode(self, state: dict) -> torch.Tensor:
        return encode_solver_state(state['env_state'], self.spec, self.n_obstacles)

    def legal_mask(self, state: dict) -> torch.Tensor:
        return torch.ones(self.n_actions, dtype=torch.bool)

    def is_terminal(self, state: dict) -> bool:
        if state['done']:
            return True
        if state['depth'] >= self.max_depth:
            return True
        return self.env._obstacles_dropped(state['env_state'])

    def terminal_value(self, state: dict) -> float:
        if self.env._is_goal(state['env_state']):
            return 1.0
        if self.env._obstacles_dropped(state['env_state']):
            return -1.0
        # depth-cap timeout: small negative — solver failed to escape
        return -0.5

    def transition(self, state: dict, action_idx: int) -> tuple[dict, bool]:
        ns_terminal, = self.batched_transition([state], [action_idx])
        return ns_terminal

    def batched_transition(self, states, action_indices):
        """Batched env step. Splits into chunks of size <= env.n_envs."""
        if not states:
            return []
        actions = [action_idx_to_solver_dict(a, s['env_state'], self.env,
                                             self.n_obstacles, self.n_z_levels)
                   for s, a in zip(states, action_indices)]
        pairs = list(zip([s['env_state'] for s in states], actions))

        max_batch = max(1, self.env.n_envs)
        all_results = []
        for c0 in range(0, len(pairs), max_batch):
            all_results.extend(self.env.batch_evaluate(pairs[c0:c0 + max_batch]))

        out = []
        for state, action, (new_env_state, _reward, done) in zip(states, actions, all_results):
            next_state = {
                'env_state': copy.deepcopy(new_env_state),
                'depth': state['depth'] + 1,
                'done': bool(done),
                'last_action': action,
            }
            out.append((next_state, self.is_terminal(next_state)))
        return out

    def initial_state(self, env_state: dict) -> dict:
        return {
            'env_state': copy.deepcopy(env_state),
            'depth': 0,
            'done': self.env._is_goal(env_state),
            'last_action': None,
        }
