"""Inference-only planner that uses the trained solver net.

Drop-in replacement for MCTSPusher: same plan() signature, same verification.
The stacker net is never loaded — at inference time we plan from a
user-provided initial_state (the env's actual world).
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from .games import SolverGame
from .grid import build_grid_spec, GridSpec
from .mcts import AZMCTS
from .networks import build_solver_net

logger = logging.getLogger(__name__)


def _load_solver_net(path: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    spec = GridSpec(**ckpt['spec'])
    # Checkpoints from before the net_arch knob are always MLPs.
    net = build_solver_net(ckpt.get('net_arch', 'mlp'),
                           in_dim=ckpt['solver_in_dim'],
                           n_actions=ckpt['solver_n_actions'],
                           n_obstacles=ckpt.get('n_obstacles', 0),
                           grid_h=spec.Gx, grid_w=spec.Gy)
    net.load_state_dict(ckpt['solver'])
    net.eval()
    return net, ckpt


class AlphaZeroPusher:
    """Inference-only planner using a trained solver network."""

    def __init__(self, env, solver_net_path: str,
                 n_simulations: int = 50, max_depth: int = 10,
                 c_puct: float = 1.5, temperature: float = 1e-3,
                 seed: int | None = 42, verify_threshold: float = 0.75,
                 n_verify_runs: int = 16,
                 verify_push_steps: int | None = None):
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.seed = seed
        self.verify_threshold = verify_threshold
        self.n_verify_runs = n_verify_runs
        self.verify_push_steps = verify_push_steps
        self.batch_size = max(1, env.n_envs)

        self.net, ckpt = _load_solver_net(solver_net_path)
        # Checkpoints from before the use_cell_onehot knob were always trained
        # with the grid one-hots included.
        self.use_cell_onehot = ckpt.get('use_cell_onehot', True)
        spec_dict = ckpt['spec']
        self.spec = GridSpec(**spec_dict)
        env_spec = build_grid_spec(env)
        if (env_spec.Gx, env_spec.Gy, env_spec.Z) != (self.spec.Gx, self.spec.Gy, self.spec.Z):
            logger.warning(
                'Grid mismatch between env (%dx%dx%d) and checkpoint (%dx%dx%d) — '
                'using checkpoint geometry; results may be off.',
                env_spec.Gx, env_spec.Gy, env_spec.Z,
                self.spec.Gx, self.spec.Gy, self.spec.Z)

        # Validate that the checkpoint action space matches the env.
        ckpt_n_z  = len(self.spec.z_levels) or 1
        ckpt_n_obs = ckpt['solver_n_actions'] // (4 * ckpt_n_z) - 1
        if ckpt_n_obs != env.n_obstacles:
            raise ValueError(
                f'Checkpoint {solver_net_path!r} was trained with n_obstacles={ckpt_n_obs} '
                f'but the environment has n_obstacles={env.n_obstacles}. '
                f'Pass n_obstacles={ckpt_n_obs} to benchmark.py.'
            )
        if ckpt_n_z != env.n_z_levels:
            logger.warning(
                'Checkpoint %r was trained with n_z_levels=%d but env has n_z_levels=%d. '
                'SolverGame will use the checkpoint value (%d) — ensure cfg.n_z_levels '
                'is set to %d so the planning env simulates the correct stack heights.',
                solver_net_path, ckpt_n_z, env.n_z_levels, ckpt_n_z, ckpt_n_z,
            )

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True, pause_before_verify: bool = False
             ) -> list[dict] | None:
        if initial_state is None:
            initial_state = self.env.get_state(0)

        game = SolverGame(self.env, self.spec, max_depth=self.max_depth,
                         use_cell_onehot=self.use_cell_onehot)
        mcts = AZMCTS(game, self.net, c_puct=self.c_puct,
                      dirichlet_eps=0.0)

        state = game.initial_state(initial_state)
        plan: list[dict] = []

        for t in range(self.max_depth):
            if game.is_terminal(state):
                break
            _, counts = mcts.run(state, self.n_simulations, add_root_noise=False)
            if counts.sum() <= 0:
                if verbose:
                    print(f'  AlphaZero: no visits at step {t}, stopping.')
                break
            a = int(counts.argmax().item())
            # We need the action *dict* (not the index) appended to the plan.
            # game.transition produces next state + we recover the action dict
            # from the returned state's last_action.
            state, _ = game.transition(state, a)
            assert state['last_action'] is not None
            plan.append(state['last_action'])
            if verbose:
                print(f'  AZ step {t+1}: [{state["last_action"]["action_type"]}] '
                      f'obj={state["last_action"]["obj_idx"]} z={state["last_action"]["push_z"]:.3f}')
            if state['done']:
                break

        if not plan:
            return None

        # Single-run gate check inside plan(); benchmark.py does the full n_verify_runs check separately.
        from planner import _verify_plan  # reuse existing verifier
        with self.env.push_steps_ctx(self.verify_push_steps) if hasattr(self.env, 'push_steps_ctx') else _NullCtx():
            n_tries = 1
            successes, avg_reward, goal_flags = _verify_plan(
                self.env, plan, copy.deepcopy(initial_state), n_tries,
                verbose=verbose, pause=pause_before_verify)

        if successes < n_tries * self.verify_threshold:
            if verbose:
                print(f'  AlphaZero: plan failed verification ({successes}/{n_tries}, avg_reward={avg_reward:.3f}).')
            return None

        if verbose:
            print(f'  AlphaZero: plan verified ({successes}/{n_tries}, avg_reward={avg_reward:.3f}).')
        self._last_verify = (successes, avg_reward, n_tries, goal_flags)
        return plan

    def verify(self, plan: list[dict], initial_state: dict,
               verbose: bool = True) -> tuple[int, float, float, bool, list]:
        # Return the cached result from plan()'s internal verification rather
        # than re-running a second stochastic pass whose results could diverge.
        successes, avg_reward, n_tries, goal_flags = self._last_verify
        rate = successes / n_tries
        passed = rate >= self.verify_threshold
        return successes, avg_reward, rate, passed, goal_flags


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
