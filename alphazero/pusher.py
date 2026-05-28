"""Inference-only planner that uses the trained solver net.

Drop-in replacement for MCTSPusher: same plan() signature, same verification.
The stacker net is never loaded — at inference time we plan from a
user-provided initial_state (the env's actual world).
"""

from __future__ import annotations

import copy
import logging

import torch

from .games import SolverGame
from .grid import build_grid_spec, GridSpec
from .mcts import AZMCTS
from .networks import SolverNet

logger = logging.getLogger(__name__)


def _load_solver_net(path: str) -> tuple[SolverNet, dict]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    net = SolverNet(in_dim=ckpt['solver_in_dim'],
                    n_actions=ckpt['solver_n_actions'])
    net.load_state_dict(ckpt['solver'])
    net.eval()
    return net, ckpt


class AlphaZeroPusher:
    """Inference-only planner using a trained solver network."""

    def __init__(self, env, solver_net_path: str,
                 n_simulations: int = 50, max_depth: int = 10,
                 c_puct: float = 1.5, temperature: float = 1e-3,
                 seed: int | None = 42, verify_threshold: float = 0.75,
                 min_verify_envs: int = 16,
                 verify_push_steps: int | None = None):
        self.env = env
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.seed = seed
        self.verify_threshold = verify_threshold
        self.min_verify_envs = min_verify_envs
        self.verify_push_steps = verify_push_steps
        self.batch_size = max(1, env.n_envs)

        self.net, ckpt = _load_solver_net(solver_net_path)
        spec_dict = ckpt['spec']
        self.spec = GridSpec(**spec_dict)
        env_spec = build_grid_spec(env)
        if (env_spec.Gx, env_spec.Gy, env_spec.Z) != (self.spec.Gx, self.spec.Gy, self.spec.Z):
            logger.warning(
                'Grid mismatch between env (%dx%dx%d) and checkpoint (%dx%dx%d) — '
                'using checkpoint geometry; results may be off.',
                env_spec.Gx, env_spec.Gy, env_spec.Z,
                self.spec.Gx, self.spec.Gy, self.spec.Z)

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True, pause_before_verify: bool = False
             ) -> list[dict] | None:
        if initial_state is None:
            initial_state = self.env.get_state(0)

        game = SolverGame(self.env, self.spec, max_depth=self.max_depth)
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

        # Verify against multiple envs (same as MCTSPusher does for goal nodes)
        from planner import _verify_plan  # reuse existing verifier
        with self.env.push_steps_ctx(self.verify_push_steps) if hasattr(self.env, 'push_steps_ctx') else _NullCtx():
            n_tries = max(self.min_verify_envs, self.batch_size)
            successes, avg_reward = _verify_plan(
                self.env, plan, copy.deepcopy(initial_state), n_tries,
                verbose=verbose, pause=pause_before_verify)

        if successes < n_tries * self.verify_threshold:
            if verbose:
                print(f'  AlphaZero: plan failed verification ({successes}/{n_tries}, avg_reward={avg_reward:.3f}).')
            return None

        if verbose:
            print(f'  AlphaZero: plan verified ({successes}/{n_tries}, avg_reward={avg_reward:.3f}).')
        return plan


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
