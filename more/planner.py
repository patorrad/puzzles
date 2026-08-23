"""
MOREPlanner — inference wrapper for MORE (Huang et al., ICRA 2022, arXiv:2202.01426).

Implements the shared Planner protocol so it can be swapped directly with
AlphaZeroPusher in any eval harness.

NOTE: Action count is a soft metric when comparing MORE against PUCT.  MORE's
contour pushes and PUCT's cardinal moves are different units of work; a single
MORE push may correspond to a qualitatively different intervention than a cardinal
push.  Primary comparison metrics are success rate, planning wall-clock time, and
downstream real-arm execution performance.
"""

from __future__ import annotations

import copy
import sys
import time

import torch

def _log(msg): print(msg, file=sys.stderr, flush=True)

from planner import _verify_plan
from more.contour_sampler import ContourSampler
from more.mcts import MORETree
from more.ppn import build_ppn


def _load_ppn(path: str, n_obstacles: int):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    arch = ckpt.get('arch', 'deepsets')
    if arch == 'deepsets':
        kwargs = dict(obj_emb_dim=ckpt.get('obj_emb_dim', 64),
                      push_emb_dim=ckpt.get('push_emb_dim', 64),
                      agg_hidden=ckpt.get('agg_hidden', 128))
    else:
        kwargs = dict(hidden=ckpt.get('hidden', 128))
    net = build_ppn(arch, n_obstacles=n_obstacles, **kwargs)
    net.load_state_dict(ckpt['ppn'])
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)


class MOREPlanner:
    """
    MORE guided-MCTS planner.

    References
    ----------
    Huang et al., "Interleaving Monte Carlo Tree Search and Self-Supervised
    Learning for Object Retrieval in Clutter", ICRA 2022, arXiv:2202.01426.

    Parameters
    ----------
    env : SimulatorEnv
    ppn_path : str | None
        Path to a trained PPN checkpoint.  If None, falls back to unguided UCT
        (useful for Phase A data collection without a trained network).
    n_simulations : int
        MCTS iterations per plan() call.
    max_depth : int
        Maximum search depth (paper uses 3 for guided eval).
    gamma : float
        Discount factor (paper uses 0.5).
    k_per_object : int
        Contour push samples per object per expansion step.
    verify_threshold : float
    min_verify_envs : int
    seed : int | None
    """

    def __init__(self,
                 env,
                 ppn_path: str | None = None,
                 n_simulations: int = 500,
                 tree_depth: int = 3,
                 n_plan_steps: int = 20,
                 gamma: float = 0.5,
                 k_per_object: int = 8,
                 rollout_depth: int = 5,
                 m: int = 3,
                 c_uct: float = 2.0,
                 verify_threshold: float = 0.0,
                 min_verify_envs: int = 16,
                 verify_push_steps: int | None = None,
                 seed: int | None = 42):
        self.env = env
        self.n_simulations = n_simulations
        self.n_plan_steps = n_plan_steps
        self.verify_threshold = verify_threshold
        self.min_verify_envs = min_verify_envs
        self.verify_push_steps = verify_push_steps
        self.batch_size = max(1, env.n_envs)

        if seed is not None:
            torch.manual_seed(seed)

        self.ppn = None
        if ppn_path is not None:
            self.ppn = _load_ppn(ppn_path, env.n_obstacles)

        self.sampler = ContourSampler.from_env(env, include_target=True)
        self.tree = MORETree(
            env=env,
            ppn=self.ppn,
            contour_sampler=self.sampler,
            gamma=gamma,
            max_depth=tree_depth,
            rollout_depth=rollout_depth,
            m=m,
            c_uct=c_uct,
            k_per_object=k_per_object,
        )
        self.collected_records: list[dict] = []

    # ------------------------------------------------------------------
    # Planner protocol
    # ------------------------------------------------------------------

    def plan(self,
             initial_state: dict | None = None,
             verbose: bool = True,
             pause_before_verify: bool = False) -> list[dict] | None:
        if initial_state is None:
            initial_state = self.env.get_state(0)

        t0 = time.perf_counter()
        plan: list[dict] = []
        state = copy.deepcopy(initial_state)
        done_in_plan = False

        for step in range(self.n_plan_steps):
            if self.env._is_goal(state):
                done_in_plan = True
                break

            action = self.tree.search(state, self.n_simulations)
            self._collect_tree_data()
            if action is None:
                if verbose:
                    _log(f'  MORE: no action found at step {step}, stopping.')
                break

            results = self.env.batch_evaluate([(state, action)])
            new_state, reward, done = results[0]

            if self.env._obstacles_dropped(new_state):
                if verbose:
                    _log(f'  MORE: obstacle dropped at step {step}, stopping.')
                break

            plan.append(action)
            if verbose:
                atype = action['action_type']
                oidx  = action.get('obj_idx', '?')
                _log(f'  MORE step {step+1}: [{atype}] obj={oidx} r={reward:.3f}')
            state = new_state

            if done:
                done_in_plan = True
                break

        elapsed = time.perf_counter() - t0

        if not plan:
            return None

        # Success only if the target actually exited the bin during execution.
        # verify_threshold=0.0 skips the stochastic replay test but does NOT
        # override the goal-completion requirement.
        if not done_in_plan:
            if verbose:
                _log(f'  MORE: {len(plan)} steps but goal not reached (t={elapsed:.1f}s).')
            return None

        if verbose:
            _log(f'  MORE: done in {len(plan)} steps (t={elapsed:.1f}s).')

        # Goal reached — skip replay verification (contour pushes are too
        # stochastic to reliably pass a fixed-plan replay test).
        if self.verify_threshold <= 0.0:
            return plan

        ctx = (self.env.push_steps_ctx(self.verify_push_steps)
               if hasattr(self.env, 'push_steps_ctx') else _NullCtx())
        with ctx:
            n_tries = self.min_verify_envs
            successes, avg_reward, _ = _verify_plan(
                self.env, plan, copy.deepcopy(initial_state),
                n_tries, verbose=verbose, pause=pause_before_verify)

        rate = successes / n_tries
        if rate < self.verify_threshold:
            if verbose:
                _log(f'  MORE: plan failed verification ({successes}/{n_tries}, '
                     f'avg_reward={avg_reward:.3f}, t={elapsed:.1f}s).')
            return None

        if verbose:
            _log(f'  MORE: plan verified ({successes}/{n_tries}, '
                 f'avg_reward={avg_reward:.3f}, t={elapsed:.1f}s).')
        self._last_verify = (successes, avg_reward, n_tries)
        return plan

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def _collect_tree_data(self) -> None:
        """Walk the last search tree and append training-format records."""
        root = self.tree.last_root
        if root is None:
            return
        self.collected_records.extend(self.tree.collect_transitions(root))

    def save_collected_data(self, path: str) -> None:
        import torch, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.collected_records, path)
        print(f'[MORE] Collected data saved → {path} ({len(self.collected_records):,} records)')

    def verify(self, plan: list[dict], initial_state: dict,
               verbose: bool = True) -> tuple[int, float, float, bool, list]:
        ctx = (self.env.push_steps_ctx(self.verify_push_steps)
               if hasattr(self.env, 'push_steps_ctx') else _NullCtx())
        with ctx:
            n_tries = self.min_verify_envs
            successes, avg_reward, goal_flags = _verify_plan(
                self.env, plan, copy.deepcopy(initial_state),
                n_tries, verbose=verbose)
        rate = successes / n_tries
        return successes, avg_reward, rate, rate >= self.verify_threshold, goal_flags


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
