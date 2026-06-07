"""
IGHA* planner for the bin-clearing task (jack branch).

Drop-in alternative to MCTSPusher / RRTPusher: same
``plan(initial_state, verbose=True, pause_before_verify=False) -> list[dict] | None``
contract and the standard push-action dicts, so verify / replay / --save all
work unchanged.

The planner uses the IGHA* *generic* environment (header-only, JIT-built) whose
dynamics/cost/validity/heuristic/goal_test callbacks wrap the puzzles simulator
through ``BinIGHAStarBridge`` (``SimulatorEnv.batch_evaluate`` as the forward
model). IGHA* searches over discretised bin-local xy state and returns an
optimal-ish state path under the grid; the bridge converts it back to actions.

Note: like MCTS/RRT this runs in-process with the simulator, so the IGHA*
extension is JIT-built against the *current* interpreter's torch (e.g. the
Isaac Lab conda env). The header-only generic env needs Boost headers; this
module auto-discovers a Boost include dir for the JIT compile.
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import torch

from planner import _PlannerBase


def _ensure_ighastar_importable() -> None:
    """Put the IGHAStar package root on sys.path and make Boost headers
    discoverable for the JIT compile (generic env uses boost::hash_combine)."""
    candidates = []
    env_root = os.environ.get("IGHASTAR_ROOT")
    if env_root:
        candidates.append(pathlib.Path(env_root))
    here = pathlib.Path(__file__).resolve().parent
    candidates += [here.parent / "IGHAStar", here.parent.parent / "IGHAStar"]
    for root in candidates:
        if (root / "ighastar" / "scripts" / "common_utils.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            os.environ.setdefault("IGHASTAR_SRC_DIR", str(root / "ighastar"))
            break

    existing = os.environ.get("CPLUS_INCLUDE_PATH", "")

    def _has_boost(inc: pathlib.Path) -> bool:
        return (inc / "boost" / "functional" / "hash.hpp").exists()

    already = any(p and _has_boost(pathlib.Path(p)) for p in existing.split(os.pathsep))
    if already or pathlib.Path("/usr/include/boost/functional/hash.hpp").exists():
        return

    search: list[pathlib.Path] = []
    # The active interpreter's own site-packages (e.g. Isaac Lab cmeel boost).
    for sp in sys.path:
        if sp.endswith("site-packages"):
            search.append(pathlib.Path(sp) / "cmeel.prefix" / "include")
    # Conda envs that ship Boost headers.
    envs_dir = pathlib.Path.home() / "miniconda3" / "envs"
    if envs_dir.is_dir():
        for e in envs_dir.iterdir():
            search.append(e / "include")
            search.append(e / "lib" / "python3.11" / "site-packages" / "cmeel.prefix" / "include")
    search.append(pathlib.Path.home() / "miniconda3" / "include")

    for inc in search:
        if _has_boost(inc):
            parts = [str(inc)] + ([existing] if existing else [])
            os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(parts)
            return


class IGHAStarPusher(_PlannerBase):
    """IGHA* generic-env planner over discretised bin xy state."""

    def __init__(self, env, max_expansions: int = 5000, hysteresis: int = 500,
                 resolution: float = 0.03, tolerance: float = 0.015,
                 grid_z: bool = True, z_resolution: float = 0.025,
                 z_tolerance: float = 0.0125,
                 max_level: int = 4, division_factor: float = 2.0,
                 preemptive_enabled: bool = False,
                 min_preemptive: int = 8, max_preemptive: int = 32,
                 seed: int | None = 42,
                 verify_threshold: float = 0.75, min_verify_envs: int = 16,
                 verify_push_steps: int | None = None,
                 prune_plan: bool = False, debug: bool = False):
        super().__init__(env, verify_threshold, min_verify_envs, seed,
                         verify_push_steps, prune_plan)
        self.debug = bool(debug)
        _ensure_ighastar_importable()
        from ighastar_bridge import BinIGHAStarBridge
        from ighastar.scripts.common_utils import create_planner

        self.bridge = BinIGHAStarBridge(env, grid_z=grid_z)
        self.max_expansions = max_expansions
        self.hysteresis = hysteresis
        self.config = self.bridge.build_config(
            max_expansions=max_expansions, hysteresis=hysteresis,
            resolution=resolution, tolerance=tolerance,
            z_resolution=z_resolution, z_tolerance=z_tolerance,
            max_level=max_level, division_factor=division_factor,
            preemptive_enabled=preemptive_enabled,
            min_preemptive=min_preemptive, max_preemptive=max_preemptive)
        self._create_planner = create_planner
        self.planner = None
        self.stats: dict = {}

    @classmethod
    def from_cfg(cls, env, cfg, seed: int) -> 'IGHAStarPusher':
        p = cfg.planner
        return cls(
            env=env,
            max_expansions=p.get('max_expansions', 5000),
            hysteresis=p.get('hysteresis', 500),
            resolution=p.get('resolution', 0.03),
            tolerance=p.get('tolerance', 0.015),
            grid_z=p.get('grid_z', True),
            z_resolution=p.get('z_resolution', 0.025),
            z_tolerance=p.get('z_tolerance', 0.0125),
            max_level=p.get('max_level', 4),
            division_factor=p.get('division_factor', 2.0),
            preemptive_enabled=bool(p.get('preemptive_expansion', {}).get('enabled', False)),
            min_preemptive=int(p.get('preemptive_expansion', {}).get('min_preemptive', 8)),
            max_preemptive=int(p.get('preemptive_expansion', {}).get('max_preemptive', 32)),
            seed=seed,
            verify_threshold=cfg.verify_threshold,
            min_verify_envs=cfg.min_verify_envs,
            verify_push_steps=cfg.get('verify_push_steps', None),
            prune_plan=cfg.get('prune_plan', False),
            debug=bool(p.get('debug', cfg.get('debug', False))),
        )

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True,
             pause_before_verify: bool = False) -> list[dict] | None:
        if initial_state is None:
            initial_state = self.env.get_state(0)

        n = self.bridge.n_dims
        start_vec = self.bridge.encode_state(initial_state)
        start = torch.tensor(start_vec, dtype=torch.float32)
        goal = torch.zeros(n, dtype=torch.float32)   # unused by generic goal_test
        world = torch.zeros(1, dtype=torch.float32)  # unused by generic env

        if verbose:
            print(f'[IGHA*] state_dim={n} (hash_dims={self.bridge.hash_dims} xy), '
                  f'controls={self.bridge.num_controls}, '
                  f'max_expansions={self.max_expansions}, hysteresis={self.hysteresis}')
            print(f'[IGHA*] start xy={np.round(start_vec[:self.bridge.hash_dims], 3)}, '
                  f'exit_y={self.bridge.exit_y}')

        self.planner = self._create_planner(self.config, bidirectional=False)
        if self.debug:
            # create_planner hardcodes debug=False; re-instantiate the already
            # built/loaded class with debug=True so IGHA* prints its per-iteration
            # stats (Expansions / level / Q_v / Seen / inactive queue).
            self.planner = type(self.planner)(self.config, True)
            print('[IGHA*] debug mode ON -- per-iteration stats below')

        success = self.planner.search(
            start, goal, world, self.max_expansions, self.hysteresis, True)

        try:
            prof = self.planner.get_profiler_info()
            expansions = int(prof[7]) if len(prof) > 7 else None
        except Exception:
            prof = None
            expansions = None
        self.stats = {'success': bool(success), 'expansions': expansions,
                      'edges': len(self.bridge.edges)}
        if verbose:
            print(f'[IGHA*] search success={success}, expansions={expansions}, '
                  f'edges_explored={len(self.bridge.edges)}')
        if self.debug and prof is not None:
            # prof: (avg_succ_t, avg_goal_t, avg_overhead_t, avg_g_update_t,
            #        switches, max_level, q_v_size, expansions, exp_list, cost_list)
            try:
                preempt = self.planner.get_preemptive_expansions()
            except Exception:
                preempt = 'n/a'
            try:
                print(f'[IGHA*] profiler: switches={int(prof[4])}, '
                      f'max_level={int(prof[5])}, Q_v_size={int(prof[6])}, '
                      f'expansions={int(prof[7])}, preemptive_expansions={preempt} | '
                      f'avg_us: successor={float(prof[0]):.1f}, goal={float(prof[1]):.1f}, '
                      f'overhead={float(prof[2]):.1f}, g_update={float(prof[3]):.1f}')
            except Exception:
                pass

        if not success:
            if verbose:
                print('[IGHA*] no goal path found.')
            return None

        path_tensor = self.planner.get_best_path().numpy()  # [P, n+1], goal-first
        state_path = path_tensor[::-1, :n].copy()            # post-push states, start->goal
        # get_best_path() emits only non-start nodes, so a k-push solution has k
        # rows. Prepend the true start so each consecutive (s_i, s_{i+1}) pair
        # maps to exactly one recorded push edge.
        if len(state_path) == 0 or not np.allclose(state_path[0], start_vec, atol=1e-4):
            state_path = np.vstack([start_vec.reshape(1, -1), state_path])
        plan = self.bridge.actions_from_state_path(state_path)
        if verbose:
            print(f'[IGHA*] path has {len(state_path)} states -> {len(plan)} actions')

        if not plan:
            return None

        # Same verification gate as MCTS/RRT.
        successes, avg_reward, rate, passed = self.verify(plan, initial_state, verbose)
        if verbose:
            print(f'[IGHA*] verify: {successes}/{self.batch_size} '
                  f'(rate={rate:.2f}, avg_reward={avg_reward:.3f}, passed={passed})')
        if passed and self.prune_plan:
            from planner import _prune_plan
            with self.env.push_steps_ctx(self.verify_push_steps):
                plan = _prune_plan(self.env, plan, initial_state,
                                   self.batch_size, self.verify_threshold, verbose)
        return plan if passed else None
