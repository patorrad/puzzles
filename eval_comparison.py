from __future__ import annotations

import sys as _sys, re as _re

class _OmniFilter:
    _pat = _re.compile(r'(\d{4}-\d{2}-\d{2}T|\[INFO\]|\[WARNING\]|\[ERROR\]).*'
                       r'(omni|isaacsim|carb|isaac|kit|nv::|physx|usd|omniverse)',
                       _re.IGNORECASE)
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)

_sys.stdout = _OmniFilter(_sys.stdout)

"""
Head-to-head evaluation: AlphaZero PUCT vs MORE.

Runs both planners on identical scenes (same seeds) and logs per-episode:
  success, plan length, planning wall-clock, #batch_calls, #total_pairs, plan

Aggregates into a comparison table (CSV + printed summary).

Usage::

    # IsaacLab, 20 runs, 2 obstacles:
    ISAACLAB_HEADLESS=1 python eval_comparison.py \\
        --sim isaaclab --n_obs 2 --n_runs 20 --n_envs 16 \\
        --az_checkpoint alphazero_latest.pt \\
        --more_checkpoint ppn.pt \\
        --output results.csv

    # Unguided MORE (no PPN — useful to sanity-check before training):
    ISAACLAB_HEADLESS=1 python eval_comparison.py \\
        --sim isaaclab --n_obs 2 --n_runs 10 \\
        --az_checkpoint alphazero_latest.pt \\
        --output results.csv
"""

import argparse
import copy
import csv
import os
import time
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    seed:            int
    planner:         str
    success:         bool
    plan_length:     int
    plan_time_s:     float
    batch_calls:     int
    total_pairs:     int


# ---------------------------------------------------------------------------
# Run one episode with a given planner
# ---------------------------------------------------------------------------

def _run_episode(env, planner, initial_state: dict, seed: int,
                 planner_name: str, verbose: bool) -> tuple['EpisodeResult', list | None]:
    env.reset_sim_counters()
    t0 = time.perf_counter()
    plan = planner.plan(copy.deepcopy(initial_state), verbose=verbose)
    elapsed = time.perf_counter() - t0

    success = plan is not None and len(plan) > 0
    result = EpisodeResult(
        seed=seed,
        planner=planner_name,
        success=success,
        plan_length=len(plan) if success else 0,
        plan_time_s=elapsed,
        batch_calls=env.batch_calls,
        total_pairs=env.total_pairs,
    )
    return result, (plan if success else None)


# ---------------------------------------------------------------------------
# Aggregate statistics over a list of EpisodeResults
# ---------------------------------------------------------------------------

def _aggregate(results: list[EpisodeResult]) -> dict:
    if not results:
        return {}

    def _s(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    n = len(results)
    successes = [r for r in results if r.success]
    success_rate = len(successes) / n

    pl_m, pl_s   = _s([r.plan_length  for r in successes]) if successes else (0., 0.)
    bc_m, bc_s   = _s([r.batch_calls  for r in results])
    tp_m, tp_s   = _s([r.total_pairs  for r in results])
    pt_m, pt_s   = _s([r.plan_time_s  for r in results])

    return {
        'planner':          results[0].planner,
        'n_runs':           n,
        'success_rate':     f'{success_rate:.1%}',
        'plan_len_mean':    f'{pl_m:.1f}',
        'plan_len_std':     f'{pl_s:.1f}',
        'time_mean_s':      f'{pt_m:.1f}',
        'time_std_s':       f'{pt_s:.1f}',
        'batch_calls_mean': f'{bc_m:.0f}',
        'batch_calls_std':  f'{bc_s:.0f}',
        'total_pairs_mean': f'{tp_m:.0f}',
        'total_pairs_std':  f'{tp_s:.0f}',
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    sep = '+-' + '-+-'.join('-' * widths[k] for k in keys) + '-+'
    header = '| ' + ' | '.join(k.ljust(widths[k]) for k in keys) + ' |'
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print('| ' + ' | '.join(str(row[k]).ljust(widths[k]) for k in keys) + ' |')
    print(sep)


def _save_csv(results: list[EpisodeResult], path: str) -> None:
    fnames = [f.name for f in fields(EpisodeResult)]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f'[eval] Per-episode CSV → {path}')


def _save_summary_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    summary_path = path.replace('.csv', '_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[eval] Summary CSV → {summary_path}')


# ---------------------------------------------------------------------------
# Build planners
# ---------------------------------------------------------------------------

def _build_az_planner(env, args):
    from alphazero.pusher import AlphaZeroPusher
    return AlphaZeroPusher(
        env=env,
        solver_net_path=args.az_checkpoint,
        n_simulations=args.az_n_sim,
        max_depth=args.max_depth,
        c_puct=args.az_c_puct,
        seed=args.seed,
        verify_threshold=args.verify_threshold,
        min_verify_envs=args.n_envs,
    )


def _build_more_planner(env, args):
    from more.planner import MOREPlanner
    ppn_path = args.more_checkpoint if args.more_checkpoint else None
    return MOREPlanner(
        env=env,
        ppn_path=ppn_path,
        n_simulations=args.more_n_sim,
        tree_depth=args.more_tree_depth,
        n_plan_steps=args.more_plan_steps,
        gamma=args.more_gamma,
        k_per_object=args.more_k,
        seed=args.seed,
        verify_threshold=0.0,   # contour pushes are stochastic; trust direct execution
        min_verify_envs=args.n_envs,
    )


def _build_mcts_planner(env, args):
    from planner import MCTSPusher
    return MCTSPusher(
        env=env,
        n_simulations=args.mcts_n_sim,
        max_depth=args.max_depth,
        c_ucb=args.mcts_c_ucb,
        seed=args.seed,
        verify_threshold=args.verify_threshold,
        min_verify_envs=args.n_envs,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Head-to-head: AlphaZero PUCT vs MORE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared
    p.add_argument('--sim',     default='isaaclab',
                   choices=['isaaclab', 'genesis'],
                   help='Simulator backend')
    p.add_argument('--n_obs',      type=int,   default=2)
    p.add_argument('--n_envs',     type=int,   default=16,
                   help='Parallel envs (for verification)')
    p.add_argument('--stackable',  action='store_true',
                   help='Enable stackable objects (must match training config)')
    p.add_argument('--n_z_levels', type=int,   default=1,
                   help='Push height levels (must match training config)')
    p.add_argument('--bin_size',   type=float, default=None,
                   help='Fixed bin side length in metres; null = auto-scale with n_obs. '
                        'Use 0.4 to keep the N=7 bin size regardless of n_obs.')
    p.add_argument('--n_runs',  type=int,   default=20)
    p.add_argument('--seed',    type=int,   default=0,
                   help='Base seed; run i uses seed+i')
    p.add_argument('--max_depth',         type=int,   default=10,
                   help='AlphaZero MCTS depth')
    p.add_argument('--verify_threshold',  type=float, default=0.75,
                   help='AlphaZero verification threshold (MORE always uses 0.0)')
    p.add_argument('--output',  default='comparison_results.csv')
    p.add_argument('--verbose', action='store_true')
    # AlphaZero
    p.add_argument('--az_checkpoint', required=True,
                   help='Path to AlphaZero checkpoint (.pt)')
    p.add_argument('--az_n_sim',   type=int,   default=500)
    p.add_argument('--az_c_puct',  type=float, default=1.5)
    # Vanilla MCTS
    p.add_argument('--vanilla_mcts',  action='store_true',
                   help='Include vanilla UCB1 MCTS (no neural net) as a third planner')
    p.add_argument('--mcts_n_sim',    type=int,   default=500,
                   help='Simulations for vanilla MCTS')
    p.add_argument('--mcts_c_ucb',    type=float, default=1.4,
                   help='UCB exploration constant for vanilla MCTS')
    # MORE
    p.add_argument('--more_checkpoint', default=None,
                   help='Path to PPN checkpoint (.pt); omit for unguided MORE')
    p.add_argument('--more_n_sim',       type=int,   default=100)
    p.add_argument('--more_tree_depth',  type=int,   default=3,
                   help='MORE MCTS tree depth (paper uses 3)')
    p.add_argument('--more_plan_steps',  type=int,   default=30,
                   help='Max planning steps in the real env (separate from tree depth)')
    p.add_argument('--more_gamma',       type=float, default=0.5)
    p.add_argument('--more_k',           type=int,   default=4,
                   help='Contour samples per object')
    # WandB
    p.add_argument('--wandb',            action='store_true')
    p.add_argument('--wandb_project',    default='puzzle-comparison')
    p.add_argument('--wandb_entity',     default=None)
    p.add_argument('--log_video',        action='store_true',
                   help='Record and upload a replay video for every episode')
    p.add_argument('--video_dir',        default='eval_videos',
                   help='Directory to write mp4 files before uploading')
    return p.parse_args()


def _build_env(sim: str, n_obs: int, n_envs: int,
               stackable: bool = False, n_z_levels: int = 1,
               bin_size: float | None = None):
    """Build a BinEnv by loading the project's Hydra YAML config files."""
    from simulators import build_env

    conf_dir = os.path.join(os.path.dirname(__file__), 'conf')
    base    = OmegaConf.load(os.path.join(conf_dir, 'config.yaml'))
    sim_cfg = OmegaConf.load(os.path.join(conf_dir, 'simulator', f'{sim}.yaml'))
    rew_cfg = OmegaConf.load(os.path.join(conf_dir, 'reward', 'default.yaml'))

    overrides = {
        'simulator':   sim_cfg,
        'reward':      rew_cfg,
        'n_obstacles': n_obs,
        'stackable':   stackable,
        'n_z_levels':  n_z_levels,
        'seed':        None,
    }
    if bin_size is not None:
        overrides['bin_size'] = bin_size

    cfg = OmegaConf.merge(base, overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.pop('defaults', None)
    return build_env(cfg, n_envs=n_envs, viewer_mode='headless')


def _record_video(env, plan, initial_state, name, seed, video_dir) -> str | None:
    """Replay plan through physics and save mp4. Returns path or None on failure."""
    if not hasattr(env, 'record_replay'):
        return None
    os.makedirs(video_dir, exist_ok=True)
    path = os.path.join(video_dir, f'{name}_seed{seed:04d}.mp4')
    try:
        return env.record_replay(plan, copy.deepcopy(initial_state), path)
    except Exception as e:
        print(f'[eval] record_replay failed ({name} seed={seed}): {e}',
              file=__import__('sys').stderr)
        return None


def main():
    args = _parse_args()

    # ---------- wandb ----------
    wb = None
    if args.wandb:
        try:
            import wandb as _wb
            wb = _wb
            wb.init(project=args.wandb_project,
                    entity=args.wandb_entity or None,
                    config=vars(args))
            print(f'[eval] wandb run: {wb.run.url}')
        except Exception as e:
            print(f'[eval] wandb init failed: {e}', file=__import__('sys').stderr)
            wb = None

    # ---------- simulator ----------
    if args.sim == 'isaaclab':
        os.environ.setdefault('ISAACLAB_HEADLESS', '1')
        if args.log_video:
            os.environ['ISAACLAB_ENABLE_CAMERAS'] = '1'

    print(f'[eval] Building env: sim={args.sim}, n_obs={args.n_obs}, '
          f'n_envs={args.n_envs}, stackable={args.stackable}, '
          f'n_z_levels={args.n_z_levels}, bin_size={args.bin_size}')
    env = _build_env(args.sim, n_obs=args.n_obs, n_envs=args.n_envs,
                     stackable=args.stackable, n_z_levels=args.n_z_levels,
                     bin_size=args.bin_size)

    # ---------- planners ----------
    print(f'[eval] Building AlphaZero planner (checkpoint={args.az_checkpoint})')
    az_planner = _build_az_planner(env, args)

    ppn_status = args.more_checkpoint or 'unguided'
    print(f'[eval] Building MORE planner (ppn={ppn_status})')
    more_planner = _build_more_planner(env, args)

    planners = [(az_planner, 'AlphaZero'), (more_planner, 'MORE')]

    if args.vanilla_mcts:
        print(f'[eval] Building vanilla MCTS planner (n_sim={args.mcts_n_sim})')
        mcts_planner = _build_mcts_planner(env, args)
        planners.append((mcts_planner, 'VanillaMCTS'))

    # ---------- run ----------
    all_results: list[EpisodeResult] = []

    for run_i in range(args.n_runs):
        seed = args.seed + run_i
        initial_state = env.reset(seed=seed)
        state_copy = copy.deepcopy(initial_state)

        print(f'\n[eval] Run {run_i + 1}/{args.n_runs}  seed={seed}')

        run_log: dict = {'run_i': run_i, 'seed': seed}
        plans: dict = {}

        for planner, name in planners:
            res, plan = _run_episode(
                env, planner, state_copy, seed, name, verbose=args.verbose)
            status = 'OK ' if res.success else 'FAIL'
            print(f'  {name:12s} [{status}] '
                  f'plan_len={res.plan_length:3d}  '
                  f'time={res.plan_time_s:6.1f}s  '
                  f'batch_calls={res.batch_calls:6d}  '
                  f'total_pairs={res.total_pairs:7d}')
            all_results.append(res)
            plans[name] = plan

            run_log[f'{name}/success']     = int(res.success)
            run_log[f'{name}/plan_length'] = res.plan_length
            run_log[f'{name}/plan_time_s'] = res.plan_time_s
            run_log[f'{name}/batch_calls'] = res.batch_calls
            run_log[f'{name}/total_pairs'] = res.total_pairs

        # Record videos before logging so they go in the same wandb step
        if args.log_video:
            for name, plan in plans.items():
                if not plan:
                    print(f'  [video] {name}: skipped (no successful plan)')
                    continue
                vid = _record_video(env, plan, state_copy, name, seed,
                                    args.video_dir)
                if vid is None:
                    print(f'  [video] {name}: record_replay returned None '
                          f'(no frames captured or error)')
                elif wb:
                    run_log[f'{name}/video'] = wb.Video(vid, fps=30,
                                                         format='mp4')
                    print(f'  [video] {name}: logged to wandb ({vid})')

        # Single wb.log call per run → step aligns with run_i
        if wb:
            wb.log(run_log, step=run_i)

    # ---------- aggregate and display ----------
    planner_names = [name for _, name in planners]
    agg_rows = [_aggregate([r for r in all_results if r.planner == name])
                for name in planner_names]
    print('\n=== Comparison Summary ===')
    _print_table(agg_rows)

    if wb:
        for row in agg_rows:
            wb.log({f"summary/{row['planner']}/{k}": v
                    for k, v in row.items() if k != 'planner'})
        wb.finish()

    # ---------- save ----------
    _save_csv(all_results, args.output)
    _save_summary_csv(agg_rows, args.output)

    print('\nNOTE: plan_len is a soft metric — MORE uses contour pushes and '
          'AlphaZero uses cardinal pushes; they are different units of work.  '
          'Primary metrics are success_rate, planning time, and real-arm execution.')


if __name__ == '__main__':
    main()
