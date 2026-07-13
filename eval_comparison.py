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

from __future__ import annotations

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
                 planner_name: str, verbose: bool) -> EpisodeResult:
    env.reset_sim_counters()
    t0 = time.perf_counter()
    plan = planner.plan(copy.deepcopy(initial_state), verbose=verbose)
    elapsed = time.perf_counter() - t0

    success = plan is not None and len(plan) > 0
    return EpisodeResult(
        seed=seed,
        planner=planner_name,
        success=success,
        plan_length=len(plan) if success else 0,
        plan_time_s=elapsed,
        batch_calls=env.batch_calls,
        total_pairs=env.total_pairs,
    )


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
        max_depth=args.max_depth,
        gamma=args.more_gamma,
        k_per_object=args.more_k,
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
    p.add_argument('--n_obs',   type=int,   default=2)
    p.add_argument('--n_envs',  type=int,   default=16,
                   help='Parallel envs (for verification)')
    p.add_argument('--n_runs',  type=int,   default=20)
    p.add_argument('--seed',    type=int,   default=0,
                   help='Base seed; run i uses seed+i')
    p.add_argument('--max_depth',         type=int,   default=10)
    p.add_argument('--verify_threshold',  type=float, default=0.75)
    p.add_argument('--output',  default='comparison_results.csv')
    p.add_argument('--verbose', action='store_true')
    # AlphaZero
    p.add_argument('--az_checkpoint', required=True,
                   help='Path to AlphaZero checkpoint (.pt)')
    p.add_argument('--az_n_sim',   type=int,   default=500)
    p.add_argument('--az_c_puct',  type=float, default=1.5)
    # MORE
    p.add_argument('--more_checkpoint', default=None,
                   help='Path to PPN checkpoint (.pt); omit for unguided MORE')
    p.add_argument('--more_n_sim', type=int,   default=500)
    p.add_argument('--more_gamma', type=float, default=0.5)
    p.add_argument('--more_k',     type=int,   default=8,
                   help='Contour samples per object')
    return p.parse_args()


def _build_env(sim: str, n_obs: int, n_envs: int):
    """Build a BinEnv by loading the project's Hydra YAML config files."""
    from simulators import build_env

    conf_dir = os.path.join(os.path.dirname(__file__), 'conf')
    base    = OmegaConf.load(os.path.join(conf_dir, 'config.yaml'))
    sim_cfg = OmegaConf.load(os.path.join(conf_dir, 'simulator', f'{sim}.yaml'))
    rew_cfg = OmegaConf.load(os.path.join(conf_dir, 'reward', 'default.yaml'))

    cfg = OmegaConf.merge(base, {
        'simulator':   sim_cfg,
        'reward':      rew_cfg,
        'n_obstacles': n_obs,
        'seed':        None,
    })
    OmegaConf.set_struct(cfg, False)
    cfg.pop('defaults', None)
    return build_env(cfg, n_envs=n_envs, viewer_mode='headless')


def main():
    args = _parse_args()

    # ---------- simulator ----------
    if args.sim == 'isaaclab':
        os.environ.setdefault('ISAACLAB_HEADLESS', '1')

    print(f'[eval] Building env: sim={args.sim}, n_obs={args.n_obs}, '
          f'n_envs={args.n_envs}')
    env = _build_env(args.sim, n_obs=args.n_obs, n_envs=args.n_envs)

    # ---------- planners ----------
    print(f'[eval] Building AlphaZero planner (checkpoint={args.az_checkpoint})')
    az_planner = _build_az_planner(env, args)

    ppn_status = args.more_checkpoint or 'unguided'
    print(f'[eval] Building MORE planner (ppn={ppn_status})')
    more_planner = _build_more_planner(env, args)

    # ---------- run ----------
    all_results: list[EpisodeResult] = []

    for run_i in range(args.n_runs):
        seed = args.seed + run_i
        initial_state = env.reset(seed=seed)
        state_copy = copy.deepcopy(initial_state)

        print(f'\n[eval] Run {run_i + 1}/{args.n_runs}  seed={seed}')

        for planner, name in [(az_planner, 'AlphaZero'),
                               (more_planner, 'MORE')]:
            res = _run_episode(
                env, planner, state_copy, seed, name, verbose=args.verbose)
            status = 'OK ' if res.success else 'FAIL'
            print(f'  {name:12s} [{status}] '
                  f'plan_len={res.plan_length:3d}  '
                  f'time={res.plan_time_s:6.1f}s  '
                  f'batch_calls={res.batch_calls:6d}  '
                  f'total_pairs={res.total_pairs:7d}')
            all_results.append(res)

    # ---------- aggregate and display ----------
    az_results   = [r for r in all_results if r.planner == 'AlphaZero']
    more_results = [r for r in all_results if r.planner == 'MORE']

    agg_rows = [_aggregate(az_results), _aggregate(more_results)]
    print('\n=== Comparison Summary ===')
    _print_table(agg_rows)

    # ---------- save ----------
    _save_csv(all_results, args.output)
    _save_summary_csv(agg_rows, args.output)

    print('\nNOTE: plan_len is a soft metric — MORE uses contour pushes and '
          'AlphaZero uses cardinal pushes; they are different units of work.  '
          'Primary metrics are success_rate, planning time, and real-arm execution.')


if __name__ == '__main__':
    main()
