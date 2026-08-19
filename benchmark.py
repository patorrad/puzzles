"""
benchmark.py  –  Multi-run evaluation with WandB logging.

Usage
-----
# Genesis + MCTS, 20 runs:
python benchmark.py --config-name=benchmark

# Override simulator / planner:
python benchmark.py --config-name=benchmark simulator=isaaclab planner=rrt n_runs=10

# Record solution videos (IsaacLab only):
python benchmark.py --config-name=benchmark simulator=isaaclab record_video=true n_runs=5

# WandB project / entity:
python benchmark.py --config-name=benchmark wandb_project=my-project wandb_entity=myname
"""

import csv
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import wandb


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_idx: int
    seed: int
    success: bool
    plan_length: int      # number of actions; 0 if planning failed
    batch_calls: int      # total env.batch_evaluate() calls (incl. verification)
    total_pairs: int      # total (state, action) pairs evaluated
    plan_time_s: float
    final_reward: float | None = None
    reward_components: dict | None = None
    video_path: str | None = None


@dataclass
class AggregateStats:
    n_runs: int
    n_success: int
    success_rate: float
    # Plan length — computed over successful runs only
    plan_length_mean: float
    plan_length_std: float
    plan_length_min: float
    plan_length_max: float
    # Sim call stats — computed over all runs
    batch_calls_mean: float
    batch_calls_std: float
    batch_calls_min: float
    batch_calls_max: float
    total_pairs_mean: float
    total_pairs_std: float
    total_pairs_min: float
    total_pairs_max: float
    # Timing — all runs
    plan_time_mean: float
    plan_time_std: float
    plan_time_min: float
    plan_time_max: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_push_distance(plan: list[dict], env) -> float:
    """Sum of pusher stroke lengths (metres) across all actions in the plan."""
    total = 0.0
    for action in plan:
        _, start, end = env._action_to_stroke(action)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _compute_object_metrics(initial_state: dict, final_state: dict, env,
                             moved_threshold: float = 0.01):
    """Return (total_displacement, n_moved, n_out_of_bin).

    Displacement is summed over all obstacles (XY plane only).
    `moved_threshold` is in metres (default 1 cm).
    """
    n = env.n_obstacles
    init_pos  = initial_state['obstacle_pos']   # (N, ≥2)
    final_pos = final_state['obstacle_pos']

    total_disp = 0.0
    n_moved    = 0
    n_out      = 0
    for i in range(n):
        dx = float(final_pos[i, 0]) - float(init_pos[i, 0])
        dy = float(final_pos[i, 1]) - float(init_pos[i, 1])
        d  = (dx * dx + dy * dy) ** 0.5
        total_disp += d
        if d > moved_threshold:
            n_moved += 1
        fx, fy = float(final_pos[i, 0]), float(final_pos[i, 1])
        if fx < 0 or fx > env.bin_w or fy < 0 or fy > env.bin_d:
            n_out += 1
    return total_disp, n_moved, n_out


def _render_state_image(state: dict, env) -> 'wandb.Image':
    from visualization import render_scenario
    fig = render_scenario(state, env.bin_w, env.bin_d, env._OBJ_SIZE, env.wall_thickness)
    return wandb.Image(fig)


def _get_final_state(env, plan, initial_state):
    """Execute plan sequentially via batch_evaluate.

    Returns (final_state, force_traces) where force_traces is a list of
    per-action force-magnitude lists (IsaacLab only; empty list otherwise).
    """
    state = initial_state
    force_traces = []
    for action in plan:
        (state, _, _), = env.batch_evaluate([(state, action)])
        if hasattr(env, 'force_trace'):
            force_traces.append(list(env.force_trace))
    return state, force_traces




# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_once(env, cfg: DictConfig, run_idx: int, seed: int, initial_state=None):
    """Run one planning attempt. Returns (result, plan, initial_state, planner)."""
    print(f'\n[Run {run_idx + 1}/{cfg.n_runs}] seed={seed}')

    if initial_state is None:
        initial_state = env.reset(seed=seed)
    env.reset_sim_counters()

    from planner import _PlannerBase
    planner = _PlannerBase.from_cfg(env, cfg, seed)

    t0 = time.time()
    plan = planner.plan(initial_state, verbose=cfg.debug)
    plan_time = time.time() - t0

    batch_calls = env.batch_calls
    total_pairs = env.total_pairs
    success = plan is not None and len(plan) > 0

    result = RunResult(
        run_idx=run_idx,
        seed=seed,
        success=success,
        plan_length=len(plan) if success else 0,
        batch_calls=batch_calls,
        total_pairs=total_pairs,
        plan_time_s=plan_time,
    )

    status = 'SUCCESS' if success else 'FAILED'
    print(f'  {status} | plan_len={result.plan_length} | '
          f'batch_calls={batch_calls} | total_pairs={total_pairs} | '
          f'time={plan_time:.1f}s')

    return result, (plan if success else None), (initial_state if success else None), planner


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _aggregate(results: list[RunResult]) -> AggregateStats:
    def _stats(vals: list[float]) -> tuple[float, float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    successes = [r for r in results if r.success]
    n = len(results)

    pl_m, pl_s, pl_lo, pl_hi = _stats([r.plan_length for r in successes])
    bc_m, bc_s, bc_lo, bc_hi = _stats([r.batch_calls for r in results])
    tp_m, tp_s, tp_lo, tp_hi = _stats([r.total_pairs for r in results])
    pt_m, pt_s, pt_lo, pt_hi = _stats([r.plan_time_s for r in results])

    return AggregateStats(
        n_runs=n,
        n_success=len(successes),
        success_rate=len(successes) / n if n else 0.0,
        plan_length_mean=pl_m, plan_length_std=pl_s,
        plan_length_min=pl_lo, plan_length_max=pl_hi,
        batch_calls_mean=bc_m, batch_calls_std=bc_s,
        batch_calls_min=bc_lo, batch_calls_max=bc_hi,
        total_pairs_mean=tp_m, total_pairs_std=tp_s,
        total_pairs_min=tp_lo, total_pairs_max=tp_hi,
        plan_time_mean=pt_m, plan_time_std=pt_s,
        plan_time_min=pt_lo, plan_time_max=pt_hi,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    # IsaacLab must be launched headless in benchmark mode (no GUI).
    # enable_cameras loads omni.replicator so record_replay() can capture frames.
    # Both env vars must be set before _build_env() imports the module.
    if cfg.simulator.name == 'isaaclab':
        if cfg.viewer == 'headless':
            os.environ['ISAACLAB_HEADLESS'] = '1'
        if cfg.record_video:
            os.environ['ISAACLAB_ENABLE_CAMERAS'] = '1'

    viewer_mode = cfg.viewer
    print(f'Building environment ({cfg.simulator.name}): '
          f'{cfg.n_obstacles} obstacle(s), parallel_envs={cfg.parallel_envs}')
    from simulators import build_env
    env = build_env(cfg, n_envs=cfg.parallel_envs, viewer_mode=viewer_mode)

    # Init wandb after the simulator — Isaac Sim's AppLauncher does process-level
    # setup (signal handlers, CUDA contexts) that can corrupt wandb's upload thread
    # if wandb is initialised first.
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity if cfg.wandb_entity else None,
        name=cfg.wandb_run_name if cfg.wandb_run_name else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    # Wait until the run object is active
    while wandb.run is None:
        time.sleep(1)


    record_video = cfg.record_video and cfg.simulator.name == 'isaaclab'
    if record_video:
        video_dir = os.path.join(cfg.video_dir, wandb.run.name)
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = cfg.video_dir

    results: list[RunResult] = []
    csv_rows: list[dict] = []
    best_reward: float | None = None

    for i in range(cfg.n_runs):
        seed = cfg.base_seed + i
        initial_state = env.reset(seed=seed)
        wandb.log({'run/initial_state': _render_state_image(initial_state, env)}, step=i)

        result, plan, initial_state, planner = _run_once(env, cfg, i, seed, initial_state=initial_state)
        verify_successes = 0
        verify_rate = 0.0
        verify_std = None
        verify_flags: list = []
        final_state = None

        if plan is not None:
            verify_successes, _, verify_rate, verify_passed, verify_flags = planner.verify(plan, initial_state)
            verify_std = float(np.std(verify_flags)) if verify_flags else None
            print(f'  Benchmark verify: {verify_successes}/{env.n_envs} '
                  f'({verify_rate:.0%}) — {"PASS" if verify_passed else "FAIL"}')

            if cfg.get('solutions_dir', None):
                from main import save_solution
                os.makedirs(cfg.solutions_dir, exist_ok=True)
                sol_path = os.path.join(cfg.solutions_dir,
                                        f'run_{i:03d}_seed_{seed}.json')
                save_solution(sol_path, plan, initial_state, cfg, env)
                print(f'  Solution saved → {sol_path}')

            if record_video:
                suffix = 'pass' if verify_passed else 'verify_fail'
                video_path = os.path.join(video_dir, f'run_{i:03d}_seed_{seed}_{suffix}.mp4')
                print(f'  Recording video → {video_path}')
                try:
                    with env.push_steps_ctx(cfg.get('verify_push_steps', None)):
                        result.video_path = env.record_replay(plan, initial_state, video_path=video_path)
                    if result.video_path is not None:
                        final_state = env.get_state(0)
                    else:
                        print('  [benchmark] record_replay returned None — no frames captured.')
                except Exception as e:
                    print(f'  [benchmark] record_replay failed: {e}')

            final_state_replay, force_traces = _get_final_state(env, plan, initial_state)
            if final_state is None:
                final_state = final_state_replay

            result.reward_components = env.compute_reward_components(final_state)
            result.final_reward = sum(result.reward_components.values())
            replay_success = env.is_goal(final_state)
            if not replay_success:
                print(f'  [benchmark] WARNING: re-execution did not reach goal '
                      f'(final_reward={result.final_reward:.3f})')
        else:
            replay_success = None
            force_traces = []

        log = {
            'run/success':           int(result.success),
            'run/verify_successes':  verify_successes,
            'run/verify_rate':       verify_rate,
            'run/plan_length':       result.plan_length,
            'run/batch_calls':       result.batch_calls,
            'run/total_pairs':       result.total_pairs,
            'run/plan_time_s':       result.plan_time_s,
            'run/seed':              seed,
        }
        if result.final_reward is not None:
            log['run/final_reward'] = result.final_reward
            if best_reward is None or result.final_reward > best_reward:
                best_reward = result.final_reward
        if replay_success is not None:
            log['run/replay_success'] = int(replay_success)
        wandb.log(log, step=i)

        if results and any(r.final_reward is not None for r in results) or result.final_reward is not None:
            import matplotlib.pyplot as plt
            all_results = results + [result]
            xs = [r.run_idx for r in all_results if r.final_reward is not None]
            ys = [r.final_reward for r in all_results if r.final_reward is not None]
            best_so_far = [max(ys[:k+1]) for k in range(len(ys))]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(xs, ys, 'o--', color='tab:blue', alpha=0.5, label='reward')
            ax.plot(xs, best_so_far, '-', color='tab:orange', linewidth=2, label='best so far')
            ax.set_xlabel('run')
            ax.set_ylabel('final reward')
            ax.set_title('Reward over runs')
            ax.legend()
            fig.tight_layout()
            wandb.log({'run/reward_progress': wandb.Image(fig)}, step=i)
            plt.close(fig)

        if result.reward_components is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            keys = list(result.reward_components.keys())
            vals = list(result.reward_components.values())
            colors = ['tab:green' if v >= 0 else 'tab:red' for v in vals]
            ax.bar(keys, vals, color=colors)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(f'Reward components — run {i} (seed {seed})')
            ax.set_ylabel('value')
            fig.tight_layout()
            wandb.log({'run/reward_breakdown': wandb.Image(fig)}, step=i)
            plt.close(fig)

        if force_traces:
            import matplotlib.pyplot as plt
            flat_forces = [f for trace in force_traces for f in trace]
            boundaries = [0]
            for trace in force_traces:
                boundaries.append(boundaries[-1] + len(trace))
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(flat_forces, color='tab:blue', linewidth=1.0)
            for b in boundaries[1:-1]:
                ax.axvline(b, color='gray', linewidth=0.7, linestyle='--')
            threshold = getattr(env, 'force_threshold', None)
            if threshold and threshold > 0:
                ax.axhline(threshold, color='tab:red', linewidth=1.0,
                           linestyle='--', label=f'threshold ({threshold:.0f} N)')
                ax.legend(fontsize=8)
            label_y = max(flat_forces) * 0.92 if flat_forces else 1.0
            for b, action in zip(boundaries, plan):
                ax.text(b + 0.3, label_y, action['action_type'], fontsize=6, color='gray')
            ax.set_xlabel('physics step')
            ax.set_ylabel('contact force (N)')
            ax.set_title(f'Pusher contact force — run {i} (seed {seed})')
            fig.tight_layout()
            wandb.log({'run/contact_force': wandb.Image(fig)}, step=i)
            plt.close(fig)

        if result.video_path:
            try:
                wandb.log({'run/video': wandb.Video(result.video_path, fps=30, format='mp4')}, step=i)
                print(f'  Logged video to wandb: {result.video_path}')
            except Exception as e:
                print(f'  [benchmark] wandb video log failed: {e}')
        # ---------------------------------------------------------------
        # SQLite results database
        # ---------------------------------------------------------------
        if cfg.get('db_path', None):
            import results_db
            from alphazero.grid import build_grid_spec
            from planner import MCTSPusher

            spec = build_grid_spec(env)

            # failure reason
            if not result.success:
                failure_reason = 'planning_failed'
            elif plan is not None and not replay_success:
                failure_reason = 'replay_failed'
            else:
                failure_reason = None

            # plan cost metrics (only if we have a plan and a final state)
            push_dist   = None
            obj_disp    = None
            n_moved     = None
            n_out       = None
            if plan and final_state is not None:
                push_dist = _compute_push_distance(plan, env)
                obj_disp, n_moved, n_out = _compute_object_metrics(
                    initial_state, final_state, env)

            # per-planner counters
            node_exp = getattr(planner, 'node_expansions', None)
            net_fwd  = getattr(planner, 'network_forward_passes', None)
            budget   = getattr(cfg.planner, 'n_simulations', None)

            run_data = {
                'method':                   cfg.planner.name,
                'n_objects':                cfg.n_obstacles,
                'stackable':                int(cfg.get('stackable', False)),
                'bin_size':                 float(env.bin_w),
                'grid_gx':                  spec.Gx,
                'grid_gy':                  spec.Gy,
                'grid_z':                   spec.Z,
                'scene_id':                 seed,
                'planning_seed':            seed,
                'planning_success':         int(result.success),
                'execution_success':        int(replay_success) if replay_success is not None else None,
                'failure_reason':           failure_reason,
                'wall_clock_s':             result.plan_time_s,
                'node_expansions':          node_exp,
                'simulator_calls':          result.total_pairs,
                'network_forward_passes':   net_fwd,
                'budget_cap':               budget,
                'n_actions':                result.plan_length,
                'plan_horizon_pre_filter':  result.plan_length,
                'plan_horizon_post_filter': result.plan_length,
                'total_push_distance':      push_dist,
                'total_object_displacement': obj_disp,
                'n_objects_moved':          n_moved,
                'objects_displaced_from_bin': n_out,
                'final_reward':             result.final_reward,
                'verify_successes':         verify_successes if plan is not None else None,
                'verify_rate':              verify_rate if plan is not None else None,
                'wandb_run_id':             wandb.run.id if wandb.run else None,
            }
            row_id = results_db.log_run(cfg.db_path, run_data)
            print(f'  [db] Run logged → {cfg.db_path} (row {row_id})')

        results.append(result)

        if cfg.get('csv_path', None):
            _push_dist = _obj_disp = _n_moved = _n_out = None
            if plan is not None and final_state is not None:
                _push_dist = _compute_push_distance(plan, env)
                _obj_disp, _n_moved, _n_out = _compute_object_metrics(
                    initial_state, final_state, env)
            csv_rows.append({
                'seed':                       seed,
                'planner':                    cfg.planner.name,
                'wandb_run_name':             wandb.run.name if wandb.run else None,
                'success':                    int(result.success),
                'plan_length':                result.plan_length,
                'plan_time_s':                result.plan_time_s,
                'batch_calls':                result.batch_calls,
                'total_pairs':                result.total_pairs,
                'simulator_calls':            result.total_pairs,
                'network_forward_passes':     getattr(planner, 'network_forward_passes', None),
                'budget_cap':                 getattr(cfg.planner, 'n_simulations', None),
                'plan_horizon_pre_filter':    result.plan_length,
                'plan_horizon_post_filter':   result.plan_length,
                'total_push_distance':        _push_dist,
                'total_object_displacement':  _obj_disp,
                'n_objects_moved':            _n_moved,
                'objects_displaced_from_bin': _n_out,
                'verify_successes':           verify_successes if plan is not None else None,
                'verify_rate':                verify_rate if plan is not None else None,
                'verify_std':                 verify_std,
            })

    # Aggregate stats
    agg = _aggregate(results)

    print('\n' + '=' * 60)
    print(f'Results: {agg.n_success}/{agg.n_runs} successful '
          f'({agg.success_rate * 100:.1f}%)')
    print(f'  Plan length (success): {agg.plan_length_mean:.1f} ± {agg.plan_length_std:.1f} '
          f'[{agg.plan_length_min:.0f}, {agg.plan_length_max:.0f}]')
    print(f'  Batch calls (all):     {agg.batch_calls_mean:.1f} ± {agg.batch_calls_std:.1f} '
          f'[{agg.batch_calls_min:.0f}, {agg.batch_calls_max:.0f}]')
    print(f'  Total pairs (all):     {agg.total_pairs_mean:.1f} ± {agg.total_pairs_std:.1f} '
          f'[{agg.total_pairs_min:.0f}, {agg.total_pairs_max:.0f}]')
    print(f'  Plan time (all):       {agg.plan_time_mean:.1f} ± {agg.plan_time_std:.1f}s '
          f'[{agg.plan_time_min:.1f}, {agg.plan_time_max:.1f}]')
    print('=' * 60)

    for key, val in asdict(agg).items():
        wandb.run.summary[key] = val

    if csv_rows and cfg.get('csv_path', None):
        csv_path = cfg.csv_path
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()),
                                    extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)
        print(f'[benchmark] CSV written → {csv_path} ({len(csv_rows)} rows, '
              f'{"new file" if write_header else "appended"})')

    wandb.finish()


if __name__ == '__main__':
    main()
