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
    show_viewer = viewer_mode == 'always'
    print(f'Building environment ({cfg.simulator.name}): '
          f'{cfg.n_obstacles} obstacle(s), parallel_envs={cfg.parallel_envs}')
    from simulators import build_env
    env = build_env(cfg, n_envs=cfg.parallel_envs, show_viewer=show_viewer,
                    viewer_mode=viewer_mode)

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
        video_dir = os.path.join(cfg.video_dir, wandb.run.id)
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = cfg.video_dir

    results: list[RunResult] = []
    best_reward: float | None = None

    for i in range(cfg.n_runs):
        seed = cfg.base_seed + i
        initial_state = env.reset(seed=seed)
        wandb.log({'run/initial_state': _render_state_image(initial_state, env)}, step=i)

        result, plan, initial_state, planner = _run_once(env, cfg, i, seed, initial_state=initial_state)
        verify_successes = 0
        verify_rate = 0.0
        final_state = None

        if plan is not None:
            verify_successes, _, verify_rate, verify_passed = planner.verify(plan, initial_state)
            print(f'  Benchmark verify: {verify_successes}/{env.n_envs} '
                  f'({verify_rate:.0%}) — {"PASS" if verify_passed else "FAIL"}')

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
        results.append(result)

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

    wandb.finish()


if __name__ == '__main__':
    main()
