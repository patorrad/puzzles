"""
pipeline.py  –  End-to-end scenario → puzzle planning → IsaacLab MPC pipeline.

Stages
------
1. Generate N random scenarios and save as YAMLs in conf/scenario/generated/.
2. For each scenario, run the puzzle planner (benchmark.py-style, direct API)
   and save successful solutions as JSON.
3. Tear down the IsaacLab AppLauncher (releases GPU context).
4. For each successful solution, run isaaclabmpc planner.py + world.py as
   subprocesses, collect rich telemetry, and log everything to WandB.

Usage
-----
# Default: 10 scenarios, 3 obstacles, IsaacLab + MCTS
python pipeline.py --config-name=pipeline

# Override scenario count / planner
python pipeline.py --config-name=pipeline n_scenarios=20 n_obstacles=5

# Different planner
python pipeline.py --config-name=pipeline planner=rrt

# Custom WandB project
python pipeline.py --config-name=pipeline wandb_project=my-project wandb_entity=myname
"""

import json
import multiprocessing
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PuzzleResult:
    scenario_idx: int
    scenario_name: str
    seed: int
    success: bool
    plan_length: int
    batch_calls: int
    total_pairs: int
    plan_time_s: float
    verify_rate: float = 0.0
    final_reward: float | None = None
    reward_components: dict | None = None
    replay_success: bool | None = None


@dataclass
class MpcResult:
    scenario_name: str
    success: bool
    steps_completed: int
    total_steps: int
    elapsed_time_s: float
    ee_trajectory: list | None = None
    block_positions_final: list | None = None
    step_completion_events: list | None = None
    mppi_cost_history: list | None = None


# ---------------------------------------------------------------------------
# Stage 2: Puzzle planning — child process worker
# ---------------------------------------------------------------------------

def _puzzle_worker(cfg: DictConfig, scenarios: list, out_dir: Path, q) -> None:
    """Runs puzzle planning for all scenarios. Executed in a spawned child process.

    The parent kills this process after reading results from q, so we intentionally
    skip IsaacLab teardown — that would hang or corrupt GPU state anyway.
    """
    import os
    record_video = cfg.get('record_video', False) and cfg.simulator.name == 'isaaclab'
    planner_viewer_mode = cfg.get("viewer", "headless")
    if cfg.simulator.name == 'isaaclab':
        if planner_viewer_mode == 'headless':
            os.environ['ISAACLAB_HEADLESS'] = '1'
        if record_video:
            os.environ['ISAACLAB_ENABLE_CAMERAS'] = '1'

    from simulators import build_env
    from main import save_solution

    out_dir = Path(out_dir)
    if record_video:
        (out_dir / 'videos').mkdir(parents=True, exist_ok=True)
    env = build_env(cfg, n_envs=cfg.parallel_envs,
                    viewer_mode=planner_viewer_mode)

    puzzle_results: list[PuzzleResult] = []
    successful_names: list[str] = []
    force_traces_map: dict[str, list] = {}
    plans_map: dict[str, list] = {}
    videos_map: dict[str, str] = {}

    for i, (scenario_name, initial_state) in enumerate(scenarios):
        seed = (cfg.seed + i) if cfg.seed is not None else i
        result, plan = _plan_scenario(env, cfg, i, scenario_name, initial_state, seed)

        if plan is not None:
            final_state, force_traces = _get_final_state(env, plan, initial_state)
            result.reward_components = env.compute_reward_components(final_state)
            result.final_reward = sum(result.reward_components.values())
            result.replay_success = env.is_goal(final_state)

            solution_path = out_dir / 'solutions' / f'{scenario_name}.json'
            save_solution(str(solution_path), plan, initial_state, cfg, env)

            if record_video:
                video_path = str(out_dir / 'videos' / f'{scenario_name}.mp4')
                try:
                    recorded = env.record_replay(plan, initial_state, video_path)
                    if recorded is not None:
                        videos_map[scenario_name] = recorded
                    else:
                        print(f'  [pipeline] record_replay returned None for {scenario_name}')
                except Exception as e:
                    print(f'  [pipeline] record_replay failed for {scenario_name}: {e}')

            successful_names.append(scenario_name)
            force_traces_map[scenario_name] = force_traces
            plans_map[scenario_name] = plan

        puzzle_results.append(result)

    # Convert torch tensors in plans to plain lists before crossing the process
    # boundary — torch's shared-memory fd mechanism doesn't work across spawn.
    def _detach_plan(plan):
        return [
            {**a, 'push_pos': a['push_pos'].tolist()}
            for a in plan
        ]

    q.put({
        'puzzle_results': puzzle_results,
        'successful': successful_names,
        'force_traces': force_traces_map,
        'plans': {k: _detach_plan(v) for k, v in plans_map.items()},
        'videos': videos_map,
    })


# ---------------------------------------------------------------------------
# Stage 1: Scenario generation
# ---------------------------------------------------------------------------

def _generate_scenarios(cfg: DictConfig, outdir: Path) -> list[tuple[str, dict]]:
    """Generate N scenario YAMLs + PNGs into outdir. Returns list of (scenario_name, initial_state)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from simulators.placement import random_initial_state
    from generate_scenarios import _state_to_dict, _InlineDumper
    from visualization import render_scenario

    outdir.mkdir(parents=True, exist_ok=True)

    bin_size = cfg.get('bin_size', 0.3)
    obj_size = cfg.get('obj_size', 0.05)
    wall_thickness = cfg.get('wall_thickness', 0.02)

    gen_args = SimpleNamespace(
        n_obstacles=cfg.n_obstacles,
        bin_size=bin_size,
        wall_thickness=wall_thickness,
        friction=cfg.get('friction', 1.2),
    )

    scenarios = []
    for i in range(cfg.n_scenarios):
        seed = cfg.seed + i if cfg.seed is not None else None
        state = random_initial_state(
            n_obstacles=cfg.n_obstacles,
            obj_size=obj_size,
            stackable=cfg.get('stackable', False),
            difficult_spawn=cfg.get('difficult_spawn', False),
            seed=seed,
            bin_w=bin_size,
            bin_d=bin_size,
            n_z_levels=cfg.get('n_z_levels', 1),
            target_z_level=cfg.get('target_z_level', None),
        )
        name = f"scenario_{i:04d}"
        yaml_path = outdir / f"{name}.yaml"
        data = _state_to_dict(state, gen_args)
        with open(yaml_path, 'w') as f:
            f.write("# Generated by pipeline.py\n")
            yaml.dump(data, f, Dumper=_InlineDumper, default_flow_style=False, sort_keys=False)

        fig = render_scenario(
            state, bin_size, bin_size, obj_size, wall_thickness,
            title=f"{name} (seed={seed})",
        )
        fig.savefig(outdir / f"{name}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)

        scenarios.append((name, state))
        print(f"  [gen] {yaml_path}")

    print(f"\nGenerated {len(scenarios)} scenarios in {outdir}/")
    return scenarios


# ---------------------------------------------------------------------------
# Stage 2 helpers (benchmark.py-style)
# ---------------------------------------------------------------------------

def _render_state_image(state: dict, bin_size: float, obj_size: float,
                        wall_thickness: float) -> 'wandb.Image':
    from visualization import render_scenario
    fig = render_scenario(state, bin_size, bin_size, obj_size, wall_thickness)
    return wandb.Image(fig)


def _get_final_state(env, plan, initial_state):
    """Execute plan sequentially. Returns (final_state, force_traces)."""
    state = initial_state
    force_traces = []
    for action in plan:
        (state, _, _), = env.batch_evaluate([(state, action)])
        if hasattr(env, 'force_trace'):
            force_traces.append(list(env.force_trace))
    return state, force_traces


def _plan_scenario(env, cfg: DictConfig, scenario_idx: int, scenario_name: str,
                   initial_state: dict, seed: int) -> tuple[PuzzleResult, list | None]:
    """Run one planning attempt. Returns (PuzzleResult, plan_or_None)."""
    print(f'\n[Scenario {scenario_idx + 1}/{cfg.n_scenarios}] {scenario_name} seed={seed}')

    env.reset_sim_counters()
    from planner import _PlannerBase
    planner = _PlannerBase.from_cfg(env, cfg, seed)

    t0 = time.time()
    plan = planner.plan(initial_state, verbose=cfg.debug)
    plan_time = time.time() - t0

    success = plan is not None and len(plan) > 0
    result = PuzzleResult(
        scenario_idx=scenario_idx,
        scenario_name=scenario_name,
        seed=seed,
        success=success,
        plan_length=len(plan) if success else 0,
        batch_calls=env.batch_calls,
        total_pairs=env.total_pairs,
        plan_time_s=plan_time,
    )

    status = 'SUCCESS' if success else 'FAILED'
    print(f'  {status} | plan_len={result.plan_length} | '
          f'batch_calls={result.batch_calls} | time={plan_time:.1f}s')

    if not success:
        return result, None

    verify_successes, _, verify_rate, verify_passed = planner.verify(plan, initial_state)
    result.verify_rate = verify_rate
    print(f'  Verify: {verify_successes}/{env.n_envs} ({verify_rate:.0%}) — '
          f'{"PASS" if verify_passed else "FAIL"}')

    return result, plan


def _log_puzzle_wandb(result: PuzzleResult, initial_state, bin_size: float, obj_size: float,
                      wall_thickness: float, force_threshold: float | None,
                      plan, force_traces, all_results: list[PuzzleResult], step: int,
                      video_path: str | None = None):
    """Log per-scenario puzzle metrics to WandB."""
    import matplotlib.pyplot as plt

    log = {
        'puzzles/success':       int(result.success),
        'puzzles/verify_rate':   result.verify_rate,
        'puzzles/plan_length':   result.plan_length,
        'puzzles/batch_calls':   result.batch_calls,
        'puzzles/total_pairs':   result.total_pairs,
        'puzzles/plan_time_s':   result.plan_time_s,
        'puzzles/seed':          result.seed,
        'puzzles/initial_state': _render_state_image(initial_state, bin_size, obj_size, wall_thickness),
    }
    if result.final_reward is not None:
        log['puzzles/final_reward'] = result.final_reward
    if result.replay_success is not None:
        log['puzzles/replay_success'] = int(result.replay_success)
    wandb.log(log, step=step)

    # Reward progress chart
    rewards = [(r.scenario_idx, r.final_reward) for r in all_results if r.final_reward is not None]
    if rewards:
        xs, ys = zip(*rewards)
        best = [max(ys[:k+1]) for k in range(len(ys))]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(xs, ys, 'o--', color='tab:blue', alpha=0.5, label='reward')
        ax.plot(xs, best, '-', color='tab:orange', linewidth=2, label='best so far')
        ax.set_xlabel('scenario')
        ax.set_ylabel('final reward')
        ax.set_title('Puzzle reward over scenarios')
        ax.legend()
        fig.tight_layout()
        wandb.log({'puzzles/reward_progress': wandb.Image(fig)}, step=step)
        plt.close(fig)

    if result.reward_components is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        keys = list(result.reward_components.keys())
        vals = list(result.reward_components.values())
        ax.bar(keys, vals, color=['tab:green' if v >= 0 else 'tab:red' for v in vals])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'Reward components — {result.scenario_name}')
        fig.tight_layout()
        wandb.log({'puzzles/reward_breakdown': wandb.Image(fig)}, step=step)
        plt.close(fig)

    if force_traces and plan:
        fig, ax = plt.subplots(figsize=(8, 3))
        flat = [f for trace in force_traces for f in trace]
        boundaries = [0]
        for trace in force_traces:
            boundaries.append(boundaries[-1] + len(trace))
        ax.plot(flat, color='tab:blue', linewidth=1.0)
        for b in boundaries[1:-1]:
            ax.axvline(b, color='gray', linewidth=0.7, linestyle='--')
        if force_threshold and force_threshold > 0:
            ax.axhline(force_threshold, color='tab:red', linewidth=1.0, linestyle='--',
                       label=f'threshold ({force_threshold:.0f} N)')
            ax.legend(fontsize=8)
        label_y = max(flat) * 0.92 if flat else 1.0
        for b, action in zip(boundaries, plan):
            ax.text(b + 0.3, label_y, action['action_type'], fontsize=6, color='gray')
        ax.set_xlabel('physics step')
        ax.set_ylabel('contact force (N)')
        ax.set_title(f'Contact force — {result.scenario_name}')
        fig.tight_layout()
        wandb.log({'puzzles/contact_force': wandb.Image(fig)}, step=step)
        plt.close(fig)

    if video_path is not None:
        try:
            wandb.log({'puzzles/replay_video': wandb.Video(video_path, fps=30, format='mp4')}, step=step)
        except Exception as e:
            print(f'  [pipeline] wandb video log failed for {result.scenario_name}: {e}')


# ---------------------------------------------------------------------------
# Stage 3: IsaacLab MPC subprocess runner
# ---------------------------------------------------------------------------

def _stream_output(src, log_path: Path, prefix: str) -> threading.Thread:
    """Daemon thread: copy lines from src to log_path and stdout with prefix."""
    log_file = open(log_path, 'w')

    def _run():
        try:
            for line in src:
                log_file.write(line)
                log_file.flush()
                print(f"{prefix}{line}", end='', flush=True)
        finally:
            log_file.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _check_port_free(port: int) -> None:
    result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
    if result.stdout.strip():
        pids = result.stdout.strip().replace('\n', ' ')
        sys.exit(f'[MPC] ERROR: port {port} is already in use (pids: {pids}). '
                 f'Kill the stale planner with: kill {pids}')


def _wait_for_planner_server(addr: str = "tcp://localhost:4242", max_wait_s: int = 120) -> None:
    """Poll until the zerorpc planner server responds to test()."""
    import zerorpc as _zerorpc
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            c = _zerorpc.Client(timeout=5, heartbeat=None)
            c.connect(addr)
            c.test("pipeline-ping")
            c.close()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Planner server at {addr} did not become ready in {max_wait_s}s")


def _run_isaaclabmpc(scenario_name: str, scenario_yaml: Path, solution_json: Path,
                     out_dir: Path, cfg: DictConfig) -> MpcResult:
    """Launch planner.py + world.py subprocesses. Returns MpcResult."""
    ilab_dir = Path(cfg.isaaclabmpc_dir)
    # Resolve all paths to absolute before passing to subprocesses, which run
    # with a different cwd (ilab_dir.parent.parent).
    solution_json  = solution_json.resolve()
    result_json    = (out_dir / 'isaaclabmpc_results' / f'{scenario_name}.json').resolve()
    telemetry_json = (out_dir / 'telemetry' / f'{scenario_name}_planner.json').resolve()
    planner_log = out_dir / 'logs' / f'{scenario_name}_planner.txt'
    world_log   = out_dir / 'logs' / f'{scenario_name}_world.txt'

    python = sys.executable
    show_planner_viewer = cfg.get('show_mpc_planner_viewer', False)
    show_world_viewer   = cfg.get('show_mpc_world_viewer', False)

    planner_cmd = [
        python,
        str(ilab_dir / 'planner.py'),
        '--scenario', str(scenario_yaml),
        '--solution_path', str(solution_json),
        '--telemetry_path', str(telemetry_json),
    ]
    if not show_planner_viewer:
        planner_cmd.append('--headless')
    world_cmd = [
        python,
        str(ilab_dir / 'world.py'),
        '--scenario', str(scenario_yaml),
        '--n_steps', str(cfg.isaaclabmpc_n_steps),
        '--output_path', str(result_json),
    ]
    if not show_world_viewer:
        world_cmd.append('--headless')

    print(f'\n[MPC] {scenario_name}: cwd={ilab_dir.parent.parent}')
    print(f'[MPC] {scenario_name}: scenario_yaml exists={scenario_yaml.exists()} path={scenario_yaml}')
    print(f'[MPC] {scenario_name}: solution_json exists={solution_json.exists()} path={solution_json}')
    print(f'[MPC] {scenario_name}: planner cmd: {" ".join(planner_cmd)}')

    _check_port_free(4242)
    print(f'[MPC] {scenario_name}: launching planner …')
    planner_proc = subprocess.Popen(
        planner_cmd,
        cwd=str(ilab_dir.parent.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(planner_proc.stdout, planner_log, f'[{scenario_name}/planner] ')

    # Give the zerorpc server time to bind before world connects
    time.sleep(3)

    print(f'[MPC] {scenario_name}: planner alive={planner_proc.poll() is None} '
          f'(returncode={planner_proc.poll()})')
    print(f'[MPC] {scenario_name}: world cmd: {" ".join(world_cmd)}')
    print(f'[MPC] {scenario_name}: launching world …')
    world_proc = subprocess.Popen(
        world_cmd,
        cwd=str(ilab_dir.parent.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(world_proc.stdout, world_log, f'[{scenario_name}/world] ')

    timed_out = False
    try:
        world_proc.wait(timeout=cfg.isaaclabmpc_timeout_s)
    except subprocess.TimeoutExpired:
        print(f'[MPC] {scenario_name}: TIMEOUT after {cfg.isaaclabmpc_timeout_s}s')
        world_proc.kill()
        timed_out = True
    finally:
        planner_proc.send_signal(signal.SIGTERM)
        try:
            planner_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            planner_proc.kill()

    print(f'[MPC] {scenario_name}: world exitcode={world_proc.returncode} '
          f'planner exitcode={planner_proc.returncode}')
    print(f'[MPC] {scenario_name}: result_json exists={result_json.exists()}')

    def _tail(path: Path, n: int = 40) -> str:
        if not path.exists():
            return '  <file not found>'
        lines = path.read_text().splitlines()
        return '\n'.join(f'  {l}' for l in lines[-n:]) or '  <empty>'

    print(f'[MPC] {scenario_name}: planner log (last 20 lines):\n{_tail(planner_log)}')
    print(f'[MPC] {scenario_name}: world log (last 20 lines):\n{_tail(world_log)}')

    result_data = {}
    if result_json.exists():
        with open(result_json) as f:
            result_data = json.load(f)

    telemetry_data = {}
    if telemetry_json.exists():
        with open(telemetry_json) as f:
            telemetry_data = json.load(f)

    mpc_result = MpcResult(
        scenario_name=scenario_name,
        success=result_data.get('success', False) and not timed_out,
        steps_completed=result_data.get('steps_completed', 0),
        total_steps=result_data.get('total_steps', 0),
        elapsed_time_s=result_data.get('elapsed_time_s', 0.0),
        ee_trajectory=result_data.get('ee_trajectory'),
        block_positions_final=result_data.get('block_positions_final'),
        step_completion_events=result_data.get('step_completion_events'),
        mppi_cost_history=telemetry_data.get('mppi_cost_history'),
    )

    status = 'SUCCESS' if mpc_result.success else ('TIMEOUT' if timed_out else 'FAILED')
    print(f'[MPC] {scenario_name}: {status} — '
          f'{mpc_result.steps_completed}/{mpc_result.total_steps} steps '
          f'in {mpc_result.elapsed_time_s:.1f}s')
    return mpc_result


def _run_real_robot_scenario(scenario_name: str, out_dir: Path,
                              cfg: DictConfig) -> tuple['MpcResult', dict | None]:
    """Real-robot mode: launch bridge_server.py as a subprocess so the bridge
    node can connect, read object state via zerorpc client, run puzzle planning,
    inject solution, then leave the server running."""
    import io
    import torch

    def _b2t(b: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(b))

    addr     = cfg.get('bridge_server_address', 'tcp://localhost:4242')
    bind_addr = addr.replace('localhost', '0.0.0.0').replace('127.0.0.1', '0.0.0.0')
    server_log = out_dir / 'logs' / f'{scenario_name}_bridge_server.txt'

    bridge_server_script = Path(__file__).parent / 'bridge_server.py'
    server_cmd = [sys.executable, str(bridge_server_script), '--address', bind_addr]

    _check_port_free(4242)
    print(f'\n[real-robot] {scenario_name}: launching bridge_server.py on {bind_addr} …')
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(server_proc.stdout, server_log, f'[{scenario_name}/bridge_server] ')

    # Wait for the server to be reachable
    try:
        _wait_for_planner_server(addr, max_wait_s=30)
    except TimeoutError as e:
        print(f'[real-robot] ERROR: {e}')
        server_proc.kill()
        return MpcResult(scenario_name=scenario_name, success=False,
                         steps_completed=0, total_steps=0, elapsed_time_s=0.0), None

    import zerorpc as _zerorpc
    client = _zerorpc.Client(timeout=10, heartbeat=None)
    client.connect(addr)

    # Wait until the bridge node has pushed at least one set of object poses
    wait_timeout = cfg.get('bridge_state_timeout_s', 30)
    deadline = time.time() + wait_timeout
    raw_poses = None
    while time.time() < deadline:
        try:
            poses_bytes = client.get_sim_object_poses()
            t = _b2t(poses_bytes)
            if t.numel() > 0:
                raw_poses = t
                break
        except Exception:
            pass
        time.sleep(0.5)

    if raw_poses is None:
        print('[real-robot] ERROR: no object poses received from bridge node within timeout')
        server_proc.kill()
        return MpcResult(scenario_name=scenario_name, success=False,
                         steps_completed=0, total_steps=0, elapsed_time_s=0.0), None

    n_objects = raw_poses.numel() // 7

    def _world_to_bin(wp):
        return [wp[1] - 0.075, wp[0] - 0.35, wp[2] - 1.025]

    positions_bin = [_world_to_bin(raw_poses[i * 7: i * 7 + 3].tolist()) for i in range(n_objects)]
    quats         = [raw_poses[i * 7 + 3: i * 7 + 7].tolist()            for i in range(n_objects)]
    n_obs = n_objects - 1
    print(f'[real-robot] {n_objects} objects received ({n_obs} obstacles)')

    initial_state = {
        'target_pos':    torch.tensor(positions_bin[0]),
        'target_quat':   torch.tensor(quats[0]),
        'obstacle_pos':  torch.tensor(positions_bin[1:]),
        'obstacle_quat': torch.tensor(quats[1:]),
    }

    print(f'[real-robot] object states (bin frame) being passed to MCTS:')
    print(f'  target   pos={[f"{v:.4f}" for v in positions_bin[0]]}  quat={[f"{v:.4f}" for v in quats[0]]}')
    for i, (pos, quat) in enumerate(zip(positions_bin[1:], quats[1:])):
        print(f'  obstacle {i} pos={[f"{v:.4f}" for v in pos]}  quat={[f"{v:.4f}" for v in quat]}')

    with open_dict(cfg):
        cfg.n_obstacles = n_obs

    print(f'[real-robot] running puzzle planning (n_obstacles={n_obs}) …')
    ctx  = multiprocessing.get_context('spawn')
    pq   = ctx.Queue()
    proc = ctx.Process(
        target=_puzzle_worker,
        args=(cfg, [(scenario_name, initial_state)], out_dir, pq),
    )
    proc.start()
    puzzle_timeout = cfg.get('puzzle_timeout_s', None)
    worker_result = None
    try:
        worker_result = pq.get(timeout=puzzle_timeout)
    except queue.Empty:
        print('[real-robot] puzzle planning timed out')
    try:
        proc.kill()
    except OSError:
        pass
    proc.join(timeout=5)

    solution_path = out_dir / 'solutions' / f'{scenario_name}.json'
    if worker_result is None or not solution_path.exists():
        print('[real-robot] puzzle planning failed — no solution injected')
        server_proc.kill()
        return MpcResult(scenario_name=scenario_name, success=False,
                         steps_completed=0, total_steps=0, elapsed_time_s=0.0), initial_state

    for pr in worker_result.get('puzzle_results', []):
        print(f'[real-robot] puzzle result: success={pr.success} plan_len={pr.plan_length}')

    with open(solution_path) as f:
        solution_data = json.load(f)
    steps_json = json.dumps(solution_data['steps'])
    print(f'[real-robot] injecting {len(solution_data["steps"])} steps via reset_episode …')
    client.reset_episode(steps_json)

    server_proc.send_signal(signal.SIGTERM)
    try:
        server_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    print(f'[real-robot] bridge_server stopped')

    return MpcResult(
        scenario_name=scenario_name,
        success=True,
        steps_completed=0,
        total_steps=len(solution_data['steps']),
        elapsed_time_s=0.0,
    ), initial_state


def _log_mpc_wandb(result: MpcResult, step: int, initial_state=None,
                   plan: list | None = None,
                   bin_size: float = 0.3, obj_size: float = 0.05,
                   wall_thickness: float = 0.02):
    """Log per-scenario MPC metrics to WandB."""
    import matplotlib.pyplot as plt
    from visualization import render_ee_trajectory_mppi

    log = {
        'isaaclabmpc/success':          int(result.success),
        'isaaclabmpc/steps_completed':  result.steps_completed,
        'isaaclabmpc/total_steps':      result.total_steps,
        'isaaclabmpc/elapsed_time_s':   result.elapsed_time_s,
    }
    if result.total_steps > 0:
        log['isaaclabmpc/step_rate'] = result.steps_completed / result.total_steps

    if result.ee_trajectory:
        if initial_state is not None:
            fig = render_ee_trajectory_mppi(
                initial_state, bin_size, bin_size, obj_size, wall_thickness,
                result.ee_trajectory,
                plan=plan,
                title=f'EE trajectory — {result.scenario_name}',
            )
        else:
            traj = np.array(result.ee_trajectory)
            fig, ax = plt.subplots(figsize=(8, 3))
            for j, label in enumerate(['x_mppi', 'y_mppi', 'z_mppi']):
                ax.plot(traj[:, j], label=label)
            ax.set_xlabel('sample (every 50 sim steps)')
            ax.set_ylabel('EE position (m)')
            ax.set_title(f'EE trajectory — {result.scenario_name}')
            ax.legend()
            fig.tight_layout()
        log['isaaclabmpc/ee_trajectory'] = wandb.Image(fig)
        plt.close(fig)

    if result.mppi_cost_history:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(result.mppi_cost_history, color='tab:purple', linewidth=1.0)
        ax.set_xlabel('MPPI iteration')
        ax.set_ylabel('min cost')
        ax.set_title(f'MPPI cost — {result.scenario_name}')
        fig.tight_layout()
        log['isaaclabmpc/mppi_cost_history'] = wandb.Image(fig)
        plt.close(fig)

    if result.block_positions_final:
        pos = np.array(result.block_positions_final)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pos[:, 0], pos[:, 1], c=pos[:, 2], cmap='viridis',
                   s=80, zorder=3, edgecolors='k', linewidths=0.5)
        for j, (x, y, _) in enumerate(pos):
            ax.text(x, y, str(j), fontsize=7, ha='center', va='center', color='white', zorder=4)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Final block positions — {result.scenario_name}')
        fig.tight_layout()
        log['isaaclabmpc/block_positions_final'] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(log, step=step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_initial_state_from_yaml(yaml_path: Path) -> dict:
    """Reconstruct a state dict from a generated scenario YAML."""
    import torch
    with open(yaml_path) as f:
        sc = yaml.safe_load(f)
    ist = sc['initial_state']
    return {
        'target_pos':    torch.tensor(ist['target_pos']),
        'target_quat':   torch.tensor(ist['target_quat']),
        'obstacle_pos':  torch.tensor([o['pos']  for o in ist['obstacles']]),
        'obstacle_quat': torch.tensor([o['quat'] for o in ist['obstacles']]),
    }


@hydra.main(version_base=None, config_path="conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    initial_state_by_name: dict = {}
    plans_by_name: dict = {}
    bin_size       = cfg.get('bin_size', 0.3)
    obj_size       = cfg.get('obj_size', 0.05)
    wall_thickness = cfg.get('wall_thickness', 0.02)

    if cfg.get('scenario_file'):
        # ------------------------------------------------------------------
        # Single-scenario mode: load one YAML, run puzzle planning + MPC.
        # Skips generation; useful for re-running or debugging a specific scene.
        # ------------------------------------------------------------------
        scenario_file = Path(cfg.scenario_file).resolve()
        if not scenario_file.exists():
            raise FileNotFoundError(f'scenario_file not found: {scenario_file}')

        scenario_name = scenario_file.stem
        initial_state = _load_initial_state_from_yaml(scenario_file)
        scenarios = [(scenario_name, initial_state)]

        with open(scenario_file) as _f:
            _sc_meta = yaml.safe_load(_f)
        _obj_size  = cfg.get('obj_size', 0.05)
        _obj_h     = _obj_size / 2
        _all_z     = [float(initial_state['target_pos'][2])] + [
                         float(initial_state['obstacle_pos'][i][2])
                         for i in range(len(initial_state['obstacle_pos']))]
        _inferred_z_levels = max(1, max(round((z - _obj_h) / _obj_size) for z in _all_z) + 1)

        with open_dict(cfg):
            cfg.n_scenarios  = 1
            cfg.n_obstacles  = len(initial_state['obstacle_pos'])
            cfg.n_z_levels   = _sc_meta.get('n_z_levels', _inferred_z_levels)
            if 'bin_size'       in _sc_meta: cfg.bin_size       = _sc_meta['bin_size']
            if 'wall_thickness' in _sc_meta: cfg.wall_thickness = _sc_meta['wall_thickness']
            if 'friction'       in _sc_meta: cfg.friction       = _sc_meta['friction']

        force_threshold = cfg.simulator.get('force_threshold', None)
        successful: list[tuple[str, Path]] = []

        if cfg.get('resume', False):
            # ------------------------------------------------------------------
            # scenario_file + resume: skip planning, use existing solution.
            # ------------------------------------------------------------------
            _resume_dir = cfg.get('resume_dir')
            if _resume_dir:
                out_dir = Path(_resume_dir)
            else:
                latest_link = Path(cfg.output_dir) / 'latest'
                if not (latest_link.exists() or latest_link.is_symlink()):
                    raise ValueError(
                        f'No latest run found at {latest_link}; set resume_dir explicitly')
                out_dir = latest_link.resolve()
                print(f'[resume] Using latest run: {out_dir}')

            solution_path = out_dir / 'solutions' / f'{scenario_name}.json'
            if not solution_path.exists():
                raise FileNotFoundError(
                    f'No solution for {scenario_name} in {out_dir / "solutions"}')

            with open(solution_path) as f:
                plans_by_name[scenario_name] = json.load(f).get('plan', [])
            initial_state_by_name[scenario_name] = initial_state
            successful = [(scenario_name, solution_path)]
            print(f'[resume] Found solution: {solution_path}')

            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity if cfg.wandb_entity else None,
                name=cfg.wandb_run_name if cfg.wandb_run_name else None,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            while wandb.run is None:
                time.sleep(1)

            for subdir in ('isaaclabmpc_results', 'telemetry', 'logs'):
                (out_dir / subdir).mkdir(parents=True, exist_ok=True)

        else:
            # ------------------------------------------------------------------
            # scenario_file normal: run puzzle planning then MPC.
            # ------------------------------------------------------------------
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity if cfg.wandb_entity else None,
                name=cfg.wandb_run_name if cfg.wandb_run_name else None,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            while wandb.run is None:
                time.sleep(1)

            run_id = wandb.run.id
            out_dir = Path(cfg.output_dir) / run_id
            for subdir in ('scenarios', 'solutions', 'isaaclabmpc_results', 'telemetry', 'logs', 'videos'):
                (out_dir / subdir).mkdir(parents=True, exist_ok=True)
            print(f'\nOutput directory: {out_dir}')

            import shutil
            shutil.copy(scenario_file, out_dir / 'scenarios' / f'{scenario_name}.yaml')

            latest_link = Path(cfg.output_dir) / 'latest'
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(out_dir.resolve())

            print('\n' + '=' * 60)
            print(f'Stage 1: Single scenario — {scenario_name}')
            print('=' * 60)

            print('\n' + '=' * 60)
            print('Stage 2: Puzzle planning (child process)')
            print('=' * 60)

            ctx = multiprocessing.get_context('spawn')
            q = ctx.Queue()
            proc = ctx.Process(target=_puzzle_worker, args=(cfg, scenarios, out_dir, q))
            proc.start()

            puzzle_timeout = cfg.get('puzzle_timeout_s', None)
            worker_result = None
            deadline = time.time() + puzzle_timeout if puzzle_timeout else None
            while proc.is_alive():
                try:
                    worker_result = q.get(timeout=0.5)
                    break
                except queue.Empty:
                    pass
                if deadline and time.time() >= deadline:
                    print(f'[pipeline] Puzzle worker timed out after {puzzle_timeout}s')
                    break

            try:
                proc.kill()
            except OSError as e:
                print(f'[pipeline] Puzzle worker already exited before kill: {e}')
            proc.join(timeout=5)

            if worker_result is not None:
                initial_state_by_name[scenario_name] = initial_state
                plans_by_name.update(worker_result['plans'])
                puzzle_results: list[PuzzleResult] = []
                for result in worker_result['puzzle_results']:
                    plan = worker_result['plans'].get(result.scenario_name)
                    force_traces = worker_result['force_traces'].get(result.scenario_name, [])
                    if result.success:
                        successful.append((result.scenario_name,
                                           out_dir / 'solutions' / f'{result.scenario_name}.json'))
                    puzzle_results.append(result)
                    _log_puzzle_wandb(
                        result, initial_state,
                        bin_size, obj_size, wall_thickness, force_threshold,
                        plan, force_traces, puzzle_results, result.scenario_idx,
                        video_path=worker_result['videos'].get(result.scenario_name),
                    )

    elif cfg.get('resume', False):
        # ------------------------------------------------------------------
        # Resume mode: skip scenario generation and puzzle planning,
        # jump straight to Stage 3 using solutions already on disk.
        # ------------------------------------------------------------------
        resume_dir = cfg.get('resume_dir')
        if resume_dir:
            out_dir = Path(resume_dir)
        else:
            latest_link = Path(cfg.output_dir) / 'latest'
            if not (latest_link.exists() or latest_link.is_symlink()):
                raise ValueError(f'No latest run found at {latest_link}; set resume_dir explicitly')
            out_dir = latest_link.resolve()
            print(f'[resume] Using latest run: {out_dir}')
        solutions_dir = out_dir / 'solutions'
        scenarios_dir = out_dir / 'scenarios'
        scenario_names = sorted(p.stem for p in scenarios_dir.glob('*.yaml'))
        successful: list[tuple[str, Path]] = [
            (name, solutions_dir / f'{name}.json')
            for name in scenario_names
            if (solutions_dir / f'{name}.json').exists()
        ]
        print(f'[resume] {len(successful)}/{len(scenario_names)} solutions found in {solutions_dir}')

        for name, sol_path in successful:
            yaml_path = scenarios_dir / f'{name}.yaml'
            if yaml_path.exists():
                initial_state_by_name[name] = _load_initial_state_from_yaml(yaml_path)
            with open(sol_path) as f:
                plans_by_name[name] = json.load(f).get('plan', [])

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity if cfg.wandb_entity else None,
            name=cfg.wandb_run_name if cfg.wandb_run_name else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        while wandb.run is None:
            time.sleep(1)
        (out_dir / 'isaaclabmpc_results').mkdir(parents=True, exist_ok=True)
        (out_dir / 'telemetry').mkdir(parents=True, exist_ok=True)
        (out_dir / 'logs').mkdir(parents=True, exist_ok=True)

    elif cfg.get('scenario_source') == 'server':
        # ------------------------------------------------------------------
        # Server mode: pull env setup from the running planner, run puzzle
        # planning, inject solution, run MPC. No scenario YAML required.
        # Usage: python pipeline.py --config-name=pipeline scenario_source=server
        # ------------------------------------------------------------------
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity if cfg.wandb_entity else None,
            name=cfg.wandb_run_name if cfg.wandb_run_name else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        while wandb.run is None:
            time.sleep(1)

        run_id  = wandb.run.id
        out_dir = Path(cfg.output_dir) / run_id
        for subdir in ('solutions', 'isaaclabmpc_results', 'telemetry', 'logs'):
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f'\nOutput directory: {out_dir}')

        latest_link = Path(cfg.output_dir) / 'latest'
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(out_dir.resolve())

        scenario_name = cfg.get('server_scenario_name', 'server_scenario')

        print('\n' + '=' * 60)
        print('Server mode: query scenario from planner, plan, inject, MPC')
        print('=' * 60)

        mpc_result, initial_state = _run_real_robot_scenario(
            scenario_name, out_dir, cfg
        )

        _log_mpc_wandb(
            mpc_result, 0,
            initial_state=initial_state,
            bin_size=bin_size,
            obj_size=obj_size,
            wall_thickness=wall_thickness,
        )

        wandb.run.summary.update({
            'n_scenarios':              1,
            'n_puzzle_success':         1 if initial_state is not None else 0,
            'n_mpc_success':            int(mpc_result.success),
            'isaaclabmpc_success_rate': int(mpc_result.success),
        })
        wandb.finish()
        return

    else:
        # ------------------------------------------------------------------
        # Stage 1: Generate scenarios (no simulator needed)
        # ------------------------------------------------------------------
        print('\n' + '=' * 60)
        print('Stage 1: Generating scenarios')
        print('=' * 60)

        # Init WandB first so we have a run_id to build the output directory
        # before scenario generation — scenarios go into the same run dir.
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity if cfg.wandb_entity else None,
            name=cfg.wandb_run_name if cfg.wandb_run_name else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        while wandb.run is None:
            time.sleep(1)

        run_id = wandb.run.id
        out_dir = Path(cfg.output_dir) / run_id
        for subdir in ('scenarios', 'solutions', 'isaaclabmpc_results', 'telemetry', 'logs', 'videos'):
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f'\nOutput directory: {out_dir}')

        latest_link = Path(cfg.output_dir) / 'latest'
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(out_dir.resolve())

        # Keep conf/scenario/generated pointing to the latest run's scenarios
        # so external tools and ad-hoc scripts find them at the familiar path.
        generated_link = Path('conf/scenario/generated')
        if generated_link.is_symlink():
            generated_link.unlink()
        elif generated_link.exists():
            import shutil
            shutil.rmtree(generated_link)
        generated_link.symlink_to((out_dir / 'scenarios').resolve())

        scenarios = _generate_scenarios(cfg, out_dir / 'scenarios')

        force_threshold = cfg.simulator.get('force_threshold', None)

        # ------------------------------------------------------------------
        # Stage 2: Puzzle planning (child process)
        #
        # IsaacLab doesn't shut down cleanly, so we run the solver in a
        # spawned child process and kill it after results are received.
        # ------------------------------------------------------------------
        print('\n' + '=' * 60)
        print('Stage 2: Puzzle planning (child process)')
        print('=' * 60)

        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        proc = ctx.Process(target=_puzzle_worker, args=(cfg, scenarios, out_dir, q))
        proc.start()

        puzzle_timeout = cfg.get('puzzle_timeout_s', None)
        worker_result = None
        deadline = time.time() + puzzle_timeout if puzzle_timeout else None
        while proc.is_alive():
            try:
                worker_result = q.get(timeout=0.5)
                break
            except queue.Empty:
                pass
            if deadline and time.time() >= deadline:
                print(f'[pipeline] Puzzle worker timed out after {puzzle_timeout}s')
                break

        try:
            proc.kill()
        except OSError as e:
            print(f'[pipeline] Puzzle worker already exited before kill: {e}')
        proc.join(timeout=5)

        successful: list[tuple[str, Path]] = []

        if worker_result is not None:
            initial_state_by_name = dict(scenarios)  # populates the outer dict
            plans_by_name.update(worker_result['plans'])
            puzzle_results: list[PuzzleResult] = []
            for result in worker_result['puzzle_results']:
                scenario_name = result.scenario_name
                initial_state = initial_state_by_name[scenario_name]
                plan = worker_result['plans'].get(scenario_name)
                force_traces = worker_result['force_traces'].get(scenario_name, [])

                if result.success:
                    solution_path = out_dir / 'solutions' / f'{scenario_name}.json'
                    successful.append((scenario_name, solution_path))

                puzzle_results.append(result)
                _log_puzzle_wandb(
                    result, initial_state,
                    bin_size, obj_size, wall_thickness, force_threshold,
                    plan, force_traces, puzzle_results, result.scenario_idx,
                    video_path=worker_result['videos'].get(scenario_name),
                )

    n_puzzle_success = len(successful)
    if not cfg.get('resume', False):
        print(f'\nPuzzle planning: {n_puzzle_success}/{cfg.n_scenarios} successful')

    # ------------------------------------------------------------------
    # Stage 3: IsaacLab MPC
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'Stage 3: IsaacLab MPC ({n_puzzle_success} scenarios)')
    print('=' * 60)

    mpc_results: list[MpcResult] = []

    show_viewer = cfg.get('show_mpc_world_viewer', False)
    for i, (scenario_name, solution_path) in enumerate(successful):
        if show_viewer and i > 0:
            input(f'\n[MPC] Press Enter to continue to scenario {i + 1}/{len(successful)} ({scenario_name}) …')

        scenario_yaml = (out_dir / 'scenarios' / f'{scenario_name}.yaml').resolve()

        mpc_result = _run_isaaclabmpc(
            scenario_name, scenario_yaml, solution_path, out_dir, cfg
        )
        mpc_results.append(mpc_result)
        _log_mpc_wandb(
            mpc_result, cfg.n_scenarios + i,
            initial_state=initial_state_by_name.get(scenario_name),
            plan=plans_by_name.get(scenario_name),
            bin_size=bin_size, obj_size=obj_size, wall_thickness=wall_thickness,
        )

    # ------------------------------------------------------------------
    # Stage 4: Summary
    # ------------------------------------------------------------------
    n_mpc_success = sum(1 for r in mpc_results if r.success)

    print('\n' + '=' * 60)
    print(f'Pipeline complete.')
    print(f'  Scenarios generated: {cfg.n_scenarios}')
    print(f'  Puzzle success:      {n_puzzle_success}/{cfg.n_scenarios} '
          f'({n_puzzle_success/cfg.n_scenarios:.0%})')
    print(f'  MPC success:         {n_mpc_success}/{n_puzzle_success} '
          f'({n_mpc_success/max(n_puzzle_success, 1):.0%} of puzzle successes)')
    print('=' * 60)

    wandb.run.summary.update({
        'n_scenarios':                     cfg.n_scenarios,
        'n_puzzle_success':                n_puzzle_success,
        'n_mpc_success':                   n_mpc_success,
        'puzzles_success_rate':            n_puzzle_success / cfg.n_scenarios,
        'isaaclabmpc_success_rate':        n_mpc_success / cfg.n_scenarios,
        'isaaclabmpc_of_puzzle_rate':      n_mpc_success / max(n_puzzle_success, 1),
    })

    sc_artifact = wandb.Artifact('scenarios', type='dataset')
    sc_artifact.add_dir(str(out_dir / 'scenarios'))
    wandb.log_artifact(sc_artifact)

    sol_artifact = wandb.Artifact('solutions', type='dataset')
    sol_artifact.add_dir(str(out_dir / 'solutions'))
    wandb.log_artifact(sol_artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
