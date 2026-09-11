"""
run_solutions.py — Execute existing puzzle solutions through IsaacLab MPC and record metrics.

Runs only Stage 3 from pipeline.py: launches isaaclabmpc subprocesses for each .json
solution file found in solutions_dir, then logs success/failure and timing to WandB
and a CSV file.

Usage
-----
# Run all solutions in a directory
python run_solutions.py solutions_dir=solutions/mcts_5obs_stacked2_difficult

# With a custom WandB run name and no viewer
python run_solutions.py solutions_dir=solutions/mcts_5obs_stacked2_difficult \
    wandb_run_name=mcts_5obs_mpc_eval show_mpc_world_viewer=false

# Run without WandB
python run_solutions.py solutions_dir=solutions/mcts_5obs_stacked2_difficult use_wandb=false

# Provide paired scenario YAMLs (same stem as solution files)
python run_solutions.py solutions_dir=solutions/mcts_5obs_stacked2_difficult \
    scenario_dir=conf/scenario/generated

# Limit to first N solutions (useful for smoke tests)
python run_solutions.py solutions_dir=solutions/mcts_5obs_stacked2_difficult max_solutions=5
"""

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from pipeline import (_log_mpc_wandb, MpcResult, _load_initial_state_from_yaml,
                      _check_port_free, _stream_output)


def _swap_xy(pos: list) -> list:
    """Swap pos[0] and pos[1].

    Puzzle solution JSONs store positions as [NS, EW, z] (pos[0] = NS = exit axis).
    _bin_to_mppi_local in scene.py expects [EW, NS, z] — confirmed by comparing its
    hardcoded _BIN_BLOCK_SPECS against the source solution JSON: the JSON stores
    target at [0.2297, 0.2127] while scene.py has [0.2297, 0.2127] → same, but the
    reference ur16e_stand_blocks.yaml (hand-crafted) has them swapped to [0.2127,
    0.2297], causing wrong MPPI positions. We correct for this here.
    """
    return [pos[1], pos[0], pos[2]]


def _make_scenario_yaml_from_solution(solution_path: Path, yaml_dir: Path) -> Path:
    """Generate a scenario YAML from a solution JSON so planner.py gets the correct
    object positions. Puzzle positions are [NS, EW, z]; _bin_to_mppi_local expects
    [EW, NS, z], so we swap x and y before writing.
    """
    with open(solution_path) as f:
        sol = json.load(f)
    ist = sol["initial_state"]

    obstacles = [
        {"pos": _swap_xy(pos), "quat": quat}
        for pos, quat in zip(ist["obstacle_pos"], ist["obstacle_quat"])
    ]
    data = {
        "initial_state": {
            "target_pos":  _swap_xy(ist["target_pos"]),
            "target_quat": ist["target_quat"],
            "obstacles":   obstacles,
        }
    }
    if "env_config" in sol:
        ec = sol["env_config"]
        if "BIN_W" in ec:
            data["bin_size"] = ec["BIN_W"]
        if "n_obstacles" in ec:
            data["n_obstacles"] = ec["n_obstacles"]

    out_path = yaml_dir / f"{solution_path.stem}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)
    return out_path


def _collect_solutions(solutions_dir: Path, scenario_dir: Path | None,
                       max_solutions: int | None) -> list[tuple[str, Path, Path | None]]:
    """Return list of (name, solution_path, scenario_yaml_or_None) sorted by name."""
    jsons = sorted(solutions_dir.glob("run_*_seed_*.json"))
    if not jsons:
        sys.exit(f"[run_solutions] No .json files found in {solutions_dir}")
    if max_solutions is not None:
        jsons = jsons[:max_solutions]

    result = []
    for p in jsons:
        name = p.stem
        scenario_yaml = None
        if scenario_dir is not None:
            candidate = scenario_dir / f"{name}.yaml"
            if candidate.exists():
                scenario_yaml = candidate
        result.append((name, p, scenario_yaml))
    return result


def _load_initial_state_from_solution(solution_path: Path):
    """Extract initial_state dict from a solution JSON (for wandb logging)."""
    import torch
    with open(solution_path) as f:
        sol = json.load(f)
    ist = sol["initial_state"]
    return {
        "target_pos":    torch.tensor(ist["target_pos"]),
        "target_quat":   torch.tensor(ist["target_quat"]),
        "obstacle_pos":  torch.tensor(ist["obstacle_pos"]),
        "obstacle_quat": torch.tensor(ist["obstacle_quat"]),
    }


def _load_plan_from_solution(solution_path: Path) -> list[dict]:
    with open(solution_path) as f:
        sol = json.load(f)
    return sol.get("plan", [])


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[run_solutions] CSV saved: {csv_path}")


def _mpc_result_from_output(name: str, out_path: Path) -> MpcResult:
    """Build an MpcResult from a written isaaclabmpc_results/<name>.json (or
    a zeroed placeholder if the file doesn't exist, e.g. never attempted)."""
    result_data = {}
    if out_path.exists():
        with open(out_path) as f:
            result_data = json.load(f)
    return MpcResult(
        scenario_name          = name,
        success                = result_data.get("success", False),
        steps_completed        = result_data.get("steps_completed", 0),
        total_steps            = result_data.get("total_steps", 0),
        elapsed_time_s         = result_data.get("elapsed_time_s", 0.0),
        ee_trajectory          = result_data.get("ee_trajectory"),
        block_positions_final  = result_data.get("block_positions_final"),
        step_completion_events = None,
        mppi_cost_history      = None,
        target_exited          = result_data.get("target_exited", False),
        intruder_exited        = result_data.get("intruder_exited", False),
    )


def _run_multi_episode(manifest_path: Path, out_dir: Path, cfg: DictConfig,
                       first_scenario: Path | None) -> list[MpcResult]:
    """Launch planner + world once and run all manifest episodes without relaunching Isaac Lab."""
    ilab_dir    = Path(cfg.isaaclabmpc_dir)
    python      = sys.executable
    show_viewer = cfg.get("show_mpc_world_viewer", False)

    planner_cmd = [python, str(ilab_dir / "planner.py"), "--defer_solution", "--headless"]
    if first_scenario is not None:
        planner_cmd += ["--scenario", str(first_scenario)]

    world_cmd = [
        python, str(ilab_dir / "world.py"),
        "--manifest", str(manifest_path),
        "--n_steps", str(cfg.isaaclabmpc_n_steps),
        "--solution_timeout_s", str(cfg.get("solution_timeout_s", 300)),
    ]
    if not show_viewer:
        world_cmd.append("--headless")
    if cfg.get("save_video", False):
        world_cmd.append("--save_video")

    with open(manifest_path) as f:
        episodes = json.load(f)["episodes"]
    per_ep_timeout = cfg.get("solution_timeout_s", cfg.isaaclabmpc_timeout_s)
    total_timeout = cfg.get("isaaclabmpc_timeout_s", len(episodes) * per_ep_timeout + 120)

    planner_log = out_dir / "logs" / "_planner.txt"
    world_log   = out_dir / "logs" / "_world.txt"
    cwd = str(ilab_dir.parent.parent)

    _check_port_free(4242)
    print(f"\n[run_solutions] Launching planner (defer_solution) …")
    planner_proc = subprocess.Popen(
        planner_cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(planner_proc.stdout, planner_log, "[planner] ")

    time.sleep(3)

    print(f"[run_solutions] Launching world ({len(episodes)} episodes, "
          f"timeout={total_timeout}s) …")
    world_proc = subprocess.Popen(
        world_cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(world_proc.stdout, world_log, "[world] ")

    try:
        world_proc.wait(timeout=total_timeout)
    except subprocess.TimeoutExpired:
        print(f"[run_solutions] GLOBAL TIMEOUT after {total_timeout}s — killing world")
        world_proc.kill()
    finally:
        planner_proc.send_signal(signal.SIGTERM)
        try:
            planner_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            planner_proc.kill()

    return [_mpc_result_from_output(ep["name"], Path(ep["output_path"])) for ep in episodes]


@hydra.main(version_base=None, config_path="conf", config_name="run_solutions")
def main(cfg: DictConfig) -> None:
    solutions_dir = Path(cfg.solutions_dir).resolve()
    scenario_dir  = Path(cfg.scenario_dir).resolve() if cfg.get("scenario_dir") else None
    max_solutions = cfg.get("max_solutions", None)
    use_wandb     = cfg.get("use_wandb", True)

    if not solutions_dir.exists():
        sys.exit(f"[run_solutions] solutions_dir not found: {solutions_dir}")

    solutions = _collect_solutions(solutions_dir, scenario_dir, max_solutions)
    print(f"\n[run_solutions] {len(solutions)} solutions in {solutions_dir}")

    run_label = cfg.get("wandb_run_name") or solutions_dir.name
    out_dir = solutions_dir
    for subdir in ("isaaclabmpc_results", "telemetry", "logs", "scenarios"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    print(f"[run_solutions] Output directory: {out_dir}")

    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity if cfg.get("wandb_entity") else None,
            name=run_label,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        while wandb.run is None:
            time.sleep(1)

    bin_size       = cfg.get("bin_size", 0.3)
    obj_size       = cfg.get("obj_size", 0.05)
    wall_thickness = cfg.get("wall_thickness", 0.02)

    csv_rows: list[dict] = []

    # Generate scenario YAMLs for all solutions upfront and write the manifest.
    # Episodes whose isaaclabmpc_results/<name>.json already shows total_steps > 0
    # were genuinely attempted in a prior run (success or failure) and are skipped;
    # only episodes that never got a chance to execute (missing output, or
    # total_steps == 0 — e.g. cut off by a prior run's global timeout) are rerun.
    # This makes re-running run_solutions.py on the same directory a resume, not
    # a full redo.
    manifest_episodes = []
    resolved_solutions = []
    already_good: dict[str, MpcResult] = {}
    for name, sol_path, scenario_yaml in solutions:
        if scenario_yaml is None:
            scenario_yaml = _make_scenario_yaml_from_solution(
                sol_path, out_dir / "scenarios"
            ).resolve()
        result_json = (out_dir / "isaaclabmpc_results" / f"{name}.json").resolve()
        video_path  = (out_dir / "videos" / f"{name}.mp4").resolve()
        resolved_solutions.append((name, sol_path, scenario_yaml))

        existing = _mpc_result_from_output(name, result_json)
        if existing.total_steps > 0:
            already_good[name] = existing
            continue

        manifest_episodes.append({
            "name":          name,
            "scenario_yaml": str(scenario_yaml),
            "solution_path": str(sol_path.resolve()),
            "output_path":   str(result_json),
            "video_path":    str(video_path),
        })

    if already_good:
        print(f"[run_solutions] Skipping {len(already_good)} already-attempted "
              f"episode(s); running {len(manifest_episodes)} remaining.")

    manifest_path = (out_dir / "manifest.json").resolve()
    with open(manifest_path, "w") as f:
        json.dump({"episodes": manifest_episodes}, f, indent=2)
    print(f"[run_solutions] Manifest written: {manifest_path}")

    if manifest_episodes:
        first_scenario = Path(manifest_episodes[0]["scenario_yaml"])
        freshly_run = {r.scenario_name: r
                      for r in _run_multi_episode(manifest_path, out_dir, cfg, first_scenario)}
    else:
        print("[run_solutions] Nothing to run — all episodes already attempted.")
        freshly_run = {}

    mpc_results = [already_good.get(name) or freshly_run[name]
                  for name, _, _ in resolved_solutions]

    # WandB logging and CSV — happens after world.py finishes all episodes.
    for i, (mpc_result, (name, sol_path, _)) in enumerate(
            zip(mpc_results, resolved_solutions)):
        initial_state = None
        try:
            initial_state = _load_initial_state_from_solution(sol_path)
        except Exception:
            pass

        plan = None
        try:
            plan = _load_plan_from_solution(sol_path)
        except Exception:
            pass

        if use_wandb:
            _log_mpc_wandb(
                mpc_result, i,
                initial_state=initial_state,
                plan=plan,
                bin_size=bin_size,
                obj_size=obj_size,
                wall_thickness=wall_thickness,
            )

        csv_rows.append({
            "scenario":        name,
            "success":         int(mpc_result.success),
            "target_exited":   int(mpc_result.target_exited),
            "intruder_exited": int(mpc_result.intruder_exited),
            "steps_completed": mpc_result.steps_completed,
            "total_steps":     mpc_result.total_steps,
            "elapsed_time_s":  round(mpc_result.elapsed_time_s, 2),
            "step_rate":       round(mpc_result.steps_completed / mpc_result.total_steps, 4)
                               if mpc_result.total_steps > 0 else 0.0,
            "solution_path":   str(sol_path),
        })

    # Summary
    n_success = sum(r.success for r in mpc_results)
    n_total   = len(mpc_results)
    success_rate = n_success / n_total if n_total > 0 else 0.0
    avg_time = sum(r.elapsed_time_s for r in mpc_results) / n_total if n_total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Run complete.")
    print(f"  Solutions:   {n_total}")
    print(f"  MPC success: {n_success}/{n_total} ({success_rate:.0%})")
    print(f"  Avg time:    {avg_time:.1f}s")
    print("=" * 60)

    _write_csv(out_dir / "results.csv", csv_rows)

    if use_wandb:
        wandb.run.summary.update({
            "n_solutions":    n_total,
            "n_mpc_success":  n_success,
            "success_rate":   success_rate,
            "avg_time_s":     avg_time,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
