"""
compare_planners.py — End-to-end pipeline comparison: Puzzle+MPPI vs CuTAMP+CuRobo.

Pipeline A (run first in the IsaacLab env via pipeline.py):
  MCTS push planner → IsaacLab MPPI execution

Pipeline B (run in the cutamp env via this script):
  CuTAMP TAMP → CuRobo collision-free motion trajectories

Workflow
--------
# Step 1 — Pipeline A (IsaacLab env)
python pipeline.py --config-name=pipeline n_scenarios=10

# Step 2 — Pipeline B (cutamp env)
python compare_planners.py --only_cutamp \\
    --scenario_dir outputs/latest/scenarios \\
    --cutamp_out cutamp_results

# Step 3 — Merge and compare (either env)
python compare_planners.py \\
    --merge outputs/latest cutamp_results \\
    --out compare_results.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import yaml

PUZZLE_DIR = os.path.dirname(os.path.abspath(__file__))
CUTAMP_DIR = os.path.abspath(os.path.join(PUZZLE_DIR, "..", "cuTAMP"))


# ---------------------------------------------------------------------------
# Pipeline B: CuTAMP runner
# ---------------------------------------------------------------------------

def run_cutamp_benchmark(
    scenario_paths: list,
    cutamp_out: str,
    num_particles: int = 1024,
    num_opt_steps: int = 1000,
    experiment_root: str = "/tmp/cutamp-bench",
) -> None:
    """Run CuTAMP on each scenario, saving one JSON per scenario to cutamp_out/."""
    sys.path.insert(0, CUTAMP_DIR)
    sys.path.insert(0, PUZZLE_DIR)
    from cutamp_adapter import run_cutamp_on_scenario

    os.makedirs(cutamp_out, exist_ok=True)

    for i, path in enumerate(scenario_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"  [cutamp] [{i+1}/{len(scenario_paths)}] {name}...")
        try:
            m = run_cutamp_on_scenario(
                path,
                num_particles=num_particles,
                num_opt_steps=num_opt_steps,
                experiment_root=experiment_root,
            )
            status = "SUCCESS" if m["execution_success"] else (
                "PLAN_OK" if m["plan_success"] else "FAILED"
            )
            print(f"    {status} | plan_time={m['plan_time_s']:.1f}s | "
                  f"satisfying={m['num_satisfying']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            with open(path) as f:
                n_obs = int(yaml.safe_load(f).get("n_obstacles", 0))
            m = {
                "scenario": os.path.basename(path),
                "n_obstacles": n_obs,
                "plan_success": False,
                "plan_time_s": 0.0,
                "execution_success": False,
                "total_time_s": 0.0,
                "num_satisfying": 0,
                "error": str(e),
            }

        out_path = os.path.join(cutamp_out, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(m, f, indent=2)


# ---------------------------------------------------------------------------
# Merge: read Pipeline A and Pipeline B results, aggregate, report
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def merge_and_report(pipeline_a_dir: str, cutamp_results_dir: str, out_csv: str) -> None:
    scenario_yamls = sorted(glob.glob(os.path.join(pipeline_a_dir, "scenarios", "*.yaml")))
    if not scenario_yamls:
        print(f"No scenario YAMLs found in {pipeline_a_dir}/scenarios/")
        sys.exit(1)

    rows_a = []
    rows_b = []

    for yaml_path in scenario_yamls:
        name = os.path.splitext(os.path.basename(yaml_path))[0]

        # n_obstacles from scenario YAML
        with open(yaml_path) as f:
            n_obs = int(yaml.safe_load(f).get("n_obstacles", 0))

        # Pipeline A
        pr = _load_json(os.path.join(pipeline_a_dir, "puzzle_results", f"{name}.json"))
        mpc = _load_json(os.path.join(pipeline_a_dir, "isaaclabmpc_results", f"{name}.json"))

        plan_success_a = pr.get("success", False)
        plan_time_a = pr.get("plan_time_s", 0.0)
        exec_success_a = mpc.get("success", False) if mpc else False
        mpc_time_a = mpc.get("elapsed_time_s", 0.0) if mpc else 0.0
        total_time_a = plan_time_a + mpc_time_a

        rows_a.append({
            "scenario": name,
            "n_obstacles": n_obs,
            "pipeline": "puzzle+mppi",
            "plan_success": plan_success_a,
            "plan_time_s": plan_time_a,
            "execution_success": exec_success_a,
            "total_time_s": total_time_a,
            "mpc_ran": bool(mpc),
        })

        # Pipeline B
        cb = _load_json(os.path.join(cutamp_results_dir, f"{name}.json"))
        if cb:
            rows_b.append({
                "scenario": name,
                "n_obstacles": n_obs,
                "pipeline": "cutamp+curobo",
                "plan_success": cb.get("plan_success", False),
                "plan_time_s": cb.get("plan_time_s", 0.0),
                "execution_success": cb.get("execution_success", False),
                "total_time_s": cb.get("total_time_s", 0.0),
                "mpc_ran": True,
            })

    _print_table(rows_a, rows_b)
    _save_csv(rows_a + rows_b, out_csv)


def _aggregate(rows: list, pipeline: str, n_obs: int) -> dict | None:
    subset = [r for r in rows if r["pipeline"] == pipeline and r["n_obstacles"] == n_obs]
    if not subset:
        return None
    n = len(subset)
    plan_succ = sum(1 for r in subset if r["plan_success"])
    exec_succ = sum(1 for r in subset if r["execution_success"])
    plan_times = np.array([r["plan_time_s"] for r in subset])
    total_times = np.array([r["total_time_s"] for r in subset])
    return {
        "pipeline": pipeline,
        "n_obstacles": n_obs,
        "n_scenarios": n,
        "plan_success_rate": plan_succ / n,
        "exec_success_rate": exec_succ / n,
        "plan_time_mean": float(plan_times.mean()),
        "plan_time_std": float(plan_times.std()),
        "total_time_mean": float(total_times.mean()),
        "total_time_std": float(total_times.std()),
    }


def _print_table(rows_a: list, rows_b: list) -> None:
    all_rows = rows_a + rows_b
    n_obs_values = sorted({r["n_obstacles"] for r in all_rows})
    pipelines = ["puzzle+mppi", "cutamp+curobo"]

    agg = [
        row
        for n_obs in n_obs_values
        for pipeline in pipelines
        for row in [_aggregate(rows_a if pipeline == "puzzle+mppi" else rows_b, pipeline, n_obs)]
        if row is not None
    ]

    print("\n" + "=" * 100)
    print(f"{'Pipeline':<18} {'N_obs':<7} {'N':<5} {'PlanSucc%':<12} "
          f"{'ExecSucc%':<12} {'PlanTime(s)':<20} {'TotalTime(s)'}")
    print("-" * 100)
    for row in sorted(agg, key=lambda r: (r["n_obstacles"], r["pipeline"])):
        print(
            f"{row['pipeline']:<18} {row['n_obstacles']:<7} {row['n_scenarios']:<5} "
            f"{row['plan_success_rate']*100:>7.1f}%     "
            f"{row['exec_success_rate']*100:>7.1f}%     "
            f"{row['plan_time_mean']:>5.1f} ± {row['plan_time_std']:.1f}s      "
            f"{row['total_time_mean']:>5.1f} ± {row['total_time_std']:.1f}s"
        )
    print("=" * 100)


def _save_csv(rows: list, path: str) -> None:
    if not rows:
        return
    fieldnames = ["scenario", "n_obstacles", "pipeline", "plan_success",
                  "plan_time_s", "execution_success", "total_time_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline comparison: Puzzle+MPPI vs CuTAMP+CuRobo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode")

    # --only_cutamp mode
    cutamp_parser = subparsers.add_parser(
        "cutamp", help="Run CuTAMP on scenarios (cutamp env)"
    )
    cutamp_parser.add_argument("--scenario_dir", required=True,
                               help="Directory of scenario YAML files (from pipeline.py outputs)")
    cutamp_parser.add_argument("--cutamp_out", default="cutamp_results",
                               help="Directory to write per-scenario CuTAMP result JSONs")
    cutamp_parser.add_argument("--num_particles", type=int, default=1024)
    cutamp_parser.add_argument("--num_opt_steps", type=int, default=1000)
    cutamp_parser.add_argument("--experiment_root", default="/tmp/cutamp-bench")

    # --merge mode
    merge_parser = subparsers.add_parser(
        "merge", help="Merge Pipeline A and B results into comparison CSV"
    )
    merge_parser.add_argument("pipeline_a_dir",
                              help="Pipeline A output dir (outputs/<run_id>/)")
    merge_parser.add_argument("cutamp_results_dir",
                              help="Directory of CuTAMP result JSONs (from cutamp mode)")
    merge_parser.add_argument("--out", default="compare_results.csv")

    # Legacy flat flags for backwards compatibility
    parser.add_argument("--only_cutamp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario_dir", help=argparse.SUPPRESS)
    parser.add_argument("--cutamp_out", default="cutamp_results", help=argparse.SUPPRESS)
    parser.add_argument("--num_particles", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--num_opt_steps", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--experiment_root", default="/tmp/cutamp-bench", help=argparse.SUPPRESS)
    parser.add_argument("--merge", nargs=2, metavar=("PIPELINE_A_DIR", "CUTAMP_RESULTS_DIR"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default="compare_results.csv", help=argparse.SUPPRESS)

    args = parser.parse_args()

    os.chdir(PUZZLE_DIR)

    # Resolve legacy flat flags
    if args.mode is None:
        if args.merge:
            args.mode = "_merge_legacy"
        elif args.only_cutamp or args.scenario_dir:
            args.mode = "_cutamp_legacy"

    if args.mode == "cutamp" or args.mode == "_cutamp_legacy":
        scenario_dir = getattr(args, "scenario_dir", None)
        if not scenario_dir:
            parser.error("--scenario_dir required for cutamp mode")
        scenario_paths = sorted(glob.glob(os.path.join(scenario_dir, "*.yaml")))
        if not scenario_paths:
            print(f"No YAML scenarios found in: {scenario_dir}")
            sys.exit(1)
        print(f"Found {len(scenario_paths)} scenario(s) in {scenario_dir}")
        run_cutamp_benchmark(
            scenario_paths,
            cutamp_out=args.cutamp_out,
            num_particles=args.num_particles,
            num_opt_steps=args.num_opt_steps,
            experiment_root=args.experiment_root,
        )

    elif args.mode == "merge" or args.mode == "_merge_legacy":
        if args.mode == "merge":
            pipeline_a_dir = args.pipeline_a_dir
            cutamp_results_dir = args.cutamp_results_dir
            out_csv = args.out
        else:
            pipeline_a_dir, cutamp_results_dir = args.merge
            out_csv = args.out
        merge_and_report(pipeline_a_dir, cutamp_results_dir, out_csv)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
