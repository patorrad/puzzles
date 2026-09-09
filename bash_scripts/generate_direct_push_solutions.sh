#!/usr/bin/env bash
# direct_push baseline (no puzzle solver — a single pull_s straight out the
# exit) across every stacked2-difficult obstacle count: 5, 7, 10, 14.
#
# For each obstacle count this generates the solution JSONs (no search, no
# benchmark harness needed) and then immediately runs them through the real
# MPC via run_solutions.py, same as run_all_solutions.sh does for the other
# planners' pre-made solutions. Safe to re-run: skips any directory that
# already has a results.csv.
#
# Run from the puzzle directory:
#   bash bash_scripts/generate_direct_push_solutions.sh
set -euo pipefail

PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"
BENCH="--config-name=benchmark n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4"

for obs in 7 10 14; do
    sol_dir="results/1_verification/direct_push_${obs}obs_stacked2_difficult"

    if [[ -f "$sol_dir/results.csv" ]]; then
        echo "[skip] $sol_dir (already run)"
        continue
    fi

    echo "========================================"
    echo "  n_obstacles=${obs} — DirectPush: generating solutions"
    echo "========================================"
    "$PYTHON" generate_direct_push_solutions.py $BENCH \
        n_obstacles=${obs} \
        solutions_dir="$sol_dir"

    echo "========================================"
    echo "  n_obstacles=${obs} — DirectPush: running MPC"
    echo "========================================"
    "$PYTHON" run_solutions.py \
        solutions_dir="$sol_dir" \
        use_wandb=false \
        save_video=false \
        show_mpc_world_viewer=true

done

echo "All direct_push solutions generated and run."
