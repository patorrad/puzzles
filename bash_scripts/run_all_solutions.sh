#!/usr/bin/env bash
# Run run_solutions.py for every solutions directory that has not yet been run.
# A directory is considered "run" if it already contains isaaclabmpc_results/ or results.csv.

set -euo pipefail

SOLUTIONS_ROOT="/home/paolo/Documents/puzzle/results/1_verification"
PUZZLE_DIR="/home/paolo/Documents/puzzle"
PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"

cd "$PUZZLE_DIR"

for sol_dir in "$SOLUTIONS_ROOT"/*/; do
    dirname=$(basename "$sol_dir")
    [[ "$dirname" == "archive" ]] && continue

    if [[ -f "$sol_dir/results.csv" ]]; then
        echo "[skip] $dirname (already run)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[run] $dirname"
    echo "=========================================="

    "$PYTHON" run_solutions.py \
        solutions_dir="$sol_dir" \
        use_wandb=false \
        save_video=false \
        show_mpc_world_viewer=false

    echo "[done] $dirname"
done

echo ""
echo "All pending solution directories processed."
