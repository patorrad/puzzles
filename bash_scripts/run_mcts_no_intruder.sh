#!/usr/bin/env bash
# Run the MPC stage (intruder_penalty=0.0) for the fresh MCTS solution
# directories in results/1_verification (mcts_*obs_stacked2_difficult,
# excluding the "_old" ones from the pre-fix runs).
#
# Assumes isaaclabmpc's intruder_penalty is already 0.0 in
#   /home/paolo/Documents/isaaclabmpc/examples/ur16e_stacked_robot_sim/config.yaml
# (set by the earlier no-intruder AlphaZero ablation run) — this script just
# checks and prints the current value rather than changing it, since it may
# already be what you want.
#
# Headless, no video (fast for a multi-directory batch); wandb left at its
# default (on). Safe to re-run: skips any directory that already has a
# results.csv.
#
# Run from the puzzle directory:
#   bash bash_scripts/run_mcts_no_intruder.sh
set -euo pipefail

PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"
CONFIG="/home/paolo/Documents/isaaclabmpc/examples/ur16e_stacked_robot_sim/config.yaml"

echo "Current intruder_penalty setting:"
grep "intruder_penalty: " "$CONFIG" | head -1
echo ""

for sol_dir in results/1_verification/mcts_*obs_stacked2_difficult; do
    [[ -d "$sol_dir" ]] || continue
    [[ "$sol_dir" == *_old ]] && continue

    dirname=$(basename "$sol_dir")

    if [[ -f "$sol_dir/results.csv" ]]; then
        echo "[skip] $dirname (already run)"
        continue
    fi

    n_solutions=$(ls "$sol_dir"/run_*_seed_*.json 2>/dev/null | wc -l)
    if [[ "$n_solutions" -eq 0 ]]; then
        echo "[skip] $dirname (no solution files yet)"
        continue
    fi

    echo "========================================"
    echo "  Running MPC (intruder_penalty=0.0) for $dirname ($n_solutions solutions)"
    echo "========================================"
    "$PYTHON" run_solutions.py \
        solutions_dir="$sol_dir" \
        save_video=false \
        show_mpc_world_viewer=false

    echo "[done] $dirname"
done

echo ""
echo "All fresh MCTS solution directories processed."
