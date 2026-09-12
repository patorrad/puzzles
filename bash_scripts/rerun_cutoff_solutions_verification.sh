#!/usr/bin/env bash
# Top up every directory in solutions/1_verification: runs MPC on any solution
# that was never attempted (missing isaaclabmpc_results entirely, e.g.
# alphazero_mlp1024_14obs_stacked3_diff_bottom_iter100) or was cut off by the
# old isaaclabmpc_timeout_s global-timeout bug (total_steps=0 in results.csv).
#
# run_solutions.py is resumable (skips episodes with total_steps > 0 and only
# reruns the rest), so this is safe to point at every directory unconditionally
# — already-complete ones just report "Nothing to run" quickly.
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_cutoff_solutions_verification.sh
set -euo pipefail

PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"

for sol_dir in solutions/1_verification/*/; do
    sol_dir="${sol_dir%/}"
    dirname=$(basename "$sol_dir")

    n_solutions=$(ls "$sol_dir"/run_*_seed_*.json 2>/dev/null | wc -l)
    if [[ "$n_solutions" -eq 0 ]]; then
        echo "[skip] $dirname (no solution files)"
        continue
    fi

    echo "========================================"
    echo "  $dirname ($n_solutions solutions) — topping up"
    echo "========================================"
    "$PYTHON" run_solutions.py \
        solutions_dir="$sol_dir" \
        use_wandb=false \
        save_video=false \
        show_mpc_world_viewer=false

    echo "[done] $dirname"
done

echo ""
echo "All solutions/1_verification directories topped up."
