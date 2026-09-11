#!/usr/bin/env bash
# Top up every results.csv in results/1_verification that has episodes cut off
# by the old isaaclabmpc_timeout_s global-timeout bug (rows with total_steps=0,
# meaning they were queued but never actually attempted before the batch got
# killed). run_solutions.py now skips already-attempted episodes (total_steps
# > 0) and only reruns the cut-off ones, so this is safe to point at any
# directory — already-complete ones just print "Nothing to run".
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_cutoff_episodes.sh
set -euo pipefail

PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"

for csv_path in results/1_verification/*/results.csv; do
    sol_dir=$(dirname "$csv_path")
    dirname=$(basename "$sol_dir")

    n_cutoff=$(python3 -c "
import csv
rows = list(csv.DictReader(open('$csv_path')))
print(sum(1 for r in rows if r.get('total_steps') == '0'))
")

    if [[ "$n_cutoff" -eq 0 ]]; then
        echo "[skip] $dirname (no cut-off episodes)"
        continue
    fi

    echo "========================================"
    echo "  $dirname: $n_cutoff cut-off episode(s) — rerunning"
    echo "========================================"
    "$PYTHON" run_solutions.py \
        solutions_dir="$sol_dir" \
        use_wandb=false \
        save_video=false \
        show_mpc_world_viewer=false

    echo "[done] $dirname"
done

echo ""
echo "All cut-off episodes across results/1_verification processed."
