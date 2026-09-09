#!/usr/bin/env bash
# Rerun the MCTS baseline (planner search stage only) across all
# stacked2-difficult obstacle counts: 5, 7, 10, 14.
#
# Needed because the previous MCTS runs reflect two bugs since fixed:
#   - conf/planner/mcts.yaml target_prob was 0.1 (90% of actions pushed
#     obstacles instead of the target) — now 0.6, matching every other planner.
#   - MCTSPusher.plan()'s end-of-budget fallback returned an unverified
#     "best effort" path instead of reporting failure — now returns None,
#     matching AlphaZero's behavior.
#
# Appends into the same per-obs CSVs and mcts_*obs_stacked2_difficult
# solutions_dir as before — run on a separate machine/checkout, so no
# cleanup of old rows is done here.
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_mcts_baseline.sh
set -euo pipefail

BENCH="--config-name=benchmark n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4"

for obs in 5 7 10 14; do

    echo "========================================"
    echo "  n_obstacles=${obs} — MCTS (rerun after target_prob/fallback fix)"
    echo "========================================"
    python benchmark.py $BENCH planner=mcts \
        n_obstacles=${obs} \
        wandb_run_name=mcts_random_stacker_${obs}obs_stacked2_difficult \
        csv_path=results/1_verification/${obs}obs_stacked2_difficult \
        solutions_dir=results/1_verification/mcts_${obs}obs_stacked2_difficult

done

echo "All MCTS baseline reruns complete."
