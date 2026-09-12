#!/usr/bin/env bash
# Rerun the (now fixed) MCTS baseline for stacked3_diff_bottom across all
# obstacle counts: 5, 7, 10, 14.
#
# Writes into a fresh directory (results/1_verification_stacked3) rather than
# results/1_verification, since that already has 100 rows/obstacle count of
# OLD pre-fix MCTS data (target_prob=0.1 + unverified-fallback bug) for
# stacked3 that would otherwise get mixed with the new, correct runs.
#
# Uses the stacked3_diff_bottom scenario config (n_z_levels=3 stackable=true
# difficult_spawn=true bin_size=0.4 force_obstacle_on_target=true), verified
# against the wandb config of the original stacked3 MCTS run (6m3o6n74) that
# produced the existing results/1_verification data.
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_mcts_stacked3.sh
set -euo pipefail

for obs in 10 14; do

    echo "========================================"
    echo "  n_obstacles=${obs} — MCTS stacked3_diff_bottom (rerun after fix)"
    echo "========================================"
    python benchmark.py --config-name=benchmark n_z_levels=3 stackable=true difficult_spawn=true bin_size=0.4 force_obstacle_on_target=true planner=mcts \
        n_obstacles=${obs} \
        wandb_run_name=mcts_random_stacker_${obs}obs_stacked3_diff_bottom \
        csv_path=results/1_verification_stacked3/${obs}obs_stacked3_diff_bottom \
        solutions_dir=results/1_verification_stacked3/mcts_${obs}obs_stacked3_diff_bottom

done

echo "MCTS stacked3_diff_bottom reruns complete -> results/1_verification_stacked3"
