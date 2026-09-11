#!/usr/bin/env bash
# Rerun the AlphaZero planner-search stage (benchmark.py only, no MPC) for
# 5 and 14 obstacles, stacked2_difficult — the two cases whose flat CSV rows
# couldn't be cleanly recovered from wandb after being accidentally overwritten
# (5obs: no single wandb run covered all 100 seeds without a success-count
# mismatch; 14obs: coverage was split across multiple crashed/resumed runs).
#
# NOTE: this overwrites solutions_dir's solution JSONs with a fresh run.
# AlphaZero's search isn't perfectly reproducible, so the new solutions may
# not exactly match the ones the existing MPC results.csv was computed
# against — this script intentionally does NOT rerun the MPC stage, so that
# results.csv is left as-is (possibly stale) until you decide to redo it.
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_alphazero_5_14obs.sh
set -euo pipefail

BENCH="--config-name=benchmark n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4"

for obs in 5 14; do

    echo "========================================"
    echo "  n_obstacles=${obs} — AlphaZero (rerun after overwrite)"
    echo "========================================"
    python benchmark.py $BENCH planner=alphazero \
        n_obstacles=${obs} \
        planner.checkpoint=outputs/${obs}obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult/alphazero_latest.pt \
        wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult \
        csv_path=results/1_verification/${obs}obs_stacked2_difficult \
        solutions_dir=results/1_verification/alphazero_mlp1024_${obs}obs_stacked2_difficult

done

echo "AlphaZero 5obs/14obs reruns complete."
