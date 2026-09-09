#!/usr/bin/env bash
# Benchmark AlphaZero at 14obs_stacked3_diff_bottom — the one (obs, variant)
# combination missing from results/1_verification because no checkpoint had
# been trained for it. Checkpoint now lives at the same path convention as
# the 5/7/10obs stacked3 checkpoints:
#   outputs/14obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_14obs_stacked3_diff_bottom/alphazero_latest.pt
#
# Mirrors bash_scripts/benchmark_alphazero_checkpoints.sh's stacked3 config
# (n_z_levels=3 force_obstacle_on_target=true) and the existing 5/7/10obs
# stacked3 solutions_dir naming convention.
#
# Run from the puzzle directory:
#   bash bash_scripts/benchmark_alphazero_14obs_stacked3.sh
#
# This copy starts at the halfway-point seed (skip_runs=50 of n_runs=100) so
# it can run on a second machine in parallel with the first half elsewhere,
# writing into its own local results/solutions dirs.
set -euo pipefail

CHECKPOINT="outputs/14obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_14obs_stacked3_diff_bottom/alphazero_latest.pt"

python benchmark.py --config-name=benchmark n_obstacles=14 n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
    planner.checkpoint="$CHECKPOINT" \
    skip_runs=50 \
    wandb_run_name=n_sim_sol200_nn1024_random_stacker_14obs_stacked3_diff_bottom_part2 \
    csv_path=results/1_verification/14obs_stacked3_diff_bottom \
    solutions_dir=solutions/1_verification/alphazero_mlp1024_14obs_stacked3_diff_bottom_iter100

echo "Done. Solutions written to solutions/1_verification/alphazero_mlp1024_14obs_stacked3_diff_bottom_iter100"
