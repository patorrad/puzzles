#!/usr/bin/env bash
# Benchmark all three planners (MCTS, AlphaZero, MORE) across all stacked2-difficult
# obstacle counts: 5, 7, 10, 14.
#
# Run from the puzzle directory:
#   bash bash_scripts/benchmark_stacked2_all.sh
set -euo pipefail

BENCH="--config-name=benchmark n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4"

for obs in 14; do

    echo "========================================"
    echo "  n_obstacles=${obs} — AlphaZero"
    echo "========================================"
    python benchmark.py $BENCH planner=alphazero \
        n_obstacles=${obs} \
        planner.checkpoint=outputs/${obs}obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult/alphazero_latest.pt \
        wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult \
        csv_path=results/1_verification/${obs}obs_stacked2_difficult \
        solutions_dir=results/1_verification/alphazero_mlp1024_${obs}obs_stacked2_difficult \
        skip_runs=35

    echo "========================================"
    echo "  n_obstacles=${obs} — MORE"
    echo "========================================"
    python benchmark.py $BENCH parallel_envs=160 planner=more \
        n_obstacles=${obs} \
        planner.ppn_checkpoint=outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/ppn_mlp1024_n${obs}_bin04_stacked2.pt \
        wandb_run_name=more_mlp1024_random_stacker_${obs}obs_stacked2_difficult \
        csv_path=results/1_verification/${obs}obs_stacked2_difficult \
        solutions_dir=results/1_verification/more_mlp1024_${obs}obs_stacked2_difficult

done

echo "All benchmarks complete."
