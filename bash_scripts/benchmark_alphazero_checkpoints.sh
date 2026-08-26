#!/usr/bin/env bash
set -euo pipefail

# Benchmark AlphaZero at 5 and 7 obstacles across training progress, using the
# checkpoints saved every 10 iterations (checkpoint_every=10). "75" isn't a
# saved checkpoint, so iter_0070 stands in for the mid-training snapshot.

benchmark_alphazero_checkpoint () {
    local obs=$1
    local iter=$2

    local ckpt_dir="outputs/${obs}obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom"
    local checkpoint="${ckpt_dir}/alphazero_iter_$(printf '%04d' "$iter").pt"

    python benchmark.py --config-name=benchmark n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
        planner.checkpoint="$checkpoint" \
        wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom_iter${iter} \
        csv_path=results/${obs}obs_stacked3_diff_bottom_iter${iter} \
        solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked3_diff_bottom_iter${iter}
}

for obs in 5 7; do
    for iter in 50 70 100; do
        benchmark_alphazero_checkpoint "$obs" "$iter"
    done
done
