#!/usr/bin/env bash
set -euo pipefail

# Train a MORE PPN and benchmark it, for both the 7-obstacle and 10-obstacle
# stacked3_diff_bottom datasets. Data was already collected + combined
# (see combine_more_data.py), so this script only does phase=train + benchmark.

train_and_benchmark () {
    local obs=$1
    local data=$2

    local out_dir="outputs/${obs}obs_stacked3_diff_bottom/more_data_n${obs}_bin04_stacked3_diff_bottom"
    local ppn_ckpt="${out_dir}/ppn_mlp1024_n${obs}_bin04_stacked3.pt"

    # python -m more.train --phase train --arch mlp --n_obs "$obs" \
    #     --data "$data" \
    #     --output "$ppn_ckpt"

    python benchmark.py --config-name=benchmark parallel_envs=160 planner=more \
        n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true \
        planner.ppn_checkpoint="$ppn_ckpt" \
        wandb_run_name=more_mlp1024_${obs}obs_stacked3_diff_bottom_iterations \
        csv_path=results/${obs}obs_stacked3_diff_bottom_iterations \
        solutions_dir=solutions/more_mlp1024_${obs}obs_stacked3_diff_bottom_iterations
}

train_and_benchmark 7 \
    outputs/7obs_stacked3_diff_bottom/more_data_n7_bin04_stacked3_diff_bottom/more_data_n7_bin04_stacked3_diff_bottom_combined.pt

train_and_benchmark 10 \
    outputs/10obs_stacked3_diff_bottom/more_data_n10_bin04_stacked3_diff_bottom/more_data_n10_bin04_stacked3_diff_bottom_combined.pt

# benchmark_alphazero () {
#     local obs=$1
#     local checkpoint=$2

#     python benchmark.py --config-name=benchmark n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
#         planner.checkpoint="$checkpoint" \
#         wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom \
#         csv_path=results/${obs}obs_stacked3_diff_bottom \
#         solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked3_diff_bottom
# }

# benchmark_alphazero 7 \
#     outputs/7obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_7obs_stacked3_diff_bottom/alphazero_latest.pt

# benchmark_alphazero 5 \
#     outputs/5obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_5obs_stacked3_diff_bottom/alphazero_latest.pt

# benchmark_mcts () {
#     local obs=$1

#     python benchmark.py --config-name=benchmark n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true planner=mcts \
#         wandb_run_name=mcts_random_stacker_${obs}obs_stacked3_diff_bottom \
#         csv_path=results/${obs}obs_stacked3_diff_bottom_iterations \
#         solutions_dir=solutions/mcts_${obs}obs_stacked3_diff_bottom_iterations
# }

# for obs in 5 7 10 14; do
#     benchmark_mcts "$obs"
# done
