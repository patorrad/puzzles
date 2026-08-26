#!/usr/bin/env bash
set -euo pipefail

# obs=7

# python benchmark.py --config-name=benchmark planner=mcts  \
#     wandb_run_name=mcts_random_stacker_${obs}obs_stacked2_difficult \
#     csv_path=results/${obs}obs_stacked2_difficult \
#     solutions_dir=solutions/mcts_${obs}obs_stacked2_difficult

# python benchmark.py --config-name=benchmark planner=alphazero \
#     planner.checkpoint=outputs/${obs}obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult/alphazero_latest.pt \
#     wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult \
#     csv_path=results/${obs}obs_stacked2_difficult \
#     solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked2_difficult

python benchmark.py --config-name=benchmark n_obstacles=5 n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
    planner.checkpoint=outputs/5obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_5obs_stacked3_diff_bottom/alphazero_latest.pt \
    wandb_run_name=n_sim_sol200_nn1024_random_stacker_5obs_stacked3_diff_bottom \
    csv_path=results/5obs_stacked3_diff_bottom \
    solutions_dir=solutions/alphazero_mlp1024_5obs_stacked3_diff_bottom

python benchmark.py --config-name=benchmark n_obstacles=7 n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
    planner.checkpoint=outputs/7obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_7obs_stacked3_diff_bottom/alphazero_latest.pt \
    wandb_run_name=n_sim_sol200_nn1024_random_stacker_7obs_stacked3_diff_bottom \
    csv_path=results/7obs_stacked3_diff_bottom \
    solutions_dir=solutions/alphazero_mlp1024_7obs_stacked3_diff_bottom