#!/usr/bin/env bash
set -euo pipefail

COMMON="--config-name=benchmark parallel_envs=160 planner=more"

# python benchmark.py $COMMON \
#     n_obstacles=10 \
#     planner.ppn_checkpoint=outputs/10obs_stacked2_difficult/more_data_n10_bin04_stacked2_difficult/ppn_deepsets_n10_bin04_stacked2.pt \
#     wandb_run_name=more_deepsets_random_stacker_10obs_stacked2_difficult \
#     csv_path=results/10obs_stacked2_difficult \
#     solutions_dir=solutions/more_deepsets_10obs_stacked2_difficult

# python benchmark.py $COMMON \
#     n_obstacles=7 \
#     planner.ppn_checkpoint=outputs/7obs_stacked2_difficult/more_data_n7_bin04_stacked2_difficult/ppn_deepsets_n7_bin04_stacked2.pt \
#     wandb_run_name=more_deepsets_random_stacker_7obs_stacked2_difficult \
#     csv_path=results/7obs_stacked2_difficult \
#     solutions_dir=solutions/more_deepsets_7obs_stacked2_difficult

# python benchmark.py $COMMON \
#     n_obstacles=14 \
#     planner.ppn_checkpoint=outputs/14obs_stacked2_difficult/more_data_n14_bin04_stacked2_difficult/ppn_mlp1024_n14_bin04_stacked2.pt \
#     wandb_run_name=more_deepsets_random_stacker_14obs_stacked2_difficult_no_zeros \
#     csv_path=results/14obs_stacked2_difficult \
#     solutions_dir=solutions/more_mlp1024_14obs_stacked2_difficult_no_zeros \
#     collect_data_dir=outputs/14obs_stacked2_difficult/benchmark_data_no_zeros

python benchmark.py $COMMON \
    n_obstacles=7 \
    planner.ppn_checkpoint=outputs/7obs_stacked2_difficult/more_data_n7_bin04_stacked2_difficult/ppn_mlp1024_n7_bin04_stacked2.pt \
    wandb_run_name=more_1024mlp_random_stacker_7obs_stacked2_difficult_no_zeros \
    csv_path=results/7obs_stacked2_difficult \
    solutions_dir=solutions/more_mlp1024_7obs_stacked2_difficult_no_zeros \
    collect_data_dir=outputs/7obs_stacked2_difficult/benchmark_data_no_zeros

python benchmark.py $COMMON \
    n_obstacles=10 \
    planner.ppn_checkpoint=outputs/10obs_stacked2_difficult/more_data_n10_bin04_stacked2_difficult/ppn_mlp1024_n10_bin04_stacked2.pt \
    wandb_run_name=more_1024mlp_random_stacker_10obs_stacked2_difficult_no_zeros \
    csv_path=results/10obs_stacked2_difficult \
    solutions_dir=solutions/more_mlp1024_10obs_stacked2_difficult_no_zeros \
    collect_data_dir=outputs/10obs_stacked2_difficult/benchmark_data_no_zeros

# python benchmark.py $COMMON \
#     planner.ppn_checkpoint=null \
#     wandb_run_name=more_no_ppn_14obs_stacked2_difficult \
#     csv_path=results/14obs_stacked2_difficult \
#     solutions_dir=solutions/more_no_ppn_14obs_stacked2_difficult

# python benchmark.py --config-name=benchmark planner=mcts  \
#     wandb_run_name=mcts_random_stacker_14obs_stacked2_difficult \
#     csv_path=results/14obs_stacked2_difficult solutions_dir=solutions/mcts_14obs_stacked2_difficult

# python benchmark.py --config-name=benchmark planner=alphazero \
#     planner.checkpoint=outputs/14obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_14obs_stacked2_difficult/alphazero_latest.pt \
#     wandb_run_name=n_sim_sol200_nn1024_random_stacker_14obs_stacked2_difficult \
#     csv_path=results/14obs_stacked2_difficult \
#     solutions_dir=solutions/alphazero_mlp1024_14obs_stacked2_difficult

# python benchmark.py $COMMON n_obstacles=10 \
#     planner.ppn_checkpoint=null \
#     wandb_run_name=more_no_ppn_10obs_stacked2_difficult \
#     csv_path=results/10obs_stacked2_difficult \
#     solutions_dir=solutions/more_no_ppn_10obs_stacked2_difficult
