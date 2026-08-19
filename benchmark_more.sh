#!/usr/bin/env bash
set -euo pipefail

COMMON="--config-name=benchmark parallel_envs=160 planner=more"

python benchmark.py $COMMON \
    n_obstacles=10 \
    planner.ppn_checkpoint=outputs/10obs_stacked2_difficult/more_data_n10_bin04_stacked2_difficult/ppn_deepsets_n10_bin04_stacked2.pt \
    wandb_run_name=more_deepsets_random_stacker_10obs_stacked2_difficult \
    csv_path=results/10obs_stacked2_difficult \
    solutions_dir=solutions/more_deepsets_10obs_stacked2_difficult

python benchmark.py $COMMON \
    n_obstacles=7 \
    planner.ppn_checkpoint=outputs/7obs_stacked2_difficult/more_data_n7_bin04_stacked2_difficult/ppn_deepsets_n7_bin04_stacked2.pt \
    wandb_run_name=more_deepsets_random_stacker_7obs_stacked2_difficult \
    csv_path=results/7obs_stacked2_difficult \
    solutions_dir=solutions/more_deepsets_7obs_stacked2_difficult

python benchmark.py $COMMON \
    n_obstacles=14 \
    planner.ppn_checkpoint=outputs/14obs_stacked2_difficult/more_data_n14_bin04_stacked2_difficult/ppn_deepsets_n14_bin04_stacked2.pt \
    wandb_run_name=more_deepsets_random_stacker_14obs_stacked2_difficult \
    csv_path=results/14obs_stacked2_difficult \
    solutions_dir=solutions/more_deepsets_14obs_stacked2_difficult
