#!/usr/bin/env bash
set -euo pipefail

COMMON="--phase collect --sim isaaclab --n_envs 64 --n_scenes 200 --n_simulations 200 --k_per_object 4 --stackable --n_z_levels 2 --bin_size 0.4 --difficult_spawn"

# 7 obstacles
for seed in 111 120 145 190 205; do
    python -m more.train $COMMON --n_obs 7 \
        --data outputs/7obs_stacked2_difficult/more_data_n7_bin04_stacked2_difficult/more_data_n7_bin04_stacked2_difficult_seed${seed}.pt \
        --seed $seed
done

# 10 obstacles
for seed in 111 120 145 190 205; do
    python -m more.train $COMMON --n_obs 10 \
        --data outputs/10obs_stacked2_difficult/more_data_n10_bin04_stacked2_difficult/more_data_n10_bin04_stacked2_difficult_seed${seed}.pt \
        --seed $seed
done

# 14 obstacles
for seed in 111 120 145 190 205; do
    python -m more.train $COMMON --n_obs 14 \
        --data outputs/14obs_stacked2_difficult/more_data_n14_bin04_stacked2_difficult/more_data_n14_bin04_stacked2_difficult_seed${seed}.pt \
        --seed $seed
done


