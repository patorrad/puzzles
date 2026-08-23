#!/usr/bin/env bash
set -euo pipefail

obs=5

COMMON="--phase collect --sim isaaclab --n_envs 64 --n_scenes 200 --n_simulations 200 --k_per_object 4 --stackable --n_z_levels 2 --bin_size 0.4 --difficult_spawn"

# 7 obstacles
for seed in 99 100 101 102 103 104 111 120 145 190 205; do
    python -m more.train $COMMON --n_obs $obs \
        --data outputs/5obs_stacked2_difficult/more_data_n5_bin04_stacked2_difficult/more_data_n5_bin04_stacked2_difficult_seed${seed}.pt \
        --seed $seed
done

python benchmark.py --config-name=benchmark planner=mcts  \
    wandb_run_name=mcts_random_stacker_${obs}obs_stacked2_difficult \
    csv_path=results/${obs}obs_stacked2_difficult \
    solutions_dir=solutions/mcts_${obs}obs_stacked2_difficult

python benchmark.py --config-name=benchmark planner=alphazero \
    planner.checkpoint=outputs/${obs}obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult/alphazero_latest.pt \
    wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult \
    csv_path=results/${obs}obs_stacked2_difficult \
    solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked2_difficult

python - <<EOF
import torch, glob, sys
pattern = "outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult_seed*.pt"
files = sorted(glob.glob(pattern))
if not files:
    sys.exit(f"No seed files found matching {pattern}")
combined = []
for f in files:
    combined.extend(torch.load(f, weights_only=False))
out = "outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult_combined.pt"
torch.save(combined, out)
print(f"Combined {len(files)} files -> {len(combined)} records -> {out}")
EOF

COMMON="--phase train --arch mlp --stackable --difficult_spawn "

python -m more.train $COMMON --n_obs $obs \
    --data outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult_combined.pt \
    --output outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/ppn_mlp1024_n${obs}_bin04_stacked2.pt

COMMON="--config-name=benchmark parallel_envs=160 planner=more"

python benchmark.py $COMMON \
    n_obstacles=5 \
    planner.ppn_checkpoint=outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/ppn_mlp1024_n${obs}_bin04_stacked2.pt \
    wandb_run_name=more_mlp1024_random_stacker_${obs}obs_stacked2_difficult \
    csv_path=results/${obs}obs_stacked2_difficult \
    solutions_dir=solutions/more_mlp1024_${obs}obs_stacked2_difficult