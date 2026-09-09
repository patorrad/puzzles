#!/usr/bin/env bash
set -euo pipefail

COMMON="--phase train --arch mlp --stackable --difficult_spawn "

# for obs in 5 14; do
#     out_dir="outputs/${obs}obs_stacked3_diff_bottom/more_data_n${obs}_bin04_stacked3_diff_bottom"
#     python -m more.train $COMMON --n_obs $obs \
#     --data ${out_dir}/more_data_n${obs}_bin04_stacked3_diff_bottom_combined.pt \
#     --output ${out_dir}/ppn_mlp1024_n${obs}_bin04_stacked3.pt
# done

# ---------------------------------------------------------------------------
# Benchmark campaign "1_verification" — MORE and MCTS, 5/7/10/14 obstacles.
# ---------------------------------------------------------------------------

# mkdir -p results/1_verification solutions/1_verification

benchmark_more () {
    local obs=$1
    local ppn_ckpt="outputs/${obs}obs_stacked3_diff_bottom/more_data_n${obs}_bin04_stacked3_diff_bottom/ppn_mlp1024_n${obs}_bin04_stacked3.pt"

    python benchmark.py --config-name=benchmark parallel_envs=160 planner=more \
        n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true \
        planner.ppn_checkpoint="$ppn_ckpt" \
        wandb_run_name=more_mlp1024_${obs}obs_stacked3_diff_bottom_1_verification \
        csv_path=results/1_verification/${obs}obs_stacked3_diff_bottom \
        solutions_dir=solutions/1_verification/more_mlp1024_${obs}obs_stacked3_diff_bottom
}

# benchmark_mcts () {
#     local obs=$1

#     python benchmark.py --config-name=benchmark n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true planner=mcts \
#         wandb_run_name=mcts_random_stacker_${obs}obs_stacked3_diff_bottom_1_verification \
#         csv_path=results/1_verification/${obs}obs_stacked3_diff_bottom \
#         solutions_dir=solutions/1_verification/mcts_${obs}obs_stacked3_diff_bottom
# }

for obs in 14; do
    benchmark_more "$obs"
    # benchmark_mcts "$obs"
done

# ---------------------------------------------------------------------------
# Benchmark AlphaZero — iter_0100 checkpoints, 5/7/10 obstacles.
# ---------------------------------------------------------------------------

# benchmark_alphazero () {
#     local obs=$1
#     local checkpoint=$2

#     python benchmark.py --config-name=benchmark n_obstacles="$obs" n_z_levels=3 force_obstacle_on_target=true planner=alphazero \
#         planner.checkpoint="$checkpoint" \
#         wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom_iter100 \
#         csv_path=results/1_verification/${obs}obs_stacked3_diff_bottom \
#         solutions_dir=solutions/1_verification/alphazero_mlp1024_${obs}obs_stacked3_diff_bottom_iter100
# }

# benchmark_alphazero 5 \
#     outputs/5obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_5obs_stacked3_diff_bottom/alphazero_iter_0100.pt

# benchmark_alphazero 7 \
#     outputs/7obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_7obs_stacked3_diff_bottom/alphazero_iter_0100.pt

# benchmark_alphazero 10 \
#     outputs/10obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_10obs_stacked3_diff_bottom/alphazero_iter_0100.pt

