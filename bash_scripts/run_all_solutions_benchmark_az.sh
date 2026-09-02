#!/usr/bin/env bash
set -euo pipefail

obs=5
SOLUTIONS_ROOT="/home/paolo/Documents/puzzle/solutions"
PUZZLE_DIR="/home/paolo/Documents/puzzle"
PYTHON="/home/paolo/miniconda3/envs/env_isaaclab/bin/python"

cd "$PUZZLE_DIR"

# for iter in 0050 0070 0100; do
#     python benchmark.py --config-name=benchmark planner=alphazero \
#         planner.checkpoint=outputs/${obs}obs_stacked3_diff_bottom/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom/alphazero_iter_${iter}.pt \
#         wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom_good${iter}_1 \
#         csv_path=results/${obs}obs_stacked3_diff_bottom_good${iter}_1 \
#         solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked3_diff_bottom_good${iter}_1
# done

for iter in 0050; do
    python benchmark.py --config-name=benchmark planner=alphazero \
        planner.checkpoint=outputs/14obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_14obs_stacked2_difficult/alphazero_iter_${iter}.pt \
        wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked3_diff_bottom_good${iter}_1 \
        csv_path=results/${obs}obs_stacked3_diff_bottom_good${iter}_1 \
        solutions_dir=solutions/alphazero_mlp1024_${obs}obs_stacked3_diff_bottom_good${iter}_1
done

# for sol_dir in "$SOLUTIONS_ROOT"/*/; do
#     dirname=$(basename "$sol_dir")
#     [[ "$dirname" == "archive" ]] && continue

#     # Only process MCTS solution directories.
#     [[ "$dirname" != *mcts* ]] && continue

#     # Preserve previous results for comparison.
#     if [[ -f "$sol_dir/results.csv" ]]; then
#         mv "$sol_dir/results.csv" "$sol_dir/results_previous.csv"
#         echo "[backup] $dirname → results_previous.csv"
#     fi

#     echo ""
#     echo "=========================================="
#     echo "[run] $dirname"
#     echo "=========================================="

#     "$PYTHON" run_solutions.py \
#         solutions_dir="$sol_dir" \
#         use_wandb=false \
#         save_video=false

#     echo "[done] $dirname"
# done

# echo ""
# echo "All MCTS solution directories processed."
