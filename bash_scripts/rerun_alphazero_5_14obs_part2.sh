#!/usr/bin/env bash
# Continue the AlphaZero 5obs/14obs stacked2_difficult rerun for the second
# half of seeds (50-99) — the first run only completed seeds 0-49 because
# conf/benchmark.yaml's n_runs was set to 50 instead of 100 at the time.
#
# Passes n_runs=100 skip_runs=50 explicitly on the command line (rather than
# fixing conf/benchmark.yaml) so this doesn't depend on — or change — that
# shared default, and doesn't redo the already-completed seeds 0-49.
#
# Run from the puzzle directory:
#   bash bash_scripts/rerun_alphazero_5_14obs_part2.sh
set -euo pipefail

BENCH="--config-name=benchmark n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4"

for obs in 5 14; do

    echo "========================================"
    echo "  n_obstacles=${obs} — AlphaZero (seeds 50-99)"
    echo "========================================"
    python benchmark.py $BENCH planner=alphazero \
        n_obstacles=${obs} \
        n_runs=100 skip_runs=50 \
        planner.checkpoint=outputs/${obs}obs_stacked2_difficult/n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult/alphazero_latest.pt \
        wandb_run_name=n_sim_sol200_nn1024_random_stacker_${obs}obs_stacked2_difficult_part2 \
        csv_path=results/1_verification/${obs}obs_stacked2_difficult \
        solutions_dir=results/1_verification/alphazero_mlp1024_${obs}obs_stacked2_difficult

done

echo "AlphaZero 5obs/14obs seeds 50-99 complete."
