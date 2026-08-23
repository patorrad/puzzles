#!/usr/bin/env bash
set -euo pipefail

COMMON="--phase train --arch mlp --stackable --difficult_spawn "

for obs in 7 10; do
    python -m more.train $COMMON --n_obs $obs \
    --data outputs/${obs}obs_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult/more_data_n${obs}_bin04_stacked2_difficult_combined.pt \
    --output ppn_mlp1024_n${obs}_bin04_stacked2.pt
done

