"""
generate_direct_push_solutions.py — write direct_push solution JSONs without
running the puzzle solver's benchmark harness at all.

The direct_push "plan" (see planner.DirectPushPlanner) needs no search — it's
always the same single pull_s action on the target. There's nothing to time
or verify at the puzzle-planner level, so this skips benchmark.py's wandb
init and per-run CSV/timing entirely, and just writes solution JSONs straight
into solutions_dir/run_XXX_seed_Y.json — the same layout run_all_solutions.sh
already expects. After this, run the normal MPC stage with:

    bash bash_scripts/run_all_solutions.sh

Usage (mirrors benchmark.py's CLI):
    python generate_direct_push_solutions.py --config-name=benchmark \\
        n_z_levels=2 stackable=true difficult_spawn=true bin_size=0.4 \\
        n_obstacles=5 solutions_dir=results/1_verification/direct_push_5obs_stacked2_difficult
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    if cfg.simulator.name == 'isaaclab' and cfg.viewer == 'headless':
        os.environ['ISAACLAB_HEADLESS'] = '1'

    from simulators import build_env
    from main import save_solution
    from planner import DirectPushPlanner

    print(f'Building environment ({cfg.simulator.name}): {cfg.n_obstacles} obstacle(s)')
    env = build_env(cfg, n_envs=cfg.parallel_envs, viewer_mode=cfg.viewer)
    planner = DirectPushPlanner(env=env, verify_threshold=cfg.verify_threshold,
                                n_verify_runs=cfg.n_verify_runs, seed=cfg.seed)

    solutions_dir = Path(cfg.solutions_dir)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    skip_runs = cfg.get('skip_runs', 0)
    for i in range(cfg.n_runs):
        if i < skip_runs:
            continue
        seed = cfg.base_seed + i
        initial_state = env.reset(seed=seed)
        plan = planner.plan(initial_state, verbose=False)

        path = solutions_dir / f'run_{i:03d}_seed_{seed}.json'
        save_solution(str(path), plan, initial_state, cfg, env)
        print(f'[{i + 1}/{cfg.n_runs}] seed={seed} -> {path}')

    print(f'\nWrote {cfg.n_runs - skip_runs} solutions to {solutions_dir}')
    print('Next: bash bash_scripts/run_all_solutions.sh')


if __name__ == '__main__':
    main()
