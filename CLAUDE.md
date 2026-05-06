# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Physics-based bin-clearing planner: move a red target object out the south opening of a bin while keeping blue obstacles inside. A kinematic pusher blade sweeps objects in a 0.3×0.3 m bin. The planner treats the physics sim as a black-box forward model.

## Environment

Conda env: `genesistest2`. Requires Genesis 0.3.11, CUDA, PyTorch 2.9.1+cu130. Genesis simulator must be installed separately.

Always use the conda environment `genesis_mpc` for genesis runs, or `isaaclab_mpc` for isaaclab runs.

## Running

Configuration is managed via [Hydra](https://hydra.cc/) with config files in `conf/`. The entry point is `main.py`.

**Default focus: IsaacLab simulator + MCTS planner.** Unless otherwise specified, assume `simulator=isaaclab` and `planner=mcts` for all suggestions, examples, and code changes.

```bash
# Default: MCTS with IsaacLab, 2 obstacles, 8 parallel envs
python main.py simulator=isaaclab

# Switch simulator or planner
python main.py simulator=isaaclab
python main.py planner=rrt

# Override values
python main.py n_obstacles=3 seed=42 parallel_envs=16

# Show viewer during planning
python main.py show_during_planning=true

# Save plan to file
python main.py save=solution.json no_replay=true

# Visualize search tree (matplotlib)
python main.py visualize=true

# Multi-z-level (stacked objects)
python main.py n_z_levels=2 target_z_level=1
```

Key config defaults (`conf/config.yaml`): `parallel_envs=8`, `push_steps=128`, `wall_thickness=0.25`, `verify_threshold=0.75`, `n_z_levels=1`, `target_z_level=null`, `bin_size_factor=0.9`.

### Benchmark

`benchmark.py` is a multi-run evaluation harness with WandB integration. It runs `n_runs` independent planning attempts and aggregates success rate, plan length, sim call counts, timing, and reward breakdown.

```bash
# 20-run benchmark: IsaacLab, 20 obstacles, WandB logging
python benchmark.py --config-name=benchmark

# Stacked scenario: 10 obstacles, target forced to z-level 1
python benchmark.py --config-name=stacked_benchmark
```

IsaacLab runs support video recording via `env.record_replay()`.

## Architecture

### Simulator abstraction

`simulators/base_env.py` — `SimulatorEnv` ABC defines the interface all backends must implement:
- `get_state(env_idx)` / `set_state(state, env_idx)` — state is a dict with `target_pos`, `target_quat`, `obstacle_pos`, `obstacle_quat` (torch tensors)
- `execute_ns_push`, `execute_ns_pull`, `execute_ew_push` — single-env action primitives
- `batch_evaluate(pairs)` — parallel evaluation of `(state, action)` pairs; returns `(new_state, reward, done)` per pair
- `_compute_reward`, `_is_goal`, `_obstacles_dropped` — reward/termination logic

Simulator backends: `BinEnvIsaacLab` (default focus), `BinEnvGenesis`, `BinEnvIsaacGym`. Imports are lazy in `simulators/__init__.py` so missing dependencies don't cause import errors.

`simulators/placement.py` — Shared, simulator-agnostic initial-state generation (pure PyTorch/NumPy). `random_initial_state()` places objects with proximity-based column stacking: if a sampled position lands within `OBJ_SIZE × 1.05` of an existing column it stacks on that column instead of retrying. `target_z_level` controls target height (`null` = random from occupied levels, `0` = floor, `1+` = stacked on an obstacle column).

### Planners (`planner.py`)

Both planners use `env.batch_evaluate()` and work in single-env or parallel-env mode transparently.

**`MCTSPusher`** — Monte-Carlo Tree Search with UCB1 selection. Phases: select → expand → rollout → backprop. Uses virtual visits to handle parallel node expansion. Key params: `n_simulations=2000`, `rollout_depth=5`, `max_depth=10`, `c_ucb`.

**`RRTPusher`** — RRT over push actions. Selects nodes weighted by reward, samples random actions, adds new nodes if they improve position. Supports Genesis debug-draw visualization during search. Key params: `max_iter=150`, `max_depth=12`.

Both planners call `_verify_plan()` when a goal is found, re-running the full plan `n_tries` times in parallel to confirm robustness before returning.

### Action space

Actions are dicts: `{action_type, obj_idx, push_pos, push_z}`.
- `action_type`: `push_n`, `pull_s`, `push_e`, `push_w`
- `obj_idx`: 0 = target, 1..N = obstacles
- `push_z`: discrete height sampled from `env.z_levels` (e.g. `[OBJ_H, OBJ_H + OBJ_SIZE]` for 2 levels)
- Sampling is biased: target gets higher probability; `pull_s`/`push_n` get higher weight (45% each) vs east/west (5% each)

### Hydra config structure

```
conf/
  config.yaml               # root defaults + runtime flags
  simulator/                # genesis.yaml, isaaclab.yaml, isaacgym.yaml
  planner/                  # mcts.yaml, rrt.yaml
  reward/                   # default.yaml (target_progress, obstacle_penalty=0.5, path_blocker=0.5)
  benchmark.yaml            # 20-run preset: IsaacLab, 20 obstacles, WandB
  stacked_benchmark.yaml    # stacked preset: 10 obstacles, 2 z-levels, target_z_level=1
  many_objects.yaml         # preset for more obstacles
```

IsaacLab-specific params in `conf/simulator/isaaclab.yaml`: `force_threshold` (N, stops pusher early on contact; 0 to disable), `position_iterations`, `velocity_iterations` (PhysX solver quality).

### Replay

For Genesis/IsaacGym, replay is launched as a subprocess (Genesis reinitializes its CUDA context). For IsaacLab, `env.replay()` is called directly. The `replay_file` config key is used internally by the subprocess mechanism.

### Tests

- `test_placement.py` — unit tests for `simulators/placement.py` (no simulator required); covers `target_z_level` values 0/1/null and placement invariants (no column collisions, within bin bounds)
- `test_rewards.py` — reward logic tests

### Value function training (`train_value.py`)

Runs MCTS over many random initial states, harvests `(state, mean_reward)` pairs from all MCTS nodes, trains a small MLP on normalized state vectors. Outputs `value_net.pt` and `dataset.npz`. Currently not working.
