# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Physics-based bin-clearing planner: move a red target object out the south opening of a bin while keeping blue obstacles inside. A kinematic pusher blade sweeps objects in a 0.3×0.3 m bin. The planner treats the physics sim as a black-box forward model.

## Environment

Conda env: `genesistest2`. Requires Genesis 0.3.11, CUDA, PyTorch 2.9.1+cu130. Genesis simulator must be installed separately.

## Running

Configuration is managed via [Hydra](https://hydra.cc/) with config files in `conf/`. The entry point is `main.py`.

Always use the conda environemnt genesis_mpc for genesis runs, or isaaclab_mpc for isaaclab runs.

```bash
# Default: MCTS with Genesis, 2 obstacles, 8 parallel envs
python main.py

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
```

Key config defaults (`conf/config.yaml`): `parallel_envs=8`, `push_steps=160`, `wall_thickness=0.25`, `verify_threshold=0.75`.

## Architecture

### Simulator abstraction

`simulators/base_env.py` — `SimulatorEnv` ABC defines the interface all backends must implement:
- `get_state(env_idx)` / `set_state(state, env_idx)` — state is a dict with `target_pos`, `target_quat`, `obstacle_pos`, `obstacle_quat` (torch tensors)
- `execute_ns_push`, `execute_ns_pull`, `execute_ew_push` — single-env action primitives
- `batch_evaluate(pairs)` — parallel evaluation of `(state, action)` pairs; returns `(new_state, reward, done)` per pair
- `_compute_reward`, `_is_goal`, `_obstacles_dropped` — reward/termination logic

Simulator backends: `BinEnvGenesis` (default), `BinEnvIsaacGym`, `BinEnvIsaacLab`. Imports are lazy in `simulators/__init__.py` so missing dependencies don't cause import errors.

### Planners (`planner.py`)

Both planners use `env.batch_evaluate()` and work in single-env or parallel-env mode transparently.

**`MCTSPusher`** — Monte-Carlo Tree Search with UCB1 selection. Phases: select → expand → rollout → backprop. Uses virtual visits to handle parallel node expansion. Key params: `n_simulations`, `rollout_depth`, `max_depth`, `c_ucb`.

**`RRTPusher`** — RRT over push actions. Selects nodes weighted by reward, samples random actions, adds new nodes if they improve position. Supports Genesis debug-draw visualization during search.

Both planners call `_verify_plan()` when a goal is found, re-running the full plan `n_tries` times in parallel to confirm robustness before returning.

### Action space

Actions are dicts: `{action_type, obj_idx, push_pos, push_z}`.
- `action_type`: `push_n`, `pull_s`, `push_e`, `push_w`
- `obj_idx`: 0 = target, 1..N = obstacles
- Sampling is biased: target gets higher probability; `pull_s`/`push_n` get higher weight (45% each) vs east/west (5% each)

### Hydra config structure

```
conf/
  config.yaml         # root defaults + runtime flags
  simulator/          # genesis.yaml, isaaclab.yaml, isaacgym.yaml
  planner/            # mcts.yaml, rrt.yaml
  reward/             # default.yaml (target_progress, obstacle_penalty, path_blocker)
  benchmark.yaml
  many_objects.yaml   # preset for more obstacles
```

### Replay

For Genesis/IsaacGym, replay is launched as a subprocess (Genesis reinitializes its CUDA context). For IsaacLab, `env.replay()` is called directly. The `replay_file` config key is used internally by the subprocess mechanism.

### Value function training (`train_value.py`)

Runs MCTS over many random initial states, harvests `(state, mean_reward)` pairs from all MCTS nodes, trains a small MLP on normalized state vectors. Outputs `value_net.pt` and `dataset.npz`. Currently not working.
