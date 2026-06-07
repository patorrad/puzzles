# Puzzle — Bin-Clearing Planning with Genesis

A physics-based planning environment for the bin-clearing task: move a target
object out of a bin through an open south side, while keeping obstacles inside.
Built on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 0.3.11.

## Task

```
  ┌─────────────┐  ← north wall
  │   [obs0]    │
  │      [tgt]  │
  │   [obs1]    │
  └──────   ────┘  ← open south side (EXIT_Y = -0.05)
```

A kinematic pusher blade sweeps objects inside a 0.3 × 0.3 m bin. The planner
must move the **red target** out the south opening without knocking any
**blue obstacle** out.

## Setup

Genesis 0.3.11, CUDA, PyTorch 2.9.1+cu130

## Files

| File | Description |
|---|---|
| [env.py](env.py) | `BinEnv` — single Genesis scene, kinematic NS+EW pusher blades |
| [planner.py](planner.py) | `MCTSPusher`, `RRTPusher`, `ParallelMCTSPusher`, `ParallelRRTPusher` |
| [planner_ighastar.py](planner_ighastar.py) | `IGHAStarPusher` — IGHA* search planner (uses the generic IGHA* env) |
| [ighastar_bridge.py](ighastar_bridge.py) | `BinIGHAStarBridge` — adapts `SimulatorEnv` to IGHA*'s generic-env callbacks |
| [main.py](main.py) | CLI entry point — plan, replay, save solution |
| [apply_solution.py](apply_solution.py) | Convert `solution.json` → genesismpc actor YAMLs + config |
| [train_value.py](train_value.py) | Collect MCTS tree data and train an MLP value function | Not working at the moment
| [viz.py](viz.py) | Matplotlib tree visualisation for MCTS / RRT

## Running the planner

```bash
# MCTS (headless planning, then viewer replay)
python main.py --planner mcts

# RRT
python main.py --planner rrt --max-iter 200

# More obstacles
python main.py --planner mcts --n-obstacles 3 --n-simulations 200

# Parallel MCTS — all n_envs rollouts evaluated in one GPU kernel
python main.py --planner mcts --parallel-envs 8 --n-simulations 200

# Save solution for MPPI simulator
python main.py --planner mcts --save solution.json --no-replay

# Visualise the search tree after planning (matplotlib 2D top view)
python main.py --planner rrt --visualize
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--planner` | `mcts` | `mcts` or `rrt` |
| `--n-obstacles` | `2` | Number of obstacle objects |
| `--n-simulations` | `80` | MCTS: simulations per plan call |
| `--max-iter` | `150` | RRT: tree expansion iterations |
| `--parallel-envs N` | `0` | GPU-parallel environments (0 = off) |
| `--friction` | `1.0` | Object/pusher friction coefficient |
| `--n-z-levels` | `1` | Discrete push-height levels |
| `--push-steps` | `80` | Pusher sweep steps per action |
| `--seed` | `None` | Random seed for reproducibility |
| `--save FILE` | — | Export solution + scene to JSON |
| `--no-replay` | — | Skip viewer replay after planning |

## IGHA* planner

`IGHAStarPusher` is a third planner (alongside `mcts` / `rrt`) that runs the
**IGHA\*** search algorithm over the discretised bin state, using the simulator
(`SimulatorEnv.batch_evaluate`) as a black-box forward model — so it works with
any backend (`isaaclab`, `genesis`, `isaacgym`) and produces the same
`list[dict]` action format (verify / replay / save all work unchanged).

Each node carries the **full object pose losslessly** — `xy + z + quaternion`
for every object (`N_DIMS = 7*(1+n_obstacles)`) — so the search never fabricates
or resets a node's true state. IGHA\* only **grids/dedups a leading subspace**
(`HASH_DIMS`): by default `grid_z=true` grids each object's `[x, y, z]`
(`HASH_DIMS = 3*(1+n_obstacles)`); set `grid_z=false` to grid `[x, y]` only
(`2*(1+n_obstacles)`). The orientation (quaternion) always rides along for the
dynamics (`set_state`) without being discretised. Controls are discrete
macro-actions `(action_type, obj_idx, z_level)`. The goal is the same as the
other planners: target out the open −y face (`target_y <= EXIT_Y`) with no
obstacle dropped.

### Setup (one-time)

IGHA\* is a separate C++/pybind package, JIT-built on first use via
`torch.utils.cpp_extension`. Requirements:

- The **IGHAStar** repo on its `generalized_version` branch (it must contain
  `ighastar/src/Environments/include/generic.h`). By default it is expected as a
  sibling of this repo (`../IGHAStar`); otherwise set `IGHASTAR_ROOT`:
  ```bash
  export IGHASTAR_ROOT=/path/to/IGHAStar
  ```
- A C++17 compiler (`g++`), `ninja`, and **Boost headers** (header-only
  `boost::hash_combine`). `planner_ighastar.py` auto-discovers a Boost include
  dir (system `/usr/include`, the active interpreter's `cmeel.prefix/include`,
  or conda envs); if your Boost lives elsewhere, prepend it to
  `CPLUS_INCLUDE_PATH`.

The first run JIT-compiles the extension (`ighastar_generic_<N>_<C>_<H>`,
~15–30 s); subsequent runs reuse the cached build.

### Running

```bash
# IGHA* on Isaac Lab (headless), 2 obstacles
python main.py simulator=isaaclab planner=ighastar

# Genesis backend instead
python main.py simulator=genesis planner=ighastar

# Smaller/faster first run
python main.py simulator=isaaclab planner=ighastar parallel_envs=16 planner.max_expansions=500
```

Worked example — hard 5-block / 3-z-level problem, saved for replay:

```bash
ISAACLAB_HEADLESS=1 python main.py \
  simulator=isaaclab planner=ighastar viewer=headless \
  n_obstacles=5 n_z_levels=3 \
  parallel_envs=72 planner.max_expansions=100 \
  push_steps=256 verify_push_steps=256 \
  save=sol.json

# then render the saved plan to a video
python replay.py sol.json --video solution.mp4 --push-steps 256
```

(`num_controls = 4·(5+1)·3 = 72`, so `parallel_envs=72` runs ~3 node expansions
per GPU sweep; `save=sol.json` writes the solution for `replay.py`.)

### Config (`conf/planner/ighastar.yaml`)

| Key | Default | Description |
|---|---|---|
| `max_expansions` | `5000` | Search budget; each expansion runs `num_controls` `batch_evaluate` sims |
| `hysteresis` | `500` | IGHA* resolution-switch threshold |
| `resolution` | `0.1` | Starting grid resolution per xy dim (m) |
| `tolerance` | `0.0125` | Dedup tolerance per xy dim (m) |
| `grid_z` | `true` | Also grid object z (height) in the hashed subspace (`HASH_DIMS = 3·(1+n_obstacles)`); set `false` to grid xy only (`2·(1+n_obstacles)`) |
| `z_resolution` | `0.1` | Starting grid resolution for z dims (m), used when `grid_z=true` |
| `z_tolerance` | `0.0125` | Dedup tolerance for z dims (m) |
| `max_level` | `4` | Number of resolution levels |
| `division_factor` | `2.0` | Resolution shrink factor per level |
| `debug` | `false` | Print IGHA* per-iteration stats (Expansions/level/Q_v/Seen) + a profiler summary |
| `preemptive_expansion.enabled` | `false` | Batch many `Q_v` vertices' successors into one `batch_evaluate` launch (helps fill `parallel_envs`; tends not to help at very large branching factors) |
| `preemptive_expansion.min_preemptive` | `1` | Launch a preemptive batch only once the stash holds ≥ this many vertices |
| `preemptive_expansion.max_preemptive` | `3` | Cap on vertices expanded per preemptive launch |

If the search finds no goal, raise `planner.max_expansions` and/or
`parallel_envs` (each expansion batches `num_controls` pushes through the
simulator, so more parallel envs is faster).

## Parallel environments

`ParallelBinEnv` wraps a single Genesis scene built with `scene.build(n_envs=N)`.
All environments share one CUDA kernel — evaluating N (state, action) pairs costs
the same wall-clock time as evaluating 1.

```python
from simulators import ParallelBinEnv
penv = ParallelBinEnv(n_envs=8, n_obstacles=2)
results = penv.batch_evaluate([(state0, action0), ..., (state7, action7)])
# results: list of (new_state, reward, done)
```

## Saving and replaying solutions

```bash
# Plan and save
python main.py --planner mcts --save solution.json

# solution.json contains:
#   env_config    — bin/object dimensions
#   initial_state — target + obstacle poses
#   actors        — ActorWrapper-compatible dicts (target, obstacles, walls)
#   plan          — action sequence
#   steps         — per-action start/end pose of displaced object + target
```

## Exporting to genesismpc

`apply_solution.py` reads `solution.json` and writes actor YAML files into the
genesismpc repo, then patches `config_ur.yaml`:

```bash
python apply_solution.py --solution solution.json \
                         --genesismpc-dir ../genesismpc
# Writes: conf/actors/puzzle_target.yaml
#         conf/actors/puzzle_obstacle_0.yaml  ...
#         conf/actors/puzzle_floor.yaml  etc.
# Patches: examples/ur5_stick_stacked_blocks/config_ur.yaml
```

## Training a value function

`train_value.py` runs MCTS over many random initial states, harvests
`(state, mean_reward)` pairs from every node in the search trees, and trains
a small MLP to predict state value. The resulting `value_net.pt` can be used
as a **terminal cost** in MPPI:

```
J(τ) = Σ running_costs(s_t, a_t)  +  α · (1 − V(s_H))
```

```bash
# Collect data and train
python train_value.py --episodes 100 --sims 300

# Retrain on existing dataset without re-collecting
python train_value.py --load-data dataset.npz --epochs 400

# Outputs: value_net.pt, dataset.npz
```

### Value network inputs

State vector — 6 floats, bin-normalised:

```
[target_x/BIN_W,  target_y/BIN_D,
 obs0_x/BIN_W,    obs0_y/BIN_D,
 obs1_x/BIN_W,    obs1_y/BIN_D]
```

Reward range `[-1.0, 1.0]` (normalised to `[0, 1]` for training):
- `< 0` — obstacle dropped (penalty)
- `= 0` — no progress
- `= 1` — target exited the bin (goal)

## Bin layout constants

```python
BIN_W  = 0.3    # x extent (m)
BIN_D  = 0.3    # y extent (m)
BIN_H  = 0.15   # wall height (m)
WALL_T = 0.02   # wall thickness (m)
OBJ_SIZE = 0.08 # object cube side (m)
EXIT_Y = -0.05  # target escapes when y < EXIT_Y
```

## Dependencies

- Genesis 0.3.11
- NumPy, PyTorch
- Conda env: `genesistest2`
