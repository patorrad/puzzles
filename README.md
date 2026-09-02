# TODO
- save_solution was written for AZ/MCTS actions (push_pos, action_type, obj_idx) but MORE uses push_dir with push_start_xy/push_end_xy. I need to make the plan serialization handle both:
- Also note: _detach_plan at line 127–131 only handles push_pos (AZ/MCTS format), so it will break for MORE plans too. Want me to fix that?




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

<!-- Now it's fully clear. parallel_envs: 1 and min_verify_envs: 16 — so it's 16 sequential single-env runs, not 16 parallel slots. The print is a bug: the denominator uses env.n_envs (=1) instead of min_verify_envs (=16), so it displays 16/1 but the rate math (successes / self.min_verify_envs) is correct.

Each of the 16 runs does a full deterministic reset — positions, quaternions, zero_all_dofs_velocity() — so every run starts from byte-identical state. There are no spatial offsets (single env slot, not parallel), no friction sampling, no noise injection anywhere.

The variance you see in the wandb fraction plots is therefore pure GPU floating-point nondeterminism across sequential identical calls: CUDA contact detection and collision resolution use parallel reductions whose operand ordering is not guaranteed to be consistent between kernel launches. The same pusher trajectory from the same starting state produces slightly different object positions depending on which GPU warps happen to process which contacts in which order.

Bottom line: verify_rate is measuring how often the simulator agrees with itself across repeated identical calls. It has no connection to physical robustness — it doesn't tell you anything about what happens if friction varies by ±10% or the pusher has positioning error. A plan that scores 15/16 isn't "robust to noise"; it's just hitting a CUDA nondeterminism edge case. -->
