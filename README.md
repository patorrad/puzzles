# Puzzle — Bin-Clearing Planning in Isaac Lab

Physics-based planning for a bin-clearing task: push a **target** cube out of a
bin through its open south side while keeping every **obstacle** inside.
All planning, verification, training and benchmarking runs in
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) with GPU-parallel
environments.

Three planners are provided:

| Planner | Config | Learned component |
|---|---|---|
| UCT MCTS | `planner=mcts` | none (random rollouts) |
| MORE | `planner=more` | Push Prediction Network (PPN) guiding the search |
| AlphaZero | `planner=alphazero` | policy/value network guiding PUCT search |

## Task

```
        north wall
  ┌─────────────────┐
  │  [o]   [o]      │
  │      [o]        │
  │   [o][T]  [o]   │   T = target, o = obstacle
  │      [o]        │   objects may be stacked up to n_z_levels high
  └──────     ──────┘
        open south side  (exit when target NS-coordinate < EXIT_Y = -0.05 m)
```

- Positions are `pos[0]` = north–south, `pos[1]` = east–west, `pos[2]` = height.
- A kinematic pusher blade sweeps objects. After each push the physics settles
  and the new state is read back.
- **Goal:** the target has exited the bin. An episode fails if any obstacle leaves.

### Reward

Configured in [conf/reward/default.yaml](conf/reward/default.yaml). The reward
is the sum of three terms:

| Term | Value |
|---|---|
| `target_progress` | `(bin_d/2 − target_ns) / (bin_d/2 − EXIT_Y)`, clipped to `[0, 2]` |
| `obstacle_penalty` | `−weight` (5.0) per obstacle that left the bin |
| `path_blocker` | `−weight` (0.5) × `max(0, 1 − ew_dist/scale)` for each obstacle between the target and the exit |

### Action spaces

- **UCT MCTS and AlphaZero** share one discrete action space of size
  `4 × (N+1) × Z`: an action type (`push_n`, `pull_s`, `push_e`, `push_w`) ×
  an object index (0 = target, 1..N = obstacles) × a push height level.
- **MORE** samples continuous `push_dir` actions along each object's contour
  ([more/contour_sampler.py](more/contour_sampler.py)). Each push starts just
  outside the object's footprint and passes through its centroid.

## Installation

Tested with Python 3.11, Isaac Sim 5.1.0, Isaac Lab 0.54.3, PyTorch 2.7.0 (CUDA 12.8).

1. Install Isaac Sim and Isaac Lab by following the
   [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/).
   Isaac Lab is installed from source (`./isaaclab.sh --install`).
2. In the same environment, install this repo's Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Isaac Lab reads `ISAACLAB_HEADLESS` and `ISAACLAB_ENABLE_CAMERAS` when
[simulators/isaaclab_env.py](simulators/isaaclab_env.py) is imported. The entry
points set these from the `viewer` and `record_video` options, so you normally
don't set them yourself.

## Running the planners

[benchmark.py](benchmark.py) runs a planner on `n_runs` random scenes (scene
`i` uses seed `base_seed + i`). It verifies each plan, then logs to W&B and,
optionally, to CSV, solution JSONs and videos. It is the entry point for all
three planners.

```bash
# UCT MCTS
python benchmark.py --config-name=benchmark planner=mcts

# MORE (needs a trained PPN)
python benchmark.py --config-name=benchmark planner=more parallel_envs=160 \
    planner.ppn_checkpoint=path/to/ppn.pt

# AlphaZero (needs a trained checkpoint)
python benchmark.py --config-name=benchmark planner=alphazero \
    planner.checkpoint=path/to/alphazero_latest.pt
```

Common overrides for the scene:

```bash
python benchmark.py --config-name=benchmark planner=mcts \
    n_obstacles=7 n_z_levels=3 force_obstacle_on_target=true bin_size=0.4 \
    n_runs=100 skip_runs=0 \
    wandb_run_name=mcts_7obs_stacked3 \
    csv_path=results/7obs_stacked3 \
    solutions_dir=solutions/mcts_7obs_stacked3
```

The `n_obstacles` and `n_z_levels` you benchmark with must match the ones the
MORE or AlphaZero network was trained on, because the network input and output
sizes depend on them.

### Outputs

| Option | Output |
|---|---|
| `csv_path` | Per-episode results (success, plan length, planning time, …), appended to this file |
| `solutions_dir` | One solution JSON per episode (initial state + plan) |
| `record_video` / `video_dir` | Replay video of each verified plan |
| `wandb_project` / `wandb_run_name` | Metrics and scene renders ([visualization.py](visualization.py)) |

## Planners

### UCT MCTS — [planner.py](planner.py) (`MCTSPusher`)

Monte Carlo tree search with UCB1 selection and random rollouts. Each simulated
push is evaluated in Isaac Lab. Once search finishes, the best path is
re-executed and accepted only if the fraction of successful runs reaches
`verify_threshold`.

| Key ([conf/planner/mcts.yaml](conf/planner/mcts.yaml)) | Default | Meaning |
|---|---|---|
| `n_simulations` | 1000 | Tree simulations per planning call |
| `rollout_depth` | 5 | Random rollout length |
| `max_depth` | 10 | Maximum plan length |
| `c_ucb` | 1.5 | UCB1 exploration constant |
| `target_prob` | 0.6 | Probability of sampling an action on the target |
| `action_weights` | `[.25,.25,.25,.25]` | Sampling weights for `push_n, pull_s, push_e, push_w` |

### MORE — [more/](more/) (`MOREPlanner`)

A re-implementation of Huang et al., *Interleaving Monte Carlo Tree Search and
Self-Supervised Learning for Object Retrieval in Clutter* (ICRA 2022). The PPN
predicts a Q-value for each contour push candidate. These predictions:

- set the initial statistics of each new node,
- decide which candidates are expanded first, and
- enter the selection rule (Eq. 3) and the final choice of action (Eq. 4).

If `ppn_checkpoint` is `null`, the tree falls back to unguided UCT, which is
how training data is collected.

| Key ([conf/planner/more.yaml](conf/planner/more.yaml)) | Default | Meaning |
|---|---|---|
| `ppn_checkpoint` | `ppn.pt` | Trained PPN |
| `n_simulations` | 50 | Tree iterations per step |
| `max_depth` | 3 | Tree depth |
| `gamma` | 0.5 | Discount factor |
| `k_per_object` | 4 | Contour pushes sampled per object per expansion |
| `rollout_depth` | 5 | Rollout length |
| `m` | 3 | Top-m rollout rewards summed in Eq. 3 |
| `c_uct` | 2.0 | UCT constant (unguided fallback only) |

### AlphaZero — [alphazero/](alphazero/) (`AlphaZeroPusher`)

PUCT search guided by a learned policy prior and value estimate
(`SolverNet`). Training is two-player self-play:

- a **stacker** builds the initial scene by placing blocks on a grid, and
- a **solver** clears it.

At inference only the solver network is used.

| Key ([conf/planner/alphazero.yaml](conf/planner/alphazero.yaml)) | Default | Meaning |
|---|---|---|
| `checkpoint` | `alphazero_latest.pt` | Trained checkpoint |
| `n_simulations` | 1000 | PUCT simulations per move |
| `max_depth` | 10 | Maximum plan length |
| `c_puct` | 1.5 | PUCT exploration constant |
| `temperature` | 1e-3 | Move-selection temperature (≈ argmax of visit counts) |

## Training

### AlphaZero — [train_alphazero.py](train_alphazero.py)

Hydra config: [conf/alphazero_train.yaml](conf/alphazero_train.yaml).

```bash
python train_alphazero.py net_arch=mlp n_obstacles=5 n_z_levels=3 \
    random_stacker=true viewer=headless \
    output_dir=outputs/5obs_stacked3/alphazero_mlp

# Resume
python train_alphazero.py resume_from=outputs/5obs_stacked3/alphazero_mlp/alphazero_latest.pt
```

Each iteration does the following:

1. Plays `episodes_per_iter` self-play episodes. The stacker places
   `n_obstacles` blocks (or, with `random_stacker=true`, the scene comes from
   `env.reset()`), then the solver plays up to `selfplay.max_depth` pushes with
   MCTS.
2. Stores `(state, visit distribution π, return z)` records in a replay buffer
   of size `buffer_size`.
3. Runs `train_steps_per_iter` minibatch updates of size `batch_size`.
4. Every `checkpoint_every` iterations, writes `alphazero_iter_XXXX.pt` and
   `alphazero_latest.pt` to `output_dir`.

The value target `z` is the discounted return, with γ = `selfplay.gamma`, of the
per-step reward `clamp(reward / reward_scale, −1, 1)`.

### MORE — [more/train.py](more/train.py)

Training runs once in two phases. It is not an iterative loop.

```bash
# Phase A: collect (state, push, Q, N) records with unguided UCT
python -m more.train --phase collect --sim isaaclab --n_envs 64 \
    --n_obs 5 --stackable --n_z_levels 3 --bin_size 0.4 --difficult_spawn \
    --n_scenes 200 --n_simulations 200 --k_per_object 4 --seed 0 \
    --data outputs/more_data_n5.pt

# Phase B: fit the PPN to the collected Q-values
python -m more.train --phase train --arch mlp --n_obs 5 --stackable --difficult_spawn \
    --data outputs/more_data_n5.pt --output outputs/ppn_mlp_n5.pt \
    --epochs 100 --batch_size 256 --lr 1e-3
```

`--phase both` runs both phases in sequence. To collect in parallel, run Phase A
with several seeds and concatenate the record lists (`torch.load` → `extend` →
`torch.save`) before Phase B.

## Networks (MLP)

All networks are small two-hidden-layer ReLU MLPs trained with Adam
(`lr=1e-3`, `weight_decay=1e-4`). The notation below is:

- `N` = `n_obstacles`
- `Z` = `n_z_levels`
- `Gx × Gy` = the placement grid ([alphazero/grid.py](alphazero/grid.py))

### Shared scene encoding

The solver and PPN encode the scene the same way. The layout is grouped by
section, with the target first in each section:

```
[ target_xyz (3) | obstacle_xyz (3N) | target_quat (4) | obstacle_quat (4N) ]   → 7(N+1) floats
```

Quaternions are made canonical (sign flipped so the largest component is
positive), so `q` and `−q` encode the same way.

### AlphaZero `SolverNet` — [alphazero/networks.py](alphazero/networks.py)

```
x ─ Linear(in, 1024) ─ ReLU ─ Linear(1024, 1024) ─ ReLU ─┬─ Linear(1024, 4(N+1)Z)  → policy logits
                                                          └─ Linear(1024, 1) ─ tanh → value ∈ [−1, 1]
```

- **Input** ([alphazero/encoders.py](alphazero/encoders.py) `encode_solver_state`):
  the scene encoding above. When `use_cell_onehot=true`, it also includes a
  one-hot of the grid cell for each object, `(N+1)·Gx·Gy` floats. Input size =
  `7(N+1) + (N+1)·Gx·Gy`.
- **Policy:** one logit per discrete action. The action index is
  `((type · (N+1)) + obj) · Z + z`. Illegal actions are set to `−inf` before
  the softmax.
- **Loss:** cross-entropy against the MCTS visit distribution `π` plus MSE
  between the value and the return `z`.

### AlphaZero `StackerNet` — [alphazero/networks.py](alphazero/networks.py)

The same trunk with `hidden = 128`. It is used only during self-play training,
unless `random_stacker=true`.

- **Input:** a `[4, Gx, Gy]` grid, flattened
  ([alphazero/encoders.py](alphazero/encoders.py) `encode_stacker_state`).
  The four channels are:
  - the occupancy heat-map,
  - the target cell mask,
  - the highest occupied level, normalized, and
  - a constant `blocks_remaining / N` plane.
- **Policy:** one logit per placement cell `(i, j, k)`, so `Gx·Gy·Z` logits.
- **Value:** tanh. It is trained toward the negative mean return of the solver
  (the stacker is adversarial).
- **Loss:** the same as the solver.

### MORE `PPNFlat` — [more/ppn.py](more/ppn.py)

```
[scene (7(N+1)) | push (5)] ─ Linear(·, H) ─ ReLU ─ Linear(H, H) ─ ReLU ─ Linear(H, 1) → Q(s, push)
```

- **Push features:** `push_start_xy (2) | unit push direction (2) | push_z (1)`.
- **Hidden width `H`:** `--hidden`. The default in `more/train.py` is 1024; it is
  stored in the checkpoint.
- **Loss:** Huber loss against the Phase A Q-values. Each sample is weighted by
  its visit count `N`, normalized to `[0, 1]`.

### Checkpoint formats

| File | Keys |
|---|---|
| `alphazero_*.pt` | `solver`, `stacker` (state dicts), `net_arch`, `use_cell_onehot`, `n_obstacles`, `solver_in_dim`, `solver_n_actions`, `stacker_grid_h/w`, `stacker_n_actions`, `spec` (GridSpec), `iter` |
| `ppn_*.pt` | `ppn` (state dict), `arch`, `hidden`, `n_obstacles` |

## Environment configuration

Scene options live in [conf/config.yaml](conf/config.yaml).
[conf/benchmark.yaml](conf/benchmark.yaml) and
[conf/alphazero_train.yaml](conf/alphazero_train.yaml) override some of them.

| Key | Meaning |
|---|---|
| `n_obstacles` | Number of obstacle cubes |
| `bin_size` | Bin side length in m (`null` = scale automatically with `n_obstacles × bin_size_factor`) |
| `obj_size` | Cube side length (m) |
| `stackable`, `n_z_levels` | Allow stacking, and the number of height levels |
| `difficult_spawn` | Spawn the target in the north half of the bin (far from the exit) |
| `force_obstacle_on_target`, `force_obstacle_on_target_prob` | Put an obstacle on top of the target |
| `push_steps`, `verify_push_steps` | Pusher sweep steps during planning and during verification |
| `parallel_envs` | Number of Isaac Lab environments evaluated in one batch |
| `verify_threshold`, `n_verify_runs` | Plan acceptance criterion |
| `viewer` | `headless` \| `replay` \| `verify` \| `always` |

Isaac Lab physics settings (PhysX iterations, depenetration velocity, settle
steps, contact-force cutoff) are in
[conf/simulator/isaaclab.yaml](conf/simulator/isaaclab.yaml).

## Repository layout

| Path | Contents |
|---|---|
| [simulators/isaaclab_env.py](simulators/isaaclab_env.py) | Isaac Lab bin environment (scene, pushers, batched `batch_evaluate`) |
| [simulators/base_env.py](simulators/base_env.py) | Environment interface, reward, goal checks |
| [planner.py](planner.py) | UCT MCTS and the `from_cfg` planner factory |
| [more/](more/) | MORE: contour sampler, guided tree, PPN, training |
| [alphazero/](alphazero/) | AlphaZero: grid, encoders, networks, games, PUCT MCTS, self-play, training |
| [benchmark.py](benchmark.py) | Multi-run evaluation |
| [train_alphazero.py](train_alphazero.py) | AlphaZero training entry point |
| [conf/](conf/) | Hydra configs (`planner/`, `simulator/`, `reward/`, `scenario/`) |
| [bash_scripts/](bash_scripts/) | Training and benchmark campaigns |
| [paper_scripts/](paper_scripts/) | Plots and tables from benchmark CSVs; tests |

## Tests

```bash
ISAACLAB_HEADLESS=1 pytest -v paper_scripts/
```
