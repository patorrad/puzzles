from __future__ import annotations

import sys as _sys, re as _re

class _OmniFilter:
    _pat = _re.compile(r'(\d{4}-\d{2}-\d{2}T|\[INFO\]|\[WARNING\]|\[ERROR\]).*'
                       r'(omni|isaacsim|carb|isaac|kit|nv::|physx|usd|omniverse)',
                       _re.IGNORECASE)
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)

_sys.stdout = _OmniFilter(_sys.stdout)

"""
MORE training pipeline — Phases A and B.

Phase A — Data collection (run once):
  Run unguided UCT MCTS on training scenes; log (state, action, Q, N) records.

Phase B — PPN training (run once, after Phase A):
  Regress PPN to the collected Q-values; weight samples by visit count N.

Intentionally a SINGLE-PASS pipeline.  There is no iterative collect→train→collect
loop — that would replicate AlphaZero, not MORE.

Usage (CLI via Hydra):

    # Phase A — data collection
    python -m more.train phase=collect output=more_data.pt \\
        n_scenes=200 n_simulations=200 seed=0

    # Phase B — PPN training
    python -m more.train phase=train data=more_data.pt output=ppn.pt \\
        epochs=100 batch_size=256 lr=1e-3

    # Both in one go
    python -m more.train phase=both ...
"""

import argparse
import copy
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from more.contour_sampler import ContourSampler
from more.mcts import MORETree
from more.ppn import PPN, build_ppn


# ---------------------------------------------------------------------------
# Env-build helper (loads the project's YAML configs, then calls build_env)
# ---------------------------------------------------------------------------

def _build_env(sim: str, n_obs: int, n_envs: int,
               stackable: bool = False, n_z_levels: int = 1,
               bin_size: float | None = None, difficult_spawn: bool = False,
               force_obstacle_on_target: bool = False,
               viewer: str = 'headless'):
    """Build a BinEnv by loading the project's Hydra YAML config files."""
    from omegaconf import OmegaConf
    from simulators import build_env

    conf_dir = os.path.join(os.path.dirname(__file__), '..', 'conf')
    base    = OmegaConf.load(os.path.join(conf_dir, 'config.yaml'))
    sim_cfg = OmegaConf.load(os.path.join(conf_dir, 'simulator', f'{sim}.yaml'))
    rew_cfg = OmegaConf.load(os.path.join(conf_dir, 'reward', 'default.yaml'))

    overrides = {
        'simulator':   sim_cfg,
        'reward':      rew_cfg,
        'n_obstacles':    n_obs,
        'stackable':      stackable,
        'n_z_levels':     n_z_levels,
        'difficult_spawn': difficult_spawn,
        'force_obstacle_on_target': force_obstacle_on_target,
        'seed':           None,
    }
    if bin_size is not None:
        overrides['bin_size'] = bin_size

    cfg = OmegaConf.merge(base, overrides)
    # Remove the 'defaults' key that Hydra uses at compose-time but
    # build_env doesn't need.
    OmegaConf.set_struct(cfg, False)
    cfg.pop('defaults', None)
    return build_env(cfg, n_envs=n_envs, viewer_mode=viewer)


# ---------------------------------------------------------------------------
# Phase A — unguided UCT data collection
# ---------------------------------------------------------------------------

@dataclass
class CollectConfig:
    n_scenes:      int   = 200
    n_simulations: int   = 200
    max_depth:     int   = 4    # paper: 4 for data collection
    gamma:         float = 0.5
    rollout_depth: int   = 5
    k_per_object:  int   = 8
    c_uct:         float = 2.0
    seed:          int   = 0
    checkpoint_path: str = None


def collect_data(env, cfg: CollectConfig) -> list[dict]:
    """
    Run unguided UCT (no PPN) on *cfg.n_scenes* random scenes and return a
    flat list of {'state', 'action', 'Q', 'N'} dicts.

    The env is used to sample initial states (env.reset) and evaluate pushes
    (env.batch_evaluate).  No PPN is used — this is pure self-supervised
    data collection as described in MORE §III-B Phase A.
    """
    sampler = ContourSampler.from_env(env, include_target=True)
    tree = MORETree(
        env=env,
        ppn=None,           # unguided
        contour_sampler=sampler,
        gamma=cfg.gamma,
        max_depth=cfg.max_depth,
        rollout_depth=cfg.rollout_depth,
        k_per_object=cfg.k_per_object,
        c_uct=cfg.c_uct,
    )

    all_records: list[dict] = []
    start_scene = 0
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        ckpt = torch.load(cfg.checkpoint_path, weights_only=False)
        if isinstance(ckpt, dict) and 'scenes_completed' in ckpt:
            all_records = ckpt['records']
            start_scene = ckpt['scenes_completed']
            print(f'[MORE] Resuming from checkpoint: {start_scene}/{cfg.n_scenes} scenes already done, '
                  f'{len(all_records)} transitions loaded.')

    rng = random.Random(cfg.seed)

    for scene_idx in tqdm(range(cfg.n_scenes), desc='[MORE] Collecting data'):
        seed = rng.randint(0, 2**31 - 1)
        if scene_idx < start_scene:
            continue  # already completed in a prior run; seed still drawn so the sequence matches
        state = env.reset(seed=seed)

        # Build tree from root, collect transitions
        from more.mcts import MORENode
        root = MORENode(state=copy.deepcopy(state))
        for _ in range(cfg.n_simulations):
            path = tree._select(root)
            leaf = path[-1]
            if leaf.done or leaf.dead_end or leaf.depth >= cfg.max_depth:
                continue
            tree._expand(leaf)
            tree._batch_rollout_children(leaf.children, leaf.depth)
            for node in path:
                node.N += 1

        records = tree.collect_transitions(root)
        all_records.extend(records)

        if cfg.checkpoint_path:
            torch.save({'records': all_records, 'scenes_completed': scene_idx + 1}, cfg.checkpoint_path)
            print(f'[MORE] Checkpoint: {len(all_records)} transitions after scene {scene_idx+1}/{cfg.n_scenes}')

        if cfg.checkpoint_path:
            torch.save(all_records, cfg.checkpoint_path)
            print(f'[MORE] Checkpoint: {len(all_records)} transitions after scene {scene_idx+1}/{cfg.n_scenes}')

    print(f'[MORE] Collected {len(all_records)} transitions from {cfg.n_scenes} scenes.')
    return all_records


# ---------------------------------------------------------------------------
# Phase B — PPN training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs:        int   = 100
    batch_size:    int   = 256
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    log_every:     int   = 10
    seed:          int   = 0
    # PPN architecture
    arch:          str   = 'deepsets'   # 'deepsets' or 'mlp'
    obj_emb_dim:   int   = 64           # deepsets only
    push_emb_dim:  int   = 64           # deepsets only
    agg_hidden:    int   = 128          # deepsets only
    hidden:        int   = 1024          # mlp only


def _records_to_tensors(records: list[dict], n_obstacles: int):
    """
    Flatten a list of transition records into batched tensors for training.

    Returns (states_dict, pushes_dict, q_targets, weights) where:
      - states_dict contains batched (B, ...) tensors for target + obstacle poses
      - pushes_dict contains batched (B, ...) push geometry tensors
      - q_targets   (B,) float Q values
      - weights     (B,) float per-sample weights ∝ N (visit count)
    """
    target_pos_list    = []
    target_quat_list   = []
    obstacle_pos_list  = []
    obstacle_quat_list = []
    push_start_list    = []
    push_end_list      = []
    push_z_list        = []
    q_list             = []
    n_list             = []

    for rec in records:
        s = rec['state']
        a = rec['action']
        target_pos_list.append(s['target_pos'][:3].float())
        target_quat_list.append(s['target_quat'][:4].float())
        obs_pos = s['obstacle_pos'][:n_obstacles, :3].float()
        obs_quat = s['obstacle_quat'][:n_obstacles, :4].float()
        # Pad if fewer obstacles than expected
        pad_n = n_obstacles - obs_pos.shape[0]
        if pad_n > 0:
            obs_pos  = F.pad(obs_pos,  (0, 0, 0, pad_n))
            obs_quat = F.pad(obs_quat, (0, 0, 0, pad_n))
        obstacle_pos_list.append(obs_pos)
        obstacle_quat_list.append(obs_quat)

        push_start_list.append(a['push_start_xy'].float())
        push_end_list.append(a['push_end_xy'].float())
        push_z_list.append(torch.tensor(float(a['push_z'])))
        q_list.append(torch.tensor(float(rec['Q'])))
        n_list.append(torch.tensor(float(rec['N'])))

    states = {
        'target_pos':    torch.stack(target_pos_list),
        'target_quat':   torch.stack(target_quat_list),
        'obstacle_pos':  torch.stack(obstacle_pos_list),
        'obstacle_quat': torch.stack(obstacle_quat_list),
    }
    pushes = {
        'push_start_xy': torch.stack(push_start_list),
        'push_end_xy':   torch.stack(push_end_list),
        'push_z':        torch.stack(push_z_list),
    }
    q_targets = torch.stack(q_list)
    weights   = torch.stack(n_list)
    weights   = weights / weights.max()    # normalise to [0, 1]

    return states, pushes, q_targets, weights


def train_ppn(records: list[dict], n_obstacles: int,
              cfg: TrainConfig,
              device: str = 'cpu') -> PPN:
    """
    Train PPN on the collected records.  Returns the trained network.

    Loss: weighted Smooth L1 (HuberLoss), weight ∝ N(s,a).
    Single training pass — no iterative data collection.
    """
    torch.manual_seed(cfg.seed)

    print(f'[MORE] Training PPN on {len(records)} transitions ...')
    states, pushes, q_targets, weights = _records_to_tensors(records, n_obstacles)

    # Move to device
    def _to(d: dict) -> dict:
        return {k: v.to(device) for k, v in d.items()}

    states_dev   = _to(states)
    pushes_dev   = _to(pushes)
    q_targets_dev = q_targets.to(device)
    weights_dev   = weights.to(device)

    if cfg.arch == 'deepsets':
        arch_kwargs = dict(obj_emb_dim=cfg.obj_emb_dim,
                           push_emb_dim=cfg.push_emb_dim,
                           agg_hidden=cfg.agg_hidden)
    else:
        arch_kwargs = dict(hidden=cfg.hidden)
    net = build_ppn(cfg.arch, n_obstacles=n_obstacles, **arch_kwargs).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    criterion = torch.nn.HuberLoss(reduction='none')

    B = len(records)

    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(B)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, B, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            sb = {k: v[idx] for k, v in states_dev.items()}
            pb = {k: v[idx] for k, v in pushes_dev.items()}
            qb = q_targets_dev[idx]
            wb = weights_dev[idx]

            optimizer.zero_grad()
            q_pred = net(sb, pb)
            raw_loss = criterion(q_pred, qb)
            loss = (raw_loss * wb).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % cfg.log_every == 0:
            print(f'  epoch {epoch:4d}/{cfg.epochs} | loss={epoch_loss/n_batches:.5f}')

    print('[MORE] PPN training complete.')
    return net


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ppn(net, path: str, cfg: TrainConfig, n_obstacles: int):
    ckpt = {
        'ppn':         net.state_dict(),
        'n_obstacles': n_obstacles,
        'arch':        cfg.arch,
    }
    if cfg.arch == 'deepsets':
        ckpt.update(obj_emb_dim=cfg.obj_emb_dim,
                    push_emb_dim=cfg.push_emb_dim,
                    agg_hidden=cfg.agg_hidden)
    else:
        ckpt['hidden'] = cfg.hidden
    torch.save(ckpt, path)
    print(f'[MORE] PPN saved → {path}')


def save_data(records: list[dict], path: str):
    torch.save(records, path)
    print(f'[MORE] Data saved → {path} ({len(records)} records)')


def load_data(path: str) -> list[dict]:
    records = torch.load(path, weights_only=False)
    print(f'[MORE] Loaded {len(records)} records from {path}')
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='MORE PPN training pipeline')
    p.add_argument('--phase',    choices=['collect', 'train', 'both'],
                   default='both')
    p.add_argument('--sim',      default='isaaclab',
                   help='Simulator backend (isaaclab | genesis)')
    p.add_argument('--n_obs',      type=int, default=2)
    p.add_argument('--n_envs',     type=int, default=8)
    p.add_argument('--viewer',     default='headless',
                   choices=['headless', 'replay', 'verify', 'always'],
                   help='Viewer mode for the collection env — use "always" to watch it live')
    p.add_argument('--difficult_spawn', action='store_true',
                   help='Use difficult initial spawn positions')
    p.add_argument('--force_obstacle_on_target', action='store_true',
                   help='Place one obstacle on top of the target (match alphazero_train.yaml)')
    p.add_argument('--stackable',  action='store_true',
                   help='Enable stackable objects (match alphazero_train.yaml)')
    p.add_argument('--n_z_levels', type=int, default=1,
                   help='Number of push height levels (match alphazero_train.yaml)')
    p.add_argument('--bin_size',   type=float, default=None,
                   help='Fixed bin side length in metres; null = auto-scale with n_obs. '
                        'Use 0.4 to keep the N=7 bin size regardless of n_obs.')
    p.add_argument('--data',     default='more_data.pt',
                   help='Input (train) or output (collect) data file')
    p.add_argument('--output',   default='ppn.pt')
    # Collection
    p.add_argument('--n_scenes',      type=int,   default=200)
    p.add_argument('--n_simulations', type=int,   default=200)
    p.add_argument('--max_depth',     type=int,   default=4)
    p.add_argument('--gamma',         type=float, default=0.5)
    p.add_argument('--c_uct',         type=float, default=2.0)
    p.add_argument('--k_per_object',  type=int,   default=8,
                   help='Contour samples per object per expansion (halving this ~halves sim calls)')
    p.add_argument('--seed',          type=int,   default=0)
    # Training
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--arch',       default='deepsets', choices=['deepsets', 'mlp'],
                   help='PPN architecture: deepsets (default) or mlp (AlphaZero-style flat MLP)')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = _parse_args()

    records = []

    if args.phase in ('collect', 'both'):
        if args.sim == 'isaaclab':
            _viewer_to_headless = {'headless': '1', 'replay': '0', 'verify': '0', 'always': '0'}
            os.environ.setdefault('ISAACLAB_HEADLESS', _viewer_to_headless[args.viewer])
        env = _build_env(args.sim, n_obs=args.n_obs, n_envs=args.n_envs,
                         stackable=args.stackable, n_z_levels=args.n_z_levels,
                         bin_size=args.bin_size, difficult_spawn=args.difficult_spawn,
                         force_obstacle_on_target=args.force_obstacle_on_target,
                         viewer=args.viewer)
        cfg_c = CollectConfig(
            n_scenes=args.n_scenes,
            n_simulations=args.n_simulations,
            max_depth=args.max_depth,
            gamma=args.gamma,
            c_uct=args.c_uct,
            k_per_object=args.k_per_object,
            seed=args.seed,
            checkpoint_path=args.data,
        )
        records = collect_data(env, cfg_c)
        if args.phase == 'collect':
            save_data(records, args.data)
            return

    if args.phase == 'train':
        records = load_data(args.data)

    if args.phase in ('train', 'both'):
        cfg_t = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            arch=args.arch,
        )
        net = train_ppn(records, args.n_obs, cfg_t, device=args.device)
        save_ppn(net, args.output, cfg_t, args.n_obs)


if __name__ == '__main__':
    main()
