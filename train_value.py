"""
train_value.py — collect MCTS tree data and train a simple MLP value function.

Data collection
---------------
For each episode we run MCTS from a fresh random initial state, then walk
every node in the resulting search tree. Each node gives a
(state_vector, mean_reward) training sample — the MCTS backpropagation has
already aggregated rollout returns into node.mean_reward.

State encoding (2 obstacles, xy only, bin-normalized)
------------------------------------------------------
    [tx, ty, o0x, o0y, o1x, o1y]   shape (6,)  values in ~[0, 1]

Usage
-----
    conda run -n genesistest2 python train_value.py
    conda run -n genesistest2 python train_value.py --episodes 200 --sims 300
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from env import BinEnv, BIN_W, BIN_D
from planner import MCTSPusher, MCTSNode


# ── state encoding ────────────────────────────────────────────────────────────

N_OBS = 2
N_IN  = 2 * (1 + N_OBS)   # 6: (tx,ty) + (o0x,o0y) + (o1x,o1y)

# Reward range: r in [-0.5*N_OBS, 1.0] — normalise to [0, 1] for training
R_MIN = -0.5 * N_OBS   # worst case: all obstacles dropped
R_MAX =  1.0            # goal reached

def normalise_reward(r: np.ndarray) -> np.ndarray:
    return (r - R_MIN) / (R_MAX - R_MIN)

def denormalise_reward(r_norm: np.ndarray) -> np.ndarray:
    return r_norm * (R_MAX - R_MIN) + R_MIN


def encode_state(state: dict) -> np.ndarray:
    """Bin-normalised xy positions — shape (N_IN,), float32."""
    tx, ty = state['target_pos'][:2]
    feats = [tx / BIN_W, ty / BIN_D]
    for i in range(N_OBS):
        ox, oy = state['obstacle_pos'][i][:2]
        feats += [ox / BIN_W, oy / BIN_D]
    return np.array(feats, dtype=np.float32)


# ── tree traversal ────────────────────────────────────────────────────────────

def collect_tree_samples(root: MCTSNode,
                         min_visits: int = 2
                         ) -> list[tuple[np.ndarray, float]]:
    """
    BFS over MCTS tree; yield (state_vec, mean_reward) for every node
    that has been visited at least min_visits times (noisier leaf nodes
    with a single visit are excluded).
    """
    samples = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node.visits >= min_visits:
            samples.append((encode_state(node.state),
                            float(node.mean_reward)))
        queue.extend(node.children)
    return samples


# ── model ─────────────────────────────────────────────────────────────────────

class ValueMLP(nn.Module):
    def __init__(self, n_in: int = N_IN, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            # No Sigmoid — targets are normalised to [0,1] but linear output
            # avoids gradient saturation and can slightly extrapolate.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Return value in original reward range [-0.5*N_OBS, 1.0]."""
        return self.forward(x) * (R_MAX - R_MIN) + R_MIN


# ── data collection ───────────────────────────────────────────────────────────

def collect_dataset(n_episodes: int, n_sims: int,
                    min_visits: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MCTS for n_episodes different initial states; harvest all tree nodes.
    Returns (X, y) arrays ready for training.
    """
    env = BinEnv(
        n_obstacles=N_OBS,
        show_viewer=False,
        friction=1.2,
        n_z_levels=1,
        push_steps=20,
        substeps=4,
    )

    all_X, all_y = [], []

    for ep in range(n_episodes):
        # New random layout each episode
        env.reset(seed=ep)
        initial_state = env.get_state(0)

        planner = MCTSPusher(
            env=env,
            n_simulations=n_sims,
            rollout_depth=4,
            max_depth=10,
            n_children=4,
            seed=ep,
        )
        planner.plan(initial_state, verbose=False)

        if planner.root is None:
            continue

        samples = collect_tree_samples(planner.root, min_visits=min_visits)
        for vec, val in samples:
            all_X.append(vec)
            all_y.append(val)

        n_nodes = len(samples)
        print(f'  ep {ep+1:3d}/{n_episodes}  nodes={n_nodes:4d}  '
              f'total={len(all_X):6d}')

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y,  dtype=np.float32)

    # Print distribution before balancing (in original reward units)
    neg  = (y < 0).sum()
    zero = (y == 0).sum()
    pos  = (y > 0).sum()
    print(f'\nRaw dataset: {len(y)} samples  '
          f'neg={neg} zero={zero} pos={pos}  '
          f'mean={y.mean():.3f}  min={y.min():.3f}  max={y.max():.3f}')

    # Normalise to [0, 1] so all reward levels are equally learnable
    y = normalise_reward(y)

    # Balance: keep all informative (non-neutral) samples;
    # subsample the neutral-zero bin to equal count so the
    # model can't collapse to predicting the neutral value.
    neutral = normalise_reward(np.float32(0.0))
    nz_idx  = np.where(y != neutral)[0]
    z_idx   = np.where(y == neutral)[0]
    n_keep  = min(len(z_idx), max(len(nz_idx), 1))
    z_keep  = np.random.choice(z_idx, size=n_keep, replace=False)
    keep    = np.concatenate([nz_idx, z_keep])
    np.random.shuffle(keep)
    X, y = X[keep], y[keep]
    print(f'Balanced dataset: {len(y)} samples  '
          f'non-neutral={len(nz_idx)} ({100*len(nz_idx)/len(y):.1f}%)')

    return X, y


# ── training ──────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray,
          epochs: int = 200,
          batch_size: int = 256,
          lr: float = 1e-3,
          val_frac: float = 0.1) -> ValueMLP:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining on {device}  |  dataset size={len(X)}')

    # Train / val split
    n_val = max(1, int(len(X) * val_frac))
    idx = np.random.permutation(len(X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.from_numpy(X[tr_idx]).to(device)
    y_tr = torch.from_numpy(y[tr_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    model = ValueMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        # Mini-batch SGD
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            batch = perm[i:i + batch_size]
            pred = model(X_tr[batch])
            loss = loss_fn(pred, y_tr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_tr)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f'  epoch {epoch:3d}/{epochs}  '
                  f'train={epoch_loss:.4f}  val={val_loss:.4f}  '
                  f'best_val={best_val:.4f}')

    model.load_state_dict(best_state)
    return model.cpu()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes',   type=int, default=50,
                    help='Number of MCTS episodes (different initial states)')
    ap.add_argument('--sims',       type=int, default=200,
                    help='MCTS simulations per episode')
    ap.add_argument('--epochs',     type=int, default=200)
    ap.add_argument('--min-visits', type=int, default=2,
                    help='Min node visits to include as training sample')
    ap.add_argument('--out-model',  default='value_net.pt')
    ap.add_argument('--out-data',   default='dataset.npz')
    ap.add_argument('--load-data',  default=None,
                    help='Skip collection and load existing dataset.npz')
    args = ap.parse_args()

    # ── data ──
    if args.load_data:
        print(f'Loading dataset from {args.load_data}')
        d = np.load(args.load_data)
        X, y = d['X'], d['y']
    else:
        print(f'Collecting data: {args.episodes} episodes × {args.sims} sims')
        X, y = collect_dataset(args.episodes, args.sims, args.min_visits)
        np.savez(args.out_data, X=X, y=y)
        print(f'Dataset saved to {args.out_data}  '
              f'({len(X)} samples, mean_reward={y.mean():.3f})')

    # ── train ──
    model = train(X, y, epochs=args.epochs)
    torch.save(model.state_dict(), args.out_model)
    print(f'\nModel saved to {args.out_model}')

    # ── quick sanity check: show spread across value range ──
    model.eval()
    sorted_idx = np.argsort(y)
    check_idx  = sorted_idx[np.linspace(0, len(y)-1, 8, dtype=int)]
    with torch.no_grad():
        preds = model(torch.from_numpy(X[check_idx])).numpy()
    # Convert back to original reward units for readability
    y_orig    = denormalise_reward(y[check_idx])
    pred_orig = denormalise_reward(preds)
    print('\nSanity check (8 evenly-spaced by target value, original reward scale):')
    print(f'  target   : {y_orig.round(3)}')
    print(f'  predicted: {pred_orig.round(3)}')


if __name__ == '__main__':
    main()
