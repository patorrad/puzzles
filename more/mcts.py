"""
MORE guided MCTS engine (Huang et al., ICRA 2022, arXiv:2202.01426).

Key differences from the UCB MCTS in planner.py:
  - Selection uses MORE Eq. 3 (no UCB exploration term; C = 0).
  - Final action choice uses MORE Eq. 4.
  - PPN Q-estimates seed every node's statistics at initialisation (N_init = 1).
  - PPN also orders which candidates are expanded first (high Q_ppn first).
  - Rollouts run to a terminal state or max_depth; discounted by gamma = 0.5.

Public helpers (tested independently):
  - MORE_Q_guide(q_ppn_samples, rollout_rewards, N, m) → float   [Eq. 3]
  - MORE_Q_best(q_ppn_samples, rollout_rewards)        → float   [Eq. 4]

Tree entry point:
  - MORETree.search(root_state, n_simulations) → best_action_dict
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Public formula helpers (Eq. 3 / Eq. 4) — dependency-free, fully testable
# ---------------------------------------------------------------------------

def MORE_Q_guide(
    q_ppn_samples: list[float],
    rollout_rewards: list[float],
    N: int,
    m: int = 3,
) -> float:
    """
    MORE Eq. 3 — guided Q estimate used during selection.

      Q_guide(s, a) = ( max(Q_ppn) + sum_{top-m} r_i ) / N

    N is clamped to at least 1 (MORE initialises N = 1 before any visit).
    m is clamped to the number of available rollout rewards.
    """
    if not q_ppn_samples:
        return 0.0
    max_ppn = max(q_ppn_samples)
    m_eff = min(m, len(rollout_rewards))
    top_m_sum = sum(sorted(rollout_rewards, reverse=True)[:m_eff])
    return (max_ppn + top_m_sum) / max(N, 1)


def MORE_Q_best(
    q_ppn_samples: list[float],
    rollout_rewards: list[float],
) -> float:
    """
    MORE Eq. 4 — final action selection (after search).

      Q_best(s, a) = max(Q_ppn) + max_i r_i
    """
    if not q_ppn_samples or not rollout_rewards:
        return 0.0
    return max(q_ppn_samples) + max(rollout_rewards)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class MORENode:
    """Tree node for the MORE guided MCTS.

    Statistics differ from MCTSNode:
      - q_ppn_samples : PPN Q-values collected for this (state, action) pair
      - rollout_rewards : discounted returns from rollouts through this node
      - N : visit count (initialised to 1 per MORE's normalisation)
    """
    state: dict
    action: dict | None = None          # action taken to reach this node
    parent: 'MORENode | None' = None
    depth: int = 0
    done: bool = False
    dead_end: bool = False              # obstacle dropped — never expand

    q_ppn_samples:   list[float] = field(default_factory=list)
    rollout_rewards: list[float] = field(default_factory=list)
    N: int = 1                          # init=1 per MORE
    children: list['MORENode'] = field(default_factory=list)
    expanded: bool = False              # True once _expand has run on this node

    def q_guide(self, m: int = 3) -> float:
        return MORE_Q_guide(self.q_ppn_samples, self.rollout_rewards, self.N, m)

    def q_best(self) -> float:
        return MORE_Q_best(self.q_ppn_samples, self.rollout_rewards)

    def is_leaf(self) -> bool:
        return not self.expanded or self.dead_end


# ---------------------------------------------------------------------------
# Tree search
# ---------------------------------------------------------------------------

class MORETree:
    """
    MORE guided MCTS.

    Parameters
    ----------
    env : SimulatorEnv
        Must support batch_evaluate() and _is_goal() / _obstacles_dropped().
    ppn : PPN | None
        Trained Push Prediction Network.  If None, falls back to unguided UCT
        (used for Phase A data collection).
    contour_sampler : ContourSampler
        Generates candidate push_dir actions for each state.
    gamma : float
        Discount factor (paper uses 0.5).
    max_depth : int
        Maximum tree depth.
    n_rollouts : int
        Random rollouts per expansion.
    rollout_depth : int
        Steps per rollout.
    m : int
        Number of top rewards summed in Eq. 3.
    c_uct : float
        UCT constant for the unguided fallback (Phase A only; ignored when ppn
        is provided).
    k_per_object : int
        Contour samples per object per search step.
    """

    def __init__(self, env, ppn, contour_sampler,
                 gamma: float = 0.5,
                 max_depth: int = 3,
                 n_rollouts: int = 1,
                 rollout_depth: int = 5,
                 m: int = 3,
                 c_uct: float = 2.0,
                 k_per_object: int = 8):
        self.env = env
        self.ppn = ppn
        self.sampler = contour_sampler
        self.gamma = gamma
        self.max_depth = max_depth
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.m = m
        self.c_uct = c_uct
        self.k_per_object = k_per_object

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(self, root_state: dict, n_simulations: int) -> dict | None:
        """
        Run MORE guided MCTS from root_state for n_simulations iterations.

        Returns the best action dict (Eq. 4), or None if no child was expanded.
        """
        import torch
        root = MORENode(state=copy.deepcopy(root_state))

        for _ in range(n_simulations):
            # Select — descend to a leaf via Q_guide
            path = self._select(root)
            leaf = path[-1]

            if leaf.done or leaf.dead_end or leaf.depth >= self.max_depth:
                continue

            # Expand — add children from contour samples, seed with PPN
            self._expand(leaf)

            # Rollout — batch all new children in one vectorised sweep
            self._batch_rollout_children(leaf.children, leaf.depth)

            # Backprop — update N up the path
            for node in path:
                node.N += 1

        if not root.children:
            return None

        best = max(root.children, key=lambda n: n.q_best())
        return best.action

    # ------------------------------------------------------------------
    # Selection (Eq. 3 — no UCB term when PPN is available)
    # ------------------------------------------------------------------

    def _select(self, root: MORENode) -> list[MORENode]:
        path = [root]
        node = root
        while not node.is_leaf():
            if self.ppn is not None:
                node = max(node.children,
                           key=lambda n: n.q_guide(self.m))
            else:
                # Unguided UCT fallback (Phase A data collection)
                node = self._uct_select(node)
            path.append(node)
            if node.done or node.dead_end:
                break
        return path

    def _uct_select(self, node: MORENode) -> MORENode:
        parent_n = sum(c.N for c in node.children) + 1
        def _uct(child):
            if not child.rollout_rewards:
                return float('inf')
            q = max(child.rollout_rewards)  # simple max
            return q + self.c_uct * math.sqrt(math.log(parent_n) / child.N)
        return max(node.children, key=_uct)

    # ------------------------------------------------------------------
    # Expansion — seed children with PPN Q-estimates
    # ------------------------------------------------------------------

    def _expand(self, node: MORENode) -> None:
        if node.expanded:
            return
        node.expanded = True

        import torch
        candidates = self.sampler.sample(node.state, k_per_object=self.k_per_object)
        if not candidates:
            return

        # Score all candidates with PPN in one batched forward pass
        if self.ppn is not None:
            B = len(candidates)
            dev = next(self.ppn.parameters()).device
            state_tensors = {
                'target_pos':    node.state['target_pos'].float().unsqueeze(0).expand(B, -1).to(dev),
                'target_quat':   node.state['target_quat'].float().unsqueeze(0).expand(B, -1).to(dev),
                'obstacle_pos':  node.state['obstacle_pos'].float().unsqueeze(0).expand(B, -1, -1).to(dev),
                'obstacle_quat': node.state['obstacle_quat'].float().unsqueeze(0).expand(B, -1, -1).to(dev),
            }
            push_tensors = {
                'push_start_xy': torch.stack([a['push_start_xy'] for a in candidates]).to(dev),
                'push_end_xy':   torch.stack([a['push_end_xy']   for a in candidates]).to(dev),
                'push_z':        torch.tensor([float(a['push_z']) for a in candidates], device=dev),
            }
            with torch.no_grad():
                q_all = self.ppn(state_tensors, push_tensors).tolist()
        else:
            q_all = [0.0] * len(candidates)

        scored: list[tuple[float, dict]] = list(zip(q_all, candidates))

        # High-Q actions are expanded first (PPN expansion priority)
        scored.sort(key=lambda t: t[0], reverse=True)

        pairs = [(node.state, action) for _, action in scored]
        results = self._chunked_batch_evaluate(pairs)

        for (q_est, action), (new_state, reward, done) in zip(scored, results):
            dropped = self.env._obstacles_dropped(new_state)
            child = MORENode(
                state=new_state,
                action=action,
                parent=node,
                depth=node.depth + 1,
                done=done,
                dead_end=dropped,
                q_ppn_samples=[q_est],
                rollout_rewards=[reward],
                N=1,
            )
            node.children.append(child)

    # ------------------------------------------------------------------
    # Rollout — batched across all children simultaneously
    # ------------------------------------------------------------------

    def _batch_rollout_children(self, children: list['MORENode'],
                                parent_depth: int) -> None:
        """
        Run one random rollout for every non-terminal child in a single
        vectorised sweep.  Each rollout step issues ONE batch_evaluate call
        for all alive children instead of one call per child per step.

        Mutates each child's rollout_rewards and N in place.
        """
        import random

        alive_idx = [i for i, c in enumerate(children)
                     if not c.done and not c.dead_end]
        if not alive_idx:
            return

        states   = {i: copy.deepcopy(children[i].state) for i in alive_idx}
        returns  = {i: 0.0 for i in alive_idx}
        discount = 1.0

        for step in range(self.rollout_depth):
            if not alive_idx:
                break
            depth = parent_depth + 1 + step
            if depth >= self.max_depth:
                break

            # Sample one random action per alive child
            pairs: list = []
            valid: list = []
            for ci in alive_idx:
                cands = self.sampler.sample(states[ci], k_per_object=4)
                if cands:
                    pairs.append((states[ci], random.choice(cands)))
                    valid.append(ci)

            if not pairs:
                break

            results = self._chunked_batch_evaluate(pairs)

            next_alive: list = []
            for ci, (new_state, reward, done) in zip(valid, results):
                returns[ci] += discount * reward
                if not done and not self.env._obstacles_dropped(new_state):
                    states[ci] = new_state
                    next_alive.append(ci)

            alive_idx = next_alive
            discount *= self.gamma

        for i, child in enumerate(children):
            child.rollout_rewards.append(returns.get(i, 0.0))
            child.N += 1

    def _chunked_batch_evaluate(self, pairs: list) -> list:
        """Split pairs into n_envs-sized chunks to avoid overflowing the env pool."""
        n = self.env.n_envs
        results = []
        for i in range(0, len(pairs), n):
            results.extend(self.env.batch_evaluate(pairs[i:i + n]))
        return results

    def _rollout(self, state: dict, start_depth: int) -> float:
        """Single-trajectory rollout (kept for testing; not used in search())."""
        import random
        total = 0.0
        discount = 1.0
        current = copy.deepcopy(state)
        for step in range(self.rollout_depth):
            if start_depth + step >= self.max_depth:
                break
            candidates = self.sampler.sample(current, k_per_object=4)
            if not candidates:
                break
            action = random.choice(candidates)
            results = self.env.batch_evaluate([(current, action)])
            new_state, reward, done = results[0]
            total += discount * reward
            discount *= self.gamma
            if done or self.env._obstacles_dropped(new_state):
                break
            current = new_state
        return total

    # ------------------------------------------------------------------
    # Data export (Phase A logging)
    # ------------------------------------------------------------------

    def collect_transitions(self, root: MORENode) -> list[dict]:
        """
        Walk the tree and yield (state, action, Q_mcts, N) records for PPN training.
        Used by the Phase A data-collection script.
        """
        records = []
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                if child.action is not None and child.rollout_rewards:
                    records.append({
                        'state':  child.parent.state,
                        'action': child.action,
                        'Q':      child.q_best(),
                        'N':      child.N,
                    })
                stack.append(child)
        return records
