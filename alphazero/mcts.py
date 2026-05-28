"""PUCT MCTS guided by a (policy, value) network.

The same runner is used for both the solver and stacker players: each provides
its own ``Game`` (encode + transition + terminal + net forward), so the search
tree is player-specific. There is no within-tree player switching.

Children are materialized **lazily**: at expansion we record priors only, and
only call ``game.transition`` for a child when it is first selected. This
matters for the solver, whose transition is an expensive ``env.batch_evaluate``
call.

Standard AlphaGo Zero formulas:
  - Selection:   argmax_a  Q(s,a) + c_puct * P(s,a) * sqrt(ΣN) / (1 + N(a))
  - Expansion:   one network forward; record (prior) for each legal action
  - Evaluation:  leaf value = network's v (or terminal_value if terminal)
  - Backup:      N += 1, W += v, Q = W / N
"""

from __future__ import annotations

import math
from typing import Protocol

import torch

from .networks import masked_softmax


class Game(Protocol):
    """Per-player game interface for PUCT MCTS."""

    n_actions: int

    def encode(self, state) -> torch.Tensor: ...
    def legal_mask(self, state) -> torch.Tensor: ...
    def is_terminal(self, state) -> bool: ...
    def terminal_value(self, state) -> float: ...
    def transition(self, state, action_idx: int) -> tuple[object, bool]: ...


class AZNode:
    __slots__ = ('state', 'is_terminal', 'parent', 'priors',
                 'children', 'N', 'W', 'expanded')

    def __init__(self, state, parent: 'AZNode | None' = None,
                 is_terminal: bool = False):
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        # priors[a] is set during expansion; children[a] is None until first visit
        self.priors: torch.Tensor | None = None
        self.children: dict[int, AZNode | None] = {}
        self.N: int = 0
        self.W: float = 0.0
        self.expanded: bool = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class AZMCTS:
    def __init__(self, game: Game, net, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.0,
                 device: str = 'cpu'):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    @torch.no_grad()
    def _evaluate(self, state) -> tuple[torch.Tensor, float]:
        """Network forward returning (priors over legal actions, value)."""
        x = self.game.encode(state).to(self.device).unsqueeze(0)
        logits, value = self.net(x)
        mask = self.game.legal_mask(state).to(self.device)
        priors = masked_softmax(logits.squeeze(0), mask)
        return priors.cpu(), float(value.item())

    def _expand(self, node: AZNode) -> float:
        priors, v = self._evaluate(node.state)
        node.priors = priors
        node.expanded = True
        for a in torch.nonzero(priors > 0, as_tuple=False).flatten().tolist():
            node.children[a] = None  # materialize on first visit
        return v

    def _select_child(self, node: AZNode) -> int:
        total_n = max(node.N, 1)
        sqrt_total = math.sqrt(total_n)
        best_a, best_score = -1, -float('inf')
        for a in node.children:
            child = node.children[a]
            q = child.Q if child is not None else 0.0
            n = child.N if child is not None else 0
            u = self.c_puct * float(node.priors[a]) * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_a, best_score = a, score
        return best_a

    def _materialize(self, parent: AZNode, action: int) -> AZNode:
        next_state, terminal = self.game.transition(parent.state, action)
        child = AZNode(state=next_state, parent=parent, is_terminal=terminal)
        parent.children[action] = child
        return child

    def _add_dirichlet_noise(self, root: AZNode):
        if self.dirichlet_eps <= 0 or root.priors is None:
            return
        legal = torch.nonzero(root.priors > 0, as_tuple=False).flatten()
        if len(legal) == 0:
            return
        noise = torch.distributions.Dirichlet(
            torch.full((len(legal),), self.dirichlet_alpha)).sample()
        for idx, a in enumerate(legal.tolist()):
            root.priors[a] = (1 - self.dirichlet_eps) * root.priors[a] + \
                             self.dirichlet_eps * float(noise[idx])

    def run(self, root_state, n_simulations: int,
            add_root_noise: bool = False) -> tuple[AZNode, torch.Tensor]:
        """Run n_simulations PUCT iterations. Returns (root, visit counts)."""
        root = AZNode(state=root_state,
                      is_terminal=self.game.is_terminal(root_state))
        if not root.is_terminal:
            v0 = self._expand(root)
            root.N += 1
            root.W += v0
            if add_root_noise:
                self._add_dirichlet_noise(root)

        for _ in range(n_simulations):
            node = root
            path = [root]

            while node.expanded and node.children and not node.is_terminal:
                a = self._select_child(node)
                if a < 0:
                    break
                child = node.children[a]
                if child is None:
                    child = self._materialize(node, a)
                node = child
                path.append(node)

            if node.is_terminal:
                v = self.game.terminal_value(node.state)
            elif not node.expanded:
                v = self._expand(node)
            else:
                v = node.Q

            for n in path:
                n.N += 1
                n.W += v

        counts = torch.zeros(self.game.n_actions, dtype=torch.float32)
        for a, c in root.children.items():
            counts[a] = c.N if c is not None else 0
        return root, counts


@torch.no_grad()
def _batched_expand(game, net, nodes, device):
    """Expand a batch of nodes via one network forward.

    For each leaf node, sets node.priors (masked softmax) and node.expanded.
    Returns the per-node value v from the network.
    """
    if not nodes:
        return []
    xs = torch.stack([game.encode(n.state) for n in nodes]).to(device)
    masks = torch.stack([game.legal_mask(n.state) for n in nodes]).to(device)
    logits, values = net(xs)
    priors_all = masked_softmax(logits, masks).cpu()
    vs = [float(v.item()) for v in values]
    for n, priors in zip(nodes, priors_all):
        n.priors = priors
        n.expanded = True
        for a in torch.nonzero(priors > 0, as_tuple=False).flatten().tolist():
            n.children[a] = None
    return vs


def _select_child_index(node: AZNode, c_puct: float) -> int:
    total_n = max(node.N, 1)
    sqrt_total = math.sqrt(total_n)
    best_a, best_score = -1, -float('inf')
    for a in node.children:
        child = node.children[a]
        q = child.Q if child is not None else 0.0
        n = child.N if child is not None else 0
        u = c_puct * float(node.priors[a]) * sqrt_total / (1 + n)
        score = q + u
        if score > best_score:
            best_a, best_score = a, score
    return best_a


def _apply_dirichlet(root: AZNode, alpha: float, eps: float):
    if eps <= 0 or root.priors is None:
        return
    legal = torch.nonzero(root.priors > 0, as_tuple=False).flatten()
    if len(legal) == 0:
        return
    noise = torch.distributions.Dirichlet(torch.full((len(legal),), alpha)).sample()
    for idx, a in enumerate(legal.tolist()):
        root.priors[a] = (1 - eps) * root.priors[a] + eps * float(noise[idx])


@torch.no_grad()
def run_parallel(game, net, root_states: list, n_simulations: int,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.0, add_root_noise: bool = False,
                 device: str = 'cpu') -> list[tuple[AZNode, torch.Tensor]]:
    """Run K independent PUCT searches in lock-step.

    All K trees share one ``game`` and ``net`` but maintain separate roots and
    independent sub-trees. Every iteration:
      1. Descend each tree to a leaf (or to a pending transition).
      2. Batch all pending env transitions through ``game.batched_transition``.
      3. Batch all leaf network forwards.
      4. Backprop per-tree.

    Returns a list of K (root_node, visit_counts) tuples.
    """
    K = len(root_states)
    if K == 0:
        return []

    roots = [AZNode(state=s, is_terminal=game.is_terminal(s)) for s in root_states]

    # Initial root expansion
    to_expand = [r for r in roots if not r.is_terminal]
    if to_expand:
        vs = _batched_expand(game, net, to_expand, device)
        for r, v in zip(to_expand, vs):
            r.N += 1
            r.W += v
        if add_root_noise:
            for r in to_expand:
                _apply_dirichlet(r, dirichlet_alpha, dirichlet_eps)

    for _ in range(n_simulations):
        paths: list[list[AZNode]] = [None] * K
        leaves: list[AZNode | None] = [None] * K
        pending_transition: list[tuple[AZNode, int] | None] = [None] * K

        # Phase 1: descend
        for i in range(K):
            if roots[i].is_terminal:
                paths[i] = [roots[i]]
                leaves[i] = roots[i]
                continue
            node = roots[i]
            path = [node]
            while node.expanded and node.children and not node.is_terminal:
                a = _select_child_index(node, c_puct)
                if a < 0:
                    break
                child = node.children[a]
                if child is None:
                    pending_transition[i] = (node, a)
                    break
                node = child
                path.append(node)
            paths[i] = path
            leaves[i] = node  # may be None semantically if pending_transition[i] set

        # Phase 2: batched transitions for pending children
        pending_idx = [i for i in range(K) if pending_transition[i] is not None]
        if pending_idx:
            parents = [pending_transition[i][0] for i in pending_idx]
            actions = [pending_transition[i][1] for i in pending_idx]
            results = game.batched_transition([p.state for p in parents], actions)
            for i, (ns, terminal) in zip(pending_idx, results):
                parent, a = pending_transition[i]
                child = AZNode(state=ns, parent=parent, is_terminal=terminal)
                parent.children[a] = child
                paths[i] = paths[i] + [child]
                leaves[i] = child

        # Phase 3: batched NN forward for non-terminal unexpanded leaves
        to_eval_idx = [i for i in range(K)
                       if leaves[i] is not None
                       and not leaves[i].is_terminal
                       and not leaves[i].expanded]
        values = [None] * K
        if to_eval_idx:
            vs = _batched_expand(game, net, [leaves[i] for i in to_eval_idx], device)
            for i, v in zip(to_eval_idx, vs):
                values[i] = v

        # Terminal / already-expanded leaves
        for i in range(K):
            if values[i] is not None:
                continue
            leaf = leaves[i]
            if leaf is None:
                values[i] = 0.0
            elif leaf.is_terminal:
                values[i] = game.terminal_value(leaf.state)
            else:
                values[i] = leaf.Q

        # Phase 4: backprop
        for i in range(K):
            for n in paths[i]:
                n.N += 1
                n.W += values[i]

    out = []
    for root in roots:
        counts = torch.zeros(game.n_actions, dtype=torch.float32)
        for a, c in root.children.items():
            counts[a] = c.N if c is not None else 0
        out.append((root, counts))
    return out


def visit_counts_to_policy(counts: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if counts.sum() == 0:
        return counts
    # Argmax for any near-zero temperature. counts.pow(1/T) overflows to inf at
    # T<<1, so the threshold must be generous, not just 1e-6.
    if temperature < 0.05:
        out = torch.zeros_like(counts)
        out[counts.argmax()] = 1.0
        return out
    # Stable log-space normalization: pi[a] ∝ counts[a]^(1/T).
    log_c = torch.where(counts > 0, counts.log(),
                        torch.full_like(counts, float('-inf')))
    return torch.softmax(log_c / temperature, dim=-1)
