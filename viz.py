"""
viz.py – Matplotlib visualizations for the bin-clearing planner.

Functions
---------
plot_state(state, ax=None, title='')
    Top-down snapshot of the bin and all objects.

plot_rrt_tree(planner, ax=None, title='RRT tree')
    RRT tree nodes/edges colored by reward, with the best path highlighted.

plot_mcts_tree(planner, ax=None, title='MCTS tree', max_nodes=800)
    MCTS tree with nodes sized by visit count and colored by mean reward,
    with the best path highlighted.

show()
    Convenience wrapper for plt.show().
"""

from collections import deque

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from env import BIN_D, BIN_W, EXIT_X, OBJ_SIZE

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HALF = OBJ_SIZE / 2


def _make_ax(ax, title: str) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(EXIT_X - 0.04, BIN_D + 0.06)
    ax.set_ylim(-0.06, BIN_W + 0.06)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)  ← exit')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    return ax


def _draw_bin(ax: plt.Axes):
    """Bin walls and exit zone."""
    kw = dict(color='steelblue', lw=2, zorder=1)
    ax.plot([BIN_D, BIN_D], [0, BIN_W],     **kw)   # north (+x)
    ax.plot([0, BIN_D],     [0, 0],         **kw)   # west (-y)
    ax.plot([0, BIN_D],     [BIN_W, BIN_W], **kw)   # east (+y)
    ax.axvspan(EXIT_X, 0, alpha=0.12, color='limegreen', zorder=0)
    ax.axvline(EXIT_X, color='limegreen', lw=1.2, ls='--', zorder=1,
               label=f'exit (x={EXIT_X})')


def _draw_objects(ax: plt.Axes, state: dict):
    """Target (red) and obstacles (blue) as labeled squares."""
    tx, ty = state['target_pos'][:2]
    ax.add_patch(mpatches.Rectangle(
        (tx - _HALF, ty - _HALF), OBJ_SIZE, OBJ_SIZE,
        color='crimson', alpha=0.85, zorder=4,
    ))
    ax.text(tx, ty, 'T', ha='center', va='center',
            fontsize=7, color='white', fontweight='bold', zorder=5)

    for i, pos in enumerate(state['obstacle_pos']):
        ox, oy = pos[:2]
        ax.add_patch(mpatches.Rectangle(
            (ox - _HALF, oy - _HALF), OBJ_SIZE, OBJ_SIZE,
            color='royalblue', alpha=0.75, zorder=4,
        ))
        ax.text(ox, oy, str(i), ha='center', va='center',
                fontsize=7, color='white', zorder=5)


def _path_nodes_rrt(best_node) -> list:
    """Walk parent pointers from best_node back to root."""
    nodes = []
    n = best_node
    while n is not None:
        nodes.append(n)
        n = n.parent
    nodes.reverse()
    return nodes


def _path_nodes_mcts(best_leaf) -> list:
    """Walk parent pointers from best_leaf back to root."""
    return _path_nodes_rrt(best_leaf)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_state(state: dict, ax: plt.Axes | None = None,
               title: str = 'Bin state') -> plt.Axes:
    """Top-down snapshot of the bin and all objects."""
    ax = _make_ax(ax, title)
    _draw_bin(ax)
    _draw_objects(ax, state)
    ax.legend(loc='upper right', fontsize=7)
    return ax


def plot_rrt_tree(planner, ax: plt.Axes | None = None,
                  title: str = 'RRT tree') -> plt.Axes:
    """
    Visualize the RRT search tree.

    Parameters
    ----------
    planner : RRTPusher  (after calling .plan())
    """
    tree = planner.tree
    best_node = planner.best_node

    ax = _make_ax(ax, title)
    _draw_bin(ax)

    if not tree:
        return ax

    rewards = np.array([n.reward for n in tree])
    vmin, vmax = rewards.min(), max(rewards.max(), 1e-6)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    # Edges
    for node in tree:
        if node.parent is None:
            continue
        px, py = node.parent.state['target_pos'][:2]
        cx, cy = node.state['target_pos'][:2]
        ax.plot([px, cx], [py, cy],
                color=cmap(norm(node.reward)), lw=0.6, alpha=0.55, zorder=2)

    # Nodes
    xs = [n.state['target_pos'][0] for n in tree]
    ys = [n.state['target_pos'][1] for n in tree]
    sc = ax.scatter(xs, ys, c=rewards, cmap='RdYlGn',
                    vmin=vmin, vmax=vmax, s=12, zorder=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='reward', shrink=0.7)

    # Best path
    if best_node is not None and best_node.depth > 0:
        path = _path_nodes_rrt(best_node)
        px = [n.state['target_pos'][0] for n in path]
        py = [n.state['target_pos'][1] for n in path]
        ax.plot(px, py, 'k-o', lw=2, ms=5, zorder=6, label='best path')

    _draw_objects(ax, tree[0].state)
    ax.legend(loc='upper right', fontsize=7)
    return ax


def plot_mcts_tree(planner, ax: plt.Axes | None = None,
                   title: str = 'MCTS tree',
                   max_nodes: int = 800) -> plt.Axes:
    """
    Visualize the MCTS search tree.

    Parameters
    ----------
    planner   : MCTSPusher  (after calling .plan())
    max_nodes : cap BFS traversal to avoid an unreadable plot
    """
    root = planner.root
    best_leaf = planner.best_leaf

    ax = _make_ax(ax, title)
    _draw_bin(ax)

    if root is None:
        return ax

    # BFS collect nodes
    queue = deque([root])
    nodes = []
    while queue and len(nodes) < max_nodes:
        n = queue.popleft()
        nodes.append(n)
        queue.extend(n.children)

    mean_rewards = np.array([n.mean_reward for n in nodes])
    visits = np.array([n.visits for n in nodes], dtype=float)
    vmin = mean_rewards.min()
    vmax = max(mean_rewards.max(), 1e-6)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    # Edges
    for n in nodes:
        if n.parent is None:
            continue
        px, py = n.parent.state['target_pos'][:2]
        cx, cy = n.state['target_pos'][:2]
        color = 'lightcoral' if n.dead_end else cmap(norm(n.mean_reward))
        ax.plot([px, cx], [py, cy],
                color=color, lw=0.5, alpha=0.5, zorder=2)

    # Nodes — size proportional to visit count
    xs = [n.state['target_pos'][0] for n in nodes]
    ys = [n.state['target_pos'][1] for n in nodes]
    sizes = 5 + 45 * (visits / max(visits.max(), 1))
    sc = ax.scatter(xs, ys, c=mean_rewards, cmap='RdYlGn',
                    vmin=vmin, vmax=vmax, s=sizes, zorder=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='mean reward', shrink=0.7)

    # Dead-end nodes
    dead_xs = [n.state['target_pos'][0] for n in nodes if n.dead_end]
    dead_ys = [n.state['target_pos'][1] for n in nodes if n.dead_end]
    if dead_xs:
        ax.scatter(dead_xs, dead_ys, marker='x', color='red',
                   s=30, zorder=5, label='dead end')

    # Best path
    if best_leaf is not None and best_leaf.depth > 0:
        path = _path_nodes_mcts(best_leaf)
        px = [n.state['target_pos'][0] for n in path]
        py = [n.state['target_pos'][1] for n in path]
        ax.plot(px, py, 'k-o', lw=2, ms=5, zorder=6, label='best path')

    _draw_objects(ax, root.state)
    ax.legend(loc='upper right', fontsize=7)
    return ax


def show():
    """Display all open figures."""
    plt.show()
