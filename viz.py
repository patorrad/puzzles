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

from env import BIN_D, BIN_W, EXIT_Y, OBJ_SIZE

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HALF = OBJ_SIZE / 2


def _make_ax(ax, title: str) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-0.06, BIN_W + 0.06)
    ax.set_ylim(EXIT_Y - 0.04, BIN_D + 0.06)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    return ax


def _draw_bin(ax: plt.Axes):
    """Bin walls and exit zone."""
    kw = dict(color='steelblue', lw=2, zorder=1)
    ax.plot([0, BIN_W], [BIN_D, BIN_D], **kw)   # north
    ax.plot([0, 0],     [0, BIN_D],     **kw)   # west
    ax.plot([BIN_W, BIN_W], [0, BIN_D], **kw)  # east
    ax.axhspan(EXIT_Y, 0, alpha=0.12, color='limegreen', zorder=0)
    ax.axhline(EXIT_Y, color='limegreen', lw=1.2, ls='--', zorder=1,
               label=f'exit (y={EXIT_Y})')


def _col_counts(state: dict) -> dict:
    """Return {obj_key: column_stack_count} for every object in state.

    Objects within OBJ_SIZE/2 of each other in (x,y) are considered the same
    column. Keys are ('target',) or ('obs', i).
    """
    _OBJ_H = OBJ_SIZE / 2
    entries = [('target', state['target_pos'])]
    for i, pos in enumerate(state['obstacle_pos']):
        entries.append((('obs', i), pos))

    columns: list[tuple[float, float, list]] = []  # (cx, cy, [keys])
    for key, pos in entries:
        x, y = float(pos[0]), float(pos[1])
        matched = None
        for col in columns:
            if ((x - col[0]) ** 2 + (y - col[1]) ** 2) ** 0.5 < OBJ_SIZE * 0.6:
                matched = col
                break
        if matched is None:
            matched = (x, y, [])
            columns.append(matched)
        matched[2].append(key)

    result = {}
    for _, _, keys in columns:
        for k in keys:
            result[k] = len(keys)
    return result


def _draw_objects(ax: plt.Axes, state: dict):
    """Target (red) and obstacles (blue) as labeled squares.

    For stacked columns the square count is shown in the top-right corner of
    each object's square. Objects with a higher z are drawn on top.
    """
    _OBJ_H = OBJ_SIZE / 2
    counts = _col_counts(state)

    def _z_level(pos) -> int:
        return round((float(pos[2]) - _OBJ_H) / OBJ_SIZE)

    def _draw_obj(x, y, z_lvl, color, alpha, label, count):
        zo = 4 + z_lvl
        ax.add_patch(mpatches.Rectangle(
            (x - _HALF, y - _HALF), OBJ_SIZE, OBJ_SIZE,
            color=color, alpha=alpha, zorder=zo,
        ))
        ax.text(x, y, label, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=zo + 1)
        if count > 1:
            ax.text(x + _HALF * 0.75, y + _HALF * 0.75, str(count),
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold', zorder=zo + 1)

    tx, ty = state['target_pos'][:2]
    _draw_obj(tx, ty, _z_level(state['target_pos']),
              'crimson', 0.85, 'T', counts['target'])

    for i, pos in enumerate(state['obstacle_pos']):
        ox, oy = pos[:2]
        _draw_obj(ox, oy, _z_level(pos),
                  'royalblue', 0.75, str(i), counts[('obs', i)])


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
