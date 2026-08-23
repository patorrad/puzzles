"""Shared visualization utilities for the bin-clearing puzzle."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np


def render_scenario(state: dict, bin_w: float, bin_d: float,
                    obj_size: float, wall_thickness: float,
                    title: str = "Initial configuration") -> plt.Figure:
    """Render a top-down 2D view of a bin scenario.

    Returns a matplotlib Figure. Caller is responsible for saving or wrapping
    (e.g. fig.savefig(...) or wandb.Image(fig)).
    """
    half = obj_size / 2
    obj_h = obj_size / 2

    fig, ax = plt.subplots(figsize=(4, 4))
    # x-axis = NS/forward (exit at left, north at right)
    # y-axis = EW/lateral
    ax.set_xlim(-0.15, bin_d + 0.05)
    ax.set_ylim(-0.05, bin_w + 0.05)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    wt = wall_thickness
    for xy, wh in [
        ((0,     -wt),  (bin_d, wt)),           # west  (EW = 0)
        ((0,  bin_w),   (bin_d, wt)),           # east  (EW = bin_w)
        ((bin_d, -wt),  (wt, bin_w + 2 * wt)), # north (NS = bin_d)
        # south/exit opening at NS = 0 is intentionally left open
    ]:
        ax.add_patch(patches.Rectangle(xy, wh[0], wh[1], color='#333'))

    n_obstacles = len(state['obstacle_pos'])
    for i in range(n_obstacles):
        x = float(state['obstacle_pos'][i][0])
        y = float(state['obstacle_pos'][i][1])
        z = float(state['obstacle_pos'][i][2])
        z_level = round((z - obj_h) / obj_size)
        ax.add_patch(patches.Rectangle(
            (x - half, y - half), obj_size, obj_size,
            color='steelblue', ec='navy', lw=0.5,
            zorder=4 + z_level,
        ))
        if z_level > 0:
            ax.text(x + half * 0.55, y + half * 0.55, str(z_level + 1),
                    fontsize=6, color='white', fontweight='bold',
                    ha='center', va='center', zorder=5 + z_level)

    tx = float(state['target_pos'][0])
    ty = float(state['target_pos'][1])
    tz = float(state['target_pos'][2])
    tz_level = round((tz - obj_h) / obj_size)
    ax.add_patch(patches.Rectangle(
        (tx - half, ty - half), obj_size, obj_size,
        color='crimson', ec='darkred', lw=0.5,
        zorder=4 + tz_level,
    ))
    if tz_level > 0:
        ax.text(tx + half * 0.55, ty + half * 0.55, str(tz_level + 1),
                fontsize=6, color='white', fontweight='bold',
                ha='center', va='center', zorder=5 + tz_level)

    ax.annotate('', xy=(-0.10, bin_w / 2), xytext=(0.02, bin_w / 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.set_xlabel('NS / forward (m)   exit ←')
    ax.set_ylabel('EW / lateral (m)')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _bin_to_mppi(x_bin: float, y_bin: float):
    """Convert bin-frame (x, y) to MPPI local (x_mppi, y_mppi).

    Bin frame: x = NS/forward, y = EW/lateral.
    x_mppi is the forward-reach axis (maps to bin x / north-south).
    y_mppi is the lateral axis (maps to bin y / east-west) with a 0.20 m
    physical gap at y_mppi ∈ (-0.10, 0.10) where the robot arm body sits.
    """
    x_mppi = x_bin + 0.10
    y_mppi = (y_bin - 0.15) + 0.10 * (1.0 if y_bin >= 0.15 else -1.0)
    return x_mppi, y_mppi


# Plot-space direction for each action type in the MPPI-frame plot.
# Axes: horizontal = y_mppi (west→east), vertical = x_mppi (south→north).
# (horiz_delta, vert_delta) per unit length along the push direction.
_MPPI_PLOT_DIRS = {
    'push_n': ( 0, +1),   # bin north = increasing x_mppi = up
    'pull_s': ( 0, -1),   # bin south = decreasing x_mppi = down
    'push_e': (+1,  0),   # bin east  = increasing y_mppi = right
    'push_w': (-1,  0),   # bin west  = decreasing y_mppi = left
}


def render_ee_trajectory_mppi(state: dict, bin_w: float, bin_d: float,
                               obj_size: float, wall_thickness: float,
                               trajectory: list,
                               plan: list | None = None,
                               title: str = "EE trajectory") -> plt.Figure:
    """Top-down overlay of EE trajectory and puzzle plan in MPPI local frame.

    Renders objects and bin walls converted to MPPI local space, then plots
    the EE trajectory (already in MPPI local) directly — no coordinate
    inversion needed, so no discontinuity artifacts from the piecewise
    bin↔MPPI transform.

    Horizontal axis = y_mppi (west left, east right).
    Vertical axis   = x_mppi (south bottom, north top).
    Trajectory color encodes EE z-height. Plan arrows colored by step order.
    """
    half = obj_size / 2
    obj_h = obj_size / 2
    wt = wall_thickness

    # Bin extents in MPPI space (x_bin=NS, y_bin=EW after coordinate swap)
    x_south = 0.10           # x_bin = 0  (NS = 0, south/exit boundary)
    x_north = bin_d + 0.10   # x_bin = bin_d (NS = bin_d, north wall)
    y_west  = -0.25          # y_bin = 0  (EW = 0, left half of robot gap)
    y_east  = bin_w - 0.05   # y_bin = bin_w (EW = bin_w, right half)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_xlim(y_west - 0.04, y_east + 0.04)
    ax.set_ylim(x_south - 0.12, x_north + 0.04)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    # Bin walls
    for xy, wh in [
        ((y_west - wt, x_south), (wt, bin_d)),                        # west
        ((y_east,      x_south), (wt, bin_d)),                        # east
        ((y_west - wt, x_north), (y_east - y_west + 2*wt, wt)),      # north
    ]:
        ax.add_patch(patches.Rectangle(xy, wh[0], wh[1], color='#333'))

    # Obstacles
    for i in range(len(state['obstacle_pos'])):
        x_b = float(state['obstacle_pos'][i][0])
        y_b = float(state['obstacle_pos'][i][1])
        z_b = float(state['obstacle_pos'][i][2])
        z_level = round((z_b - obj_h) / obj_size)
        xm, ym = _bin_to_mppi(x_b, y_b)
        ax.add_patch(patches.Rectangle(
            (ym - half, xm - half), obj_size, obj_size,
            color='steelblue', ec='navy', lw=0.5, zorder=4 + z_level,
        ))
        if z_level > 0:
            ax.text(ym + half*0.55, xm + half*0.55, str(z_level + 1),
                    fontsize=6, color='white', fontweight='bold',
                    ha='center', va='center', zorder=5 + z_level)

    # Target
    tx = float(state['target_pos'][0])
    ty = float(state['target_pos'][1])
    tz = float(state['target_pos'][2])
    tz_level = round((tz - obj_h) / obj_size)
    txm, tym = _bin_to_mppi(tx, ty)
    ax.add_patch(patches.Rectangle(
        (tym - half, txm - half), obj_size, obj_size,
        color='crimson', ec='darkred', lw=0.5, zorder=4 + tz_level,
    ))

    # Exit arrow (target leaves south = decreasing x_mppi)
    ax.annotate('', xy=(tym, x_south - 0.08), xytext=(tym, x_south - 0.01),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Plan arrows (push_pos in bin frame → convert to MPPI)
    if plan:
        n = len(plan)
        step_colors = plt.cm.cool(np.linspace(0.15, 0.85, max(n, 1)))
        arr_len = obj_size * 0.75

        for i, action in enumerate(plan):
            px_b = float(action['push_pos'][0])
            py_b = float(action['push_pos'][1])
            xm_a, ym_a = _bin_to_mppi(px_b, py_b)
            dh, dv = _MPPI_PLOT_DIRS[action['action_type']]
            color = step_colors[i]

            ax.annotate('',
                xy=(ym_a + dh * arr_len, xm_a + dv * arr_len),
                xytext=(ym_a, xm_a),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                mutation_scale=10),
                zorder=13)

            obj_label = 'T' if action['obj_idx'] == 0 else str(action['obj_idx'])
            ax.text(ym_a + dh * (arr_len + 0.013),
                    xm_a + dv * (arr_len + 0.013),
                    f'{i+1}{obj_label}',
                    fontsize=5.5, color=color, fontweight='bold',
                    ha='center', va='center', zorder=14)

    # EE trajectory — already in MPPI local (x_mppi, y_mppi, z_mppi)
    traj = np.array(trajectory)
    x_mppi_t = traj[:, 0]
    y_mppi_t = traj[:, 1]
    z_t      = traj[:, 2]

    if len(traj) > 1:
        z_min, z_max = float(z_t.min()), float(z_t.max())
        z_span = z_max - z_min
        z_norm = (z_t - z_min) / (z_span + 1e-8)

        # Plot: horizontal = y_mppi, vertical = x_mppi
        pts = np.column_stack([y_mppi_t, x_mppi_t]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        z_seg = (z_norm[:-1] + z_norm[1:]) / 2

        lc = LineCollection(segs, cmap='plasma', linewidth=2, zorder=10, alpha=0.85)
        lc.set_array(z_seg)
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        ax.autoscale_view()

        z_label = (f'EE z (m)  [{z_min:.3f} – {z_max:.3f}]'
                   if z_span > 1e-4 else f'EE z = {z_min:.3f} m')
        fig.colorbar(lc, ax=ax, label=z_label, fraction=0.046, pad=0.04)

    ax.scatter(y_mppi_t[0],  x_mppi_t[0],  color='lime',   s=50, zorder=11,
               marker='o', edgecolors='black', linewidths=0.5, label='EE start')
    ax.scatter(y_mppi_t[-1], x_mppi_t[-1], color='yellow', s=60, zorder=11,
               marker='*', edgecolors='black', linewidths=0.5, label='EE end')
    ax.legend(fontsize=6, loc='upper left')

    ax.set_xlabel('lateral / y_mppi (m)   ← W  ·  E →')
    ax.set_ylabel('reach / x_mppi (m)   S ↓  ·  N ↑')
    ax.set_title(title)
    fig.tight_layout()
    return fig
