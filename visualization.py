"""Shared visualization utilities for the bin-clearing puzzle."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    ax.set_xlim(-0.05, bin_w + 0.05)
    ax.set_ylim(-0.15, bin_d + 0.05)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    wt = wall_thickness
    for xy, wh in [
        ((-wt, 0),      (wt, bin_d)),   # west
        ((bin_w, 0),    (wt, bin_d)),   # east
        ((0, bin_d),    (bin_w, wt)),   # north
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

    ax.annotate('', xy=(bin_w / 2, -0.10), xytext=(bin_w / 2, 0.02),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    fig.tight_layout()
    return fig
