"""
Plot benchmark results from results/ directory.

Usage:
    python plot_results.py                        # all files in results/
    python plot_results.py --results_dir results  # explicit dir
    python plot_results.py --output plots.png     # save instead of show
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_results(results_dir: str) -> pd.DataFrame:
    dfs = []
    for path in sorted(Path(results_dir).iterdir()):
        if not path.is_file():
            continue
        # Infer n_obstacles from filename (e.g. "7obs_stacked2_difficult" → 7)
        m = re.search(r'(\d+)obs', path.name)
        if m is None:
            print(f'  [skip] {path.name}: no obs count in name')
            continue
        n_obs = int(m.group(1))
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
        except Exception as e:
            print(f'  [skip] {path.name}: {e}')
            continue
        df['n_obstacles'] = n_obs
        # Clean tensor(...) strings that leak from torch tensors — numeric cols only
        skip_cols = {'planner', 'wandb_run_name', 'action_type'}
        for col in df.select_dtypes(include='object').columns:
            if col in skip_cols:
                continue
            cleaned = df[col].astype(str).str.extract(r'([-\d.]+)', expand=False)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().any():
                df[col] = converted
        df['success'] = pd.to_numeric(df['success'], errors='coerce').fillna(0).astype(int)
        df['plan_time_s'] = pd.to_numeric(df['plan_time_s'], errors='coerce')
        df['total_pairs'] = pd.to_numeric(df['total_pairs'], errors='coerce')
        dfs.append(df)
        print(f'  Loaded {path.name}: {len(df)} rows, n_obstacles={n_obs}')

    if not dfs:
        raise ValueError(f'No valid result files found in {results_dir}')
    return pd.concat(dfs, ignore_index=True)


def plot(df: pd.DataFrame, output: str | None):
    # Group key: strip the _Nobs_ part from wandb_run_name so the same
    # planner config connects across obstacle counts as a single line.
    # e.g. "n_sim_sol200_nn1024_random_stacker_7obs_stacked2_difficult"
    #   → "n_sim_sol200_nn1024_random_stacker_stacked2_difficult"
    df['run_label'] = (df['wandb_run_name']
                       .fillna(df['planner'])
                       .str.replace(r'_\d+obs', '', regex=True))
    groups = df.groupby(['run_label', 'n_obstacles'])

    agg = groups.agg(
        success_rate=('success', 'mean'),
        verify_rate_mean=('verify_rate', 'mean'),
        verify_rate_std=('verify_rate', 'std'),
        plan_time_median=('plan_time_s', 'median'),
        plan_time_mean=('plan_time_s', 'mean'),
        total_pairs_median=('total_pairs', 'median'),
        n=('success', 'count'),
    ).reset_index()
    agg['verify_rate_std'] = agg['verify_rate_std'].fillna(0)

    run_labels = sorted(agg['run_label'].unique())
    n_obs_vals = sorted(agg['n_obstacles'].unique())
    cmap = plt.colormaps['tab10']
    colors = {label: cmap(i % 10) for i, label in enumerate(run_labels)}
    _markers = ['o', 's', '^', 'D', 'v', 'P']
    obs_markers = {n: _markers[i % len(_markers)] for i, n in enumerate(n_obs_vals)}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Benchmark results', fontsize=13, fontweight='bold', y=1.01)

    # ── Plot 1: Success rate vs n_obstacles ──────────────────────────────────
    ax = axes[0]
    n_runs = len(run_labels)
    jitter_range = 0.3  # total width spread across all runs
    jitter_step = jitter_range / max(n_runs - 1, 1)
    for i, label in enumerate(run_labels):
        offset = -jitter_range / 2 + i * jitter_step
        sub = agg[agg['run_label'] == label].sort_values('n_obstacles')
        x = sub['n_obstacles'] + offset
        ax.errorbar(x, sub['success_rate'] * 100,
                    yerr=sub['verify_rate_std'] * 100,
                    marker='o', linewidth=2, capsize=4, capthick=1.5,
                    label=label, color=colors[label])
        for (_, row), xi in zip(sub.iterrows(), x):
            ax.annotate(f'n={row["n"]}', (xi, row['success_rate'] * 100),
                        textcoords='offset points', xytext=(4, 4), fontsize=7,
                        color=colors[label])
    ax.set_xlabel('Number of obstacles')
    ax.set_ylabel('Success rate (%) ± std verify rate')
    ax.set_title('Success vs obstacles')
    ax.set_xticks(n_obs_vals)
    ax.set_xlim(min(n_obs_vals) - 1.5, max(n_obs_vals) + 1.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')

    # Shared bar layout for grouped bar charts (plots 2, 3, 4, 5)
    n_runs = len(run_labels)
    bar_width = 0.8 / max(n_runs, 1)
    x_base = np.arange(len(n_obs_vals))
    offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * bar_width

    # ── Plot 2: Median plan time vs n_obstacles — grouped bar chart ──────────
    ax = axes[1]
    for i, label in enumerate(run_labels):
        sub_df = df[df['run_label'] == label].dropna(subset=['plan_time_s'])
        means, stds, xs = [], [], []
        for j, n_obs in enumerate(n_obs_vals):
            pts = sub_df[sub_df['n_obstacles'] == n_obs]['plan_time_s']
            if pts.empty:
                continue
            means.append(pts.mean())
            stds.append(pts.std() if len(pts) > 1 else 0)
            xs.append(x_base[j] + offsets[i])
        if xs:
            ax.bar(xs, means, width=bar_width * 0.9, yerr=stds,
                   color=colors[label], alpha=0.8, label=label,
                   capsize=3, error_kw={'elinewidth': 1})
    ax.set_xlabel('Number of obstacles')
    ax.set_ylabel('Mean planning time (s)')
    ax.set_title('Planning time vs obstacles')
    ax.set_xticks(x_base)
    ax.set_xticklabels(n_obs_vals)
    ax.legend(fontsize=6, loc='upper left', title='run')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Plot 3: Simulator pairs (successful runs) — grouped bar chart ─────────
    ax = axes[2]
    for i, label in enumerate(run_labels):
        sub_df = df[(df['run_label'] == label) & (df['success'] == 1)].dropna(subset=['total_pairs'])
        means, stds, xs = [], [], []
        for j, n_obs in enumerate(n_obs_vals):
            pts = sub_df[sub_df['n_obstacles'] == n_obs]['total_pairs']
            if pts.empty:
                continue
            means.append(pts.mean())
            stds.append(pts.std() if len(pts) > 1 else 0)
            xs.append(x_base[j] + offsets[i])
        if xs:
            ax.bar(xs, means, width=bar_width * 0.9, yerr=stds,
                   color=colors[label], alpha=0.8, label=label,
                   capsize=3, error_kw={'elinewidth': 1})

    ax.set_xlabel('Number of obstacles')
    ax.set_ylabel('Mean simulator pairs')
    ax.set_title('Simulator pairs (successful runs)')
    ax.set_xticks(x_base)
    ax.set_xticklabels(n_obs_vals)
    ax.legend(fontsize=6, loc='upper left', title='run')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Plot 4: Plan time scatter per episode vs n_obstacles ─────────────────
    ax = axes[3]
    for i, label in enumerate(run_labels):
        sub_df = df[df['run_label'] == label].dropna(subset=['plan_time_s'])
        means, stds, xs = [], [], []
        for j, n_obs in enumerate(n_obs_vals):
            pts = sub_df[sub_df['n_obstacles'] == n_obs]['plan_time_s']
            if pts.empty:
                continue
            means.append(pts.mean())
            stds.append(pts.std() if len(pts) > 1 else 0)
            xs.append(x_base[j] + offsets[i])
        if xs:
            ax.bar(xs, means, width=bar_width * 0.9, yerr=stds,
                   color=colors[label], alpha=0.8, label=label,
                   capsize=3, error_kw={'elinewidth': 1})
    ax.set_xlabel('Number of obstacles')
    ax.set_ylabel('Mean planning time (s)')
    ax.set_title('Planning time vs obstacles')
    ax.set_xticks(x_base)
    ax.set_xticklabels(n_obs_vals)
    ax.legend(fontsize=6, loc='upper left', title='run')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Plot 5: Number of actions vs n_obstacles ──────────────────────────────
    ax = axes[4]
    for i, label in enumerate(run_labels):
        sub_df = df[df['run_label'] == label].dropna(subset=['plan_length'])
        means, stds, xs = [], [], []
        for j, n_obs in enumerate(n_obs_vals):
            pts = sub_df[sub_df['n_obstacles'] == n_obs]['plan_length']
            if pts.empty:
                continue
            means.append(pts.mean())
            stds.append(pts.std() if len(pts) > 1 else 0)
            xs.append(x_base[j] + offsets[i])
        if xs:
            ax.bar(xs, means, width=bar_width * 0.9, yerr=stds,
                   color=colors[label], alpha=0.8, label=label,
                   capsize=3, error_kw={'elinewidth': 1})
    ax.set_xlabel('Number of obstacles')
    ax.set_ylabel('Mean number of actions')
    ax.set_title('Actions vs obstacles')
    ax.set_xticks(x_base)
    ax.set_xticklabels(n_obs_vals)
    ax.legend(fontsize=6, loc='upper left', title='run')
    ax.grid(True, alpha=0.3, axis='y')

    axes[5].set_visible(False)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved → {output}')
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--output', default='plots.png', help='Output file (default: plots.png)')
    args = p.parse_args()

    print(f'Loading from {args.results_dir}/')
    df = load_results(args.results_dir)
    print(f'Total rows: {len(df)}')
    plot(df, args.output)


if __name__ == '__main__':
    main()
