"""
cloth_main.py  –  Cloth folding puzzle demo with Genesis.

Usage
-----
# Run with viewer, 3 random fold actions:
python cloth_main.py

# Headless, 5 folds, custom grid:
python cloth_main.py --no-viewer --folds 5 --grid-n 6

# Slow visual replay (0.02 s between steps):
python cloth_main.py --step-delay 0.02

Options
-------
--grid-n        : checkerboard rows/columns (default 4)
--cloth-w       : cloth width  in metres   (default 0.4)
--cloth-h       : cloth height in metres   (default 0.4)
--particle-size : PBD particle spacing     (default 0.025)
--fold-steps    : arc steps per fold       (default 40)
--settle-steps  : steps to settle after fold (default 80)
--folds         : number of sequential folds to demo (default 3)
--no-viewer     : disable Genesis viewer
--step-delay    : seconds between steps for visual replay (default 0)
--seed          : random seed for action selection
"""

import argparse

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grid-n',        type=int,   default=4)
    p.add_argument('--cloth-w',       type=float, default=0.4)
    p.add_argument('--cloth-h',       type=float, default=0.4)
    p.add_argument('--particle-size', type=float, default=0.025)
    p.add_argument('--fold-steps',    type=int,   default=40)
    p.add_argument('--settle-steps',  type=int,   default=80)
    p.add_argument('--folds',         type=int,   default=3,
                   help='Number of sequential folds to demo')
    p.add_argument('--no-viewer',     action='store_true')
    p.add_argument('--step-delay',    type=float, default=0.0,
                   help='Seconds between simulation steps (slows replay)')
    p.add_argument('--seed',          type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    from cloth_env import ClothEnv

    env = ClothEnv(
        grid_n=args.grid_n,
        cloth_w=args.cloth_w,
        cloth_h=args.cloth_h,
        show_viewer=not args.no_viewer,
        particle_size=args.particle_size,
        fold_steps=args.fold_steps,
        settle_steps=args.settle_steps,
    )

    n_particles = env.cloth.n_particles
    print(f'Cloth {args.cloth_w:.2f} m × {args.cloth_h:.2f} m, '
          f'{args.grid_n}×{args.grid_n} grid, '
          f'{n_particles} PBD particles')

    actions = env.get_valid_actions()
    chosen  = [actions[i] for i in rng.choice(len(actions), size=args.folds, replace=False)]

    for step_i, action in enumerate(chosen):
        r, c, d = action['cell_row'], action['cell_col'], action['direction']
        print(f'  Fold {step_i + 1}/{args.folds}: '
              f'cell ({r}, {c}), direction={d}')
        env.execute_fold(
            cell_row=r,
            cell_col=c,
            direction=d,
            step_delay=args.step_delay,
        )

    print('Done.')
    if not args.no_viewer:
        input('Press Enter to close...')


if __name__ == '__main__':
    main()