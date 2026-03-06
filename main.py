"""
main.py  –  Bin-clearing planning demo with Genesis.

Usage
-----
# Run MCTS planner (headless, then replay with viewer):
python main.py --planner mcts

# Run RRT planner:
python main.py --planner rrt

# Show viewer during planning (slower):
python main.py --planner mcts --show-during-planning

# Change number of obstacles:
python main.py --planner mcts --n-obstacles 3

Options
-------
--planner            : mcts | rrt  (default: mcts)
--n-obstacles        : int (default: 2)
--n-simulations      : MCTS simulations (default: 80)
--max-iter           : RRT iterations (default: 150)
--show-during-planning : open viewer while planning (slow)
--no-replay          : skip replay of solution
--seed               : random seed (default: 0)
"""

import argparse
import copy
import time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', default='mcts', choices=['mcts', 'rrt'])
    p.add_argument('--n-obstacles', type=int, default=2)
    p.add_argument('--n-simulations', type=int, default=80,
                   help='MCTS: number of simulations')
    p.add_argument('--max-iter', type=int, default=150,
                   help='RRT: maximum iterations')
    p.add_argument('--stackable', action='store_true',
                   help='Allow objects to be stacked on top of each other')
    p.add_argument('--friction', type=float, default=1.0,
                   help='Friction coefficient for all objects (default: 1.0)')
    p.add_argument('--show-during-planning', action='store_true')
    p.add_argument('--visualize-search', action='store_true',
                   help='RRT: draw each explored branch in the Genesis viewer live')
    p.add_argument('--pause-search', action='store_true',
                   help='RRT: pause for Enter after drawing each branch (implies --visualize-search)')
    p.add_argument('--no-replay', action='store_true')
    p.add_argument('--visualize', action='store_true',
                   help='Show matplotlib tree visualization after planning')
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


def replay_solution(plan: list[dict], env):
    """Re-run the planned action sequence. Reuses the existing env."""
    print('\n=== Replaying solution ===')

    env.reset()
    if env.show_viewer:
        input('Press Enter to start replay...')

    done = False
    for step_i, action in enumerate(plan):
        atype = action.get('action_type', 'push')
        print(f'  Step {step_i+1}/{len(plan)}: '
              f'{atype} obj {action["obj_idx"]} '
              f'pos {np.round(action["push_pos"], 3)} '
              + (f'dir {np.round(action["push_dir"], 2)}' if atype == 'push' else ''))
        if atype == 'pull':
            _, reward, done = env.execute_pull(
                action['push_pos'], pull_z=action.get('push_z'), step_delay=0.02)
        else:
            _, reward, done = env.execute_push(
                action['push_pos'], action['push_dir'],
                push_z=action.get('push_z'), step_delay=0.02)
        print(f'    -> reward={reward:.3f}, done={done}')
        if done:
            print('  Target escaped the bin!')
            break

    if not done:
        print('  Plan executed (target may not have fully escaped).')
    if env.show_viewer:
        input('Press Enter to close viewer...')


def main():
    args = parse_args()

    # ---- build planning environment (headless) ----
    from env import BinEnv
    from planner import RRTPusher, MCTSPusher

    show_viewer = args.show_during_planning or args.visualize_search or args.pause_search or not args.no_replay
    print(f'Building environment: {args.n_obstacles} obstacle(s), seed={args.seed}')
    env = BinEnv(
        n_obstacles=args.n_obstacles,
        show_viewer=show_viewer,
        seed=args.seed,
        stackable=args.stackable,
        friction=args.friction,
    )
    initial_state = env.get_state()
    target_start = initial_state['target_pos']
    print(f'Target start position: {np.round(target_start, 3)}')
    from env import EXIT_Y
    print(f'Exit condition: target_y < {EXIT_Y}')

    # ---- run planner ----
    t0 = time.time()

    if args.planner == 'mcts':
        print(f'\nRunning MCTS ({args.n_simulations} simulations)...')
        planner = MCTSPusher(
            env=env,
            n_simulations=args.n_simulations,
            rollout_depth=4,
            max_depth=10,
            n_children=4,
            seed=args.seed,
        )
        plan = planner.plan(initial_state, verbose=True)

    else:  # rrt
        print(f'\nRunning RRT ({args.max_iter} iterations)...')
        planner = RRTPusher(
            env=env,
            max_iter=args.max_iter,
            max_depth=12,
            seed=args.seed,
        )
        plan = planner.plan(initial_state, verbose=True,
                            visualize_search=args.visualize_search,
                            pause_each_iter=args.pause_search)

    elapsed = time.time() - t0
    print(f'\nPlanning took {elapsed:.1f}s')

    if args.visualize:
        import viz
        if args.planner == 'rrt':
            viz.plot_rrt_tree(planner)
        else:
            viz.plot_mcts_tree(planner)
        viz.show()

    if plan is None or len(plan) == 0:
        print('No plan found.')
        return

    print(f'Plan found: {len(plan)} actions')
    for i, a in enumerate(plan):
        atype = a.get('action_type', 'push')
        dir_str = f'dir={np.round(a["push_dir"], 2)} ' if atype == 'push' else ''
        print(f'  {i+1}. [{atype}] obj={a["obj_idx"]} pos={np.round(a["push_pos"], 3)} '
              f'{dir_str}z={a.get("push_z", 0.04):.3f}')

    # ---- evaluate plan ----
    print('\nEvaluating plan...')
    if env.show_viewer:
        input('Press Enter to start evaluation...')
    env.reset()
    done = False
    for i, action in enumerate(plan):
        print(action.get('action_type'))
        if action.get('action_type') == 'pull':
            state, reward, done = env.execute_pull(
                action['push_pos'], pull_z=action.get('push_z'))
        else:
            state, reward, done = env.execute_push(
                action['push_pos'], action['push_dir'], push_z=action.get('push_z'))
        print(f'  Step {i+1}: reward={reward:.3f}, target_y={state["target_pos"][1]:.3f}, done={done}')
        
        if done:
            break

    if done:
        print('\nSuccess! Target moved out of the bin.')
    else:
        print('\nPlan did not fully solve the task (partial progress).')
        final_y = state['target_pos'][1]
        from env import BIN_D
        progress = (BIN_D / 2 - final_y) / (BIN_D / 2 - EXIT_Y)
        print(f'Progress toward exit: {progress:.1%}')

    # ---- replay with viewer ----
    if not args.no_replay:
        replay_solution(plan, env)


if __name__ == '__main__':
    main()
