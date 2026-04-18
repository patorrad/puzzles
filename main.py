"""
main.py  –  Bin-clearing planning demo with Genesis.

Usage
-----
# Run MCTS planner (headless, then replay with viewer):
python main.py --planner mcts

# Run RRT planner:
python main.py --planner rrt

# Parallel MCTS (8 envs evaluated simultaneously on GPU):
python main.py --planner mcts --parallel-envs 8

Options
-------
--planner            : mcts | rrt  (default: mcts)
--n-obstacles        : int (default: 2)
--n-simulations      : MCTS simulations (default: 80)
--max-iter           : RRT iterations (default: 150)
--show-during-planning : open viewer while planning (single-env mode only)
--parallel-envs N    : MCTS parallel GPU environments (0 = disabled)
--no-replay          : skip replay of solution
--seed               : random seed
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
    p.add_argument('--stackable', action='store_true')
    p.add_argument('--friction', type=float, default=1.0)
    p.add_argument('--n-z-levels', type=int, default=1)
    p.add_argument('--push-steps', type=int, default=80)
    p.add_argument('--substeps', type=int, default=4)
    p.add_argument('--show-during-planning', action='store_true')
    p.add_argument('--visualize-search', action='store_true')
    p.add_argument('--pause-search', action='store_true')
    p.add_argument('--parallel-envs', type=int, default=0,
                   help='MCTS: parallel Genesis envs for batch evaluation (0=off)')
    p.add_argument('--no-replay', action='store_true')
    p.add_argument('--visualize', action='store_true',
                   help='Show matplotlib tree after planning')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--save', type=str, default=None, metavar='FILE',
                   help='Save solution + env to a JSON file for the MPPI sim')
    p.add_argument('--load-state', type=str, default=None, metavar='FILE',
                   help='Load initial block positions from a solution JSON')
    p.add_argument('--target-pos', type=float, nargs=2, default=None,
                   metavar=('X', 'Y'),
                   help='Fixed starting position of the target block (m)')
    p.add_argument('--obstacle-pos', type=float, nargs=2, action='append',
                   default=None, metavar=('X', 'Y'),
                   dest='obstacle_pos',
                   help='Fixed position for an obstacle (repeat for each obstacle)')
    p.add_argument('--bin-center', type=float, nargs=2, default=None,
                   metavar=('X', 'Y'),
                   help='Centre of the bin in world coordinates (default: 0.15 0.15)')
    # Internal: used when this script relaunches itself just for replay
    p.add_argument('--_replay-file', default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Replay (runs in a subprocess with a clean Genesis context)
# ---------------------------------------------------------------------------

def _simulate_plan_steps(env, plan: list[dict], initial_state: dict) -> list[dict]:
    """
    Re-simulate each action and record:
      - start/end pose of the displaced object
      - start/end pose of the target (for full state visibility at every step)

    Returns a list of dicts with:
        obj_idx      : which object moved
        obj_name     : 'target' or 'obstacle_N'
        start_pos    : [x, y, z] of displaced object before push
        start_quat   : [w, x, y, z] of displaced object before push
        end_pos      : [x, y, z] of displaced object after push
        end_quat     : [w, x, y, z] of displaced object after push
        target_start_pos  : [x, y, z] of target before push
        target_start_quat : [w, x, y, z] of target before push
        target_end_pos    : [x, y, z] of target after push
        target_end_quat   : [w, x, y, z] of target after push
    """
    from planner import _execute_action

    steps = []
    state = initial_state
    env.set_state(state)

    for action in plan:
        obj_idx = int(action['obj_idx'])
        obj_name = 'target' if obj_idx == 0 else f'obstacle_{obj_idx - 1}'

        # Poses before
        if obj_idx == 0:
            start_pos  = state['target_pos'].tolist()
            start_quat = state['target_quat'].tolist()
        else:
            start_pos  = state['obstacle_pos'][obj_idx - 1].tolist()
            start_quat = state['obstacle_quat'][obj_idx - 1].tolist()
        target_start_pos  = state['target_pos'].tolist()
        target_start_quat = state['target_quat'].tolist()

        new_state, _, _ = _execute_action(env, action)
        state = new_state

        # Poses after
        if obj_idx == 0:
            end_pos  = state['target_pos'].tolist()
            end_quat = state['target_quat'].tolist()
        else:
            end_pos  = state['obstacle_pos'][obj_idx - 1].tolist()
            end_quat = state['obstacle_quat'][obj_idx - 1].tolist()
        target_end_pos  = state['target_pos'].tolist()
        target_end_quat = state['target_quat'].tolist()

        steps.append({
            'obj_idx':           obj_idx,
            'obj_name':          obj_name,
            'start_pos':         start_pos,
            'start_quat':        start_quat,
            'end_pos':           end_pos,
            'end_quat':          end_quat,
            'target_start_pos':  target_start_pos,
            'target_start_quat': target_start_quat,
            'target_end_pos':    target_end_pos,
            'target_end_quat':   target_end_quat,
        })

    return steps


def save_solution(path: str, plan: list[dict], initial_state: dict, args, env):
    """
    Export the plan and environment to JSON for the MPPI robotics simulator.

    Schema
    ------
    env_config      : bin/object dimensions, friction
    initial_state   : target and obstacle poses (pos + quat)
    actors          : GenesisWrapper-compatible ActorWrapper dicts for each
                      object (target + obstacles), ready to drop into a scene
    plan            : action sequence (action_type, push_pos, push_z, obj_idx)
    steps           : per-action start/end pose of the displaced object
    """
    from env import (BIN_W, BIN_D, BIN_H, WALL_T, OBJ_SIZE, OBJ_H,
                     PUSHER_T, PUSHER_W, EXIT_Y)

    # ActorWrapper-compatible dicts (matches GenesisWrapper's ActorWrapper fields)
    def make_actor(name: str, size: list, pos: list, color: list,
                   fixed: bool = False, rho: float = 500.0) -> dict:
        return {
            'type':     'Box',
            'name':     name,
            'init_pos': [float(v) for v in pos],
            'init_ori': [0.0, 0.0, 0.0, 1.0],
            'size':     [float(v) for v in size],
            'rho':      rho,
            'friction': float(args.friction),
            'fixed':    fixed,
            'color':    color,
        }

    obj_size  = [float(OBJ_SIZE)] * 3
    wall_color = [0.5, 0.5, 0.8]
    floor_color = [0.7, 0.6, 0.5]

    actors = [
        # movable objects
        make_actor('target', obj_size, initial_state['target_pos'].tolist(),
                   [0.9, 0.2, 0.2], rho=50.0),
        *[make_actor(f'obstacle_{i}', obj_size, pos.tolist(), [0.3, 0.5, 0.9])
          for i, pos in enumerate(initial_state['obstacle_pos'])],
        # static bin geometry
        make_actor('floor', [BIN_W + 2*WALL_T, BIN_D + 2*WALL_T, WALL_T],
                   [BIN_W/2, BIN_D/2, -WALL_T/2], floor_color, fixed=True),
        make_actor('wall_north', [BIN_W + 2*WALL_T, WALL_T, BIN_H],
                   [BIN_W/2, BIN_D + WALL_T/2, BIN_H/2], wall_color, fixed=True),
        make_actor('wall_west', [WALL_T, BIN_D, BIN_H],
                   [-WALL_T/2, BIN_D/2, BIN_H/2], wall_color, fixed=True),
        make_actor('wall_east', [WALL_T, BIN_D, BIN_H],
                   [BIN_W + WALL_T/2, BIN_D/2, BIN_H/2], wall_color, fixed=True),
    ]

    data = {
        'env_config': {
            'BIN_W':      float(BIN_W),
            'BIN_D':      float(BIN_D),
            'BIN_H':      float(BIN_H),
            'WALL_T':     float(WALL_T),
            'OBJ_SIZE':   float(OBJ_SIZE),
            'OBJ_H':      float(OBJ_H),
            'EXIT_Y':     float(EXIT_Y),
            'PUSHER_T':   float(PUSHER_T),
            'PUSHER_W':   float(PUSHER_W),
            'friction':   float(args.friction),
            'n_obstacles': args.n_obstacles,
        },
        'initial_state': {
            'target_pos':    initial_state['target_pos'].tolist(),
            'target_quat':   initial_state['target_quat'].tolist(),
            'obstacle_pos':  initial_state['obstacle_pos'].tolist(),
            'obstacle_quat': initial_state['obstacle_quat'].tolist(),
        },
        'actors': actors,
        'plan': [
            {
                'action_type': a['action_type'],
                'obj_idx':     int(a['obj_idx']),
                'push_pos':    [float(v) for v in a['push_pos']],
                'push_z':      float(a['push_z']),
            }
            for a in plan
        ],
        'steps': _simulate_plan_steps(env, plan, initial_state),
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Solution saved to {path}')


def _do_replay(args):
    """Load plan from file and replay with viewer. Called in a fresh process."""
    with open(args._replay_file) as f:
        data = json.load(f)

    plan = [
        {**a, 'push_pos': np.array(a['push_pos'])}
        for a in data['plan']
    ]
    initial_state = {
        'target_pos':    np.array(data['initial_state']['target_pos']),
        'target_quat':   np.array(data['initial_state']['target_quat']),
        'obstacle_pos':  np.array(data['initial_state']['obstacle_pos']),
        'obstacle_quat': np.array(data['initial_state']['obstacle_quat']),
    }

    from env import BinEnv
    env = BinEnv(
        n_obstacles=args.n_obstacles,
        show_viewer=True,
        seed=args.seed,
        stackable=args.stackable,
        friction=args.friction,
        n_z_levels=args.n_z_levels,
        push_steps=args.push_steps,
        substeps=args.substeps,
        bin_center=tuple(args.bin_center) if args.bin_center else None,
    )

    # Restore the exact initial state the planner used
    env.set_state(initial_state)
    if env.show_viewer:
        input('Press Enter to start replay...')

    done = False
    for step_i, action in enumerate(plan):
        print(f'  Step {step_i+1}/{len(plan)}: '
              f'[{action["action_type"]}] obj {action["obj_idx"]} '
              f'pos {np.round(action["push_pos"], 3)} z={action["push_z"]:.3f}')
        atype = action['action_type']
        if atype == 'push_n':
            _, reward, done = env.execute_ns_push(
                action['push_pos'], action['push_z'], step_delay=0.02)
        elif atype == 'pull_s':
            _, reward, done = env.execute_ns_pull(
                action['push_pos'], action['push_z'], step_delay=0.02)
        elif atype == 'push_e':
            _, reward, done = env.execute_ew_push(
                action['push_pos'], action['push_z'], direction=+1, step_delay=0.02)
        else:
            _, reward, done = env.execute_ew_push(
                action['push_pos'], action['push_z'], direction=-1, step_delay=0.02)
        print(f'    -> reward={reward:.3f}, done={done}')
        if done:
            print('  Target escaped the bin!')
            break

    if not done:
        print('  Plan executed (target may not have fully escaped).')
    if env.show_viewer:
        input('Press Enter to close viewer...')


def _launch_replay(plan, initial_state, args):
    """Serialize plan+state to a temp file, relaunch this script for replay."""
    data = {
        'plan': [
            {**a, 'push_pos': [float(v) for v in a['push_pos']],
                  'push_z': float(a['push_z']),
                  'obj_idx': int(a['obj_idx'])}
            for a in plan
        ],
        'initial_state': {
            'target_pos':    initial_state['target_pos'].tolist(),
            'target_quat':   initial_state['target_quat'].tolist(),
            'obstacle_pos':  initial_state['obstacle_pos'].tolist(),
            'obstacle_quat': initial_state['obstacle_quat'].tolist(),
        },
    }
    fd, path = tempfile.mkstemp(suffix='.json', prefix='puzzle_plan_')
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f)

    cmd = [
        sys.executable, __file__,
        '--_replay-file', path,
        '--n-obstacles',  str(args.n_obstacles),
        '--friction',     str(args.friction),
        '--n-z-levels',   str(args.n_z_levels),
        '--push-steps',   str(args.push_steps),
        '--substeps',     str(args.substeps),
    ]
    if args.seed is not None:
        cmd += ['--seed', str(args.seed)]
    if args.stackable:
        cmd += ['--stackable']
    if args.bin_center is not None:
        cmd += ['--bin-center', str(args.bin_center[0]), str(args.bin_center[1])]

    print(f'\nLaunching replay subprocess (plan saved to {path})...')
    subprocess.run(cmd)
    os.unlink(path)


# ---------------------------------------------------------------------------
# Build initial_positions from CLI args / --load-state
# ---------------------------------------------------------------------------

def _build_initial_positions(args) -> dict | None:
    """Return an initial_positions dict for BinEnv, or None for random placement."""
    pos = {}

    if args.load_state:
        with open(args.load_state) as f:
            data = json.load(f)
        state = data.get('initial_state', {})
        if 'target_pos' in state:
            pos['target'] = state['target_pos'][:2]
        if 'obstacle_pos' in state:
            pos['obstacles'] = [p[:2] for p in state['obstacle_pos']]

    # CLI flags override anything from --load-state
    if args.target_pos is not None:
        pos['target'] = args.target_pos
    if args.obstacle_pos is not None:
        pos['obstacles'] = args.obstacle_pos

    return pos if pos else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Replay mode: invoked by _launch_replay() in a clean process
    if args._replay_file:
        _do_replay(args)
        return

    from env import BinEnv
    from planner import RRTPusher, MCTSPusher

    using_parallel = args.parallel_envs > 0

    # In parallel mode always run headless — replay launches a fresh process
    show_viewer = (not using_parallel and
                   (args.show_during_planning or args.visualize_search
                    or args.pause_search or not args.no_replay))

    initial_positions = _build_initial_positions(args)
    print(f'Building environment: {args.n_obstacles} obstacle(s), seed={args.seed}')
    if initial_positions:
        if 'target' in initial_positions:
            print(f'  target pos: {initial_positions["target"]}')
        for i, p in enumerate(initial_positions.get('obstacles', [])):
            print(f'  obstacle_{i} pos: {p}')
    env = BinEnv(
        n_obstacles=args.n_obstacles,
        show_viewer=show_viewer,
        seed=args.seed,
        stackable=args.stackable,
        friction=args.friction,
        n_z_levels=args.n_z_levels,
        push_steps=args.push_steps,
        substeps=args.substeps,
        initial_positions=initial_positions,
        bin_center=tuple(args.bin_center) if args.bin_center else None,
    )
    initial_state = env.get_state()
    print(f'Target start: {np.round(initial_state["target_pos"], 3)}')

    # ---- run planner ----
    t0 = time.time()

    if args.planner == 'mcts':
        if using_parallel:
            from parallel_env import ParallelBinEnv
            from planner import ParallelMCTSPusher
            print(f'\nBuilding ParallelBinEnv ({args.parallel_envs} envs)...')
            penv = ParallelBinEnv(
                n_envs=args.parallel_envs,
                n_obstacles=args.n_obstacles,
                friction=args.friction,
                n_z_levels=args.n_z_levels,
                push_steps=args.push_steps,
                substeps=args.substeps,
            )
            print(f'Running Parallel MCTS ({args.n_simulations} sims, '
                  f'{args.parallel_envs} envs)...')
            planner = ParallelMCTSPusher(
                env=env,
                parallel_env=penv,
                n_simulations=args.n_simulations,
                rollout_depth=4,
                max_depth=10,
                n_children=args.parallel_envs,
                n_rollouts=args.parallel_envs,
                seed=args.seed,
            )
        else:
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
        if using_parallel:
            from parallel_env import ParallelBinEnv
            from planner import ParallelRRTPusher
            print(f'\nBuilding ParallelBinEnv ({args.parallel_envs} envs)...')
            penv = ParallelBinEnv(
                n_envs=args.parallel_envs,
                n_obstacles=args.n_obstacles,
                friction=args.friction,
                n_z_levels=args.n_z_levels,
                push_steps=args.push_steps,
                substeps=args.substeps,
            )
            print(f'Running Parallel RRT ({args.max_iter} batch iters × '
                  f'{args.parallel_envs} envs = '
                  f'~{args.max_iter * args.parallel_envs} evals)...')
            planner = ParallelRRTPusher(
                env=env,
                parallel_env=penv,
                max_iter=args.max_iter,
                max_depth=12,
                seed=args.seed,
            )
        else:
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

    print(f'\nPlanning took {time.time() - t0:.1f}s')

    if args.visualize:
        import viz
        viz.plot_rrt_tree(planner) if args.planner == 'rrt' else viz.plot_mcts_tree(planner)
        viz.show()

    if not plan:
        print('No plan found.')
        return

    print(f'Plan found: {len(plan)} actions')
    for i, a in enumerate(plan):
        print(f'  {i+1}. [{a["action_type"]}] obj={a["obj_idx"]} '
              f'pos={np.round(a["push_pos"], 3)} z={a["push_z"]:.3f}')

    if args.save:
        save_solution(args.save, plan, initial_state, args, env)

    # ---- replay ----
    if not args.no_replay:
        if using_parallel:
            # Fresh subprocess = clean Genesis context, no scene conflicts
            _launch_replay(plan, initial_state, args)
        else:
            # Single-env mode: reuse the existing env
            env.reset()
            if env.show_viewer:
                input('Press Enter to start replay...')
            for step_i, action in enumerate(plan):
                atype = action['action_type']
                print(f'  Step {step_i+1}: [{atype}]')
                if atype == 'push_n':
                    env.execute_ns_push(action['push_pos'], action['push_z'], step_delay=0.02)
                elif atype == 'pull_s':
                    env.execute_ns_pull(action['push_pos'], action['push_z'], step_delay=0.02)
                elif atype == 'push_e':
                    env.execute_ew_push(action['push_pos'], action['push_z'],
                                        direction=+1, step_delay=0.02)
                else:
                    env.execute_ew_push(action['push_pos'], action['push_z'],
                                        direction=-1, step_delay=0.02)
            if env.show_viewer:
                input('Press Enter to close viewer...')


if __name__ == '__main__':
    main()
