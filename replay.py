#!/usr/bin/env python3
"""Replay a saved MCTS solution JSON with the IsaacLab viewer."""
import argparse
import json
import logging

import torch

logger = logging.getLogger(__name__)


def run_replay(env, plan: list[dict], initial_state: dict) -> None:
    """Execute plan actions on env with viewer. Shared by main.py and CLI."""
    env.set_state(initial_state)
    if env.show_viewer:
        input('Press Enter to start replay...')

    done = False
    for i, action in enumerate(plan):
        atype = action['action_type']
        logger.info('Step %d/%d: [%s] obj=%s z=%.3f',
                    i + 1, len(plan), atype, action['obj_idx'], action['push_z'])
        if atype == 'push_n':
            _, r, done = env.execute_ns_push(action['push_pos'], action['push_z'])
        elif atype == 'pull_s':
            _, r, done = env.execute_ns_pull(action['push_pos'], action['push_z'])
        elif atype == 'push_e':
            _, r, done = env.execute_ew_push(action['push_pos'], action['push_z'], direction=+1)
        else:
            _, r, done = env.execute_ew_push(action['push_pos'], action['push_z'], direction=-1)
        logger.info('  reward=%.3f done=%s', r, done)
        if done:
            logger.info('Target escaped!')
            break

    if not done:
        logger.info('Plan complete (target may not have fully escaped).')
    if env.show_viewer:
        input('Press Enter to close...')


def main():
    import os
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    p = argparse.ArgumentParser(description='Replay a saved MCTS solution.')
    p.add_argument('solution', help='Path to solution JSON')
    p.add_argument('--push-steps', type=int, default=256)
    p.add_argument('--video', metavar='PATH', help='Record video to this .mp4 path (headless)')
    args = p.parse_args()

    # Both env vars must be set before the IsaacLab module is imported.
    if args.video:
        os.environ['ISAACLAB_ENABLE_CAMERAS'] = '1'
        os.environ['ISAACLAB_HEADLESS'] = '1'

    with open(args.solution) as f:
        data = json.load(f)

    ec = data['env_config']
    plan = [{**a, 'push_pos': torch.tensor(a['push_pos'])} for a in data['plan']]
    initial_state = {k: torch.tensor(v) for k, v in data['initial_state'].items()}

    from simulators.isaaclab_env import BinEnvIsaacLab
    viewer_mode = 'headless' if args.video else 'always'
    env = BinEnvIsaacLab(
        n_obstacles=ec['n_obstacles'],
        n_envs=1,
        friction=ec['friction'],
        wall_thickness=ec['WALL_T'],
        obj_size=ec['OBJ_SIZE'],
        bin_size=ec['BIN_W'],
        push_steps=args.push_steps,
        viewer_mode=viewer_mode,
    )

    if args.video:
        logger.info('Recording → %s', args.video)
        env.record_replay(plan, initial_state, video_path=args.video)
    else:
        run_replay(env, plan, initial_state)


if __name__ == '__main__':
    main()
