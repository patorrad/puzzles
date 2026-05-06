"""
main.py  –  Bin-clearing planning demo.

Usage
-----
# Run MCTS with Genesis (defaults):
python main.py

# Switch simulator or planner via config group override:
python main.py simulator=isaaclab
python main.py planner=rrt

# Override individual values:
python main.py n_obstacles=3 wall_thickness=0.1 seed=42

# Parallel MCTS:
python main.py parallel_envs=8

# Show viewer during planning:
python main.py viewer=always

# Full example:
python main.py simulator=isaaclab planner=mcts n_obstacles=2 wall_thickness=0.15

# Run on a fixed preset environment (poses defined in conf/scenario/example.yaml):
python main.py scenario=example

# Preset + override simulator/planner:
python main.py scenario=example simulator=isaaclab planner=rrt
"""

import os
import sys
import json
import subprocess
import tempfile
import time

import torch
import hydra
from omegaconf import DictConfig

from simulators import build_env


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _simulate_plan_steps(env, plan: list[dict], initial_state: dict) -> list[dict]:
    steps = []
    state = initial_state
    env.set_state(state)

    for action in plan:
        obj_idx  = int(action['obj_idx'])
        obj_name = 'target' if obj_idx == 0 else f'obstacle_{obj_idx - 1}'

        if obj_idx == 0:
            start_pos  = state['target_pos'].tolist()
            start_quat = state['target_quat'].tolist()
        else:
            start_pos  = state['obstacle_pos'][obj_idx - 1].tolist()
            start_quat = state['obstacle_quat'][obj_idx - 1].tolist()
        target_start_pos  = state['target_pos'].tolist()
        target_start_quat = state['target_quat'].tolist()

        (new_state, _, _), = env.batch_evaluate([(state, action)])
        state = new_state

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


def save_solution(path: str, plan: list[dict], initial_state: dict,
                  cfg: DictConfig, env):
    BIN_W    = env.bin_w
    BIN_D    = env.bin_d
    BIN_H    = 0.5
    OBJ_SIZE = env._OBJ_SIZE
    OBJ_H    = OBJ_SIZE / 2
    EXIT_Y   = -0.05
    PUSHER_T = 0.012
    PUSHER_W = OBJ_SIZE * 0.88
    wt       = cfg.wall_thickness

    def make_actor(name, size, pos, color, fixed=False, rho=500.0):
        return {
            'type': 'Box', 'name': name,
            'init_pos': [float(v) for v in pos],
            'init_ori': [0.0, 0.0, 0.0, 1.0],
            'size': [float(v) for v in size],
            'rho': rho, 'friction': float(cfg.friction),
            'fixed': fixed, 'color': color,
        }

    wall_color  = [0.5, 0.5, 0.8]
    floor_color = [0.7, 0.6, 0.5]
    actors = [
        make_actor('target', [OBJ_SIZE]*3, initial_state['target_pos'].tolist(),
                   [0.9, 0.2, 0.2], rho=50.0),
        *[make_actor(f'obstacle_{i}', [OBJ_SIZE]*3, pos.tolist(), [0.3, 0.5, 0.9])
          for i, pos in enumerate(initial_state['obstacle_pos'])],
        make_actor('floor', [BIN_W+2*wt, BIN_D+2*wt, wt],
                   [BIN_W/2, BIN_D/2, -wt/2], floor_color, fixed=True),
        make_actor('wall_north', [BIN_W+2*wt, wt, BIN_H],
                   [BIN_W/2, BIN_D+wt/2, BIN_H/2], wall_color, fixed=True),
        make_actor('wall_west',  [wt, BIN_D, BIN_H],
                   [-wt/2, BIN_D/2, BIN_H/2], wall_color, fixed=True),
        make_actor('wall_east',  [wt, BIN_D, BIN_H],
                   [BIN_W+wt/2, BIN_D/2, BIN_H/2], wall_color, fixed=True),
    ]

    data = {
        'env_config': {
            'BIN_W': BIN_W, 'BIN_D': BIN_D, 'BIN_H': BIN_H,
            'WALL_T': wt, 'OBJ_SIZE': OBJ_SIZE, 'OBJ_H': OBJ_H,
            'EXIT_Y': EXIT_Y, 'PUSHER_T': PUSHER_T, 'PUSHER_W': PUSHER_W,
            'friction': float(cfg.friction), 'n_obstacles': cfg.n_obstacles,
        },
        'initial_state': {
            'target_pos':    initial_state['target_pos'].tolist(),
            'target_quat':   initial_state['target_quat'].tolist(),
            'obstacle_pos':  initial_state['obstacle_pos'].tolist(),
            'obstacle_quat': initial_state['obstacle_quat'].tolist(),
        },
        'actors': actors,
        'plan': [
            {'action_type': a['action_type'], 'obj_idx': int(a['obj_idx']),
             'push_pos': [float(v) for v in a['push_pos']], 'push_z': float(a['push_z'])}
            for a in plan
        ],
        'steps': _simulate_plan_steps(env, plan, initial_state),
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Solution saved to {path}')




def _do_replay(cfg: DictConfig):
    """Load plan from file and replay with viewer. Called in a fresh process."""
    with open(cfg.replay_file) as f:
        data = json.load(f)

    plan = [{**a, 'push_pos': torch.tensor(a['push_pos'])} for a in data['plan']]
    initial_state = {
        'target_pos':    torch.tensor(data['initial_state']['target_pos']),
        'target_quat':   torch.tensor(data['initial_state']['target_quat']),
        'obstacle_pos':  torch.tensor(data['initial_state']['obstacle_pos']),
        'obstacle_quat': torch.tensor(data['initial_state']['obstacle_quat']),
    }

    env = build_env(cfg, n_envs=1, show_viewer=True)
    env.set_state(initial_state)
    if env.show_viewer:
        input('Press Enter to start replay...')

    done = False
    for step_i, action in enumerate(plan):
        atype = action['action_type']
        print(f'  Step {step_i+1}/{len(plan)}: [{atype}] obj {action["obj_idx"]} '
              f'pos {torch.round(action["push_pos"], decimals=3)} z={action["push_z"]:.3f}')
        if atype == 'push_n':
            _, reward, done = env.execute_ns_push(action['push_pos'], action['push_z'])
        elif atype == 'pull_s':
            _, reward, done = env.execute_ns_pull(action['push_pos'], action['push_z'])
        elif atype == 'push_e':
            _, reward, done = env.execute_ew_push(action['push_pos'], action['push_z'], direction=+1)
        else:
            _, reward, done = env.execute_ew_push(action['push_pos'], action['push_z'], direction=-1)
        print(f'    -> reward={reward:.3f}, done={done}')
        if done:
            print('  Target escaped the bin!')
            break

    if not done:
        print('  Plan executed (target may not have fully escaped).')
    if env.show_viewer:
        input('Press Enter to close viewer...')


def _launch_replay(plan, initial_state, cfg: DictConfig):
    """Serialize plan+state to a temp file, relaunch this script for replay."""
    data = {
        'plan': [
            {**a, 'push_pos': [float(v) for v in a['push_pos']],
             'push_z': float(a['push_z']), 'obj_idx': int(a['obj_idx'])}
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
        f'replay_file={path}',
        f'simulator={cfg.simulator.name}',
        f'n_obstacles={cfg.n_obstacles}',
        f'friction={cfg.friction}',
        f'n_z_levels={cfg.n_z_levels}',
        f'push_steps={cfg.push_steps}',
        f'substeps={cfg.substeps}',
        f'wall_thickness={cfg.wall_thickness}',
        f'reward.target_progress.enabled={cfg.reward.target_progress.enabled}',
        f'reward.obstacle_penalty.enabled={cfg.reward.obstacle_penalty.enabled}',
        f'reward.obstacle_penalty.weight={cfg.reward.obstacle_penalty.weight}',
        f'reward.path_blocker.enabled={cfg.reward.path_blocker.enabled}',
        f'reward.path_blocker.weight={cfg.reward.path_blocker.weight}',
        f'reward.path_blocker.scale={cfg.reward.path_blocker.scale}',
        'hydra.run.dir=.',
        'hydra.output_subdir=null',
    ]
    if cfg.seed is not None:
        cmd.append(f'seed={cfg.seed}')
    if cfg.stackable:
        cmd.append('stackable=true')

    print(f'\nLaunching replay subprocess (plan saved to {path})...')
    subprocess.run(cmd)
    os.unlink(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Replay mode: invoked by _launch_replay() in a clean process
    if cfg.replay_file is not None:
        _do_replay(cfg)
        return

    from planner import RRTPusher, MCTSPusher

    if cfg.get('debug'):
        from omegaconf import OmegaConf
        print('=== Config ===')
        print(OmegaConf.to_yaml(cfg))
        print('==============')

    sim_name = cfg.simulator.name
    viewer_mode = cfg.viewer  # 'headless' | 'replay' | 'verify' | 'always'

    if sim_name == 'isaaclab' and viewer_mode == 'headless':
        os.environ['ISAACLAB_HEADLESS'] = '1'

    show_viewer = viewer_mode == 'always'

    # Scenario config groups land under cfg.scenario.* (no @package _global_).
    # Propagate env-relevant keys into root cfg before build_env reads them.
    if cfg.get('scenario') is not None:
        from omegaconf import OmegaConf, open_dict
        sc = cfg.scenario
        with open_dict(cfg):
            for key in ('n_obstacles', 'n_z_levels', 'target_z_level',
                        'stackable', 'difficult_spawn',
                        'bin_size', 'wall_thickness', 'friction'):
                if key in sc:
                    cfg[key] = sc[key]

    print(f'Building environment ({sim_name}): {cfg.n_obstacles} obstacle(s), '
          f'wall_thickness={cfg.wall_thickness}, seed={cfg.seed}')
    env = build_env(cfg, n_envs=cfg.parallel_envs, show_viewer=show_viewer,
                    viewer_mode=viewer_mode)

    if cfg.get('scenario') is not None:
        from omegaconf import OmegaConf
        sc = cfg.scenario
        obs_list = OmegaConf.to_container(sc.initial_state.obstacles, resolve=True)
        initial_state = {
            'target_pos':    torch.tensor(sc.initial_state.target_pos),
            'target_quat':   torch.tensor(sc.initial_state.target_quat),
            'obstacle_pos':  torch.tensor([o['pos']  for o in obs_list]),
            'obstacle_quat': torch.tensor([o['quat'] for o in obs_list]),
        }
        env.set_state(initial_state)
    else:
        initial_state = env.get_state(0)

    print(f'Target start: {torch.round(initial_state["target_pos"].cpu(), decimals=3)}')

    # ---- run planner ----
    t0 = time.time()

    if cfg.planner.name == 'mcts':
        print(f'\nRunning MCTS ({cfg.planner.n_simulations} simulations)...')
        planner = MCTSPusher(
            env=env,
            n_simulations=cfg.planner.n_simulations,
            rollout_depth=cfg.planner.rollout_depth,
            max_depth=cfg.planner.max_depth,
            seed=cfg.seed,
            verify_threshold=cfg.verify_threshold,
        )
        plan = planner.plan(initial_state, verbose=True,
                            pause_before_verify=cfg.pause_before_verify)

    else:  # rrt
        print(f'\nRunning RRT ({cfg.planner.max_iter} iterations)...')
        planner = RRTPusher(
            env=env,
            max_iter=cfg.planner.max_iter,
            max_depth=cfg.planner.max_depth,
            seed=cfg.seed,
            verify_threshold=cfg.verify_threshold,
        )
        plan = planner.plan(initial_state, verbose=True,
                            pause_before_verify=cfg.pause_before_verify)

    print(f'\nPlanning took {time.time() - t0:.1f}s')

    if not plan:
        print('No plan found.')
        return

    print(f'Plan found: {len(plan)} actions')
    for i, a in enumerate(plan):
        print(f'  {i+1}. [{a["action_type"]}] obj={a["obj_idx"]} '
              f'pos={torch.round(a["push_pos"], decimals=3)} z={a["push_z"]:.3f}')

    if cfg.save:
        save_solution(cfg.save, plan, initial_state, cfg, env)

    if cfg.mppi_output:
        save_solution(cfg.mppi_output, plan, initial_state, cfg, env)

    # ---- replay ----
    if viewer_mode != 'headless':
        if sim_name == 'isaaclab':
            env.replay(plan, initial_state)
        else:
            _launch_replay(plan, initial_state, cfg)


if __name__ == '__main__':
    main()
