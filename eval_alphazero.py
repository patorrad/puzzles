"""Hydra entrypoint for evaluating a trained AlphaZero checkpoint.

Usage:
  # Run all three modes on the latest checkpoint
  python eval_alphazero.py

  # Specific checkpoint, more episodes
  python eval_alphazero.py checkpoint=outputs/alphazero/alphazero_iter_0095.pt n_episodes=128

  # Single mode
  python eval_alphazero.py mode=self_play

  # Stronger MCTS at eval time
  python eval_alphazero.py selfplay.n_simulations_solver=200
"""

import logging
import os

import hydra
from omegaconf import DictConfig


_VIEWER_TO_HEADLESS = {'headless': '1', 'replay': '0', 'verify': '0', 'always': '0'}


@hydra.main(version_base=None, config_path='conf', config_name='alphazero_eval')
def main(cfg: DictConfig) -> None:
    # view=true overrides viewer/parallel_envs so the user can watch episodes
    # play out cleanly, one at a time.
    if bool(cfg.get('view', False)):
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.viewer = 'always'
            cfg.parallel_envs = 1

    if cfg.simulator.name == 'isaaclab':
        os.environ.setdefault('ISAACLAB_HEADLESS',
                              _VIEWER_TO_HEADLESS.get(cfg.viewer, '1'))

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Deferred imports: AppLauncher must read env vars first.
    from simulators import build_env
    from alphazero.eval import evaluate, load_checkpoint, UniformNet
    from alphazero.selfplay import SelfPlayConfig

    logger.info('Loading checkpoint: %s', cfg.checkpoint)
    solver_net, stacker_net, spec = load_checkpoint(cfg.checkpoint)
    solver_net.to(cfg.device); stacker_net.to(cfg.device)

    logger.info('Building env (%s): n_obstacles=%d parallel_envs=%d',
                cfg.simulator.name, cfg.n_obstacles, cfg.parallel_envs)
    env = build_env(cfg, n_envs=cfg.parallel_envs, viewer_mode=cfg.viewer)

    sp_cfg = SelfPlayConfig(**dict(cfg.selfplay))
    render = bool(cfg.get('view', False))
    pause = render and bool(cfg.get('pause_between_eps', True))

    modes = (['self_play', 'vs_random_stacker', 'vs_random_solver']
             if cfg.mode == 'all' else [cfg.mode])
    if render and len(modes) > 1:
        logger.info('view=true: limiting to mode=self_play for viewing.')
        modes = ['self_play']

    print()
    print('=' * 72)
    print(f'AlphaZero eval — checkpoint: {cfg.checkpoint}')
    print(f'  n_episodes per mode: {cfg.n_episodes}  '
          f'sims (solver/stacker): {sp_cfg.n_simulations_solver}/{sp_cfg.n_simulations_stacker}')
    print('=' * 72)

    for mode in modes:
        if mode == 'self_play':
            label = 'self-play (trained solver vs trained stacker)'
            r = evaluate(env, solver_net, stacker_net, spec, sp_cfg,
                         n_episodes=cfg.n_episodes, device=cfg.device,
                         random_stacker=False, random_solver=False,
                         seed=cfg.seed, render=render,
                         pause_between_eps=pause)
        elif mode == 'vs_random_stacker':
            label = 'trained solver vs UNIFORM RANDOM stacker'
            r = evaluate(env, solver_net, stacker_net, spec, sp_cfg,
                         n_episodes=cfg.n_episodes, device=cfg.device,
                         random_stacker=True, random_solver=False,
                         seed=cfg.seed, render=render,
                         pause_between_eps=pause)
        elif mode == 'vs_random_solver':
            label = 'trained stacker vs UNIFORM RANDOM solver'
            r = evaluate(env, solver_net, stacker_net, spec, sp_cfg,
                         n_episodes=cfg.n_episodes, device=cfg.device,
                         random_stacker=False, random_solver=True,
                         seed=cfg.seed, render=render,
                         pause_between_eps=pause)
        else:
            raise ValueError(f'unknown mode: {mode}')

        print()
        print(f'[{mode}] {label}')
        print(f'  {r.summary()}')

    print()


if __name__ == '__main__':
    main()
