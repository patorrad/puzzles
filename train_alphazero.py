"""Hydra entrypoint for AlphaZero-style training.

Usage:
  python train_alphazero.py
  python train_alphazero.py simulator=genesis n_iterations=50 device=cuda
  python train_alphazero.py selfplay.n_simulations_solver=50 episodes_per_iter=16

NOTE: Isaac Lab reads ISAACLAB_HEADLESS / ISAACLAB_ENABLE_CAMERAS at module-load
time (simulators/isaaclab_env.py launches the app at import). So this script
sets those env vars BEFORE importing simulators — defer that import inside
main() rather than placing it at the top of the file.
"""

import logging
import os

import hydra
from omegaconf import DictConfig


_VIEWER_TO_HEADLESS = {'headless': '1', 'replay': '0', 'verify': '0', 'always': '0'}


@hydra.main(version_base=None, config_path='conf', config_name='alphazero_train')
def main(cfg: DictConfig) -> None:
    # Configure Isaac Lab before its module is imported.
    if cfg.simulator.name == 'isaaclab':
        os.environ.setdefault('ISAACLAB_HEADLESS',
                              _VIEWER_TO_HEADLESS.get(cfg.viewer, '1'))
        # Cameras are required by record_replay (omni.replicator).
        if bool(cfg.get('log_video', False)):
            os.environ['ISAACLAB_ENABLE_CAMERAS'] = '1'

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Deferred imports — must come after the env-var setup above.
    from simulators import build_env
    from alphazero.train import train

    logger.info('Building env (%s): n_obstacles=%d parallel_envs=%d viewer=%s',
                cfg.simulator.name, cfg.n_obstacles, cfg.parallel_envs, cfg.viewer)
    env = build_env(cfg, n_envs=cfg.parallel_envs, viewer_mode=cfg.viewer)
    train(env, cfg)


if __name__ == '__main__':
    main()
