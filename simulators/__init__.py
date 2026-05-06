"""
Simulator implementations for the puzzles project.

Imports are lazy so that missing simulator dependencies (e.g. genesis, isaacgym)
don't cause errors when using a different backend.
"""

from .base_env import SimulatorEnv


def build_env(cfg, n_envs: int, show_viewer: bool = False, viewer_mode: str = 'replay'):
    """Instantiate the correct simulator backend from a Hydra config."""
    sim = cfg.simulator.name
    if sim == 'genesis':
        from .genesis_env import BinEnv as BinEnvGenesis
        BinEnv = BinEnvGenesis
    elif sim == 'isaacgym':
        from .isaacgym_env import BinEnvIsaacGym
        BinEnv = BinEnvIsaacGym
    else:
        from .isaaclab_env import BinEnvIsaacLab
        BinEnv = BinEnvIsaacLab

    return BinEnv(
        n_obstacles=cfg.n_obstacles,
        show_viewer=show_viewer,
        viewer_mode=viewer_mode,
        seed=cfg.seed,
        stackable=cfg.stackable,
        friction=cfg.friction,
        n_z_levels=cfg.n_z_levels,
        push_steps=cfg.push_steps,
        substeps=cfg.substeps,
        wall_thickness=cfg.wall_thickness,
        difficult_spawn=cfg.difficult_spawn,
        n_envs=n_envs,
        reward_cfg=cfg.reward,
        force_threshold=cfg.simulator.force_threshold,
        bin_size=cfg.get('bin_size', None),
        bin_size_factor=cfg.get('bin_size_factor', 0.9),
        obj_size=cfg.get('obj_size', 0.05),
        debug=cfg.get('debug', False),
        target_z_level=cfg.get('target_z_level', None),
        force_obstacle_on_target=cfg.get('force_obstacle_on_target', False),
    )


def __getattr__(name: str):
    if name == 'BinEnvGenesis':
        from .genesis_env import BinEnv as BinEnvGenesis
        return BinEnvGenesis
    if name == 'BinEnvIsaacGym':
        from .isaacgym_env import BinEnvIsaacGym
        return BinEnvIsaacGym
    if name == 'BinEnvIsaacLab':
        from .isaaclab_env import BinEnvIsaacLab
        return BinEnvIsaacLab
    raise AttributeError(f"module 'simulators' has no attribute {name!r}")


__all__ = [
    'SimulatorEnv',
    'BinEnvGenesis',
    'BinEnvIsaacGym',
    'BinEnvIsaacLab',
]
