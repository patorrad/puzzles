"""
Simulator implementations for the puzzles project.

Imports are lazy so that missing simulator dependencies (e.g. genesis, isaacgym)
don't cause errors when using a different backend.
"""

from .base_env import SimulatorEnv


def build_env(cfg, n_envs: int, viewer_mode: str = 'replay'):
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
        viewer_mode=viewer_mode,
        seed=cfg.seed,
        stackable=cfg.stackable,
        friction=cfg.friction,
        max_stack_height=cfg.max_stack_height,
        push_steps=cfg.push_steps,
        substeps=cfg.substeps,
        wall_thickness=cfg.wall_thickness,
        difficult_spawn=cfg.difficult_spawn,
        n_envs=n_envs,
        reward_cfg=cfg.reward,
        force_threshold=cfg.simulator.force_threshold,
        env_spacing_factor=getattr(cfg.simulator, 'env_spacing_factor', 2.5),
        post_teleport_steps=getattr(cfg.simulator, 'post_teleport_steps', 10),
        teleport_settle_steps=getattr(cfg.simulator, 'teleport_settle_steps', 3),
        post_push_steps=getattr(cfg.simulator, 'post_push_steps', 15),
        max_depenetration_velocity=getattr(cfg.simulator, 'max_depenetration_velocity', 5.0),
        settle_depenetration_velocity=getattr(cfg.simulator, 'settle_depenetration_velocity', None),
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
