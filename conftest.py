"""
Shared pytest configuration.

Isaac Lab allows only one simulation instance per process, so all tests that
need a live environment share a single session-scoped `env` fixture defined
here.  test_rewards.py and test_smoke.py both import it automatically.

Run the full suite:
    conda run -n isaaclab_mpc ISAACLAB_HEADLESS=1 pytest -v
"""

import os
import pytest

os.environ.setdefault('ISAACLAB_HEADLESS', '1')
os.environ.setdefault('SIM', 'isaaclab')

SIM = os.environ['SIM']


@pytest.fixture(scope='session')
def env():
    """Single env shared across the whole test session (GPU init once)."""
    if SIM == 'isaaclab':
        from simulators.isaaclab_env import BinEnvIsaacLab
        return BinEnvIsaacLab(n_obstacles=2, viewer_mode='headless', n_envs=1, bin_size=1.0,
                              push_steps=128)
    else:
        from simulators.genesis_env import BinEnv as BinEnvGenesis
        return BinEnvGenesis(n_obstacles=2, viewer_mode='headless', n_envs=1, bin_size=1.0,
                             push_steps=128)
