"""
Headless unit tests for the three reward terms:
  - target_progress
  - obstacle_penalty
  - path_blocker

Supports Genesis (default) and IsaacLab via the SIM environment variable.
One env instance is shared across the whole test session. Each test:
  1. Sets objects to a known position via env.set_state()
  2. Steps the sim for SETTLE_STEPS with no actions (verifies reward is stable)
  3. Reads back actual physics state via env.get_state()
  4. Asserts _compute_reward() matches the expected value

Run with:
    # Genesis (default)
    conda activate genesis_mpc
    pytest test_rewards.py -v

    # IsaacLab
    conda activate isaaclab_mpc
    SIM=isaaclab ISAACLAB_HEADLESS=1 pytest test_rewards.py -v
"""

import math
import os
import types
import pytest
import torch

from simulators.placement import make_state

SIM = os.environ.get('SIM', 'genesis')

# Physics constants (must match genesis_env.py)
BIN_D    = 1.0
EXIT_Y   = -0.05
SETTLE_STEPS = 20  # steps with no actions before reading back state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(tp=True, op=True, op_w=0.5, pb=True, pb_w=0.5, pb_s=0.16):
    """Build a reward config namespace."""
    ns = types.SimpleNamespace
    return ns(
        target_progress  = ns(enabled=tp),
        obstacle_penalty = ns(enabled=op, weight=op_w),
        path_blocker     = ns(enabled=pb, weight=pb_w, scale=pb_s),
    )


def _settle(env):
    for _ in range(SETTLE_STEPS):
        env.step_physics()


def _place_and_settle(env, target_xy, obs_xys):
    """Place objects, settle, return read-back state."""
    env.set_state(make_state(target_xy, obs_xys))
    _settle(env)
    return env.get_state(0)


def _progress_expected(y):
    """Analytical target_progress reward for a given y (before clamping is applied)."""
    raw = (BIN_D / 2 - y) / (BIN_D / 2 - EXIT_Y)
    return float(max(0.0, min(2.0, raw)))


# Obstacles parked far north — out of path-blocker zone and clearly inside bin
_PARKED = [(0.1, 0.85), (0.9, 0.85)]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def env():
    """Single env for the whole test session (GPU init once). Simulator selected by SIM env var."""
    if SIM == 'isaaclab':
        os.environ.setdefault('ISAACLAB_HEADLESS', '1')
        from simulators.isaaclab_env import BinEnvIsaacLab
        return BinEnvIsaacLab(n_obstacles=2, show_viewer=False, n_envs=1, bin_size=1.0)
    else:
        from simulators.genesis_env import BinEnv as BinEnvGenesis
        return BinEnvGenesis(n_obstacles=2, show_viewer=False, n_envs=1, bin_size=1.0)


# ---------------------------------------------------------------------------
# target_progress
# ---------------------------------------------------------------------------

class TestTargetProgress:
    """obstacle_penalty and path_blocker disabled so only target_progress fires."""

    CFG = make_cfg(tp=True, op=False, pb=False)

    @pytest.mark.parametrize('req_y,expect_clamped', [
        (0.85, False),   # north area  → raw < 0, clamped to 0
        (0.50, False),   # centre      → raw = 0
        (0.10, False),   # south area  → raw ~ 0.73
        (-0.07, False),  # past exit   → raw ~ 1.04
        (-0.60, True),   # far past    → clamped to 2.0
    ])
    def test_values(self, env, req_y, expect_clamped):
        env.reward_cfg = self.CFG
        state = _place_and_settle(env, (0.5, req_y), _PARKED)
        actual_y = float(state['target_pos'][1])
        expected  = _progress_expected(actual_y)
        reward    = env._compute_reward(state)
        assert reward == pytest.approx(expected, abs=1e-3)
        if expect_clamped:
            assert reward == pytest.approx(2.0, abs=1e-2)

    def test_disabled(self, env):
        env.reward_cfg = make_cfg(tp=False, op=False, pb=False)
        for req_y in [0.5, 0.1, -0.1]:
            state = _place_and_settle(env, (0.5, req_y), _PARKED)
            assert env._compute_reward(state) == pytest.approx(0.0, abs=1e-5)

    def test_increases_toward_exit(self, env):
        """Reward should be monotonically higher closer to exit (all below BIN_D/2 so none are clamped to 0)."""
        env.reward_cfg = self.CFG
        rewards = []
        for req_y in [0.45, 0.25, 0.05]:
            state = _place_and_settle(env, (0.5, req_y), _PARKED)
            rewards.append(env._compute_reward(state))
        assert rewards[0] < rewards[1] < rewards[2]


# ---------------------------------------------------------------------------
# obstacle_penalty
# ---------------------------------------------------------------------------

class TestObstaclePenalty:
    """target_progress and path_blocker disabled; target parked at centre."""

    TARGET = (0.5, 0.5)

    CFG = make_cfg(tp=False, op=True, op_w=0.5, pb=False)

    def test_none_dropped(self, env):
        env.reward_cfg = self.CFG
        state = _place_and_settle(env, self.TARGET, [(0.3, 0.8), (0.7, 0.8)])
        assert env._compute_reward(state) == pytest.approx(0.0, abs=1e-4)

    def test_one_dropped(self, env):
        env.reward_cfg = self.CFG
        # one obstacle clearly past EXIT_Y
        state = _place_and_settle(env, self.TARGET, [(0.5, -0.20), (0.5, 0.8)])
        reward = env._compute_reward(state)
        assert reward == pytest.approx(-0.5, abs=1e-2)

    def test_two_dropped(self, env):
        env.reward_cfg = self.CFG
        state = _place_and_settle(env, self.TARGET, [(0.3, -0.20), (0.7, -0.20)])
        reward = env._compute_reward(state)
        assert reward == pytest.approx(-1.0, abs=1e-2)

    def test_just_inside_not_counted(self, env):
        """Obstacle at y = EXIT_Y + 0.05 should NOT be penalised."""
        env.reward_cfg = self.CFG
        state = _place_and_settle(env, self.TARGET, [(0.5, EXIT_Y + 0.05), (0.5, 0.8)])
        reward = env._compute_reward(state)
        assert reward == pytest.approx(0.0, abs=2e-2)

    def test_custom_weight(self, env):
        env.reward_cfg = make_cfg(tp=False, op=True, op_w=1.0, pb=False)
        state = _place_and_settle(env, self.TARGET, [(0.5, -0.20), (0.5, 0.8)])
        assert env._compute_reward(state) == pytest.approx(-1.0, abs=1e-2)

    def test_disabled(self, env):
        env.reward_cfg = make_cfg(tp=False, op=False, pb=False)
        state = _place_and_settle(env, self.TARGET, [(0.3, -0.20), (0.7, -0.20)])
        assert env._compute_reward(state) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# path_blocker
# ---------------------------------------------------------------------------

class TestPathBlocker:
    """target_progress and obstacle_penalty disabled; target at (0.5, 0.6)."""

    TARGET = (0.5, 0.60)
    CFG    = make_cfg(tp=False, op=False, pb=True, pb_w=0.5, pb_s=0.16)

    def _reward(self, env, obs_xys, cfg=None):
        env.reward_cfg = cfg or self.CFG
        state = _place_and_settle(env, self.TARGET, obs_xys)
        return env._compute_reward(state)

    def test_perfect_alignment(self, env):
        """Obstacle directly in front of target → full penalty."""
        r = self._reward(env, [(0.5, 0.30), (0.9, 0.9)])
        assert r == pytest.approx(-0.5, abs=2e-2)

    def test_at_scale_boundary(self, env):
        """x_dist == scale → max(0, 1-1) = 0 → no penalty."""
        r = self._reward(env, [(0.5 + 0.16, 0.30), (0.1, 0.9)])
        assert r == pytest.approx(0.0, abs=2e-2)

    def test_half_scale(self, env):
        """x_dist == scale/2 → penalty = weight × 0.5."""
        r = self._reward(env, [(0.5 + 0.08, 0.30), (0.1, 0.9)])
        assert r == pytest.approx(-0.25, abs=2e-2)

    def test_obstacle_behind_target(self, env):
        """Obstacle north of target (y_obs > y_target) → no penalty."""
        r = self._reward(env, [(0.5, 0.80), (0.1, 0.9)])
        assert r == pytest.approx(0.0, abs=1e-4)

    def test_obstacle_already_exited(self, env):
        """Obstacle at y < 0 → not in zone → no penalty."""
        r = self._reward(env, [(0.5, -0.20), (0.1, 0.9)])
        assert r == pytest.approx(0.0, abs=1e-4)

    def test_two_aligned_blockers(self, env):
        """Two perfectly-aligned blockers → penalty = 2 × weight."""
        r = self._reward(env, [(0.5, 0.30), (0.5, 0.20)])
        assert r == pytest.approx(-1.0, abs=3e-2)

    def test_custom_weight_and_scale(self, env):
        """pb_w=1.0, pb_s=0.08, x_dist=0.04 → penalty = 1.0 × (1 - 0.04/0.08) = 0.5."""
        cfg = make_cfg(tp=False, op=False, pb=True, pb_w=1.0, pb_s=0.08)
        r = self._reward(env, [(0.5 + 0.04, 0.30), (0.1, 0.9)], cfg=cfg)
        assert r == pytest.approx(-0.5, abs=2e-2)

    def test_disabled(self, env):
        env.reward_cfg = make_cfg(tp=False, op=False, pb=False)
        state = _place_and_settle(env, self.TARGET, [(0.5, 0.30), (0.9, 0.9)])
        assert env._compute_reward(state) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Combination and edge cases
# ---------------------------------------------------------------------------

class TestCombined:

    def test_all_terms_sum(self, env):
        """Total reward == sum of individually-isolated rewards."""
        target  = (0.5, 0.30)   # south area — non-trivial progress
        obs     = [(0.5, 0.15), (0.4, -0.20)]  # one blocker, one dropped

        def iso_reward(cfg):
            env.reward_cfg = cfg
            state = _place_and_settle(env, target, obs)
            return env._compute_reward(state)

        r_tp = iso_reward(make_cfg(tp=True,  op=False, pb=False))
        r_op = iso_reward(make_cfg(tp=False, op=True,  pb=False))
        r_pb = iso_reward(make_cfg(tp=False, op=False, pb=True))
        r_all = iso_reward(make_cfg(tp=True,  op=True,  pb=True))

        assert r_all == pytest.approx(r_tp + r_op + r_pb, abs=3e-2)

    def test_cfg_none_no_crash(self, env):
        """reward_cfg=None uses hardcoded defaults — should not raise."""
        env.reward_cfg = None
        state = _place_and_settle(env, (0.5, 0.3), _PARKED)
        reward = env._compute_reward(state)
        assert math.isfinite(reward)

    def test_cfg_none_matches_defaults(self, env):
        """reward_cfg=None should give the same result as the default config."""
        target, obs = (0.5, 0.3), _PARKED

        env.reward_cfg = None
        state = _place_and_settle(env, target, obs)
        r_none = env._compute_reward(state)

        env.reward_cfg = make_cfg()  # all enabled, same weights as hardcoded defaults
        state = _place_and_settle(env, target, obs)
        r_default = env._compute_reward(state)

        assert r_none == pytest.approx(r_default, abs=1e-4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
