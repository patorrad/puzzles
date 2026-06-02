"""
Smoke tests: verify the push/pull mechanics move the target in the correct
physical direction and that the exit condition fires correctly.

These tests share the session env from conftest.py (IsaacLab allows only one
sim instance per process).  The shared env has n_obstacles=2, bin_size=1.0;
obstacles are parked far north so they don't interfere.

Run headless:
    conda run -n isaaclab_mpc ISAACLAB_HEADLESS=1 pytest test_smoke.py -v
"""

import torch
import pytest

from simulators.placement import make_state

BIN_D   = 1.0      # matches shared conftest env
EXIT_NS = -0.05    # target exits when pos[0] (NS/forward) crosses this

# Obstacles parked far north, out of the way
_PARKED = [(0.85, 0.1), (0.85, 0.9)]


def test_pull_s_exits_target(env):
    """A single pull_s from bin centre should push the target past the exit."""
    state = make_state((0.5, 0.5), _PARKED)
    env.set_state(state)

    action = {
        'action_type': 'pull_s',
        'obj_idx':     0,
        'push_pos':    torch.tensor([0.5, 0.5]),
        'push_z':      0.025,
    }

    results = env.batch_evaluate([(state, action)])
    new_state, reward, done = results[0]

    ns = float(new_state['target_pos'][0])
    assert done, f"Expected done=True after pull_s, got target NS={ns:.4f}"
    assert ns <= EXIT_NS, f"Target should be past exit ({EXIT_NS}), got NS={ns:.4f}"
    assert reward > 0, f"Expected positive reward, got {reward:.4f}"


def test_push_n_moves_target_forward(env):
    """A push_n should increase the target's NS position (move away from exit)."""
    state = make_state((0.5, 0.5), _PARKED)
    env.set_state(state)

    action = {
        'action_type': 'push_n',
        'obj_idx':     0,
        'push_pos':    torch.tensor([0.5, 0.5]),
        'push_z':      0.025,
    }

    results = env.batch_evaluate([(state, action)])
    new_state, reward, done = results[0]

    ns_after = float(new_state['target_pos'][0])
    assert ns_after > 0.5, (
        f"push_n should move target north (increase NS), got 0.5 → {ns_after:.4f}"
    )
    assert not done, "push_n should not exit the target"
