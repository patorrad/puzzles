"""
Unit tests for the MORE planner components.

Tier-1 (always runs, no GPU / sim):
  - ContourSampler: geometry checks
  - PPN: I/O shapes and gradient flow
  - Guided-MCTS selection: MORE Eq.3 and Eq.4 formulas
  - Shared-interface isinstance check for MOREPlanner

Run:
    pytest test_more.py -v
"""

import math
import pytest
import torch

# ---------------------------------------------------------------------------
# ContourSampler
# ---------------------------------------------------------------------------

from more.contour_sampler import ContourSampler


def _make_state(n_obs: int = 2):
    return {
        'target_pos':   torch.tensor([0.0, 0.0, 0.025]),
        'target_quat':  torch.tensor([1.0, 0.0, 0.0, 0.0]),
        'obstacle_pos': torch.zeros(n_obs, 3),
        'obstacle_quat': torch.zeros(n_obs, 4),
    }


def test_contour_sampler_count():
    """sample() returns exactly K * n_objects * n_z_levels actions."""
    sampler = ContourSampler(obj_half_extent=0.025, approach_dist=0.02,
                             z_levels=[0.025], include_target=True)
    state = _make_state(n_obs=2)
    actions = sampler.sample(state, k_per_object=8)
    expected = 8 * 3 * 1  # 3 objects (target + 2 obstacles), 1 z level
    assert len(actions) == expected, f'expected {expected}, got {len(actions)}'


def test_contour_sampler_excludes_target():
    sampler = ContourSampler(obj_half_extent=0.025, approach_dist=0.02,
                             z_levels=[0.025], include_target=False)
    state = _make_state(n_obs=2)
    actions = sampler.sample(state, k_per_object=4)
    # Only 2 obstacles
    assert len(actions) == 4 * 2


def test_contour_sampler_directions_point_inward():
    """push_start → push_end direction should point generally toward object center."""
    sampler = ContourSampler(obj_half_extent=0.025, approach_dist=0.02,
                             z_levels=[0.025])
    center = torch.tensor([0.1, 0.05])
    for action in sampler.sample_for_object(center, obj_idx=1, k=8):
        s = action['push_start_xy']
        e = action['push_end_xy']
        # Vector from start to end should have positive projection onto
        # vector from start toward center
        to_center = center - s
        stroke_dir = e - s
        dot = float(torch.dot(to_center, stroke_dir))
        assert dot >= 0.0, (
            f'Push direction ({stroke_dir}) does not point toward center '
            f'from start ({s}), center={center}'
        )


def test_contour_sampler_starts_outside_object():
    """Start points must be outside the object footprint (beyond half-extent)."""
    half = 0.025
    approach = 0.02
    sampler = ContourSampler(obj_half_extent=half, approach_dist=approach,
                             z_levels=[0.025])
    center = torch.tensor([0.0, 0.0])
    for action in sampler.sample_for_object(center, obj_idx=0, k=16):
        s = action['push_start_xy']
        dist = float(torch.norm(s - center))
        assert dist >= half, f'Start point {s} is inside the object (dist={dist:.4f} < {half})'


def test_contour_sampler_action_format():
    """Every action must be a valid push_dir dict."""
    sampler = ContourSampler(obj_half_extent=0.025, z_levels=[0.025])
    state = _make_state(n_obs=1)
    for a in sampler.sample(state, k_per_object=4):
        assert a['action_type'] == 'push_dir'
        assert 'push_start_xy' in a
        assert 'push_end_xy' in a
        assert 'push_z' in a
        assert 'obj_idx' in a
        assert a['push_start_xy'].shape == (2,)
        assert a['push_end_xy'].shape == (2,)


# ---------------------------------------------------------------------------
# PPN — Push Prediction Network
# ---------------------------------------------------------------------------

from more.ppn import PPN


def _toy_state_batch(batch: int = 4, n_obs: int = 2):
    """Batch of random solver states as dict-of-tensors (batched)."""
    return {
        'target_pos':    torch.randn(batch, 3),
        'target_quat':   torch.randn(batch, 4),
        'obstacle_pos':  torch.randn(batch, n_obs, 3),
        'obstacle_quat': torch.randn(batch, n_obs, 4),
    }


def _toy_push_batch(batch: int = 4):
    """Batch of push descriptors."""
    return {
        'push_start_xy': torch.randn(batch, 2),
        'push_end_xy':   torch.randn(batch, 2),
        'push_z':        torch.rand(batch),
    }


def test_ppn_output_shape():
    """PPN.forward returns (batch,) Q-values."""
    net = PPN(n_obstacles=2)
    states = _toy_state_batch(batch=4, n_obs=2)
    pushes = _toy_push_batch(batch=4)
    q = net(states, pushes)
    assert q.shape == (4,), f'expected (4,), got {q.shape}'


def test_ppn_gradient_flows():
    """Gradients must flow back through all PPN parameters."""
    net = PPN(n_obstacles=2)
    states = _toy_state_batch(batch=2, n_obs=2)
    pushes = _toy_push_batch(batch=2)
    q = net(states, pushes)
    loss = q.mean()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f'No gradient for {name}'
        assert not torch.isnan(param.grad).any(), f'NaN gradient for {name}'


def test_ppn_scalar_forward():
    """Single-sample unbatched forward (used during tree search)."""
    net = PPN(n_obstacles=2)
    net.eval()
    state = {
        'target_pos':    torch.randn(3),
        'target_quat':   torch.randn(4),
        'obstacle_pos':  torch.randn(2, 3),
        'obstacle_quat': torch.randn(2, 4),
    }
    push = {
        'push_start_xy': torch.randn(2),
        'push_end_xy':   torch.randn(2),
        'push_z':        torch.tensor(0.025),
    }
    with torch.no_grad():
        q = net.forward_single(state, push)
    assert q.shape == (), f'expected scalar, got {q.shape}'
    assert torch.isfinite(q)


# ---------------------------------------------------------------------------
# Guided MCTS selection formulas (MORE Eq. 3 & 4)
# ---------------------------------------------------------------------------

from more.mcts import MORE_Q_guide, MORE_Q_best


def test_q_guide_formula():
    """
    MORE Eq. 3:  Q_guide = (max_Q_ppn + sum_{top-m} r_i) / N
    With N_init=1 and m=3.
    """
    q_ppn_samples = [0.8, 0.6, 0.9]     # from three rollouts
    rollout_rewards = [0.5, 0.3, 0.7, 0.1, 0.6]
    N = 5                                # visit count

    q_guide = MORE_Q_guide(q_ppn_samples, rollout_rewards, N, m=3)

    expected_max_ppn = 0.9
    sorted_r = sorted(rollout_rewards, reverse=True)[:3]  # top-3
    expected_sum_top = sum(sorted_r)                       # 0.7+0.6+0.5 = 1.8
    expected = (expected_max_ppn + expected_sum_top) / N
    assert abs(q_guide - expected) < 1e-5, f'{q_guide} != {expected}'


def test_q_guide_n_init_one():
    """When N=1 (never visited), Q_guide uses N=1 as per MORE."""
    q_ppn_samples = [0.5]
    rollout_rewards = [0.4]
    q_guide = MORE_Q_guide(q_ppn_samples, rollout_rewards, N=1, m=3)
    # sum top-3 of a single reward = 0.4; max_ppn = 0.5
    expected = (0.5 + 0.4) / 1
    assert abs(q_guide - expected) < 1e-5


def test_q_best_formula():
    """MORE Eq. 4:  Q_best = max_Q_ppn + max_i r_i"""
    q_ppn_samples = [0.3, 0.7, 0.5]
    rollout_rewards = [0.2, 0.9, 0.4]
    q_best = MORE_Q_best(q_ppn_samples, rollout_rewards)
    expected = 0.7 + 0.9
    assert abs(q_best - expected) < 1e-5


def test_q_guide_fewer_rollouts_than_m():
    """m is clamped to number of available rollout rewards."""
    q_ppn_samples = [0.5]
    rollout_rewards = [0.3]   # only 1 rollout; m=3 clamped to 1
    q_guide = MORE_Q_guide(q_ppn_samples, rollout_rewards, N=1, m=3)
    expected = (0.5 + 0.3) / 1
    assert abs(q_guide - expected) < 1e-5


# ---------------------------------------------------------------------------
# Shared interface — MOREPlanner satisfies Planner Protocol
# ---------------------------------------------------------------------------

from planner_protocol import Planner


def test_more_planner_satisfies_protocol():
    """MOREPlanner duck-types as a Planner (no sim needed)."""
    from more.planner import MOREPlanner
    assert callable(getattr(MOREPlanner, 'plan', None))
    assert callable(getattr(MOREPlanner, 'verify', None))

    # Mock a minimal MOREPlanner instance without constructing a real env
    class _MockMorePlanner:
        def plan(self, initial_state=None, verbose=True, pause_before_verify=False):
            return None
        def verify(self, plan, initial_state, verbose=True):
            return (0, 0.0, 0.0, False)

    assert isinstance(_MockMorePlanner(), Planner)
