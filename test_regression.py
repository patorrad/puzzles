"""
Regression guard for the PUCT planner.

Two tiers:
  1. Protocol conformance (no sim, always runs) — verifies that existing and
     new planners satisfy the shared Planner Protocol via duck-typing.
  2. Integration regression (requires sim + checkpoint) — pins AlphaZeroPusher
     output on a fixed seed so any refactor that silently changes planning
     behaviour is caught.

Run tier-1 only (fast, no GPU):
    pytest test_regression.py -m "not integration" -v

Run all (needs GPU + checkpoint at CHECKPOINT env-var path):
    CHECKPOINT=alphazero_latest.pt pytest test_regression.py -v
"""

import os
import pytest
import torch

from planner_protocol import Planner


# ---------------------------------------------------------------------------
# Tier 1 — Protocol conformance (no sim, no GPU)
# ---------------------------------------------------------------------------

class _MockPlanner:
    """Minimal duck-typed Planner used only for Protocol checks."""
    def plan(self, initial_state=None, verbose=True, pause_before_verify=False):
        return []
    def verify(self, plan, initial_state, verbose=True):
        return (0, 0.0, 0.0, False)


def test_mock_satisfies_protocol():
    p = _MockPlanner()
    assert isinstance(p, Planner), '_MockPlanner must satisfy Planner Protocol'


def test_mcts_pusher_satisfies_protocol():
    """MCTSPusher satisfies Planner without constructing a real env."""
    from planner import MCTSPusher, _PlannerBase

    # Verify static duck-typing: the class defines both required methods
    assert callable(getattr(MCTSPusher, 'plan', None)), 'MCTSPusher.plan missing'
    assert callable(getattr(MCTSPusher, 'verify', None)), 'MCTSPusher.verify missing'

    # verify() is inherited from _PlannerBase — check that too
    assert issubclass(MCTSPusher, _PlannerBase)


def test_alphazero_pusher_satisfies_protocol():
    """AlphaZeroPusher satisfies Planner without constructing a real env."""
    from alphazero.pusher import AlphaZeroPusher

    assert callable(getattr(AlphaZeroPusher, 'plan', None)), 'AlphaZeroPusher.plan missing'
    assert callable(getattr(AlphaZeroPusher, 'verify', None)), 'AlphaZeroPusher.verify missing'


def test_plan_return_types():
    """plan() returns list[dict] | None (structural — uses mock)."""
    p = _MockPlanner()
    result = p.plan(initial_state={})
    assert result is None or isinstance(result, list)
    if isinstance(result, list):
        for item in result:
            assert isinstance(item, dict)


def test_verify_return_type():
    """verify() returns a 4-tuple (successes, avg_reward, rate, passed)."""
    p = _MockPlanner()
    result = p.verify([], {})
    assert isinstance(result, tuple) and len(result) == 4
    successes, avg_reward, rate, passed = result
    assert isinstance(successes, int)
    assert isinstance(avg_reward, float)
    assert isinstance(rate, float)
    assert isinstance(passed, bool)


# ---------------------------------------------------------------------------
# Tier 2 — Integration regression (requires real sim + checkpoint)
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get('CHECKPOINT', '')
INTEGRATION_REASON = (
    'Set CHECKPOINT=<path> and use an Isaac Lab / Genesis conda env to run.'
)
needs_sim = pytest.mark.skipif(not CHECKPOINT, reason=INTEGRATION_REASON)


@needs_sim
@pytest.mark.integration
def test_alphazero_plan_stable_across_refactors(tmp_path):
    """
    Run AlphaZeroPusher on a fixed seed and assert the plan is unchanged.

    On first run (no snapshot file), this records the plan.
    On subsequent runs it checks that the recorded plan is reproduced.

    If you intentionally change planning behaviour, delete the snapshot:
        rm test_regression_snapshot.pt
    """
    import copy

    SIM = os.environ.get('SIM', 'isaaclab')
    SEED = 12345
    N_SIM = 50          # keep small so the test is fast
    SNAPSHOT = 'test_regression_snapshot.pt'

    from simulators import build_env
    from alphazero.pusher import AlphaZeroPusher

    env = build_env(SIM, n_obstacles=2, n_envs=1, seed=SEED)
    torch.manual_seed(SEED)
    state = env.reset(seed=SEED)

    planner = AlphaZeroPusher(
        env=env,
        solver_net_path=CHECKPOINT,
        n_simulations=N_SIM,
        seed=SEED,
    )

    plan = planner.plan(initial_state=copy.deepcopy(state), verbose=False)
    # Serialise plan as a list of (action_type, obj_idx, push_z) tuples —
    # push_pos is snapped to the object so it's deterministic given the seed.
    signature = [
        (a['action_type'], a['obj_idx'], round(float(a['push_z']), 4))
        for a in (plan or [])
    ]

    if os.path.exists(SNAPSHOT):
        recorded = torch.load(SNAPSHOT, weights_only=False)
        assert signature == recorded, (
            f'PUCT plan changed after refactoring!\n'
            f'  recorded : {recorded}\n'
            f'  current  : {signature}\n'
            'If this is intentional, delete test_regression_snapshot.pt.'
        )
    else:
        torch.save(signature, SNAPSHOT)
        print(f'[regression] baseline recorded → {SNAPSHOT}')
        print(f'[regression] plan signature: {signature}')
