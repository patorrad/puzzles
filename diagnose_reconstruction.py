"""
Diagnose scene reconstruction fidelity in BinEnvIsaacLab.

Three tests:

  Test 1 — Round-trip fidelity
    _set_state(S) → N settle steps → _get_state() → compare with S.
    Sweeps N_settle = [0, 2, 5, 10, 20, 50, 100].
    Reports pos/rot errors broken out by z-level (floor vs. stacked).

  Test 2 — Cross-env variance
    Same (state, action) submitted to every env slot in one batch_evaluate call.
    All results should be identical; any spread is physics non-determinism.

  Test 3 — Sequential reconstruction noise
    Chains batch_evaluate calls using the same action each time, restoring
    the state to all envs at each step. Measures per-step cross-env divergence.

Usage:
    conda run -n isaaclab_mpc python diagnose_reconstruction.py \
        [--seed SEED] [--n_obstacles N] [--n_z_levels Z] [--n_envs E] \
        [--n_seeds S] [--n_steps STEPS]
"""

import argparse
import math
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_err(a: dict, b: dict) -> torch.Tensor:
    """Per-obstacle position error (metres) between two state dicts."""
    return (a['obstacle_pos'] - b['obstacle_pos']).norm(dim=-1)   # (n_obstacles,)


def _rot_err_deg(a: dict, b: dict) -> torch.Tensor:
    """Per-obstacle rotation error (degrees) via quaternion dot product."""
    dot = (a['obstacle_quat'] * b['obstacle_quat']).sum(dim=-1).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot.abs()) * (180.0 / math.pi)        # (n_obstacles,)


def _target_pos_err(a: dict, b: dict) -> float:
    return (a['target_pos'] - b['target_pos']).norm().item()


def _fmt(t: torch.Tensor) -> str:
    return f'mean={t.mean().item()*1000:.2f}mm  max={t.max().item()*1000:.2f}mm'


def _z_label(state: dict, obj_size: float) -> list[str]:
    """Return 'floor' or 'stacked' for each obstacle based on z position."""
    floor_z = obj_size / 2
    labels = []
    for z in state['obstacle_pos'][:, 2].tolist():
        labels.append('floor' if abs(z - floor_z) < obj_size * 0.3 else 'stacked')
    return labels


def _make_env(n_obstacles: int, n_envs: int, n_z_levels: int, seed: int | None):
    """Build a BinEnvIsaacLab with sensible defaults for diagnostics."""
    from simulators.isaaclab_env import BinEnvIsaacLab
    obj_size = 0.05
    bin_size_factor = 0.2
    bin_size = (n_obstacles + 1) * obj_size * bin_size_factor
    return BinEnvIsaacLab(
        n_obstacles=n_obstacles,
        n_envs=n_envs,
        show_viewer=False,
        seed=seed,
        stackable=False,
        n_z_levels=n_z_levels,
        push_steps=128,
        substeps=4,
        wall_thickness=0.25,
        bin_size=bin_size,
        obj_size=obj_size,
        post_teleport_steps=0,   # we control settling manually in this script
        post_push_steps=50,
    )


# ---------------------------------------------------------------------------
# Test 1 — round-trip fidelity
# ---------------------------------------------------------------------------

def test_round_trip(env, state: dict, settle_counts: list[int]):
    """Restore state, settle N steps, read back, report error."""
    obj_size = env._OBJ_SIZE
    labels = _z_label(state, obj_size)
    n_stacked = labels.count('stacked')
    n_floor   = labels.count('floor')

    print(f'\n  z breakdown: {n_floor} floor, {n_stacked} stacked')
    print(f'  {"N_settle":>8}  {"tgt_pos_err":>12}  floor pos              stacked pos')

    for n in settle_counts:
        env._set_state(state, 0)
        for _ in range(n):
            env._step_sim(render=False)
        env._refresh_all()
        restored = env._get_state(0)

        tgt_err = _target_pos_err(state, restored)
        pos_e   = _pos_err(state, restored)
        rot_e   = _rot_err_deg(state, restored)

        floor_pos   = pos_e[[i for i, l in enumerate(labels) if l == 'floor']]
        stacked_pos = pos_e[[i for i, l in enumerate(labels) if l == 'stacked']]

        floor_str   = _fmt(floor_pos)   if floor_pos.numel()   > 0 else 'n/a'
        stacked_str = _fmt(stacked_pos) if stacked_pos.numel() > 0 else 'n/a'

        print(f'  {n:>8}  {tgt_err*1000:>10.2f}mm  {floor_str}  |  {stacked_str}')


# ---------------------------------------------------------------------------
# Test 2 — cross-env variance
# ---------------------------------------------------------------------------

def test_cross_env_variance(env, state: dict):
    """Run same (state, dummy_action) in all envs; compare results."""
    n_envs = env.n_envs
    if n_envs < 2:
        print('\n  [skip] need n_envs >= 2 for cross-env test')
        return

    # Use a push_n on the target as a generic action
    cx = state['target_pos'][0].item()
    cy = state['target_pos'][1].item()
    action = {
        'action_type': 'push_n',
        'obj_idx': 0,
        'push_pos': torch.tensor([cx, cy]),
        'push_z':   env.z_levels[0],
    }
    pairs = [(state, action)] * n_envs

    import copy
    results = env.batch_evaluate([
        (copy.deepcopy(state), action) for _ in range(n_envs)
    ])

    pos_list = torch.stack([r[0]['obstacle_pos'] for r in results])   # (n_envs, n_obs, 3)
    tgt_list = torch.stack([r[0]['target_pos']   for r in results])   # (n_envs, 3)

    obs_spread = (pos_list - pos_list[0:1]).norm(dim=-1).max()
    tgt_spread = (tgt_list - tgt_list[0:1]).norm(dim=-1).max()
    rewards    = [r[1] for r in results]
    reward_std = torch.tensor(rewards).std().item()

    print(f'\n  Cross-env variance ({n_envs} envs, same state+action):')
    print(f'    obstacle pos spread: {obs_spread.item()*1000:.3f}mm')
    print(f'    target   pos spread: {tgt_spread.item()*1000:.3f}mm')
    print(f'    reward std:          {reward_std:.4f}')
    if obs_spread.item() < 1e-4:
        print('    → results are DETERMINISTIC across envs ✓')
    else:
        print('    → results DIFFER across envs — physics non-determinism detected ✗')


# ---------------------------------------------------------------------------
# Test 3 — sequential noise accumulation
# ---------------------------------------------------------------------------

def test_sequential_noise(env, state: dict, n_steps: int):
    """Restore state to all envs at each step; measure per-step spread."""
    import copy

    n_envs = env.n_envs
    if n_envs < 2:
        print('\n  [skip] need n_envs >= 2 for sequential test')
        return

    cx = state['target_pos'][0].item()
    cy = state['target_pos'][1].item()
    action = {
        'action_type': 'push_n',
        'obj_idx': 0,
        'push_pos': torch.tensor([cx, cy]),
        'push_z':   env.z_levels[0],
    }

    print(f'\n  Sequential noise over {n_steps} push steps (same action each time):')
    print(f'  {"step":>4}  {"obs_spread(mm)":>16}  {"tgt_spread(mm)":>16}  {"reward_std":>12}')

    current_state = copy.deepcopy(state)
    for step in range(n_steps):
        results = env.batch_evaluate([
            (copy.deepcopy(current_state), action) for _ in range(n_envs)
        ])
        pos_list = torch.stack([r[0]['obstacle_pos'] for r in results])
        tgt_list = torch.stack([r[0]['target_pos']   for r in results])
        rewards  = torch.tensor([r[1] for r in results])

        obs_spread = (pos_list - pos_list[0:1]).norm(dim=-1).max().item()
        tgt_spread = (tgt_list - tgt_list[0:1]).norm(dim=-1).max().item()
        reward_std = rewards.std().item()

        print(f'  {step+1:>4}  {obs_spread*1000:>16.3f}  {tgt_spread*1000:>16.3f}  {reward_std:>12.4f}')

        # Advance using env 0's result as the canonical next state
        current_state = results[0][0]
        if results[0][2]:   # done
            print('  (goal reached, stopping)')
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--n_obstacles', type=int, default=3)
    parser.add_argument('--n_z_levels',  type=int, default=2)
    parser.add_argument('--n_envs',      type=int, default=128)
    parser.add_argument('--n_seeds',     type=int, default=3,
                        help='Number of different initial placements to test')
    parser.add_argument('--n_steps',     type=int, default=4,
                        help='Steps for sequential noise test')
    parser.add_argument('--skip_cross',  action='store_true')
    parser.add_argument('--skip_seq',    action='store_true')
    args = parser.parse_args()

    settle_counts = [0, 2, 5, 10, 20, 50, 100, 1000]

    print(f'Building env: n_obstacles={args.n_obstacles}, '
          f'n_z_levels={args.n_z_levels}, n_envs={args.n_envs}')
    env = _make_env(args.n_obstacles, args.n_envs, args.n_z_levels, args.seed)

    for s in range(args.n_seeds):
        seed = args.seed + s
        print(f'\n{"="*70}')
        print(f'Seed {seed} | n_obstacles={args.n_obstacles} | n_z_levels={args.n_z_levels}')
        print('='*70)

        torch.manual_seed(seed)
        env._place_objects()
        state = env._get_state(0)

        # Test 1
        print('\n[Test 1] Round-trip fidelity (env 0):')
        test_round_trip(env, state, settle_counts)

        # Test 2
        if not args.skip_cross:
            print('\n[Test 2] Cross-env variance:')
            test_cross_env_variance(env, state)

        # Test 3
        if not args.skip_seq:
            print('\n[Test 3] Sequential reconstruction noise:')
            test_sequential_noise(env, state, args.n_steps)

    print('\nDone.')


if __name__ == '__main__':
    main()
