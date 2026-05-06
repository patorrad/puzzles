"""
Unit tests for the placement logic in simulators/placement.py.

Pure-PyTorch — no simulator required. Run with:
    pytest test_placement.py -v
"""

import pytest
import torch

from simulators.placement import random_initial_state, BIN_W, BIN_D

OBJ_SIZE = 0.05
OBJ_H    = OBJ_SIZE / 2

MARGIN  = OBJ_SIZE * 0.7
LEVEL_Z = [OBJ_H + i * OBJ_SIZE for i in range(4)]  # [0.025, 0.075, 0.125, 0.175]


class TestTargetZLevelExplicit:

    def test_level0_always_floor(self):
        """target_z_level=0 always places target at floor height."""
        for seed in range(20):
            state = random_initial_state(n_obstacles=2, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=0, seed=seed)
            z = state['target_pos'][2].item()
            assert z == pytest.approx(OBJ_H, abs=1e-4), f"seed={seed}: got z={z}"

    def test_level1_height_and_xy_match_obstacle(self):
        """target_z_level=1 places target on top of an obstacle (same x,y)."""
        for seed in range(20):
            state = random_initial_state(n_obstacles=1, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=1, seed=seed)
            tz = state['target_pos'][2].item()
            assert tz == pytest.approx(LEVEL_Z[1], abs=1e-4), f"seed={seed}: got z={tz}"

            tx = state['target_pos'][0].item()
            ty = state['target_pos'][1].item()
            ox = state['obstacle_pos'][0][0].item()
            oy = state['obstacle_pos'][0][1].item()
            assert tx == pytest.approx(ox, abs=1e-4), f"seed={seed}: target x={tx} != obstacle x={ox}"
            assert ty == pytest.approx(oy, abs=1e-4), f"seed={seed}: target y={ty} != obstacle y={oy}"

    def test_level1_no_obstacles_falls_back_to_floor(self):
        """target_z_level=1 with no obstacles falls back to floor (no eligible column)."""
        for seed in range(10):
            state = random_initial_state(n_obstacles=0, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=1, seed=seed)
            z = state['target_pos'][2].item()
            assert z == pytest.approx(OBJ_H, abs=1e-4), f"seed={seed}: got z={z}"

    def test_level1_two_obstacles_valid_z(self):
        """target_z_level=1 with 2 obstacles yields level 1 whenever any obstacle is at
        level 1 (swap guarantees this). Falls back to floor only when both obstacles land
        at floor level in separate columns (no level-1 slot exists anywhere).
        In either case the z must be a recognised level height."""
        valid_zs = [LEVEL_Z[0], LEVEL_Z[1]]
        for seed in range(30):
            state = random_initial_state(n_obstacles=2, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=1, seed=seed)
            tz = state['target_pos'][2].item()
            assert any(abs(tz - vz) < 1e-4 for vz in valid_zs), (
                f"seed={seed}: z={tz} not in {valid_zs}"
            )
            # When target is stacked (z > floor), it must share x,y with an obstacle.
            if tz > OBJ_H + 1e-4:
                tx, ty = state['target_pos'][0].item(), state['target_pos'][1].item()
                match = any(
                    abs(tx - state['obstacle_pos'][i][0].item()) < 1e-4
                    and abs(ty - state['obstacle_pos'][i][1].item()) < 1e-4
                    for i in range(2)
                )
                assert match, f"seed={seed}: stacked target ({tx:.4f},{ty:.4f}) not on any obstacle"

    def test_swap_guarantees_target_at_level_when_obstacle_exists(self):
        """When an obstacle ends up at the target level but direct column placement
        failed, the swap mechanism must move the target to that level."""
        for seed in range(50):
            state = random_initial_state(n_obstacles=2, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=1, seed=seed)
            obs_at_level_1 = any(
                abs(state['obstacle_pos'][i][2].item() - LEVEL_Z[1]) < 1e-4
                for i in range(2)
            )
            if obs_at_level_1:
                tz = state['target_pos'][2].item()
                assert abs(tz - LEVEL_Z[1]) < 1e-4, (
                    f"seed={seed}: obstacle at level 1 exists but target at z={tz}"
                )


class TestTargetZLevelNone:

    def test_single_level_always_floor(self):
        """With obj_size=OBJ_SIZE, n_z_levels=1, randomised target must land at floor."""
        for seed in range(20):
            state = random_initial_state(n_obstacles=2, obj_size=OBJ_SIZE, n_z_levels=1,
                                         target_z_level=None, seed=seed)
            z = state['target_pos'][2].item()
            assert z == pytest.approx(OBJ_H, abs=1e-4), f"seed={seed}: got z={z}"

    def test_two_levels_valid_z(self):
        """With obj_size=OBJ_SIZE, n_z_levels=2, target z must be one of the two valid level heights."""
        valid = LEVEL_Z[:2]
        for seed in range(20):
            state = random_initial_state(n_obstacles=1, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=None, seed=seed)
            z = state['target_pos'][2].item()
            assert any(abs(z - vz) < 1e-4 for vz in valid), (
                f"seed={seed}: z={z} not in {valid}"
            )

    def test_target_quat_is_identity(self):
        """Target quaternion should always be identity regardless of z-level."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for seed in range(10):
            state = random_initial_state(n_obstacles=2, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=None, seed=seed)
            assert torch.allclose(state['target_quat'], identity, atol=1e-6)


class TestPlacementInvariants:

    def test_target_within_bin_bounds(self):
        """Target x,y always within [margin, bin_w/d - margin]."""
        for seed in range(50):
            state = random_initial_state(n_obstacles=3, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=None, seed=seed)
            tx, ty = state['target_pos'][0].item(), state['target_pos'][1].item()
            assert MARGIN <= tx <= BIN_W - MARGIN, f"seed={seed}: tx={tx:.4f} out of bounds"
            assert MARGIN <= ty <= BIN_D - MARGIN, f"seed={seed}: ty={ty:.4f} out of bounds"

    def test_obstacles_within_bin_bounds(self):
        """All obstacle x,y within [margin, bin_w/d - margin]."""
        for seed in range(50):
            state = random_initial_state(n_obstacles=3, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=None, seed=seed)
            for i in range(3):
                ox = state['obstacle_pos'][i][0].item()
                oy = state['obstacle_pos'][i][1].item()
                assert MARGIN <= ox <= BIN_W - MARGIN, f"seed={seed} obs={i}: ox={ox:.4f} out of bounds"
                assert MARGIN <= oy <= BIN_D - MARGIN, f"seed={seed} obs={i}: oy={oy:.4f} out of bounds"

    def test_no_same_column_same_z_overlaps(self):
        """No two objects share the same (x,y) column at the same z-height."""
        sep = OBJ_SIZE * 1.05
        for seed in range(50):
            state = random_initial_state(n_obstacles=3, obj_size=OBJ_SIZE, n_z_levels=2,
                                         target_z_level=None, seed=seed)
            all_pos = torch.cat([
                state['obstacle_pos'],
                state['target_pos'].unsqueeze(0),
            ], dim=0)  # (4, 3)

            n = all_pos.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    dxy = ((all_pos[i, 0] - all_pos[j, 0]) ** 2
                           + (all_pos[i, 1] - all_pos[j, 1]) ** 2) ** 0.5
                    dz  = abs(all_pos[i, 2].item() - all_pos[j, 2].item())
                    same_col = dxy.item() < sep
                    same_z   = dz < 1e-4
                    assert not (same_col and same_z), (
                        f"seed={seed}: objects {i} and {j} overlap at same column+z"
                    )
