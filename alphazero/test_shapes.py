"""Shape + smoke tests for the AlphaZero package.

Run:  python -m alphazero.test_shapes
"""

import torch

from .encoders import (encode_solver_state, encode_stacker_state,
                       solver_action_dim, solver_state_dim)
from .games import SolverGame, StackerGame
from .grid import GridSpec, legal_mask, realize_state
from .mcts import AZMCTS, run_parallel, visit_counts_to_policy
from .networks import SolverNet, StackerNet, masked_softmax
from .replay_buffer import ReplayBuffer


def _toy_spec(Gx=3, Gy=3, Z=1, cell=0.045):
    return GridSpec(Gx=Gx, Gy=Gy, Z=Z, cell_w=cell, cell_d=cell,
                    z_levels=[0.025])


def test_grid_unflatten_flatten():
    spec = _toy_spec()
    for a in range(spec.n_actions):
        i, j, k = spec.unflatten(a)
        assert spec.flat_index(i, j, k) == a
    print('[ok] grid flat/unflatten round-trip')


def test_legal_mask():
    spec = _toy_spec()
    occ = torch.zeros(spec.Gx, spec.Gy, spec.Z, dtype=torch.int8)
    occ[0, 0, 0] = 1
    mask = legal_mask(spec, occ, target_cell=(2, 2))
    # Target cell blocked, (0,0,0) full → 9 - 2 = 7 legal
    assert mask.sum().item() == 7, mask.sum().item()
    print('[ok] legal_mask blocks target + occupied')


def test_solver_net_shapes():
    spec = _toy_spec()
    n_obs = 2
    in_dim = solver_state_dim(spec, n_obs)
    A = solver_action_dim(n_obs, 1)
    net = SolverNet(in_dim=in_dim, n_actions=A)
    B = 4
    logits, v = net(torch.randn(B, in_dim))
    assert logits.shape == (B, A), logits.shape
    assert v.shape == (B,), v.shape
    assert (v >= -1).all() and (v <= 1).all()
    print(f'[ok] SolverNet: in_dim={in_dim} A={A} (B,A)={tuple(logits.shape)}')


def test_stacker_net_shapes():
    spec = _toy_spec()
    net = StackerNet(grid_h=spec.Gx, grid_w=spec.Gy, n_actions=spec.n_actions)
    B = 4
    x = torch.randn(B, 4, spec.Gx, spec.Gy)
    logits, v = net(x)
    assert logits.shape == (B, spec.n_actions), logits.shape
    assert v.shape == (B,), v.shape
    # Masked softmax respects legality
    mask = torch.zeros(B, spec.n_actions, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = True
    pi = masked_softmax(logits, mask)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (pi[:, 2:] == 0).all()
    print('[ok] StackerNet + masked_softmax')


def test_encode_dims():
    spec = _toy_spec()
    n_obs = 2
    state = {
        'target_pos': torch.tensor([0.07, 0.07, 0.025]),
        'target_quat': torch.tensor([0., 0., 0., 1.]),
        'obstacle_pos': torch.tensor([[0.04, 0.04, 0.025], [0.10, 0.10, 0.025]]),
        'obstacle_quat': torch.tensor([[0., 0., 0., 1.]] * 2),
    }
    x = encode_solver_state(state, spec, n_obs)
    assert x.shape == (solver_state_dim(spec, n_obs),), x.shape

    occ = torch.zeros(spec.Gx, spec.Gy, spec.Z, dtype=torch.int8)
    occ[1, 1, 0] = 1
    sx = encode_stacker_state(occ, (2, 2), blocks_remaining=1, n_obstacles=n_obs, spec=spec)
    assert sx.shape == (4, spec.Gx, spec.Gy), sx.shape
    print('[ok] encoders shapes')


def test_replay_buffer():
    buf = ReplayBuffer(maxlen=100)
    for _ in range(20):
        buf.push(torch.randn(10), torch.softmax(torch.randn(4), -1), 1.0,
                 torch.ones(4, dtype=torch.bool))
    xs, pis, zs, masks = buf.sample(8)
    assert xs.shape == (8, 10)
    assert pis.shape == (8, 4)
    assert zs.shape == (8,)
    assert masks.shape == (8, 4)
    print('[ok] ReplayBuffer sample')


class _DummyEnv:
    """Minimal env stub for testing SolverGame transitions without a simulator."""
    n_envs = 4
    n_obstacles = 2
    n_z_levels = 1
    z_levels = [0.025]
    bin_w = 0.135
    bin_d = 0.135
    _OBJ_SIZE = 0.05
    _EXIT_Y = -0.05

    def batch_evaluate(self, pairs):
        # Trivial: move target -y by 0.02 every step (deterministic).
        out = []
        for state, _action in pairs:
            ns = {k: v.clone() if hasattr(v, 'clone') else v for k, v in state.items()}
            ns['target_pos'] = ns['target_pos'].clone()
            ns['target_pos'][1] -= 0.02
            done = bool(ns['target_pos'][1] <= self._EXIT_Y)
            out.append((ns, 0.0, done))
        return out

    def _is_goal(self, state):
        return bool(state['target_pos'][1] <= self._EXIT_Y)

    def _obstacles_dropped(self, state):
        return any(float(state['obstacle_pos'][i][1]) < self._EXIT_Y
                   for i in range(self.n_obstacles))

    def get_state(self, env_idx):
        return {
            'target_pos': torch.tensor([0.07, 0.09, 0.025]),
            'target_quat': torch.tensor([0., 0., 0., 1.]),
            'obstacle_pos': torch.tensor([[0.04, 0.04, 0.025], [0.10, 0.10, 0.025]]),
            'obstacle_quat': torch.tensor([[0., 0., 0., 1.]] * 2),
        }

    def set_state(self, state, env_idx=None):
        pass


def test_stacker_mcts_smoke():
    spec = _toy_spec()
    game = StackerGame(spec, n_obstacles=2)
    net = StackerNet(grid_h=spec.Gx, grid_w=spec.Gy, n_actions=spec.n_actions)
    mcts = AZMCTS(game, net, c_puct=1.5, dirichlet_eps=0.25)
    s0 = game.initial_state(target_cell=(2, 2))
    _, counts = mcts.run(s0, n_simulations=20, add_root_noise=True)
    assert counts.sum() > 0
    pi = visit_counts_to_policy(counts, 1.0)
    assert torch.isclose(pi.sum(), torch.tensor(1.0), atol=1e-4)
    print(f'[ok] stacker MCTS smoke: visits={int(counts.sum())} over {int((counts>0).sum())} actions')


def test_solver_mcts_smoke():
    spec = _toy_spec()
    env = _DummyEnv()
    game = SolverGame(env, spec, max_depth=10)
    net = SolverNet(in_dim=solver_state_dim(spec, env.n_obstacles),
                    n_actions=game.n_actions)
    mcts = AZMCTS(game, net, c_puct=1.5, dirichlet_eps=0.25)
    s0 = game.initial_state(env.get_state(0))
    _, counts = mcts.run(s0, n_simulations=10, add_root_noise=True)
    assert counts.sum() > 0
    print(f'[ok] solver MCTS smoke: visits={int(counts.sum())} actions_visited={int((counts>0).sum())}')


def test_parallel_solver_mcts():
    """run_parallel should produce K trees with comparable visit counts."""
    spec = _toy_spec()
    env = _DummyEnv()
    game = SolverGame(env, spec, max_depth=10)
    net = SolverNet(in_dim=solver_state_dim(spec, env.n_obstacles),
                    n_actions=game.n_actions)
    K = 3
    root_states = [game.initial_state(env.get_state(0)) for _ in range(K)]
    results = run_parallel(game, net, root_states, n_simulations=8,
                           c_puct=1.5, dirichlet_eps=0.25, add_root_noise=True)
    assert len(results) == K
    for root, counts in results:
        assert counts.sum() > 0
        assert root.N >= 1
    print(f'[ok] parallel solver MCTS: {K} trees, '
          f'visits={[int(c.sum()) for _, c in results]}')


def test_batched_episodes_smoke():
    from .selfplay import SelfPlayConfig, play_batched_episodes
    spec = _toy_spec()
    env = _DummyEnv()
    solver_net = SolverNet(in_dim=solver_state_dim(spec, env.n_obstacles),
                           n_actions=4 * (env.n_obstacles + 1) * 1)
    stacker_net = StackerNet(grid_h=spec.Gx, grid_w=spec.Gy, n_actions=spec.n_actions)

    # Patch set_state / get_state so the dummy env behaves under parallel access.
    initial = env.get_state(0)
    slots = {i: {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in initial.items()}
             for i in range(env.n_envs)}
    env.set_state = lambda s, env_idx=None: slots.__setitem__(env_idx or 0, s)
    env.get_state = lambda i: slots[i]

    cfg = SelfPlayConfig(n_simulations_solver=4, n_simulations_stacker=4,
                         max_depth=4, settle_steps=0)
    target_cells = [(2, 2), (1, 2), (0, 2)]
    out = play_batched_episodes(env, solver_net, stacker_net, spec,
                                target_cells, cfg)
    assert len(out) == 3
    for records, steps, won in out:
        assert all(z in (-1.0, 1.0) for _, z in records)
        assert any(r.player == 'stacker' for r, _ in records)
    print(f'[ok] batched episodes: K=3 → won={[w for _, _, w in out]} '
          f'steps={[s for _, s, _ in out]}')


def test_episode_smoke():
    from .selfplay import SelfPlayConfig, play_episode
    spec = _toy_spec()
    env = _DummyEnv()
    solver_net = SolverNet(in_dim=solver_state_dim(spec, env.n_obstacles),
                           n_actions=4 * (env.n_obstacles + 1) * 1)
    stacker_net = StackerNet(grid_h=spec.Gx, grid_w=spec.Gy, n_actions=spec.n_actions)
    cfg = SelfPlayConfig(n_simulations_solver=5, n_simulations_stacker=5,
                         max_depth=5, settle_steps=0)
    # Inject spec into the dummy env so build_grid_spec would return our toy spec
    pairs, steps, won = play_episode(env, solver_net, stacker_net,
                                     spec, target_cell=(2, 2), cfg=cfg)
    assert any(r.player == 'stacker' for r, _ in pairs)
    assert all(z in (-1.0, 1.0) for _, z in pairs)
    print(f'[ok] episode smoke: {len(pairs)} records, solver steps={steps}, won={won}')


def test_eval_smoke():
    from .eval import evaluate, UniformNet
    from .selfplay import SelfPlayConfig
    spec = _toy_spec()
    env = _DummyEnv()

    initial = env.get_state(0)
    slots = {i: {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in initial.items()}
             for i in range(env.n_envs)}
    env.set_state = lambda s, env_idx=None: slots.__setitem__(env_idx or 0, s)
    env.get_state = lambda i: slots[i]

    A_solver = 4 * (env.n_obstacles + 1) * 1
    solver = SolverNet(in_dim=solver_state_dim(spec, env.n_obstacles),
                       n_actions=A_solver)
    stacker = StackerNet(grid_h=spec.Gx, grid_w=spec.Gy, n_actions=spec.n_actions)
    cfg = SelfPlayConfig(n_simulations_solver=4, n_simulations_stacker=4,
                         max_depth=4, settle_steps=0,
                         temperature_init=1e-3, temperature_final=1e-3,
                         dirichlet_eps=0.0)

    r_self = evaluate(env, solver, stacker, spec, cfg, n_episodes=4)
    assert 0 <= r_self.solver_wins <= 4
    assert r_self.solver_wins + r_self.stacker_wins == 4

    r_rs = evaluate(env, solver, stacker, spec, cfg, n_episodes=4,
                    random_stacker=True)
    assert r_rs.n_episodes == 4

    r_rsolver = evaluate(env, solver, stacker, spec, cfg, n_episodes=4,
                         random_solver=True)
    assert r_rsolver.n_episodes == 4
    print(f'[ok] eval smoke: self_play={r_self.summary()} | '
          f'vs_rs solver_win={r_rs.solver_win_rate:.2f} | '
          f'vs_rsolver solver_win={r_rsolver.solver_win_rate:.2f}')


def test_realize_state():
    spec = _toy_spec()
    s = realize_state(spec, [(0, 0, 0), (1, 1, 0)],
                      target_cell=(2, 2), obj_size=0.05)
    assert s['target_pos'].shape == (3,)
    assert s['obstacle_pos'].shape == (2, 3)
    assert s['target_quat'].shape == (4,)
    print('[ok] realize_state')


if __name__ == '__main__':
    test_grid_unflatten_flatten()
    test_legal_mask()
    test_solver_net_shapes()
    test_stacker_net_shapes()
    test_encode_dims()
    test_replay_buffer()
    test_realize_state()
    test_stacker_mcts_smoke()
    test_solver_mcts_smoke()
    test_parallel_solver_mcts()
    test_episode_smoke()
    test_batched_episodes_smoke()
    test_eval_smoke()
    print('\nAll tests passed.')
