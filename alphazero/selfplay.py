"""Self-play episode driver.

One episode:
  1. Stacker plays n_obstacles MCTS-guided turns, placing one block each turn.
  2. State is realized into the env (set_state + brief physics settle).
  3. Solver plays up to max_depth MCTS-guided pushes.
  4. Value targets: per-step shaped returns computed from env._compute_reward.
     When reward_scale > 0, r_t = clamp(_compute_reward(s_t) / reward_scale, -1, 1)
     and G_t = sum_{k>=0} gamma^k * r_{t+k} (discounted return from step t).
     Stacker records get z_stacker = -mean(solver_returns).
     When reward_scale == 0 (disabled), falls back to sparse ±1/-0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .games import SolverGame, StackerGame
from .grid import GridSpec, realize_state
from .mcts import AZMCTS, run_parallel, visit_counts_to_policy


@dataclass
class SelfPlayConfig:
    n_simulations_solver: int = 25
    n_simulations_stacker: int = 25
    max_depth: int = 10
    temperature_init: float = 1.0
    temperature_final: float = 1e-3
    temperature_decay_after: int = 4   # turns played at high temp before going deterministic
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    settle_steps: int = 10
    gamma: float = 0.95  # discount applied to value targets along the episode
    reward_scale: float = 2.0  # divides env._compute_reward; 0 = sparse ±1 fallback
    use_cell_onehot: bool = True  # ablation: drop per-object grid-cell one-hots from solver_state


@dataclass
class Record:
    x: torch.Tensor          # encoded state
    pi: torch.Tensor         # visit-count policy
    legal_mask: torch.Tensor # for masked CE loss
    player: str              # 'solver' | 'stacker'


def _temperature(turn: int, cfg: SelfPlayConfig) -> float:
    if turn < cfg.temperature_decay_after:
        return cfg.temperature_init
    return cfg.temperature_final


def _discounted_returns(env, step_env_states: list,
                        cfg: SelfPlayConfig) -> list[float]:
    """Discounted shaped returns for one episode.

    r_t = clamp(env._compute_reward(s_t) / reward_scale, -1, 1)
    G_t = r_t + gamma * r_{t+1} + ... (backwards accumulation), clamped.
    """
    scale = cfg.reward_scale
    rewards = [max(-1.0, min(1.0, float(env._compute_reward(es)) / scale))
               for es in step_env_states]
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + cfg.gamma * G
        returns.append(G)
    returns.reverse()
    return [max(-1.0, min(1.0, g)) for g in returns]


def _settle(env, n_steps: int):
    step_fn = getattr(env, '_step_sim', None)
    if step_fn is None:
        return
    for _ in range(n_steps):
        try:
            step_fn(render=False)
        except TypeError:
            step_fn()


def play_batched_episodes(env, solver_net, stacker_net,
                          spec: GridSpec, target_cells: list[tuple[int, int]],
                          cfg: SelfPlayConfig, device: str = 'cpu',
                          ) -> list[tuple[list, int, bool]]:
    """Run K = len(target_cells) self-play episodes in lock-step.

    Stacker MCTS is symbolic and kept sequential (already cheap). Solver MCTS
    is parallelized across the K trees via ``mcts.run_parallel``, so each PUCT
    iteration batches all K env transitions through ``env.batch_evaluate`` (in
    chunks of ``env.n_envs``).

    Returns a list of K tuples ``(records_with_z, solver_steps, solver_won)``.
    Each record is a tuple ``(Record, z)``.
    """
    K = len(target_cells)
    n_obstacles = env.n_obstacles

    stacker_game = StackerGame(spec, n_obstacles)
    stacker_mcts = AZMCTS(stacker_game, stacker_net,
                          c_puct=cfg.c_puct,
                          dirichlet_alpha=cfg.dirichlet_alpha,
                          dirichlet_eps=cfg.dirichlet_eps,
                          device=device)

    stacker_records: list[list[Record]] = [[] for _ in range(K)]
    placed_lists: list[list[tuple[int, int, int]]] = [[] for _ in range(K)]
    stacker_states = [stacker_game.initial_state(tc) for tc in target_cells]

    for turn in range(n_obstacles):
        for i in range(K):
            _, counts = stacker_mcts.run(stacker_states[i], cfg.n_simulations_stacker,
                                         add_root_noise=True)
            T = _temperature(turn, cfg)
            pi = visit_counts_to_policy(counts, T)
            legal = stacker_game.legal_mask(stacker_states[i])
            if not torch.isfinite(pi).all() or pi.sum() <= 0:
                pi = legal.float() / max(legal.sum().item(), 1)
            x = stacker_game.encode(stacker_states[i])
            stacker_records[i].append(Record(x=x, pi=pi, legal_mask=legal, player='stacker'))

            action = int(torch.multinomial(pi, 1).item())
            ix, jx, kx = spec.unflatten(action)
            placed_lists[i].append((ix, jx, kx))
            stacker_states[i] = stacker_states[i].with_placement(ix, jx, kx)

    # Realize K initial states and load them into K env slots
    init_states = [realize_state(spec, placed_lists[i], target_cells[i], env._OBJ_SIZE)
                   for i in range(K)]
    for i, s in enumerate(init_states):
        env.set_state(s, env_idx=i)
    _settle(env, cfg.settle_steps)

    env_states = [env.get_state(i) for i in range(K)]

    solver_game = SolverGame(env, spec, max_depth=cfg.max_depth,
                             reward_scale=cfg.reward_scale,
                             use_cell_onehot=cfg.use_cell_onehot)
    solver_records: list[list[Record]] = [[] for _ in range(K)]
    step_env_states: list[list[dict]] = [[] for _ in range(K)]
    states = [solver_game.initial_state(es) for es in env_states]
    finished = [solver_game.is_terminal(s) for s in states]
    final_states = list(states)

    for turn in range(cfg.max_depth):
        active = [i for i in range(K) if not finished[i]]
        if not active:
            break

        results = run_parallel(solver_game, solver_net,
                               [states[i] for i in active],
                               n_simulations=cfg.n_simulations_solver,
                               c_puct=cfg.c_puct,
                               dirichlet_alpha=cfg.dirichlet_alpha,
                               dirichlet_eps=cfg.dirichlet_eps,
                               add_root_noise=True,
                               device=device)
        T = _temperature(turn, cfg)
        chosen: list[int] = []
        for idx, i in enumerate(active):
            _, counts = results[idx]
            pi = visit_counts_to_policy(counts, T)
            legal = solver_game.legal_mask(states[i])
            if not torch.isfinite(pi).all() or pi.sum() <= 0:
                pi = legal.float() / max(legal.sum().item(), 1)
            x = solver_game.encode(states[i])
            solver_records[i].append(Record(x=x, pi=pi, legal_mask=legal, player='solver'))
            # Record the env state *before* the action for the shaped reward at this step.
            step_env_states[i].append(states[i]['env_state'])
            chosen.append(int(torch.multinomial(pi, 1).item()))

        # Batched env step for the chosen actions
        new_results = solver_game.batched_transition(
            [states[i] for i in active], chosen)
        for idx, i in enumerate(active):
            next_state, terminal = new_results[idx]
            states[i] = next_state
            final_states[i] = next_state
            if terminal:
                finished[i] = True

    outputs = []
    for i in range(K):
        fs = final_states[i]['env_state']
        won = bool(env._is_goal(fs))
        T = len(solver_records[i])
        records: list[tuple[Record, float]] = []
        if cfg.reward_scale > 0.0 and T > 0:
            returns = _discounted_returns(env, step_env_states[i], cfg)
            for t, r in enumerate(solver_records[i]):
                records.append((r, returns[t]))
            z_stacker = -sum(returns) / len(returns)
            z_stacker = max(-1.0, min(1.0, z_stacker))
        else:
            if won:
                z = 1.0
            elif env._obstacles_dropped(fs):
                z = -1.0
            else:
                z = -0.5
            for t, r in enumerate(solver_records[i]):
                records.append((r, z * cfg.gamma ** (T - 1 - t)))
            z_stacker = -z
        for r in stacker_records[i]:
            records.append((r, z_stacker))
        outputs.append((records, T, won))
    return outputs


def play_batched_episodes_random(env, solver_net, spec: GridSpec, K: int,
                                 cfg: SelfPlayConfig, device: str = 'cpu',
                                 ) -> list[tuple[list, int, bool]]:
    """Run K solver episodes with env-generated random initial states (no stacker).

    Each env slot is independently reset so the solver trains on the same
    distribution that main.py sees at planning time.  No stacker records are
    produced.  Return format is identical to play_batched_episodes so the
    caller can treat them interchangeably.
    """
    for i in range(K):
        env.reset(env_idx=i)
    _settle(env, cfg.settle_steps)
    env_states = [env.get_state(i) for i in range(K)]

    solver_game = SolverGame(env, spec, max_depth=cfg.max_depth,
                             reward_scale=cfg.reward_scale,
                             use_cell_onehot=cfg.use_cell_onehot)
    solver_records: list[list[Record]] = [[] for _ in range(K)]
    step_env_states: list[list[dict]] = [[] for _ in range(K)]
    states = [solver_game.initial_state(es) for es in env_states]
    finished = [solver_game.is_terminal(s) for s in states]
    final_states = list(states)

    for turn in range(cfg.max_depth):
        active = [i for i in range(K) if not finished[i]]
        if not active:
            break

        results = run_parallel(solver_game, solver_net,
                               [states[i] for i in active],
                               n_simulations=cfg.n_simulations_solver,
                               c_puct=cfg.c_puct,
                               dirichlet_alpha=cfg.dirichlet_alpha,
                               dirichlet_eps=cfg.dirichlet_eps,
                               add_root_noise=True,
                               device=device)
        T = _temperature(turn, cfg)
        chosen: list[int] = []
        for idx, i in enumerate(active):
            _, counts = results[idx]
            pi = visit_counts_to_policy(counts, T)
            legal = solver_game.legal_mask(states[i])
            if not torch.isfinite(pi).all() or pi.sum() <= 0:
                pi = legal.float() / max(legal.sum().item(), 1)
            x = solver_game.encode(states[i])
            solver_records[i].append(Record(x=x, pi=pi, legal_mask=legal, player='solver'))
            step_env_states[i].append(states[i]['env_state'])
            chosen.append(int(torch.multinomial(pi, 1).item()))

        new_results = solver_game.batched_transition(
            [states[i] for i in active], chosen)
        for idx, i in enumerate(active):
            next_state, terminal = new_results[idx]
            states[i] = next_state
            final_states[i] = next_state
            if terminal:
                finished[i] = True

    outputs = []
    for i in range(K):
        fs = final_states[i]['env_state']
        won = bool(env._is_goal(fs))
        T = len(solver_records[i])
        if cfg.reward_scale > 0.0 and T > 0:
            returns = _discounted_returns(env, step_env_states[i], cfg)
            records = [(r, returns[t]) for t, r in enumerate(solver_records[i])]
        else:
            if won:
                z = 1.0
            elif env._obstacles_dropped(fs):
                z = -1.0
            else:
                z = -0.5
            records = [(r, z * cfg.gamma ** (T - 1 - t))
                       for t, r in enumerate(solver_records[i])]
        outputs.append((records, T, won))
    return outputs


def play_episode(env, solver_net, stacker_net,
                 spec: GridSpec, target_cell: tuple[int, int],
                 cfg: SelfPlayConfig, device: str = 'cpu',
                 ) -> tuple[list[Record], int, bool]:
    """Run one stacker+solver episode. Returns (records, solver_steps, solver_won)."""
    n_obstacles = env.n_obstacles
    stacker_game = StackerGame(spec, n_obstacles)
    stacker_mcts = AZMCTS(stacker_game, stacker_net,
                          c_puct=cfg.c_puct,
                          dirichlet_alpha=cfg.dirichlet_alpha,
                          dirichlet_eps=cfg.dirichlet_eps,
                          device=device)

    records: list[Record] = []
    placed: list[tuple[int, int, int]] = []
    stacker_state = stacker_game.initial_state(target_cell)

    for turn in range(n_obstacles):
        _, counts = stacker_mcts.run(stacker_state, cfg.n_simulations_stacker,
                                     add_root_noise=True)
        T = _temperature(turn, cfg)
        pi = visit_counts_to_policy(counts, T)
        legal = stacker_game.legal_mask(stacker_state)
        if pi.sum() <= 0:
            # All visits zero — fall back to legal-prior uniform
            pi = legal.float() / max(legal.sum().item(), 1)

        x = stacker_game.encode(stacker_state)
        records.append(Record(x=x, pi=pi, legal_mask=legal, player='stacker'))

        action = int(torch.multinomial(pi, 1).item())
        i, j, k = spec.unflatten(action)
        placed.append((i, j, k))
        stacker_state = stacker_state.with_placement(i, j, k)

    # Realize and settle
    init_state = realize_state(spec, placed, target_cell, env._OBJ_SIZE)
    env.set_state(init_state)
    _settle(env, cfg.settle_steps)

    env_state = env.get_state(0)

    # Solver play
    solver_game = SolverGame(env, spec, max_depth=cfg.max_depth,
                             reward_scale=cfg.reward_scale,
                             use_cell_onehot=cfg.use_cell_onehot)
    solver_mcts = AZMCTS(solver_game, solver_net,
                         c_puct=cfg.c_puct,
                         dirichlet_alpha=cfg.dirichlet_alpha,
                         dirichlet_eps=cfg.dirichlet_eps,
                         device=device)

    state = solver_game.initial_state(env_state)
    solver_steps = 0
    solver_won = False
    step_env_states_ep: list[dict] = []

    for turn in range(cfg.max_depth):
        if solver_game.is_terminal(state):
            break
        _, counts = solver_mcts.run(state, cfg.n_simulations_solver,
                                    add_root_noise=True)
        T = _temperature(turn, cfg)
        pi = visit_counts_to_policy(counts, T)
        legal = solver_game.legal_mask(state)
        if pi.sum() <= 0:
            pi = legal.float() / max(legal.sum().item(), 1)

        x = solver_game.encode(state)
        records.append(Record(x=x, pi=pi, legal_mask=legal, player='solver'))
        step_env_states_ep.append(state['env_state'])

        action = int(torch.multinomial(pi, 1).item())
        state, _ = solver_game.transition(state, action)
        solver_steps += 1

    env_state_final = state.get('env_state')
    solver_won = bool(env._is_goal(env_state_final)) if env_state_final is not None else False

    # Attach value targets
    solver_recs = [r for r in records if r.player == 'solver']
    T = len(solver_recs)
    annotated: list[tuple[Record, float]] = []

    if cfg.reward_scale > 0.0 and T > 0:
        returns = _discounted_returns(env, step_env_states_ep, cfg)
        z_stacker = max(-1.0, min(1.0, -sum(returns) / len(returns)))
        solver_iter = iter(range(T))
        for r in records:
            if r.player == 'solver':
                t = next(solver_iter)
                annotated.append((r, returns[t]))
            else:
                annotated.append((r, z_stacker))
    else:
        if solver_won:
            z_solver = 1.0
        elif env_state_final is not None and env._obstacles_dropped(env_state_final):
            z_solver = -1.0
        else:
            z_solver = -0.5
        solver_iter = iter(range(T))
        for r in records:
            if r.player == 'solver':
                t = next(solver_iter)
                annotated.append((r, z_solver * cfg.gamma ** (T - 1 - t)))
            else:
                annotated.append((r, -z_solver))

    return annotated, solver_steps, solver_won
