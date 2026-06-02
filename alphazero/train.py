"""AlphaZero-style training loop for the puzzle.

Per iteration:
  1. Collect ``episodes_per_iter`` self-play episodes (each with one stacker
     phase + one solver phase). Push records to two replay buffers.
  2. Train both nets for ``train_steps_per_iter`` minibatches.
  3. Log metrics (loss, value MSE, policy entropy, win-rate window) to wandb
     and console.
  4. Checkpoint every ``checkpoint_every`` iterations.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .encoders import solver_state_dim, solver_action_dim
from .games import SolverGame, StackerGame, action_idx_to_solver_dict
from .grid import build_grid_spec, realize_state
from .mcts import AZMCTS, run_parallel
from .networks import SolverNet, StackerNet, masked_log_softmax
from .replay_buffer import ReplayBuffer
from .selfplay import SelfPlayConfig, play_batched_episodes, play_episode

logger = logging.getLogger(__name__)


def _build_networks(env, spec):
    n_obs = env.n_obstacles
    Z = max(1, env.n_z_levels)
    solver = SolverNet(
        in_dim=solver_state_dim(spec, n_obs),
        n_actions=solver_action_dim(n_obs, Z),
    )
    stacker = StackerNet(
        grid_h=spec.Gx, grid_w=spec.Gy,
        n_actions=spec.n_actions,
        in_channels=4,
    )
    return solver, stacker


def _az_loss(net, xs, pi_targets, z_targets, legal_masks):
    logits, values = net(xs)
    log_p = masked_log_softmax(logits, legal_masks)
    policy_loss = -(pi_targets * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, z_targets)
    entropy = -(pi_targets * torch.where(pi_targets > 0,
                                          pi_targets.log(),
                                          torch.zeros_like(pi_targets))).sum(dim=-1).mean()
    return policy_loss + value_loss, policy_loss, value_loss, entropy


def _train_step(net, buf, opt, batch_size):
    if len(buf) < batch_size:
        return None
    xs, pis, zs, masks = buf.sample(batch_size)
    loss, pl, vl, ent = _az_loss(net, xs, pis, zs, masks)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {'loss': float(loss), 'policy_loss': float(pl),
            'value_loss': float(vl), 'entropy': float(ent)}


def _record_demo_episode(env, solver_net, stacker_net, spec, sp_cfg, device,
                         video_path: str, rng_seed: int = 0) -> str | None:
    """Run one stacker→solver episode greedily and capture a replay video.

    Returns the saved path, or None if the env doesn't support record_replay
    (Genesis / IsaacGym) or if frame capture failed.
    """
    if not hasattr(env, 'record_replay'):
        return None

    rng = random.Random(rng_seed)
    tcell = _sample_target_cell(spec, rng)
    n_obstacles = env.n_obstacles

    # Stacker phase — greedy
    stacker_game = StackerGame(spec, n_obstacles)
    stacker_mcts = AZMCTS(stacker_game, stacker_net,
                          c_puct=sp_cfg.c_puct, dirichlet_eps=0.0, device=device)
    stacker_state = stacker_game.initial_state(tcell)
    placed: list[tuple[int, int, int]] = []
    for _ in range(n_obstacles):
        _, counts = stacker_mcts.run(stacker_state, sp_cfg.n_simulations_stacker)
        if counts.sum() <= 0:
            mask = stacker_game.legal_mask(stacker_state)
            legal_idx = torch.nonzero(mask, as_tuple=False).flatten()
            if len(legal_idx) == 0:
                break
            a = int(legal_idx[0].item())
        else:
            a = int(counts.argmax().item())
        i, j, k = spec.unflatten(a)
        placed.append((i, j, k))
        stacker_state = stacker_state.with_placement(i, j, k)

    init_state = realize_state(spec, placed, tcell, env._OBJ_SIZE)
    env.set_state(init_state, env_idx=0)
    step_fn = getattr(env, '_step_sim', None)
    if step_fn is not None:
        for _ in range(sp_cfg.settle_steps):
            try:
                step_fn(render=False)
            except TypeError:
                step_fn()
    settled_state = env.get_state(0)

    # Solver phase — greedy plan
    solver_game = SolverGame(env, spec, max_depth=sp_cfg.max_depth)
    state = solver_game.initial_state(settled_state)
    plan: list[dict] = []
    for _ in range(sp_cfg.max_depth):
        if solver_game.is_terminal(state):
            break
        results = run_parallel(solver_game, solver_net, [state],
                               n_simulations=sp_cfg.n_simulations_solver,
                               c_puct=sp_cfg.c_puct, dirichlet_eps=0.0,
                               add_root_noise=False, device=device)
        _, counts = results[0]
        if counts.sum() <= 0:
            break
        a = int(counts.argmax().item())
        action = action_idx_to_solver_dict(a, state['env_state'], env,
                                           solver_game.n_obstacles,
                                           solver_game.n_z_levels)
        plan.append(action)
        state, _ = solver_game.transition(state, a)

    if not plan:
        return None

    try:
        return env.record_replay(plan, settled_state, video_path)
    except Exception as e:
        logger.warning('record_replay failed: %s', e)
        return None


def _sample_target_cell(spec, rng: random.Random) -> tuple[int, int]:
    """Sample a target cell in the northern half of the bin (far from exit)."""
    i = rng.randrange(spec.Gx)
    # Bias toward y >= Gy//2 (top half away from the south exit at y=0).
    j = rng.randrange(max(1, spec.Gy // 2), spec.Gy)
    return i, j


def train(env, cfg):
    """Run the full training loop.

    cfg fields (Hydra DictConfig compatible):
      n_iterations, episodes_per_iter, train_steps_per_iter, batch_size,
      buffer_size, lr, weight_decay, checkpoint_every, output_dir,
      seed, device, use_wandb,
      and a nested 'selfplay' subconfig of SelfPlayConfig fields.
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    spec = build_grid_spec(env)
    logger.info('Grid: %dx%dx%d cells (cell %.3fx%.3f m), %d z-levels',
                spec.Gx, spec.Gy, spec.Z, spec.cell_w, spec.cell_d, spec.Z)

    solver_net, stacker_net = _build_networks(env, spec)
    solver_net.to(cfg.device)
    stacker_net.to(cfg.device)

    solver_opt = torch.optim.Adam(solver_net.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    stacker_opt = torch.optim.Adam(stacker_net.parameters(),
                                   lr=cfg.lr, weight_decay=cfg.weight_decay)

    solver_buf = ReplayBuffer(maxlen=cfg.buffer_size)
    stacker_buf = ReplayBuffer(maxlen=cfg.buffer_size)

    sp_cfg = SelfPlayConfig(**dict(cfg.selfplay))

    use_wandb = bool(cfg.use_wandb)
    wandb = None
    if use_wandb:
        try:
            import wandb as _wb
            wandb = _wb
            wandb.init(project=cfg.get('wandb_project', 'puzzle-alphazero'),
                       entity=cfg.get('wandb_entity') or None,
                       name=cfg.get('wandb_run_name') or None,
                       config={'cfg': dict(cfg),
                               'spec': asdict(spec),
                               'solver_in_dim': solver_net.in_dim,
                               'solver_actions': solver_net.n_actions,
                               'stacker_actions': stacker_net.n_actions})
        except Exception as e:
            logger.warning('wandb init failed (%s); disabling.', e)
            use_wandb = False

    os.makedirs(cfg.output_dir, exist_ok=True)
    win_window: list[int] = []
    win_window_size = 50

    # K episodes run in lock-step per "batch", batching env steps across them.
    # Cap K at env.n_envs so env.batch_evaluate never has to split internally.
    batch_K = max(1, min(cfg.episodes_per_iter, env.n_envs))

    for it in range(cfg.n_iterations):
        t0 = time.time()
        ep_wins, ep_steps = 0, 0
        eps_played = 0

        while eps_played < cfg.episodes_per_iter:
            k = min(batch_K, cfg.episodes_per_iter - eps_played)
            target_cells = [_sample_target_cell(spec, rng) for _ in range(k)]
            results = play_batched_episodes(
                env, solver_net, stacker_net, spec, target_cells, sp_cfg,
                device=cfg.device)
            for records, steps, won in results:
                for rec, z in records:
                    buf = solver_buf if rec.player == 'solver' else stacker_buf
                    buf.push(rec.x, rec.pi, z, rec.legal_mask)
                ep_wins += int(won)
                ep_steps += steps
                win_window.append(int(won))
                if len(win_window) > win_window_size:
                    win_window.pop(0)
            eps_played += k

        # Train
        solver_metrics = stacker_metrics = None
        for _ in range(cfg.train_steps_per_iter):
            m = _train_step(solver_net, solver_buf, solver_opt, cfg.batch_size)
            if m is not None:
                solver_metrics = m
            m = _train_step(stacker_net, stacker_buf, stacker_opt, cfg.batch_size)
            if m is not None:
                stacker_metrics = m

        dt = time.time() - t0
        win_rate = sum(win_window) / max(len(win_window), 1)
        logger.info(
            '[iter %3d] %d eps in %.1fs | win_rate(win%d)=%.2f | '
            'solver_buf=%d stacker_buf=%d | solver_loss=%s stacker_loss=%s',
            it, cfg.episodes_per_iter, dt, len(win_window), win_rate,
            len(solver_buf), len(stacker_buf),
            f"{solver_metrics['loss']:.3f}" if solver_metrics else 'n/a',
            f"{stacker_metrics['loss']:.3f}" if stacker_metrics else 'n/a',
        )
        if use_wandb:
            log = {'iter': it, 'win_rate_solver': win_rate,
                   'avg_solver_steps': ep_steps / max(cfg.episodes_per_iter, 1),
                   'solver_buffer': len(solver_buf),
                   'stacker_buffer': len(stacker_buf)}
            if solver_metrics:
                log.update({f'solver/{k}': v for k, v in solver_metrics.items()})
            if stacker_metrics:
                log.update({f'stacker/{k}': v for k, v in stacker_metrics.items()})
            wandb.log(log)

        if (it + 1) % cfg.checkpoint_every == 0 or it == cfg.n_iterations - 1:
            ckpt = {
                'iter': it,
                'solver': solver_net.state_dict(),
                'stacker': stacker_net.state_dict(),
                'solver_in_dim': solver_net.in_dim,
                'solver_n_actions': solver_net.n_actions,
                'stacker_grid_h': stacker_net.grid_h,
                'stacker_grid_w': stacker_net.grid_w,
                'stacker_n_actions': stacker_net.n_actions,
                'spec': asdict(spec),
            }
            path = os.path.join(cfg.output_dir, f'alphazero_iter_{it+1:04d}.pt')
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(cfg.output_dir, 'alphazero_latest.pt'))
            logger.info('  Checkpoint saved to %s', path)

            if use_wandb and bool(cfg.get('log_video', False)):
                video_path = os.path.join(cfg.output_dir,
                                          f'demo_iter_{it+1:04d}.mp4')
                solver_net.eval(); stacker_net.eval()
                try:
                    saved = _record_demo_episode(
                        env, solver_net, stacker_net, spec, sp_cfg,
                        cfg.device, video_path, rng_seed=it)
                finally:
                    solver_net.train(); stacker_net.train()
                if saved is not None and os.path.exists(saved):
                    try:
                        wandb.log({'demo/episode': wandb.Video(
                                       saved, fps=30, format='mp4'),
                                   'iter': it})
                        logger.info('  Demo video logged to wandb: %s', saved)
                    except Exception as e:
                        logger.warning('wandb.Video log failed: %s', e)

    return solver_net, stacker_net
