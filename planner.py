"""
Planners for the bin-clearing task.

Unified single and parallel planners
-------------------------------------
All planners now work with both single-env (n_envs=1) and parallel-env (n_envs>1) modes.
The environment's batch_evaluate() method is used for all action evaluation:
  - Single mode: evaluates 1 (state, action) pair per call
  - Parallel mode: evaluates up to n_envs pairs per call

Planners
--------
RRTPusher  : Rapidly-Exploring Random Tree over push actions.
MCTSPusher : Monte-Carlo Tree Search with UCB selection.

Both planners treat the Genesis simulation as a black-box forward model.
State save/restore is done via env.get_state() / env.set_state().

Action space
------------
Each "action" is a push:
  - direction: one of 8 discrete directions (N, NE, E, SE, S, SW, W, NW)
               plus optionally random continuous directions.
  - push target: one of the objects in the bin (usually focus on target first,
                 but can also push obstacles out of the way).
"""

import copy
import math
import random
import time
import torch
from tqdm import tqdm

from simulators import SimulatorEnv


# Action types and their sampling weights
_ACTION_TYPES   = ['push_n', 'pull_s', 'push_e', 'push_w']
_WEIGHTS_DEFAULT = torch.tensor([0.25, 0.25, 0.25, 0.25])

def _hash_action(action: dict) -> tuple:
    return action['action_type'], action['obj_idx'], tuple(action['push_pos'].tolist()), action['push_z']


def _sample_action(state: dict, env: SimulatorEnv,
                   target_prob: float = 0.6,
                   action_weights: torch.Tensor | None = None,
                   recurse_depth: int = 0, max_recurse_depth: int = 10,
                   sampled_actions: dict = {}) -> dict:
    """Sample a random action from the discrete action space.

    Returns a dict with keys:
      action_type : 'push_n' | 'pull_s' | 'push_e' | 'push_w'
      obj_idx     : int   – 0 = target, 1..N = obstacles
      push_pos    : (2,)  – xy position of chosen object
      push_z      : float – z-center of the selected object
    """
    # Build (N+1, 3) position array: [target, obs0, obs1, ...]
    all_pos_3d = torch.stack([
        state['target_pos'][:3],
        *(state['obstacle_pos'][i][:3] for i in range(env.n_obstacles)),
    ])

    # Bias toward acting on the target
    n_objs = len(all_pos_3d)
    if n_objs > 1:
        obs_prob = (1.0 - target_prob) / (n_objs - 1)
        obj_probs = torch.tensor([target_prob] + [obs_prob] * (n_objs - 1))
    else:
        obj_probs = torch.tensor([1.0])
    obj_idx = torch.multinomial(obj_probs, 1).item()

    push_pos = all_pos_3d[obj_idx, :2]

    push_z = float(all_pos_3d[obj_idx, 2])

    # Sample action type
    weights = action_weights if action_weights is not None else _WEIGHTS_DEFAULT
    atype   = _ACTION_TYPES[torch.multinomial(weights, 1).item()]

    action = {'action_type': atype, 'obj_idx': obj_idx,
              'push_pos': push_pos, 'push_z': push_z}

    if recurse_depth < max_recurse_depth and _hash_action(action) in sampled_actions:
        return _sample_action(state, env, target_prob, action_weights,
                              recurse_depth + 1, max_recurse_depth, sampled_actions)
    else:
        return action


def _verify_plan(env: SimulatorEnv, plan: list[dict], root_state: dict,
                 n_tries: int, verbose: bool = True, pause: bool = False) -> int:
    """
    Re-run a full plan n_tries times in parallel from root_state.

    Each try occupies one env slot and executes every action in sequence via
    batch_evaluate, so the cost is len(plan) batch calls regardless of n_tries.

    The viewer is enabled for the duration of verification so the trajectory
    can be watched, then restored to its previous state afterwards.

    Returns the number of tries that reached the goal.
    """
    if verbose:
        print(f'  Verifying plan ({n_tries} parallel tries)...')

    render_verify = getattr(env, 'viewer_mode', 'replay') in ('always', 'verify')
    needs_viewer = render_verify or pause
    if needs_viewer:
        prev_show_viewer = env.show_viewer
        env.show_viewer = True

    try:
        states = [copy.deepcopy(root_state) for _ in range(n_tries)]

        if pause:
            # Reset every env slot to root_state and render so the user can
            # inspect the initial configuration before any push begins.
            for i, state in enumerate(states):
                env.set_state(state, i)

            step_fn = getattr(env, '_step_sim', None)
            if step_fn is not None:
                n_settle = getattr(env, 'post_teleport_steps', 10)
                for _ in range(n_settle):
                    step_fn(render=True)
            _wait = getattr(env, 'wait_for_input', None)
            if _wait is not None:
                _wait('  [Envs reset to initial state — Press Enter to start verification...]')
            else:
                input('  [Envs reset to initial state — Press Enter to start verification...]')

        for action in plan:
            pairs = [(state, action) for state in states]
            results = env.batch_evaluate(pairs)
            states = [new_state for new_state, _, _ in results]
    finally:
        if needs_viewer:
            env.show_viewer = prev_show_viewer

    rewards = [env._compute_reward(s) for s in states]
    avg_reward = sum(rewards) / len(rewards)
    goal_flags = [env._is_goal(s) for s in states]
    successes = sum(goal_flags)
    if verbose:
        print(f'  Verification: {successes}/{n_tries} succeeded. avg_reward={avg_reward:.3f}')
    return successes, avg_reward, goal_flags


def _prune_plan(env: SimulatorEnv, plan: list[dict], root_state: dict,
                n_tries: int, verify_threshold: float, verbose: bool = True) -> list[dict]:
    """
    Remove unnecessary steps from a verified plan.

    For each step, checks whether the plan still passes verification without it.
    If so, the step is dropped and the scan restarts from the beginning (since
    earlier steps may now also be removable). Stops when no further steps can be
    removed without dropping below verify_threshold.
    """
    original_len = len(plan)
    changed = True
    while changed:
        changed = False
        for i in range(len(plan)):
            candidate = plan[:i] + plan[i + 1:]
            if not candidate:
                break
            successes, _ = _verify_plan(env, candidate, root_state, n_tries, verbose=False)
            if successes / n_tries >= verify_threshold:
                if verbose:
                    print(f'  Pruned step {i} ({plan[i]["action_type"]} obj={plan[i]["obj_idx"]})')
                plan = candidate
                changed = True
                break
    if verbose:
        print(f'  Pruning complete: {original_len} → {len(plan)} steps')
    return plan


def _verify_all_plans(
    env: SimulatorEnv,
    plans_and_nodes: list,
    root_state: dict,
    total_envs: int,
    n_verify_runs: int,
    verbose: bool = True,
    pause: bool = False,
) -> list:
    """Verify multiple plans simultaneously by interleaving their env slots.

    Within each round all plans run in a single pass: one batch_evaluate call
    per timestep with every plan's env slots included together. Plans shorter
    than the longest stop contributing pairs after their final action; their
    states are frozen at the correct terminal position.

    Rounds are used when N * n_verify_runs > total_envs so that every plan
    always receives at least n_verify_runs environments.

    Returns list of (plan, node, successes, n_tries, avg_reward).
    """
    n = len(plans_and_nodes)
    states_per_round = max(1, total_envs // n_verify_runs)
    all_results = []

    render_verify = getattr(env, 'viewer_mode', 'replay') in ('always', 'verify')
    needs_viewer = render_verify or pause
    if needs_viewer:
        prev_show_viewer = env.show_viewer
        env.show_viewer = True
    try:
        for round_start in range(0, n, states_per_round):
            batch = plans_and_nodes[round_start : round_start + states_per_round]
            envs_each = max(n_verify_runs, total_envs // len(batch))

            if verbose:
                print(f'  Verifying {len(batch)} plan(s) in parallel ({envs_each} envs each)...')

            plans = [p for p, _ in batch]
            plan_offsets = [i * envs_each for i in range(len(batch))]
            flat_states = [copy.deepcopy(root_state)
                           for _ in range(len(batch) * envs_each)]

            max_len = max(len(p) for p in plans)
            for step in range(max_len):
                pairs = []
                active = []  # flat index for each pair, used to splice results back
                for pi, plan in enumerate(plans):
                    if step < len(plan):
                        action = plan[step]
                        base = plan_offsets[pi]
                        for j in range(envs_each):
                            pairs.append((flat_states[base + j], action))
                            active.append(base + j)
                if not pairs:
                    break
                results = env.batch_evaluate(pairs)
                for flat_idx, (new_state, _, _) in zip(active, results):
                    flat_states[flat_idx] = new_state

            for pi, (plan, node) in enumerate(batch):
                base = plan_offsets[pi]
                plan_states = flat_states[base : base + envs_each]
                rewards = [env._compute_reward(s) for s in plan_states]
                avg_reward = sum(rewards) / len(rewards)
                successes = sum(1 for s in plan_states if env._is_goal(s))
                if verbose:
                    print(f'    -> {successes}/{envs_each} succeeded, avg_reward={avg_reward:.3f}')
                all_results.append((plan, node, successes, envs_each, avg_reward))
    finally:
        if needs_viewer:
            env.show_viewer = prev_show_viewer
    return all_results


# ===========================================================================
# Planner base
# ===========================================================================

class _PlannerBase:
    """Shared behaviour for RRTPusher and MCTSPusher."""

    def __init__(self, env: SimulatorEnv, verify_threshold: float,
                 n_verify_runs: int, seed: int | None,
                 verify_push_steps: int | None = None,
                 prune_plan: bool = False):
        self.env = env
        self.verify_threshold = verify_threshold
        self.n_verify_runs = n_verify_runs
        self.verify_push_steps = verify_push_steps
        self.prune_plan = prune_plan
        self.batch_size = env.n_envs  # 1 for single, n_envs for parallel
        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def _extract_path(node) -> list[dict]:
        path = []
        while node is not None and node.action is not None:
            path.append(node.action)
            node = node.parent
        path.reverse()
        return path

    @classmethod
    def from_cfg(cls, env: SimulatorEnv, cfg, seed: int) -> '_PlannerBase':
        """Instantiate the correct planner from a Hydra config."""
        aw = cfg.planner.get('action_weights', None)
        if cfg.planner.name == 'alphazero':
            from alphazero.pusher import AlphaZeroPusher
            return AlphaZeroPusher(
                env=env,
                solver_net_path=cfg.planner.checkpoint,
                n_simulations=cfg.planner.n_simulations,
                max_depth=cfg.planner.max_depth,
                c_puct=cfg.planner.c_puct,
                temperature=cfg.planner.temperature,
                seed=seed,
                verify_threshold=cfg.verify_threshold,
                n_verify_runs=cfg.n_verify_runs,
                verify_push_steps=cfg.get('verify_push_steps', None),
            )
        prune = cfg.get('prune_plan', False)
        if cfg.planner.name == 'mcts':
            return MCTSPusher(
                env=env,
                n_simulations=cfg.planner.n_simulations,
                rollout_depth=cfg.planner.rollout_depth,
                max_depth=cfg.planner.max_depth,
                c_ucb=cfg.planner.c_ucb,
                target_prob=cfg.planner.target_prob,
                seed=seed,
                verify_threshold=cfg.verify_threshold,
                n_verify_runs=cfg.n_verify_runs,
                verify_push_steps=cfg.get('verify_push_steps', None),
                action_weights=list(aw) if aw is not None else None,
                prune_plan=prune,
            )
        if cfg.planner.name == 'more':
            from more.planner import MOREPlanner
            return MOREPlanner(
                env=env,
                ppn_path=cfg.planner.get('ppn_checkpoint', None),
                n_simulations=cfg.planner.n_simulations,
                tree_depth=cfg.planner.max_depth,
                gamma=cfg.planner.gamma,
                k_per_object=cfg.planner.k_per_object,
                rollout_depth=cfg.planner.rollout_depth,
                m=cfg.planner.m,
                c_uct=cfg.planner.c_uct,
                verify_threshold=cfg.verify_threshold,
                n_verify_runs=cfg.n_verify_runs,
                verify_push_steps=cfg.get('verify_push_steps', None),
                seed=seed,
            )
        if cfg.planner.name == 'direct_push':
            return DirectPushPlanner(
                env=env,
                verify_threshold=cfg.verify_threshold,
                n_verify_runs=cfg.n_verify_runs,
                seed=seed,
                verify_push_steps=cfg.get('verify_push_steps', None),
            )
        else:
            return RRTPusher(
                env=env,
                max_iter=cfg.planner.max_iter,
                max_depth=cfg.planner.max_depth,
                goal_bias=cfg.planner.goal_bias,
                target_prob=cfg.planner.target_prob,
                seed=seed,
                verify_threshold=cfg.verify_threshold,
                n_verify_runs=cfg.n_verify_runs,
                verify_push_steps=cfg.get('verify_push_steps', None),
                action_weights=list(aw) if aw is not None else None,
                prune_plan=prune,
            )

    def verify(self, plan: list[dict], initial_state: dict,
               verbose: bool = True) -> tuple[int, float, float, bool]:
        """Re-run plan self.n_verify_runs times in parallel and return (successes, avg_reward, rate, passed, goal_flags)."""
        with self.env.push_steps_ctx(self.verify_push_steps):
            successes, avg_reward, goal_flags = _verify_plan(
                self.env, plan, initial_state,
                n_tries=self.n_verify_runs, verbose=verbose,
            )
        rate = successes / self.n_verify_runs
        passed = rate >= self.verify_threshold
        return successes, avg_reward, rate, passed, goal_flags


# ===========================================================================
# Direct-push baseline (no search)
# ===========================================================================

class DirectPushPlanner(_PlannerBase):
    """Baseline planner: skip search entirely and always emit a single
    'pull_s' action that drags the target straight out through the south
    exit in one stroke (see SimulatorEnv._action_to_stroke). Ignores every
    obstacle — useful as a no-planning lower bound to compare against
    RRT/MCTS/AlphaZero/MORE."""

    def plan(self, initial_state: dict | None = None, verbose: bool = True,
             pause_before_verify: bool = False) -> list[dict] | None:
        state = initial_state if initial_state is not None else self.env.get_state(0)
        target_pos = state['target_pos'][:3]
        action = {
            'action_type': 'pull_s',
            'obj_idx': 0,
            'push_pos': target_pos[:2],
            'push_z': float(target_pos[2]),
        }
        if verbose:
            print(f'  [DirectPushPlanner] pull_s target from {target_pos[:2].tolist()}')
        return [action]


# ===========================================================================
# RRT Planner
# ===========================================================================

class RRTNode:
    __slots__ = ('state', 'action', 'parent', 'reward', 'depth')

    def __init__(self, state: dict, action: dict | None = None,
                 parent: 'RRTNode | None' = None, reward: float = 0.0,
                 depth: int = 0):
        self.state = state
        self.action = action      # action that led here
        self.parent = parent
        self.reward = reward
        self.depth = depth


class RRTPusher(_PlannerBase):
    """
    RRT-style planner for the bin-clearing task.
    Works in both single-env (n_envs=1) and parallel-env (n_envs>1) modes.

    In each iteration (or batch of iterations for parallel mode):
      1. Select node(s) to expand based on reward + exploration.
      2. Sample random action(s).
      3. Evaluate action(s) via env.batch_evaluate() (1 action for single, up to n_envs for parallel).
      4. Add new node(s) to tree if they improve position or are novel.

    Parameters
    ----------
    env : BinEnv (with n_envs=1 for single mode or n_envs>1 for parallel)
    max_iter : int - maximum iterations (batch iterations if parallel)
    max_depth : int - max push depth per branch
    goal_bias : float - probability of biasing toward the exit
    seed : int | None
    """

    def __init__(self, env: SimulatorEnv, max_iter: int = 200, max_depth: int = 15,
                 goal_bias: float = 0.3, target_prob: float = 0.6, seed: int | None = 42,
                 verify_threshold: float = 0.75, n_verify_runs: int = 16,
                 verify_push_steps: int | None = None,
                 action_weights: list[float] | None = None,
                 prune_plan: bool = False):
        super().__init__(env, verify_threshold, n_verify_runs, seed, verify_push_steps, prune_plan)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.goal_bias = goal_bias
        self.target_prob = target_prob
        self.action_weights = torch.tensor(action_weights) if action_weights is not None else None
        self.tree: list[RRTNode] = []   # populated after plan()
        self.best_node: RRTNode | None = None

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True,
             pause_before_verify: bool = False) -> list[dict] | None:
        """
        Run RRT and return the action sequence to the goal, or None if not found.

        Handles both single-env (n_envs=1) and parallel (n_envs>1) modes:
        - Single mode: expands one node per iteration
        - Parallel mode: expands batch_size nodes per iteration
        """
        if initial_state is None:
            initial_state = self.env.get_state(0)

        root = RRTNode(state=copy.deepcopy(initial_state))
        tree: list[RRTNode] = [root]
        best_node = root
        best_reward = self.env._compute_reward(initial_state)
        draw = False

        t0 = time.time()
        for i in range(self.max_iter):
            # --- select batch_size nodes to expand ---
            weights = torch.tensor([n.reward + 0.01 for n in tree])
            weights /= weights.sum()

            expand_nodes = []
            for _ in range(self.batch_size):
                if torch.rand(1).item() < self.goal_bias:
                    expand_nodes.append(best_node)
                else:
                    expand_nodes.append(tree[torch.multinomial(weights, 1).item()])

            expand_nodes = [n if n.depth < self.max_depth else best_node
                            for n in expand_nodes]

            # --- draw current branch before push (single-env with viewer only) ---
            if draw:
                self._draw_branch(expand_nodes[0])

            # --- sample one action per node and batch-evaluate ---
            actions = [_sample_action(n.state, self.env, target_prob=self.target_prob,
                                      action_weights=self.action_weights)
                       for n in expand_nodes]
            pairs = list(zip([n.state for n in expand_nodes], actions))
            results = self.env.batch_evaluate(pairs)

            # --- incorporate results ---
            goal_node = None
            for expand_node, action, (new_state, reward, done) in \
                    zip(expand_nodes, actions, results):

                if self.env._obstacles_dropped(new_state):
                    if verbose and draw:
                        print("Branch dropped...")
                    continue

                new_node = RRTNode(
                    state=copy.deepcopy(new_state),
                    action=action,
                    parent=expand_node,
                    reward=reward,
                    depth=expand_node.depth + 1,
                )
                tree.append(new_node)

                if draw:
                    self._draw_new_node(expand_node, new_node)

                if reward > best_reward:
                    best_reward = reward
                    best_node = new_node

                if done and goal_node is None:
                    goal_node = new_node

            if goal_node is not None:
                prev_best_reward = best_reward
                prev_best_node = best_node
                if verbose:
                    print(f'  Goal reached at iter {i+1}!')
                path = self._extract_path(goal_node)
                with self.env.push_steps_ctx(self.verify_push_steps):
                    verified, avg_reward, _ = _verify_plan(self.env, path, root.state,
                                                            self.batch_size, verbose,
                                                         pause=pause_before_verify)
                if verified >= self.batch_size * self.verify_threshold:
                    goal_node.reward = avg_reward
                    if verbose:
                        components = self.env.compute_reward_components(goal_node.state)
                        print(f'  Plan final reward: {sum(components.values()):.3f}')
                        if self.env.debug:
                            for k, v in components.items():
                                print(f'    {k}: {v:.4f}')
                    if self.prune_plan:
                        with self.env.push_steps_ctx(self.verify_push_steps):
                            path = _prune_plan(self.env, path, root.state,
                                               self.batch_size, self.verify_threshold, verbose)
                    if draw:
                        self._draw_solution(goal_node)
                    self.tree = tree
                    self.best_node = goal_node
                    return path
                best_reward = prev_best_reward
                best_node = prev_best_node
                goal_node.dead_end = True  # prune so RRT doesn't revisit this failed path
                goal_node = None

            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                target_y = best_node.state['target_pos'][1]
                print(f'  RRT iter {i+1:3d}/{self.max_iter} | '
                      f'tree={len(tree)} | best_reward={best_reward:.3f} | '
                      f'target_y={target_y:.3f} | {elapsed:.1f}s')

        if verbose:
            print(f'  RRT finished. Best reward={best_reward:.3f}')

        if draw and best_node.depth > 0:
            self._draw_solution(best_node)
        self.tree = tree
        self.best_node = best_node
        return self._extract_path(best_node) if best_node.depth > 0 else None

    # ------------------------------------------------------------------
    # Debug draw helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ancestor_path(node: RRTNode) -> list[RRTNode]:
        """Return [root, ..., node] via parent pointers."""
        path: list[RRTNode] = []
        n = node
        while n is not None:
            path.append(n)
            n = n.parent
        path.reverse()
        return path

    def _draw_branch(self, node: RRTNode):
        """Clear debug objects and draw path from root to *node* in yellow."""
        if not hasattr(self.env, 'scene'):
            return
        scene = self.env.scene
        scene.clear_debug_objects()
        path = self._ancestor_path(node)
        yellow = (0.95, 0.80, 0.10, 0.9)
        for i in range(len(path) - 1):
            p1 = path[i].state['target_pos'].tolist()
            p2 = path[i + 1].state['target_pos'].tolist()
            scene.draw_debug_line(p1, p2, radius=0.004, color=yellow)
        for n in path:
            scene.draw_debug_sphere(n.state['target_pos'].tolist(),
                                    radius=0.008, color=yellow)
        scene.visualizer.update()

    def _draw_new_node(self, parent: RRTNode, child: RRTNode):
        """Append a green sphere+line for the freshly explored node."""
        if not hasattr(self.env, 'scene'):
            return
        scene = self.env.scene
        green = (0.15, 0.90, 0.25, 1.0)
        p1 = parent.state['target_pos'].tolist()
        p2 = child.state['target_pos'].tolist()
        scene.draw_debug_line(p1, p2, radius=0.004, color=green)
        scene.draw_debug_sphere(p2, radius=0.010, color=green)
        scene.visualizer.update()

    def _draw_solution(self, node: RRTNode):
        """Redraw the final solution path in bright cyan."""
        if not hasattr(self.env, 'scene'):
            return
        scene = self.env.scene
        scene.clear_debug_objects()
        path = self._ancestor_path(node)
        cyan = (0.10, 0.85, 0.95, 1.0)
        for i in range(len(path) - 1):
            p1 = path[i].state['target_pos'].tolist()
            p2 = path[i + 1].state['target_pos'].tolist()
            scene.draw_debug_line(p1, p2, radius=0.006, color=cyan)
        for n in path:
            scene.draw_debug_sphere(n.state['target_pos'].tolist(),
                                    radius=0.012, color=cyan)
        scene.visualizer.update()



# ===========================================================================
# MCTS Planner
# ===========================================================================

class MCTSNode:
    # __slots__ = ('state', 'action', 'parent', 'children',
    #              'visits', 'total_reward', 'depth', 'done', 'dead_end', 'sampled_actions'
    #              'virtual_visits')

    def __init__(self, state: dict, action: dict | None = None,
                 parent: 'MCTSNode | None' = None, depth: int = 0,
                 done: bool = False, dead_end: bool = False):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: list['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.depth = depth
        self.done = done
        self.dead_end = dead_end  # obstacle dropped — never expand
        self.sampled_actions = {}
        self.virtual_visits = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / (max(1, self.visits) + self.virtual_visits)

    def increment_virtual_visits(self):
        self.virtual_visits += 1
        if self.parent:
            self.parent.increment_virtual_visits()

    def reset_virtual_visits(self):
        self.virtual_visits = 0

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.mean_reward + c * math.sqrt(2*math.log(parent_visits + self.parent.virtual_visits) / (self.visits + self.virtual_visits))

    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb(c))

    def is_leaf(self) -> bool:
        return len(self.children) == 0 or self.dead_end


class MCTSPusher(_PlannerBase):
    """
    MCTS planner for the bin-clearing task.

    Handles both single-env (n_envs=1) and parallel (n_envs>1) modes:
    - Single mode: operates on individual nodes, performs standard MCTS
    - Parallel mode: operates on node batches, evaluates expansions and rollouts in parallel

    Uses UCB1 for tree policy and random rollouts for simulation.

    Parameters
    ----------
    env : BinEnv
    n_simulations : int  - total MCTS simulations (select+expand+rollout+backup)
    rollout_depth  : int - max steps per random rollout
    max_depth      : int - max tree depth
    n_children     : int - number of children to expand per node
    c_ucb          : float - UCB exploration constant
    seed           : int | None
    """

    def __init__(self, env: SimulatorEnv, n_simulations: int = 100,
                 rollout_depth: int = 5, max_depth: int = 10,
                 c_ucb: float = 1.4, target_prob: float = 0.6, seed: int | None = 42,
                 verify_threshold: float = 0.75, n_verify_runs: int = 16,
                 verify_push_steps: int | None = None,
                 action_weights: list[float] | None = None,
                 prune_plan: bool = False):
        super().__init__(env, verify_threshold, n_verify_runs, seed, verify_push_steps, prune_plan)
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.max_depth = max_depth
        self.c_ucb = c_ucb
        self.target_prob = target_prob
        self.action_weights = torch.tensor(action_weights) if action_weights is not None else None
        self.root: MCTSNode | None = None    # populated after plan()
        self.best_leaf: MCTSNode | None = None
        self.node_expansions: int = 0        # total (state,action) pairs expanded across all _expand calls

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True,
             pause_before_verify: bool = False) -> list[dict] | None:
        """Run MCTS and return the best action sequence found."""
        if initial_state is None:
            initial_state = self.env.get_state(0)

        self.node_expansions = 0
        root = MCTSNode(state=copy.deepcopy(initial_state), depth=0)
        best_leaf: MCTSNode | None = None
        best_reward = self.env._compute_reward(initial_state)
        best_node_reward = best_reward

        t0 = time.time()
        n_sims = self.n_simulations // self.batch_size

        pbar = tqdm(range(n_sims))
        for sim_i in pbar:
            nodes = self._select(root)
            nodes = self._expand(nodes)
            rollout_rewards = self._rollout(nodes)
            self._backprop(nodes, rollout_rewards)

            goal_nodes = []
            for node, reward in zip(nodes, rollout_rewards):
                node_reward = self.env._compute_reward(node.state)
                if reward > best_reward:
                    best_reward = reward
                    best_leaf = node
                if node_reward > best_node_reward:
                    best_node_reward = node_reward
                    pbar.set_postfix(best=f'{best_node_reward:.3f}')
                    if verbose:
                        comps = self.env.compute_reward_components(node.state)
                        print(f'  New best node={node_reward:.3f} (rollout={reward:.3f}) (sim {sim_i+1}) | '
                              + ' | '.join(f'{k}={v:.3f}' for k, v in comps.items())
                              + f' | done={node.done} obstacles_dropped={self.env._obstacles_dropped(node.state)}')
                if node.done and not node.dead_end:
                    goal_nodes.append(node)

            if goal_nodes:
                prev_best_reward = best_reward
                prev_best_leaf = best_leaf
                if verbose:
                    print(f'  MCTS: {len(goal_nodes)} goal(s) reached at simulation {sim_i+1}!')

                # Deduplicate by action sequence
                seen: set = set()
                unique: list = []
                for node in goal_nodes:
                    path = self._extract_path(node)
                    sig = tuple(tuple(sorted(a.items())) for a in path)
                    if sig not in seen:
                        seen.add(sig)
                        unique.append((path, node))

                with self.env.push_steps_ctx(self.verify_push_steps):
                    verify_results = _verify_all_plans(
                        self.env, unique, root.state,
                        self.batch_size, self.n_verify_runs, verbose,
                        pause=pause_before_verify,
                    )

                best_verified = max(
                    ((p, n, s, t, r) for p, n, s, t, r in verify_results
                     if s >= t * self.verify_threshold),
                    key=lambda x: x[4],
                    default=None,
                )
                if best_verified is not None:
                    path, goal_node, _, _, avg_reward = best_verified
                    goal_node.total_reward = avg_reward
                    goal_node.visits = 1
                    if verbose:
                        components = self.env.compute_reward_components(goal_node.state)
                        print(f'  Plan final reward: {sum(components.values()):.3f}')
                        if self.env.debug:
                            for k, v in components.items():
                                print(f'    {k}: {v:.4f}')
                    if self.prune_plan:
                        with self.env.push_steps_ctx(self.verify_push_steps):
                            path = _prune_plan(self.env, path, root.state,
                                               self.batch_size, self.verify_threshold, verbose)
                    self.root = root
                    self.best_leaf = goal_node
                    return path

                # None passed — restore and prune all
                best_reward = prev_best_reward
                best_leaf = prev_best_leaf
                for _, node, _, _, _ in verify_results:
                    node.dead_end = True

            if verbose and (sim_i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f'  MCTS sim {sim_i+1:3d}/{n_sims} | '
                      f'best_reward={best_reward:.3f} | {elapsed:.1f}s')

        if verbose:
            print(f'  MCTS finished. Best reward={best_reward:.3f}')

        self.root = root
        self.best_leaf = best_leaf
        return self._extract_path(best_leaf) if best_leaf and best_leaf.depth > 0 else None

    # ------------------------------------------------------------------
    # MCTS phases - handle both single and parallel modes
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> list[MCTSNode]:
        """Traverse tree using UCB until a leaf or unexpanded node for each slot."""
        nodes = []
        for _ in range(self.batch_size):
            node = root
            while not node.is_leaf() and not node.done:
                node = node.best_child(self.c_ucb)
            nodes.append(node)
            node.increment_virtual_visits()
        for node in nodes:
            node.reset_virtual_visits()
        return nodes

    def _expand(self, nodes: list[MCTSNode]) -> list[MCTSNode]:
        """Generate n_children children per node via batch_evaluate, return best child per node."""
        pairs: list[tuple[dict, dict]] = []
        node_for_pair: list[MCTSNode] = []

        for node in nodes:
            if node.done or node.dead_end or node.depth >= self.max_depth:
                continue
            action = _sample_action(node.state, self.env,
                                    target_prob=self.target_prob,
                                    action_weights=self.action_weights,
                                    sampled_actions=node.sampled_actions)
            node.sampled_actions[_hash_action(action)] = True
            pairs.append((node.state, action))
            node_for_pair.append(node)

        if pairs:
            self.node_expansions += len(pairs)
            results = self.env.batch_evaluate(pairs)
            for parent_node, (_, action), (new_state, reward, done) in \
                    zip(node_for_pair, pairs, results):
                child = MCTSNode(
                    state=copy.deepcopy(new_state),
                    action=action,
                    parent=parent_node,
                    depth=parent_node.depth + 1,
                    done=done,
                    dead_end=self.env._obstacles_dropped(new_state),
                )
                child.total_reward = reward
                child.visits = 1
                parent_node.children.append(child)

        expanded = []
        for node in nodes:
            if node.done or node.dead_end or node.depth >= self.max_depth:
                expanded.append(node)
            elif node.children:
                expanded.append(max(node.children, key=lambda c: c.mean_reward))
            else:
                expanded.append(node)
        return expanded

    def _rollout(self, nodes: list[MCTSNode]) -> list[float]:
        """Random rollout from each node's state for rollout_depth steps."""
        states = [copy.deepcopy(n.state) for n in nodes]
        active = list(range(len(states)))
        best_rewards = [self.env._compute_reward(s) for s in states]
        slot_to_node = list(range(len(states)))
        total_terminals = 0

        for step in range(self.rollout_depth):
            if not active:
                break
            pairs = [(states[i], _sample_action(states[i], self.env, target_prob=self.target_prob,
                                                action_weights=self.action_weights)) for i in active]
            results = self.env.batch_evaluate(pairs)

            still_active = []
            terminated = []
            for slot, i in enumerate(active):
                new_state, reward, done = results[slot]
                states[i] = new_state
                node_idx = slot_to_node[i]
                best_rewards[node_idx] = max(best_rewards[node_idx], reward)
                if done:
                    best_rewards[node_idx] = 1.0
                if done or self.env._obstacles_dropped(new_state):
                    terminated.append(i)
                else:
                    still_active.append(i)

            if self.env.debug and terminated:
                print(f'  [rollout step {step}] {len(terminated)} terminal(s): {len(still_active)} still active')
            total_terminals += len(terminated)

            for i in terminated:
                if still_active:
                    j = random.choice(still_active)
                    states[i] = copy.deepcopy(states[j])
                    slot_to_node[i] = slot_to_node[j]
                    still_active.append(i)
            active = still_active

        if self.env.debug:
            print(f'  [rollout] {total_terminals} terminal(s) across {self.rollout_depth} steps ({len(nodes)} slots)')

        for i, node in enumerate(nodes):
            if node.done:
                best_rewards[i] = 1.0
            if node.dead_end:
                best_rewards[i] = self.env._compute_reward(node.state)

        return best_rewards

    def _backprop(self, nodes: list[MCTSNode], rewards: list[float]):
        """Propagate rewards up to root, deduplicating nodes that appear multiple times."""
        seen: dict[int, tuple[MCTSNode, float]] = {}
        for n, r in zip(nodes, rewards):
            nid = id(n)
            if nid in seen:
                seen[nid] = (n, max(r, seen[nid][1]))
            else:
                seen[nid] = (n, r)

        for n, r in seen.values():
            while n is not None:
                n.visits += 1
                n.total_reward += r
                n = n.parent

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------




