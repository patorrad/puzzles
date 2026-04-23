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
import time
import torch
from tqdm import tqdm

from simulators import SimulatorEnv


# Action types and their sampling weights (bias_toward_exit=True)
# pull_s gets highest weight since it directly moves target to exit
_ACTION_TYPES  = ['push_n', 'pull_s', 'push_e', 'push_w']
_WEIGHTS_BIASED = torch.tensor([0.45, 0.45, 0.05, 0.05])
_WEIGHTS_FLAT   = torch.tensor([0.25, 0.25, 0.25, 0.25])

def _hash_action(action: dict) -> tuple:
    return action['action_type'], action['obj_idx'], tuple(action['push_pos'].tolist()), action['push_z']


def _sample_action(state: dict, env: SimulatorEnv,
                   bias_toward_exit: bool = True, recurse_depth = 0, max_recurse_depth = 10, sampled_actions = {}) -> dict:
    """Sample a random action from the discrete action space.

    Returns a dict with keys:
      action_type : 'push_n' | 'pull_s' | 'push_e' | 'push_w'
      obj_idx     : int   – 0 = target, 1..N = obstacles
      push_pos    : (2,)  – xy position of chosen object
      push_z      : float – discrete z level from env.z_levels
    """

    
    # Build (N+1, 3) position array: [target, obs0, obs1, ...]
    all_pos_3d = torch.stack([
        state['target_pos'][:3],
        *(state['obstacle_pos'][i][:3] for i in range(env.n_obstacles)),
    ])

    # Bias toward acting on the target
    obj_probs = torch.ones(len(all_pos_3d))
    obj_probs[0] *= len(all_pos_3d)
    obj_probs /= obj_probs.sum()
    obj_idx = torch.multinomial(obj_probs, 1).item()

    push_pos = all_pos_3d[obj_idx, :2]

    # Discrete z level
    z_idx  = torch.randint(len(env.z_levels), (1,)).item()
    push_z = env.z_levels[z_idx]

    # Sample action type
    weights = _WEIGHTS_BIASED if bias_toward_exit else _WEIGHTS_FLAT
    atype   = _ACTION_TYPES[torch.multinomial(weights, 1).item()]
    
    action = {'action_type': atype, 'obj_idx': obj_idx,
            'push_pos': push_pos, 'push_z': push_z}

    if recurse_depth < max_recurse_depth and _hash_action(action) in sampled_actions:
        return _sample_action(state, env, bias_toward_exit, recurse_depth + 1, max_recurse_depth, sampled_actions)
    else:
        return action


def _verify_plan(env: SimulatorEnv, plan: list[dict], root_state: dict,
                 n_tries: int, verbose: bool = True) -> int:
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

    prev_show_viewer = env.show_viewer
    env.show_viewer = True

    try:
        states = [copy.deepcopy(root_state) for _ in range(n_tries)]

        for action in plan:
            pairs = [(state, action) for state in states]
            results = env.batch_evaluate(pairs)
            states = [new_state for new_state, _, _ in results]
    finally:
        env.show_viewer = prev_show_viewer

    rewards = [env._compute_reward(s) for s in states]
    avg_reward = sum(rewards) / len(rewards)
    successes = sum(1 for s in states if env._is_goal(s))
    if verbose:
        print(f'  Verification: {successes}/{n_tries} succeeded. avg_reward={avg_reward:.3f}')
    return successes, avg_reward


# ===========================================================================
# Planner base
# ===========================================================================

class _PlannerBase:
    """Shared behaviour for RRTPusher and MCTSPusher."""

    def __init__(self, env: SimulatorEnv, verify_threshold: float, seed: int | None):
        self.env = env
        self.verify_threshold = verify_threshold
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
        if cfg.planner.name == 'mcts':
            return MCTSPusher(
                env=env,
                n_simulations=cfg.planner.n_simulations,
                rollout_depth=cfg.planner.rollout_depth,
                max_depth=cfg.planner.max_depth,
                seed=seed,
                verify_threshold=cfg.verify_threshold,
            )
        else:
            return RRTPusher(
                env=env,
                max_iter=cfg.planner.max_iter,
                max_depth=cfg.planner.max_depth,
                seed=seed,
                verify_threshold=cfg.verify_threshold,
            )

    def verify(self, plan: list[dict], initial_state: dict,
               verbose: bool = True) -> tuple[int, float, float, bool]:
        """Re-run plan self.batch_size times in parallel and return (successes, avg_reward, rate, passed)."""
        successes, avg_reward = _verify_plan(
            self.env, plan, initial_state,
            n_tries=self.batch_size, verbose=verbose,
        )
        rate = successes / self.batch_size
        passed = rate >= self.verify_threshold
        return successes, avg_reward, rate, passed


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
                 goal_bias: float = 0.3, seed: int | None = 42,
                 verify_threshold: float = 0.75):
        super().__init__(env, verify_threshold, seed)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.goal_bias = goal_bias
        self.tree: list[RRTNode] = []   # populated after plan()
        self.best_node: RRTNode | None = None

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True,
             visualize_search: bool = False,
             pause_each_iter: bool = False) -> list[dict] | None:
        """
        Run RRT and return the action sequence to the goal, or None if not found.

        Handles both single-env (n_envs=1) and parallel (n_envs>1) modes:
        - Single mode: expands one node per iteration, supports visualization
        - Parallel mode: expands batch_size nodes per iteration, skips visualization

        Returns list of action dicts: [{'push_pos', 'push_dir', 'obj_idx'}, ...]

        Parameters
        ----------
        visualize_search : bool
            If True and the env has a viewer open, draw the current
            branch being explored in the Genesis viewer using debug draw tools.
            Yellow lines/spheres = established path to expand_node.
            Green sphere/line   = newly explored node.
        pause_each_iter : bool
            If True, pause for Enter after drawing each branch (single mode only).
        """
        if pause_each_iter:
            visualize_search = True

        if self.batch_size > 1 and (visualize_search or pause_each_iter):
            if verbose:
                print('  [RRT] visualize_search/pause_each_iter ignored in parallel mode.')

        if initial_state is None:
            initial_state = self.env.get_state(0)

        root = RRTNode(state=copy.deepcopy(initial_state))
        tree: list[RRTNode] = [root]
        best_node = root
        best_reward = self.env._compute_reward(initial_state)
        draw = visualize_search and self.env.show_viewer and self.batch_size == 1

        t0 = time.time()
        for i in range(self.max_iter):
            # --- select batch_size nodes to expand ---
            weights = torch.tensor([n.reward + 0.01 for n in tree])
            weights /= weights.sum()

            expand_nodes = []
            for _ in range(self.batch_size):
                if torch.rand(1).item() < 0.3:
                    expand_nodes.append(best_node)
                else:
                    expand_nodes.append(tree[torch.multinomial(weights, 1).item()])

            expand_nodes = [n if n.depth < self.max_depth else best_node
                            for n in expand_nodes]

            # --- draw current branch before push (single-env with viewer only) ---
            if draw:
                self._draw_branch(expand_nodes[0])
                if pause_each_iter:
                    input(f'  iter {i+1}: depth={expand_nodes[0].depth} '
                          f'best={best_reward:.3f}  [Enter to push]')

            # --- sample one action per node and batch-evaluate ---
            actions = [_sample_action(n.state, self.env, bias_toward_exit=True)
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
                if verbose:
                    print(f'  Goal reached at iter {i+1}!')
                path = self._extract_path(goal_node)
                verified, avg_reward = _verify_plan(self.env, path, root.state,
                                                     self.batch_size, verbose)
                if verified >= self.batch_size * self.verify_threshold:
                    goal_node.reward = avg_reward
                    if verbose:
                        components = self.env.compute_reward_components(goal_node.state)
                        print(f'  Plan final reward: {sum(components.values()):.3f}')
                        if self.env.debug:
                            for k, v in components.items():
                                print(f'    {k}: {v:.4f}')
                    if draw:
                        self._draw_solution(goal_node)
                    self.tree = tree
                    self.best_node = goal_node
                    return path
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
                 c_ucb: float = 1.4, seed: int | None = 42,
                 verify_threshold: float = 0.75):
        super().__init__(env, verify_threshold, seed)
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.max_depth = max_depth
        self.c_ucb = c_ucb
        self.root: MCTSNode | None = None    # populated after plan()
        self.best_leaf: MCTSNode | None = None

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True) -> list[dict] | None:
        """Run MCTS and return the best action sequence found."""
        if initial_state is None:
            initial_state = self.env.get_state(0)

        root = MCTSNode(state=copy.deepcopy(initial_state), depth=0)
        best_leaf: MCTSNode | None = None
        best_reward = self.env._compute_reward(initial_state)

        t0 = time.time()
        n_sims = self.n_simulations // self.batch_size

        pbar = tqdm(range(n_sims))
        for sim_i in pbar:
            nodes = self._select(root)
            nodes = self._expand(nodes)
            rollout_rewards = self._rollout(nodes)
            self._backprop(nodes, rollout_rewards)

            goal_node = None
            for node, reward in zip(nodes, rollout_rewards):
                if reward > best_reward:
                    best_reward = reward
                    best_leaf = node
                    pbar.set_postfix(best_reward=f'{best_reward:.3f}')
                if node.done and goal_node is None:
                    goal_node = node

            if goal_node is not None:
                if verbose:
                    print(f'  MCTS: Goal reached at simulation {sim_i+1}!')
                path = self._extract_path(goal_node)
                verified, avg_reward = _verify_plan(self.env, path, root.state,
                                                     self.batch_size, verbose)
                if verified >= self.batch_size * self.verify_threshold:
                    goal_node.total_reward = avg_reward
                    goal_node.visits = 1
                    if verbose:
                        components = self.env.compute_reward_components(goal_node.state)
                        print(f'  Plan final reward: {sum(components.values()):.3f}')
                        if self.env.debug:
                            for k, v in components.items():
                                print(f'    {k}: {v:.4f}')
                    self.root = root
                    self.best_leaf = goal_node
                    return path
                goal_node = None

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
            for _ in range(self.env.n_envs):
                action = _sample_action(node.state, self.env,
                                        sampled_actions=node.sampled_actions)
                node.sampled_actions[_hash_action(action)] = True
                pairs.append((node.state, action))
                node_for_pair.append(node)

        if pairs:
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

        for _ in range(self.rollout_depth):
            if not active:
                break
            pairs = [(states[i], _sample_action(states[i], self.env)) for i in active]
            results = self.env.batch_evaluate(pairs)

            still_active = []
            for slot, i in enumerate(active):
                new_state, reward, done = results[slot]
                states[i] = new_state
                best_rewards[i] = max(best_rewards[i], reward)
                if done:
                    best_rewards[i] = 1.0
                if not self.env._obstacles_dropped(new_state):
                    still_active.append(i)
            active = still_active

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




