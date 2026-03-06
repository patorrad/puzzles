"""
Planners for the bin-clearing task.

Both planners treat the Genesis simulation as a black-box forward model.
State save/restore is done via env.get_state() / env.set_state().

Planners
--------
RRTPusher  : Rapidly-Exploring Random Tree over push actions.
MCTSPusher : Monte-Carlo Tree Search with UCB selection.

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
import numpy as np
from typing import Optional

from env import BinEnv, BIN_W, BIN_D, EXIT_Y, OBJ_SIZE


# Push directions: N, NE, E, SE, NW, W — excluding southward (pull handles that)
_PUSH_ANGLES = [a for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
                if np.sin(a) > -0.5]   # drop pure-south and nearby directions
DISCRETE_DIRS = np.stack([np.cos(_PUSH_ANGLES), np.sin(_PUSH_ANGLES)], axis=1)

# Probability of choosing a pull action instead of a push
_PULL_PROB = 0.35


def _execute_action(env: BinEnv, action: dict) -> tuple[dict, float, bool]:
    """Dispatch a sampled action to execute_push or execute_pull."""
    if action.get('action_type') == 'pull':
        return env.execute_pull(action['push_pos'], pull_z=action.get('push_z'))
    return env.execute_push(
        action['push_pos'], action['push_dir'], push_z=action.get('push_z')
    )


def _sample_action(rng: np.random.Generator, state: dict, env: BinEnv,
                   bias_toward_exit: bool = True) -> dict:
    """Sample a random action (push or pull).

    Returns a dict with keys:
      action_type : 'push' | 'pull'
      obj_idx  : int    - 0 = target, 1..N = obstacles
      push_pos : (2,)   - xy position to act at (object center)
      push_dir : (2,)   - unit push direction  (push only)
      push_z   : float  - z height of pusher center
    """
    # Build (N+1, 3) position array: [target, obs0, obs1, ...]
    all_pos_3d = np.vstack([
        state['target_pos'][:3],
        *(state['obstacle_pos'][i][:3] for i in range(len(env.obstacles))),
    ])  # shape (N+1, 3)

    # Bias toward acting on the target
    obj_probs = np.ones(len(all_pos_3d))
    obj_probs[0] *= 3.0
    obj_probs /= obj_probs.sum()
    obj_idx = rng.choice(len(all_pos_3d), p=obj_probs)

    push_pos = all_pos_3d[obj_idx, :2]
    push_z = float(all_pos_3d[obj_idx, 2])

    # Choose pull or push
    if bias_toward_exit and rng.random() < _PULL_PROB:
        return {'action_type': 'pull', 'obj_idx': obj_idx,
                'push_pos': push_pos, 'push_z': push_z}

    # Sample non-southward push direction
    dir_idx = rng.choice(len(DISCRETE_DIRS))
    push_dir = DISCRETE_DIRS[dir_idx].copy()
    push_dir += rng.normal(0, 0.15, 2)
    push_dir /= np.linalg.norm(push_dir) + 1e-9

    return {'action_type': 'push', 'obj_idx': obj_idx,
            'push_pos': push_pos, 'push_dir': push_dir, 'push_z': push_z}


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


class RRTPusher:
    """
    RRT-style planner for the bin-clearing task.

    At each iteration:
      1. Sample a random state (goal-biased) or use current best node.
      2. Find the nearest node in the tree (by target y distance).
      3. Extend by simulating a random push from that node.
      4. Add new node to tree if it improves position or is novel.

    Parameters
    ----------
    env : BinEnv
    max_iter : int  - maximum tree expansions
    max_depth : int - max push depth per branch
    goal_bias : float - probability of biasing toward the exit
    seed : int | None
    """

    def __init__(self, env: BinEnv, max_iter: int = 200, max_depth: int = 15,
                 goal_bias: float = 0.3, seed: int | None = 42):
        self.env = env
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.goal_bias = goal_bias
        self.rng = np.random.default_rng(seed)
        self.tree: list[RRTNode] = []   # populated after plan()
        self.best_node: RRTNode | None = None

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True,
             visualize_search: bool = False,
             pause_each_iter: bool = False) -> list[dict] | None:
        """
        Run RRT and return the action sequence to the goal, or None if not found.

        Returns list of action dicts: [{'push_pos', 'push_dir', 'obj_idx'}, ...]

        Parameters
        ----------
        visualize_search : bool
            If True and the env has a viewer open, draw the current branch being
            explored in the Genesis viewer using debug draw tools.
            Yellow lines/spheres = established path to expand_node.
            Green sphere/line   = newly explored node.
        pause_each_iter : bool
            If True, pause for Enter after drawing each branch (implies visualize_search).
        """
        if pause_each_iter:
            visualize_search = True

        if initial_state is None:
            initial_state = self.env.get_state()

        root = RRTNode(state=copy.deepcopy(initial_state))
        tree: list[RRTNode] = [root]
        best_node = root
        best_reward = self.env._compute_reward(initial_state)
        draw = visualize_search and self.env.show_viewer

        t0 = time.time()
        for i in range(self.max_iter):
            # --- select node to expand ---
            if self.rng.random() < 0.3:
                expand_node = best_node
            else:
                weights = np.array([n.reward + 0.01 for n in tree])
                weights /= weights.sum()
                expand_node = tree[self.rng.choice(len(tree), p=weights)]

            if expand_node.depth >= self.max_depth:
                continue

            # --- draw current branch before push ---
            if draw:
                self._draw_branch(expand_node)
                if pause_each_iter:
                    input(f'  iter {i+1}: depth={expand_node.depth} '
                          f'best={best_reward:.3f}  [Enter to push]')

            # --- sample action ---
            action = _sample_action(self.rng, expand_node.state, self.env,
                                    bias_toward_exit=True)

            # --- simulate ---
            self.env.set_state(expand_node.state)
            new_state, reward, done = _execute_action(self.env, action)

            # Drop branches where obstacles fell out of the bin
            if self.env._obstacles_dropped(new_state):
                print("Branch dropped...:-()")
                continue

            new_node = RRTNode(
                state=copy.deepcopy(new_state),
                action=action,
                parent=expand_node,
                reward=reward,
                depth=expand_node.depth + 1,
            )
            tree.append(new_node)

            # --- highlight new node ---
            if draw:
                self._draw_new_node(expand_node, new_node)

            if reward > best_reward:
                best_reward = reward
                best_node = new_node

            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                target_y = new_state['target_pos'][1]
                print(f'  RRT iter {i+1:3d}/{self.max_iter} | '
                      f'tree={len(tree)} | best_reward={best_reward:.3f} | '
                      f'target_y={target_y:.3f} | {elapsed:.1f}s')

            if done:
                if verbose:
                    print(f'  Goal reached at iter {i+1}!')
                if draw:
                    self._draw_solution(new_node)
                self.tree = tree
                self.best_node = new_node
                return self._extract_path(new_node)

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
        scene = self.env.scene
        green = (0.15, 0.90, 0.25, 1.0)
        p1 = parent.state['target_pos'].tolist()
        p2 = child.state['target_pos'].tolist()
        scene.draw_debug_line(p1, p2, radius=0.004, color=green)
        scene.draw_debug_sphere(p2, radius=0.010, color=green)
        scene.visualizer.update()

    def _draw_solution(self, node: RRTNode):
        """Redraw the final solution path in bright cyan."""
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

    @staticmethod
    def _extract_path(node: RRTNode) -> list[dict]:
        path = []
        while node.action is not None:
            path.append(node.action)
            node = node.parent
        path.reverse()
        return path


# ===========================================================================
# MCTS Planner
# ===========================================================================

class MCTSNode:
    __slots__ = ('state', 'action', 'parent', 'children',
                 'visits', 'total_reward', 'depth', 'done', 'dead_end')

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

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.visits)

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.mean_reward + c * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb(c))

    def is_leaf(self) -> bool:
        return len(self.children) == 0 or self.dead_end


class MCTSPusher:
    """
    MCTS planner for the bin-clearing task.

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

    def __init__(self, env: BinEnv, n_simulations: int = 100,
                 rollout_depth: int = 5, max_depth: int = 10,
                 n_children: int = 5, c_ucb: float = 1.4,
                 seed: int | None = 42):
        self.env = env
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.max_depth = max_depth
        self.n_children = n_children
        self.c_ucb = c_ucb
        self.rng = np.random.default_rng(seed)
        self.root: MCTSNode | None = None    # populated after plan()
        self.best_leaf: MCTSNode | None = None

    def plan(self, initial_state: dict | None = None,
             verbose: bool = True) -> list[dict] | None:
        """Run MCTS and return the best action sequence found."""
        if initial_state is None:
            initial_state = self.env.get_state()

        root = MCTSNode(state=copy.deepcopy(initial_state), depth=0)
        best_leaf: MCTSNode | None = None
        best_reward = self.env._compute_reward(initial_state)

        t0 = time.time()
        for sim_i in range(self.n_simulations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.done and not node.dead_end and node.depth < self.max_depth:
                node = self._expand(node)

            # 3. Rollout
            rollout_reward = self._rollout(node)

            # 4. Backpropagation
            self._backprop(node, rollout_reward)

            # Track best goal
            if rollout_reward > best_reward:
                best_reward = rollout_reward
                best_leaf = node

            if node.done:
                if verbose:
                    print(f'  MCTS: Goal reached at simulation {sim_i+1}!')
                self.root = root
                self.best_leaf = node
                return self._extract_path(node)

            if verbose and (sim_i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f'  MCTS sim {sim_i+1:3d}/{self.n_simulations} | '
                      f'best_reward={best_reward:.3f} | {elapsed:.1f}s')

        if verbose:
            print(f'  MCTS finished. Best reward={best_reward:.3f}')

        self.root = root
        self.best_leaf = best_leaf
        return self._extract_path(best_leaf) if best_leaf and best_leaf.depth > 0 else None

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB until a leaf or unexpanded node."""
        node = root
        while not node.is_leaf() and not node.done:
            node = node.best_child(self.c_ucb)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Generate children by simulating n_children random pushes."""
        for _ in range(self.n_children):
            action = _sample_action(self.rng, node.state, self.env)
            self.env.set_state(node.state)
            new_state, reward, done = _execute_action(self.env, action)
            child = MCTSNode(
                state=copy.deepcopy(new_state),
                action=action,
                parent=node,
                depth=node.depth + 1,
                done=done,
                dead_end=self.env._obstacles_dropped(new_state),
            )
            child.total_reward = reward
            child.visits = 1
            node.children.append(child)

        if node.children:
            # Return the most promising child
            return max(node.children, key=lambda c: c.mean_reward)
        return node

    def _rollout(self, node: MCTSNode) -> float:
        """Random rollout from node's state for rollout_depth steps."""
        if node.done:
            return 1.0
        if node.dead_end:
            return self.env._compute_reward(node.state)  # already penalized

        self.env.set_state(node.state)
        state = node.state
        best_r = self.env._compute_reward(state)

        for _ in range(self.rollout_depth):
            action = _sample_action(self.rng, state, self.env)
            state, reward, done = _execute_action(self.env, action)
            if self.env._obstacles_dropped(state):
                return reward  # penalized reward; stop rollout
            best_r = max(best_r, reward)
            if done:
                return 1.0

        return best_r

    def _backprop(self, node: MCTSNode, reward: float):
        """Propagate reward up to root."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_path(node: MCTSNode) -> list[dict]:
        path = []
        while node is not None and node.action is not None:
            path.append(node.action)
            node = node.parent
        path.reverse()
        return path
