"""
Abstract base class for simulator environments in the puzzles project.
This allows switching between different simulators (Genesis, IsaacGym, etc.)
while maintaining a consistent interface for planners.

Supports both single-env mode (n_envs=1) and parallel mode (n_envs>1).
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import torch
from typing import Dict, Tuple, List, Optional


class SimulatorEnv(ABC):
    """
    Abstract base class for bin environments with pushers and objects.

    Subclasses implement simulator-specific scene construction, physics stepping,
    and state management, while providing a consistent interface for planners.

    Can operate in two modes:
    - n_envs=1 (default): Single environment with RNG and checkpoint support
    - n_envs>1: Parallel environments for GPU-accelerated batch processing

    Parameters
    ----------
    n_obstacles : int
        Number of obstacle objects
    n_envs : int
        Number of environments. n_envs=1 is single mode, n_envs>1 is parallel mode.
    show_viewer : bool
        Whether to show viewer
    dt : float
        Simulation timestep
    seed : int | None
        Random seed (only used in single mode, n_envs=1)
    stackable : bool
        Whether objects can be stacked
    friction : float
        Friction coefficient
    n_z_levels : int
        Number of discrete push heights
    push_steps : int
        Number of steps for a push action
    substeps : int
        Physics substeps per scene.step()
    """

    _EXIT_Y   = -0.05  # target exits when its y < _EXIT_Y

    def __init__(self, n_obstacles: int = 2, n_envs: int = 1,
                 dt: float = 0.01, seed: Optional[int] = None,
                 stackable: bool = False, friction: float = 1.0,
                 n_z_levels: int = 1,
                 push_steps: int = 20, substeps: int = 4,
                 wall_thickness: float = 0.05,
                 difficult_spawn: bool = False,
                 reward_cfg=None,
                 bin_size: Optional[float] = None,
                 bin_size_factor: float = 0.9,
                 obj_size: float = 0.05,
                 debug: bool = False,
                 target_z_level: Optional[int] = None,
                 force_obstacle_on_target: bool = False,
                 force_obstacle_on_target_prob: float = 1.0,
                 viewer_mode: str = 'replay'):
        self._OBJ_SIZE = obj_size
        self.n_obstacles = n_obstacles
        self.n_envs = n_envs
        self.viewer_mode = viewer_mode
        self.show_viewer = viewer_mode == 'always'
        self.friction = friction
        self.dt = dt
        self.stackable = stackable
        self.n_z_levels = n_z_levels
        self.target_z_level = target_z_level
        self.force_obstacle_on_target = force_obstacle_on_target
        self.force_obstacle_on_target_prob = force_obstacle_on_target_prob
        self.push_steps = push_steps
        self.substeps = substeps
        self.wall_thickness = wall_thickness
        self.difficult_spawn = difficult_spawn
        self.reward_cfg = reward_cfg
        self.debug = debug

        if bin_size is None:
            bin_size = (n_obstacles + 1) * self._OBJ_SIZE * bin_size_factor
        self.bin_w = bin_size
        self.bin_d = bin_size

        # Single-env only (n_envs=1)
        if self.n_envs == 1 and seed is not None:
            torch.manual_seed(seed)

        # Sim run counters (incremented by batch_evaluate; reset by benchmark)
        self.batch_calls: int = 0
        self.total_pairs: int = 0

    # ------------------------------------------------------------------
    # State management (shared interface)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state(self, env_idx: int) -> Dict:
        """
        Get current state as dict with 'target_pos', 'target_quat', 'obstacle_pos', 'obstacle_quat'.

        Parameters
        ----------
        env_idx : int
            Index of the env slot to read from (use 0 for single-env mode).
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict, env_idx: Optional[int] = None):
        """
        Set state from dict.

        Parameters
        ----------
        state : dict
            State dict with 'target_pos', 'target_quat', 'obstacle_pos', 'obstacle_quat'
        env_idx : int | None
            If provided and in parallel mode, set state in that env index.
            If None or in single mode, set state in current env.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None, env_idx: Optional[int] = None) -> Dict:
        """
        Reset environment to initial state.

        Parameters
        ----------
        seed : int | None
            Random seed for new randomization (single mode only)
        env_idx : int | None
            If provided and in parallel mode, reset that env index only.

        Returns
        -------
        State dict
        """
        pass

    # ------------------------------------------------------------------
    # Action primitives (single-env mode only)
    # ------------------------------------------------------------------

    @abstractmethod
    def execute_ns_push(self, pos_2d: torch.Tensor, z: float,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: Optional[int] = None, step_delay: float = 0.0) -> Tuple[Dict, float, bool]:
        """
        Execute north-south push action. (Single env mode only)

        Parameters
        ----------
        pos_2d : torch.Tensor
            2D position (x, y) of target
        z : float
            Push height
        push_dist : float
            Distance to push
        approach_dist : float
            Distance to approach from
        push_steps : int | None
            Number of steps (uses default if None)
        step_delay : float
            Delay between steps

        Returns
        -------
        (state_dict, reward, done)
        """
        pass

    @abstractmethod
    def execute_ns_pull(self, pos_2d: torch.Tensor, z: float,
                        approach_dist: float = 0.12,
                        pull_steps: Optional[int] = None, step_delay: float = 0.0) -> Tuple[Dict, float, bool]:
        """Execute north-south pull action. (Single env mode only)"""
        pass

    @abstractmethod
    def execute_ew_push(self, pos_2d: torch.Tensor, z: float, direction: int,
                        push_dist: float = 0.25, approach_dist: float = 0.12,
                        push_steps: Optional[int] = None, step_delay: float = 0.0) -> Tuple[Dict, float, bool]:
        """Execute east-west push action. (Single env mode only)"""
        pass

    # ------------------------------------------------------------------
    # Batch evaluation (parallel mode only)
    # ------------------------------------------------------------------

    def batch_evaluate(self, pairs: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, float, bool]]:
        """
        Evaluate multiple (state, action) pairs in parallel. (Parallel mode only)

        All pairs are advanced together on GPU for efficiency. Increments
        self.batch_calls and self.total_pairs for benchmarking; call
        reset_sim_counters() at the start of each timed run.

        Parameters
        ----------
        pairs : list of (state_dict, action_dict)
            Up to n_envs pairs

        Returns
        -------
        list of (new_state_dict, reward, done)
        """
        self.batch_calls += 1
        self.total_pairs += len(pairs)
        return self._batch_evaluate_impl(pairs)

    def reset_sim_counters(self) -> None:
        """Reset batch_calls and total_pairs to zero (call before each benchmark run)."""
        self.batch_calls = 0
        self.total_pairs = 0

    def _action_to_stroke(self, action: Dict,
                          approach_dist: float = 0.12,
                          push_dist: float = 0.25) -> Tuple[str, list, list]:
        """Convert action dict to (pusher_type, start_3d, end_3d). (Parallel mode only)

        Supports two action dict formats:

        Cardinal (existing PUCT / MCTS actions):
            action_type : 'push_n' | 'pull_s' | 'push_e' | 'push_w'
            push_pos    : (2,) xy position of the object being pushed
            push_z      : float height

        Arbitrary-direction (MORE contour pushes):
            action_type   : 'push_dir'
            push_start_xy : (2,) world XY of the pusher approach start
            push_end_xy   : (2,) world XY of the pusher stroke end
            push_z        : float height
            obj_idx       : int (used by PPN; not consumed by sim)

        For 'push_dir' the pusher arm (ns vs ew) is chosen by the dominant
        axis of the stroke so the arm orientation roughly aligns with the
        direction of motion.
        """
        atype = action['action_type']
        z     = action['push_z']

        if atype == 'push_dir':
            s = action['push_start_xy']
            e = action['push_end_xy']
            dx = abs(float(e[0]) - float(s[0]))
            dy = abs(float(e[1]) - float(s[1]))
            pusher = 'ns' if dx >= dy else 'ew'
            return (pusher,
                    [float(s[0]), float(s[1]), z],
                    [float(e[0]), float(e[1]), z])

        pos = action['push_pos']
        if atype == 'push_n':
            return ('ns',
                    [pos[0] - approach_dist, pos[1], z],
                    [pos[0] + push_dist,     pos[1], z])
        if atype == 'pull_s':
            return ('ns',
                    [pos[0] + approach_dist,       pos[1], z],
                    [self._EXIT_Y - approach_dist, pos[1], z])
        if atype == 'push_e':
            return ('ew',
                    [pos[0], pos[1] - approach_dist, z],
                    [pos[0], pos[1] + push_dist,     z])
        # push_w
        return ('ew',
                [pos[0], pos[1] + approach_dist, z],
                [pos[0], pos[1] - push_dist,     z])

    @abstractmethod
    def _batch_evaluate_impl(self, pairs: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, float, bool]]:
        """Simulator-specific implementation of batch_evaluate."""
        pass

    # ------------------------------------------------------------------
    # Reward / goal (shared interface)
    # ------------------------------------------------------------------

    def _compute_reward(self, state: Dict) -> float:
        """Compute reward from state."""
        return sum(self.compute_reward_components(state).values())

    @abstractmethod
    def _is_goal(self, state: Dict) -> bool:
        """Check if state is goal."""
        pass

    @abstractmethod
    def _obstacles_dropped(self, state: Dict) -> bool:
        """Return True if any obstacle has left the bin (penalty condition)."""
        pass

    def is_goal(self, state: Dict) -> bool:
        """Public goal check method."""
        return self._is_goal(state)

    def compute_reward_components(self, state: Dict) -> Dict[str, float]:
        """Return each reward term separately: target_progress, obstacle_penalty, path_blocker."""
        EXIT_Y = self._EXIT_Y
        cfg = self.reward_cfg

        # pos[0] = NS/forward, pos[1] = EW/lateral
        target_progress = 0.0
        if cfg is None or cfg.target_progress.enabled:
            ns = float(state['target_pos'][0])
            target_progress = float(min(max((self.bin_d / 2 - ns) / (self.bin_d / 2 - EXIT_Y), 0.0), 2.0))

        obstacle_penalty = 0.0
        if cfg is None or cfg.obstacle_penalty.enabled:
            weight = 0.5 if cfg is None else cfg.obstacle_penalty.weight
            n_dropped = sum(1 for i in range(self.n_obstacles)
                            if float(state['obstacle_pos'][i][0]) < EXIT_Y)
            obstacle_penalty = -weight * n_dropped

        path_blocker = 0.0
        if cfg is None or cfg.path_blocker.enabled:
            weight = 0.5 if cfg is None else cfg.path_blocker.weight
            scale  = 0.16 if cfg is None else cfg.path_blocker.scale
            t_ns = float(state['target_pos'][0])
            t_ew = float(state['target_pos'][1])
            for i in range(self.n_obstacles):
                o_ns = float(state['obstacle_pos'][i][0])
                if 0.0 < o_ns < t_ns:
                    ew_dist = abs(float(state['obstacle_pos'][i][1]) - t_ew)
                    path_blocker -= weight * max(0.0, 1.0 - ew_dist / scale)

        return {
            'target_progress': target_progress,
            'obstacle_penalty': obstacle_penalty,
            'path_blocker': path_blocker,
        }

    @abstractmethod
    def step_physics(self) -> None:
        """Advance physics one tick with no action."""
        pass

    @contextmanager
    def push_steps_ctx(self, steps: Optional[int]):
        """Temporarily override self.push_steps. No-op if steps is None."""
        if steps is None:
            yield
        else:
            old = self.push_steps
            self.push_steps = steps
            try:
                yield
            finally:
                self.push_steps = old

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_target_pos_2d(self, state: Optional[Dict] = None,
                          env_idx: Optional[int] = None) -> torch.Tensor:
        """Get target position in 2D."""
        pass

    @abstractmethod
    def get_all_obj_positions_2d(self, state: Optional[Dict] = None,
                                 env_idx: Optional[int] = None) -> torch.Tensor:
        """Get all object positions in 2D."""
        pass