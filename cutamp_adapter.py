"""
cutamp_adapter.py — Convert a puzzle scenario YAML into a CuTAMP TAMPEnvironment.

The puzzle uses a kinematic pusher blade to move objects out of a bin; CuTAMP
uses a robot arm for pick-and-place. This adapter creates an equivalent
rearrangement problem so both systems can be evaluated on the same initial
object arrangements.

Goal mapping:
  - target cube  → On(target, goal)   [exit region south of bin]
  - obstacle i   → On(obs_i, floor)   [anywhere on the table]
  - robot hand   → HandEmpty

Coordinate mapping (places the bin in the Panda's comfortable workspace):
  CuTAMP x = puzzle x + (0.5 - bin_size/2)   # bin x-centre at CuTAMP x=0.5
  CuTAMP y = puzzle y - bin_size/2            # bin y-centre at CuTAMP y=0
  CuTAMP z = puzzle z                         # same height above floor
"""

import os
import sys
import time

import yaml

CUTAMP_DIR = os.path.join(os.path.dirname(__file__), "..", "cuTAMP")
if CUTAMP_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(CUTAMP_DIR))

from curobo.geom.types import Cuboid
from cutamp.envs.utils import TAMPEnvironment
from cutamp.tamp_domain import On, HandEmpty

OBJ_SIZE = 0.05
WALL_HEIGHT = 0.15

_UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]
_FLOOR_COLOR = [235, 196, 145]
_EXIT_COLOR = [186, 255, 201]
_TARGET_COLOR = [244, 67, 54]
_OBS_COLOR = [33, 150, 243]
_WALL_COLOR = [150, 150, 150]


def _pose(xyz, wxyz=_UNIT_QUAT):
    return [xyz[0], xyz[1], xyz[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3]]


def puzzle_scenario_to_cutamp_env(scenario_path: str) -> TAMPEnvironment:
    """Load a puzzle scenario YAML and build an equivalent TAMPEnvironment.

    The bin's centre is placed at CuTAMP (0.5, 0) so the Panda arm can
    reach all objects comfortably.
    """
    with open(scenario_path) as f:
        data = yaml.safe_load(f)

    init = data["initial_state"]
    n_obstacles = int(data["n_obstacles"])
    bin_size = float(data.get("bin_size", 0.3))
    wall_thickness = float(data.get("wall_thickness", 0.02))

    # Coordinate offsets: puzzle bin [0, bin_size]x[0, bin_size] → CuTAMP centred at (0.5, 0)
    x_off = 0.5 - bin_size / 2
    y_off = -bin_size / 2

    def to_cutamp(pos):
        return [pos[0] + x_off, pos[1] + y_off, pos[2]]

    # ---- Movables ----
    tp = init["target_pos"]
    tq = init["target_quat"]   # [w, x, y, z]
    target = Cuboid(
        name="target",
        dims=[OBJ_SIZE, OBJ_SIZE, OBJ_SIZE],
        pose=_pose(to_cutamp(tp), tq),
        color=_TARGET_COLOR,
    )

    obs_list = []
    for i, o in enumerate(init["obstacles"]):
        oq = o["quat"]
        obs_list.append(Cuboid(
            name=f"obs_{i}",
            dims=[OBJ_SIZE, OBJ_SIZE, OBJ_SIZE],
            pose=_pose(to_cutamp(o["pos"]), oq),
            color=_OBS_COLOR,
        ))

    movables = [target] + obs_list

    # ---- Statics ----
    # Flat table surface (same dims as the blocks benchmark env)
    floor = Cuboid(
        name="floor",
        dims=[1.3, 0.85, 0.04],
        pose=_pose([0.25, 0.0, -0.02]),
        color=_FLOOR_COLOR,
    )

    # Exit region: thin marker south of the bin opening (puzzle y=0 → CuTAMP y=y_off)
    exit_cx = 0.5                  # x-centre matches bin x-centre
    exit_cy = y_off - 0.075        # 75 mm south of the south bin edge
    goal_surface = Cuboid(
        name="goal",
        dims=[bin_size, 0.15, 0.01],
        pose=_pose([exit_cx, exit_cy, 0.005]),
        color=_EXIT_COLOR,
    )

    # Bin walls: N (+y), E (+x), W (-x). No south wall — that is the exit.
    bin_cx = 0.5
    bin_cy = 0.0

    wall_n = Cuboid(
        name="wall_n",
        dims=[bin_size + 2 * wall_thickness, wall_thickness, WALL_HEIGHT],
        pose=_pose([bin_cx, y_off + bin_size + wall_thickness / 2, WALL_HEIGHT / 2]),
        color=_WALL_COLOR,
    )
    wall_e = Cuboid(
        name="wall_e",
        dims=[wall_thickness, bin_size, WALL_HEIGHT],
        pose=_pose([x_off + bin_size + wall_thickness / 2, bin_cy, WALL_HEIGHT / 2]),
        color=_WALL_COLOR,
    )
    wall_w = Cuboid(
        name="wall_w",
        dims=[wall_thickness, bin_size, WALL_HEIGHT],
        pose=_pose([x_off - wall_thickness / 2, bin_cy, WALL_HEIGHT / 2]),
        color=_WALL_COLOR,
    )

    statics = [floor, goal_surface, wall_n, wall_e, wall_w]

    # ---- Types ----
    type_to_objects = {
        "Movable": movables,
        "Surface": [floor, goal_surface],
    }

    # ---- Goal ----
    goal_atoms = {On.ground("target", "goal"), HandEmpty.ground()}
    for i in range(n_obstacles):
        goal_atoms.add(On.ground(f"obs_{i}", "floor"))

    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    return TAMPEnvironment(
        name=f"puzzle_{scenario_name}",
        movables=movables,
        statics=statics,
        type_to_objects=type_to_objects,
        goal_state=frozenset(goal_atoms),
    )


def run_cutamp_on_scenario(
    scenario_path: str,
    num_particles: int = 1024,
    num_opt_steps: int = 1000,
    experiment_root: str = "/tmp/cutamp-bench",
) -> dict:
    """Run CuTAMP on a single puzzle scenario. Returns a metrics dict with 4 shared metrics."""
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

    with open(scenario_path) as f:
        scenario_data = yaml.safe_load(f)
    n_obstacles = int(scenario_data["n_obstacles"])

    env = puzzle_scenario_to_cutamp_env(scenario_path)
    config = TAMPConfiguration(
        num_particles=num_particles,
        num_opt_steps=num_opt_steps,
        enable_visualizer=False,
        experiment_root=experiment_root,
        enable_experiment_logging=True,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    t0 = time.time()
    curobo_plan, num_satisfying = run_cutamp(env, config, cost_reducer, constraint_checker)
    elapsed = time.time() - t0

    plan_success = num_satisfying > 0
    execution_success = curobo_plan is not None and len(curobo_plan) > 0

    return {
        "scenario": os.path.basename(scenario_path),
        "n_obstacles": n_obstacles,
        "plan_success": plan_success,
        "plan_time_s": elapsed,
        "execution_success": execution_success,
        "total_time_s": elapsed,
        "num_satisfying": int(num_satisfying),
    }
