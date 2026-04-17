"""
apply_solution.py – Convert puzzle solution.json into genesismpc actor configs.

Usage
-----
    python apply_solution.py [--solution solution.json] [--genesismpc-dir /path/to/genesismpc]

What it does
------------
1. Reads solution.json produced by `main.py --save`.
2. Transforms positions from puzzle frame → robot world frame (see below).
3. Writes puzzle_target.yaml and puzzle_obstacle_N.yaml under
   <genesismpc>/conf/actors/.
4. Rewrites config_ur.yaml replacing the actor list with the puzzle actors.
5. Writes a <solution>_robot.json with all step positions in robot frame.

Coordinate mapping
------------------
Puzzle frame          Robot world frame
  x  (east–west)  →  y  (centered at 0)
  y  (south–north →  x  (exit = BIN_EXIT_X, north wall = BIN_EXIT_X + BIN_D)
  z  (from floor)  →  z  (table top + puzzle_z)

North–south pushing (NS pusher) thus moves along the robot x-axis,
which is the direction toward the robot base at x ≈ 0.208.
"""

import argparse
import json
import os
import re

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_SOLUTION  = os.path.join(os.path.dirname(__file__), "solution_obs_3_simple_extraction.json")
DEFAULT_GMPC_DIR  = os.path.join(os.path.dirname(__file__), "../genesismpc")
ACTORS_SUBDIR     = "conf/actors"
CONFIG_REL        = "examples/ur5_stick_stacked_blocks_stand/config_ur.yaml"

# ── coordinate transform constants ──────────────────────────────────────────
# table.yaml: pos_z=0.75, size_z=0.14  →  table top z = 0.75 + 0.07 = 0.82
TABLE_TOP_Z = 0.72

# Robot x-coordinate of the bin's south/exit face (puzzle y=0).
# Robot base is at x=0.208; the bin exit should be a comfortable reach away.
BIN_EXIT_X  = 0.40

# ── coordinate transform ────────────────────────────────────────────────────

def puzzle_to_robot_pos(pos: list, bin_w: float) -> list:
    """
    Transform a position from puzzle frame to robot world frame.

    Puzzle (px, py, pz)  →  Robot (rx, ry, rz):
        rx = BIN_EXIT_X + py      (south face of bin at BIN_EXIT_X)
        ry = px - bin_w / 2       (bin centred at robot y = 0)
        rz = TABLE_TOP_Z + pz
    """
    px, py, pz = pos
    return [
        BIN_EXIT_X + py,
        px - bin_w / 2,
        TABLE_TOP_Z + pz,
    ]


def puzzle_to_robot_size(size: list) -> list:
    """
    Swap the x and y extents of a box to match the axis remapping.
    Puzzle size (sx, sy, sz)  →  Robot size (sy, sx, sz).
    """
    sx, sy, sz = size
    return [sy, sx, sz]


# ── actor YAML generation ────────────────────────────────────────────────────

def _actor_yaml(name: str, size: list, pos: list, color: list,
                rho: float, friction: float, fixed: bool = False) -> str:
    """Return a YAML string matching the block.yaml schema."""
    def fmt(lst):
        return "[" + ", ".join(f"{v:.4g}" for v in lst) + "]"

    return (
        f'type: "box"\n'
        f'name: "{name}"\n'
        f'size: {fmt(size)}\n'
        f'init_pos: {fmt(pos)}\n'
        f'mass: 1.0\n'
        f'rho: {rho:.4g}\n'
        f'fixed: {"True" if fixed else "False"}\n'
        f'handle: None\n'
        f'color: {fmt(color)}\n'
        f'friction: {friction:.4g}\n'
        f'noise_sigma_size: [0.0, 0.0, 0.0]\n'
        f'noise_percentage_friction: 0.0\n'
        f'noise_percentage_mass: 0.0\n'
    )


# ── config_ur.yaml patching ──────────────────────────────────────────────────

def _patch_config(config_path: str, actor_names: list):
    """Replace the `actors:` line and `goal:` line in config_ur.yaml."""
    with open(config_path) as f:
        text = f.read()

    # Build new actors list string (YAML flow sequence)
    actors_str = "[" + ", ".join(f"'{a}'" for a in actor_names) + "]"
    text = re.sub(r"^actors:.*$", f"actors: {actors_str}", text, flags=re.MULTILINE)

    with open(config_path, "w") as f:
        f.write(text)

    print(f"  Updated actors list in {config_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution",     default=DEFAULT_SOLUTION,
                    help="Path to solution.json")
    ap.add_argument("--genesismpc-dir", default=DEFAULT_GMPC_DIR,
                    help="Root of the genesismpc repo")
    args = ap.parse_args()

    # Load solution
    with open(args.solution) as f:
        sol = json.load(f)

    env_cfg   = sol["env_config"]
    actors_js = sol["actors"]          # list of actor dicts from save_solution()
    bin_w     = env_cfg["BIN_W"]

    def tpos(p): return puzzle_to_robot_pos(p, bin_w)
    def tsz(s):  return puzzle_to_robot_size(s)

    gmpc = os.path.abspath(args.genesismpc_dir)
    actors_dir  = os.path.join(gmpc, ACTORS_SUBDIR)
    config_path = os.path.join(gmpc, CONFIG_REL)

    os.makedirs(actors_dir, exist_ok=True)

    puzzle_actor_names = []

    for actor in actors_js:
        # solution.json uses bare names like "target", "obstacle_0"
        # prefix with "puzzle_" so they're easy to identify in the repo
        bare_name = actor["name"]
        aname     = f"puzzle_{bare_name}"
        size      = tsz(actor["size"])
        pos       = tpos(actor["init_pos"])
        color     = actor["color"]
        rho       = actor.get("rho", 500.0)   # default if not saved
        friction  = actor["friction"]
        fixed     = actor.get("fixed", False)

        yaml_str = _actor_yaml(aname, size, pos, color, rho, friction, fixed)
        out_path = os.path.join(actors_dir, f"{aname}.yaml")
        with open(out_path, "w") as f:
            f.write(yaml_str)
        print(f"  Wrote {out_path}")
        puzzle_actor_names.append(aname)

    # Keep ur5_suction + goal first, then puzzle objects
    full_actor_list = ["ur5_suction", "goal"] + puzzle_actor_names
    _patch_config(config_path, full_actor_list)

    # Write robot-frame steps so the MPPI executor works in world coordinates
    if "steps" in sol:
        robot_steps = []
        for step in sol["steps"]:
            robot_steps.append({
                "obj_idx":           step["obj_idx"],
                "obj_name":          step["obj_name"],
                "start_pos":         tpos(step["start_pos"]),
                "start_quat":        step["start_quat"],
                "end_pos":           tpos(step["end_pos"]),
                "end_quat":          step["end_quat"],
                "target_start_pos":  tpos(step["target_start_pos"]),
                "target_start_quat": step["target_start_quat"],
                "target_end_pos":    tpos(step["target_end_pos"]),
                "target_end_quat":   step["target_end_quat"],
            })
        steps_path = os.path.splitext(args.solution)[0] + "_robot.json"
        with open(steps_path, "w") as f:
            json.dump({"steps": robot_steps,
                       "frame": "robot_world",
                       "BIN_EXIT_X": BIN_EXIT_X,
                       "TABLE_TOP_Z": TABLE_TOP_Z}, f, indent=2)
        print(f"\n  Robot-frame steps written to {steps_path}")

    print("\nDone. Actor names added to config:")
    for n in full_actor_list:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
