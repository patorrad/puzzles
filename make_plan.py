"""
make_plan.py – Convert puzzle solution.json into genesismpc actor configs
                without any axis remapping.

Use this when the puzzle was planned in the same x/y frame as the robot world
(i.e. block x/y positions already match real-world coordinates).  Only the z
height of each object is overridden to place it on the real table surface.

Usage
-----
    python make_plan.py --solution solution.json \\
                        --object-z 0.86 \\
                        [--genesismpc-dir ../genesismpc]

    --object-z   Z coordinate of the centre of a block resting on the table.
                 Defaults to TABLE_TOP_Z + OBJ_H (table surface + half block height).

What it does
------------
1. Reads solution.json produced by `main.py --save`.
2. Keeps x/y positions from the solution unchanged.
3. Replaces the z of every object position with --object-z.
4. Writes puzzle_target.yaml and puzzle_obstacle_N.yaml under
   <genesismpc>/conf/actors/.
5. Rewrites config_ur.yaml replacing the actor list with the puzzle actors.
6. Writes a <solution>_robot.json with step positions using the patched z.
"""

import argparse
import json
import os
import re

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SOLUTION = os.path.join(os.path.dirname(__file__),
                                "solution_obs_3_simple_extraction.json")
DEFAULT_GMPC_DIR = os.path.join(os.path.dirname(__file__), "../genesismpc")
ACTORS_SUBDIR    = "conf/actors"
CONFIG_REL       = "examples/ur5_stick_stacked_blocks_stand/config_ur.yaml"

# Default table-top + half-block height (matches apply_solution.py defaults)
TABLE_TOP_Z = 0.72
OBJ_H_DEFAULT = 0.04   # OBJ_SIZE / 2 = 0.08 / 2


# ── helpers ───────────────────────────────────────────────────────────────────

def patch_z(pos: list, object_z: float) -> list:
    """Return [x, y, object_z] — x/y unchanged, z replaced."""
    return [pos[0], pos[1], object_z]


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


def _patch_config(config_path: str, actor_names: list):
    """Replace the `actors:` line in config_ur.yaml."""
    with open(config_path) as f:
        text = f.read()
    actors_str = "[" + ", ".join(f"'{a}'" for a in actor_names) + "]"
    text = re.sub(r"^actors:.*$", f"actors: {actors_str}", text, flags=re.MULTILINE)
    with open(config_path, "w") as f:
        f.write(text)
    print(f"  Updated actors list in {config_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default=DEFAULT_SOLUTION,
                    help="Path to solution.json")
    ap.add_argument("--genesismpc-dir", default=DEFAULT_GMPC_DIR,
                    help="Root of the genesismpc repo")
    ap.add_argument("--object-z", type=float,
                    default=TABLE_TOP_Z + OBJ_H_DEFAULT,
                    help="Real-world z of the centre of a block resting on the "
                         f"table (default: TABLE_TOP_Z + OBJ_H = "
                         f"{TABLE_TOP_Z + OBJ_H_DEFAULT:.4g})")
    args = ap.parse_args()

    with open(args.solution) as f:
        sol = json.load(f)

    actors_js = sol["actors"]
    object_z  = args.object_z

    print(f"Object centre z: {object_z:.4g} m")

    gmpc        = os.path.abspath(args.genesismpc_dir)
    actors_dir  = os.path.join(gmpc, ACTORS_SUBDIR)
    config_path = os.path.join(gmpc, CONFIG_REL)
    os.makedirs(actors_dir, exist_ok=True)

    puzzle_actor_names = []

    for actor in actors_js:
        bare_name = actor["name"]
        aname     = f"puzzle_{bare_name}"
        size      = actor["size"]                         # no axis swap
        pos       = patch_z(actor["init_pos"], object_z) # x/y unchanged, z patched
        color     = actor["color"]
        rho       = actor.get("rho", 500.0)
        friction  = actor["friction"]
        fixed     = actor.get("fixed", False)

        # Fixed geometry (walls, floor) keeps its original z — only movable
        # objects land on the table surface.
        if fixed:
            pos = actor["init_pos"]

        yaml_str = _actor_yaml(aname, size, pos, color, rho, friction, fixed)
        out_path = os.path.join(actors_dir, f"{aname}.yaml")
        with open(out_path, "w") as f:
            f.write(yaml_str)
        print(f"  Wrote {out_path}")
        puzzle_actor_names.append(aname)

    full_actor_list = ["ur5_suction", "goal"] + puzzle_actor_names
    _patch_config(config_path, full_actor_list)

    # Write steps with patched z for movable objects, original z for fixed.
    if "steps" in sol:
        patched_steps = []
        for step in sol["steps"]:
            patched_steps.append({
                "obj_idx":           step["obj_idx"],
                "obj_name":          step["obj_name"],
                "start_pos":         patch_z(step["start_pos"], object_z),
                "start_quat":        step["start_quat"],
                "end_pos":           patch_z(step["end_pos"], object_z),
                "end_quat":          step["end_quat"],
                "target_start_pos":  patch_z(step["target_start_pos"], object_z),
                "target_start_quat": step["target_start_quat"],
                "target_end_pos":    patch_z(step["target_end_pos"], object_z),
                "target_end_quat":   step["target_end_quat"],
            })
        steps_path = os.path.splitext(args.solution)[0] + "_robot.json"
        with open(steps_path, "w") as f:
            json.dump({"steps": patched_steps,
                       "object_z": object_z}, f, indent=2)
        print(f"\n  Steps written to {steps_path}")

    print("\nDone. Actor names added to config:")
    for n in full_actor_list:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
