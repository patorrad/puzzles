"""
apply_solution.py – Convert puzzle solution.json into genesismpc actor configs.

Usage
-----
    python apply_solution.py [--solution solution.json] [--genesismpc-dir /path/to/genesismpc]

What it does
------------
1. Reads solution.json produced by `main.py --save`.
2. Writes puzzle_target.yaml and puzzle_obstacle_N.yaml under
   <genesismpc>/conf/actors/.
3. Rewrites <genesismpc>/examples/ur5_stick_stacked_blocks/config_ur.yaml,
   replacing the actor list with the puzzle actors (keeping ur5_suction + goal).
"""

import argparse
import json
import os
import re

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_SOLUTION  = os.path.join(os.path.dirname(__file__), "2_objects.json")
DEFAULT_GMPC_DIR  = os.path.join(os.path.dirname(__file__), "../genesismpc")
ACTORS_SUBDIR     = "conf/actors"
CONFIG_REL        = "examples/ur5_stick_stacked_blocks_value/config_ur.yaml"

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
        size      = actor["size"]
        pos       = actor["init_pos"]
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

    print("\nDone. Actor names added to config:")
    for n in full_actor_list:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
