"""
apply_solution.py – Extract movement steps from a puzzle solution JSON.

Usage
-----
    python apply_solution.py [--solution solution.json] [--output steps.json]

Output
------
JSON with steps: [{obj_idx, obj_name, start_pos, start_quat, end_pos, end_quat}]

No coordinate transform is applied.  The puzzle planner is expected to run
with --isaaclabmpc-config so that its bin_center and floor_z match the
isaaclabmpc world frame directly.
"""

import argparse
import json
import os

_DIR = os.path.dirname(__file__)
DEFAULT_SOLUTION = os.path.join(_DIR, "solution.json")
DEFAULT_OUTPUT   = os.path.join(_DIR,
    "../isaaclabmpc/examples/ur16e_reach_stand_blocks_sim/extraction_robot_sim.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default=DEFAULT_SOLUTION,
                    help="Path to solution.json produced by main.py --save")
    ap.add_argument("--output",   default=DEFAULT_OUTPUT,
                    help="Destination steps JSON (read by isaaclabmpc planner)")
    args = ap.parse_args()

    with open(args.solution) as f:
        sol = json.load(f)

    steps = [
        {
            "obj_idx":    step["obj_idx"],
            "obj_name":   step["obj_name"],
            "start_pos":  step["start_pos"],
            "start_quat": step["start_quat"],
            "end_pos":    step["end_pos"],
            "end_quat":   step["end_quat"],
        }
        for step in sol["steps"]
    ]

    with open(args.output, "w") as f:
        json.dump({"steps": steps}, f, indent=2)
    print(f"Steps written to {args.output}")


if __name__ == "__main__":
    main()
