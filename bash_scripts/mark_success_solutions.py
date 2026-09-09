#!/usr/bin/env python3
"""
For each solution JSON in solutions/1_verification, check whether its seed
appears in results/1_verification with a non-null plan_length.  If so, copy
it as <stem>_success.json so run_solutions.py can find it.
"""

import re
import shutil
from pathlib import Path

import pandas as pd

SOLUTIONS_ROOT = Path("/home/paolo/Documents/puzzle/solutions/1_verification")
RESULTS_ROOT   = Path("/home/paolo/Documents/puzzle/results/1_verification")

# Map obs count → results DataFrame (all planners combined).
results_by_obs: dict[int, pd.DataFrame] = {}
for csv_path in RESULTS_ROOT.iterdir():
    m = re.search(r"(\d+)obs", csv_path.name)
    if m:
        results_by_obs[int(m.group(1))] = pd.read_csv(csv_path)

created = skipped = missing = 0

for sol_dir in sorted(SOLUTIONS_ROOT.iterdir()):
    if not sol_dir.is_dir():
        continue

    # Extract obs count and planner from directory name.
    m_obs = re.search(r"(\d+)obs", sol_dir.name)
    if not m_obs:
        print(f"[skip] {sol_dir.name}: can't parse obs count")
        continue
    obs = int(m_obs.group(1))

    df = results_by_obs.get(obs)
    if df is None:
        print(f"[skip] {sol_dir.name}: no results CSV for {obs}obs")
        continue

    # Detect planner: directory name starts with planner prefix.
    planner = None
    for p in ("alphazero", "mcts", "more"):
        if sol_dir.name.startswith(p):
            planner = p
            break
    if planner is None:
        print(f"[skip] {sol_dir.name}: unknown planner")
        continue

    # Seeds that have a plan for this planner+obs combo.
    valid = df[(df["planner"] == planner) & df["plan_length"].notna()]["seed"]
    valid_seeds = set(valid.astype(int).tolist())

    for sol_file in sorted(sol_dir.glob("*.json")):
        if "_success" in sol_file.name:
            continue  # already marked

        m_seed = re.search(r"seed_(\d+)", sol_file.name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        if seed not in valid_seeds:
            missing += 1
            continue

        dst = sol_file.with_name(sol_file.stem + "_success.json")
        if dst.exists():
            skipped += 1
            continue

        shutil.copy2(sol_file, dst)
        print(f"  [copy] {dst.relative_to(SOLUTIONS_ROOT)}")
        created += 1

print(f"\nDone. Created: {created}  Already existed: {skipped}  Seed not in results: {missing}")
