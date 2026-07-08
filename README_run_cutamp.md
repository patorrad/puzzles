All three files are done. Here's a summary of what changed:

pipeline.py: Added asdict to the existing dataclasses import, added puzzle_results/ to both directory creation loops, and added 4 lines in _puzzle_worker to save each PuzzleResult to puzzle_results/{name}.json immediately after planning.

cutamp_adapter.py: run_cutamp_on_scenario now reads n_obstacles from the YAML, unpacks curobo_plan from run_cutamp()'s return, and returns the 4 shared metrics: plan_success, plan_time_s, execution_success, total_time_s.

compare_planners.py: Complete rewrite with two modes:

cutamp --scenario_dir <dir> --cutamp_out cutamp_results — runs Pipeline B, saves one JSON per scenario
merge <pipeline_a_dir> <cutamp_results_dir> — reads both pipelines' disk outputs, prints aggregated table, saves CSV
The workflow is exactly as planned:


# Pipeline A (IsaacLab env)
python pipeline.py --config-name=pipeline n_scenarios=10

# Pipeline B (cutamp env)
python compare_planners.py cutamp --scenario_dir outputs/latest/scenarios --cutamp_out cutamp_results

# Merge
python compare_planners.py merge outputs/latest cutamp_results --out compare_results.csv