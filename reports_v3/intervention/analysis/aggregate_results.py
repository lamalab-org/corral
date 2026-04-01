"""Aggregate all intervention experiment results into a single DataFrame.

Scans runs/ subdirectories for *_report.json files, parses condition
metadata from directory structure, and outputs a consolidated CSV.

Output columns:
    env, agent, intervention, num_steps, dir_name,
    task_id, trial_id, score, success, surrendered
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT


def parse_condition_from_path(report_path: Path) -> dict | None:
    """Extract condition metadata from the directory path.

    Expected: runs/{env}/{agent}/{dir_name}/{report}.json
    """
    parts = report_path.relative_to(INTERVENTION_ROOT / "runs").parts
    if len(parts) < 4:
        return None

    env, agent, dir_name = parts[0], parts[1], parts[2]

    # Parse intervention type and num_steps from dir_name
    if dir_name == "baseline":
        intervention = "none"
        num_steps = 0
    else:
        match = re.match(r"(success|failed)_step(n?\d+)", dir_name)
        if not match:
            return None
        intervention = match.group(1)
        step_str = match.group(2)
        num_steps = -int(step_str[1:]) if step_str.startswith("n") else int(step_str)

    return {
        "env": env,
        "agent": agent,
        "intervention": intervention,
        "num_steps": num_steps,
        "dir_name": dir_name,
    }


def load_report(report_path: Path) -> list[dict]:
    """Load a report JSON and extract per-trial rows."""
    with report_path.open() as f:
        report = json.load(f)

    rows = []
    for task_id, task_data in report.get("task_results", {}).items():
        rows.extend(
            {
                "task_id": task_id,
                "trial_id": trial.get("trial_id"),
                "score": trial.get("score", 0.0),
                "success": trial.get("success", False),
                "surrendered": trial.get("surrendered", False),
            }
            for trial in task_data.get("trials", [])
        )
    return rows


def aggregate() -> pd.DataFrame:
    """Scan all run directories and aggregate results."""
    runs_dir = INTERVENTION_ROOT / "runs"
    if not runs_dir.exists():
        logger.info(f"No runs directory found at {runs_dir}")
        return pd.DataFrame()

    all_rows = []
    report_files = list(runs_dir.rglob("*_report.json"))
    logger.info(f"Found {len(report_files)} report files")

    for report_path in sorted(report_files):
        condition = parse_condition_from_path(report_path)
        if condition is None:
            logger.info(f"  SKIP: Could not parse condition from {report_path}")
            continue

        trials = load_report(report_path)
        for trial in trials:
            trial.update(condition)
            all_rows.append(trial)

        logger.info(
            f"  {condition['env']}/{condition['agent']}/{condition['dir_name']}: "
            f"{len(trials)} trials"
        )

    return pd.DataFrame(all_rows)


def main():
    results_df = aggregate()

    if results_df.empty:
        logger.info("No results to aggregate.")
        return

    # Save
    output_path = INTERVENTION_ROOT / "analysis" / "results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(results_df)} rows to {output_path}")

    # Summary
    logger.info("\n=== Summary ===")
    summary = (
        results_df.groupby(["env", "agent", "intervention", "num_steps"])
        .agg(
            n_trials=("success", "count"),
            success_rate=("success", "mean"),
            avg_score=("score", "mean"),
        )
        .round(3)
    )
    logger.info(summary.to_string())


if __name__ == "__main__":
    main()
