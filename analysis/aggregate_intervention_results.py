"""Aggregate intervention reports into trial-level CSV.

Reads ``results/data/intervention_reports.jsonl`` (downloaded from HF)
and extracts one row per trial from the Task Results column.

Output: ``results/data/intervention_results.csv``

Columns:
    env, agent, intervention, num_steps, dir_name,
    task_id, trial_id, score, success, surrendered
"""

import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent

_CONDITION_RE = re.compile(r"^(baseline|success|failed)(?:_step(n?\d+))?$")

# HF uses "tool_calling"; original analysis scripts use "toolcalling"
_AGENT_REVERSE = {"tool_calling": "toolcalling", "react": "react"}


def _parse_condition(condition: str) -> tuple[str, int] | None:
    """Parse condition string -> (intervention_type, num_steps)."""
    if condition == "baseline":
        return ("none", 0)
    m = _CONDITION_RE.match(condition)
    if not m:
        return None
    ctype = m.group(1)
    raw_step = m.group(2)
    if raw_step is None:
        return None
    num_steps = -int(raw_step[1:]) if raw_step.startswith("n") else int(raw_step)
    return (ctype, num_steps)


def aggregate(jsonl_path: Path) -> pd.DataFrame:
    """Read intervention reports JSONL and extract per-trial rows."""
    reports = pd.read_json(jsonl_path, lines=True)
    logger.info(f"Loaded {len(reports)} report rows from {jsonl_path}")

    all_rows = []
    for _, row in reports.iterrows():
        env = row["environment"]
        agent = _AGENT_REVERSE.get(row["agent_type"], row["agent_type"])
        condition = row["condition"]

        parsed = _parse_condition(condition)
        if parsed is None:
            logger.warning(f"  SKIP unrecognised condition: {condition}")
            continue
        intervention, num_steps = parsed

        task_results = row.get("Task Results")
        if isinstance(task_results, str):
            task_results = json.loads(task_results)
        if not task_results:
            continue

        all_rows.extend(
            {
                "task_id": task_id,
                "trial_id": trial.get("trial_id"),
                "score": trial.get("score", 0.0),
                "success": trial.get("success", False),
                "surrendered": trial.get("surrendered", False),
                "env": env,
                "agent": agent,
                "intervention": intervention,
                "num_steps": num_steps,
                "dir_name": condition,
            }
            for task_id, task_data in task_results.items()
            for trial in task_data.get("trials", [])
        )

    return pd.DataFrame(all_rows)


def main():
    jsonl_path = _SCRIPT_DIR / "results" / "data" / "intervention_reports.jsonl"
    if not jsonl_path.exists():
        logger.error(
            f"{jsonl_path} not found. Run download_intervention_reports_from_hf.py first."
        )
        raise SystemExit(1)

    results_df = aggregate(jsonl_path)
    if results_df.empty:
        logger.info("No results to aggregate.")
        return

    output_path = _SCRIPT_DIR / "results" / "data" / "intervention_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(results_df)} rows to {output_path}")

    # Summary
    summary = (
        results_df.groupby(["env", "agent", "intervention", "num_steps"])
        .agg(
            n_trials=("success", "count"),
            success_rate=("success", "mean"),
            avg_score=("score", "mean"),
        )
        .round(3)
    )
    logger.info(f"\n{summary.to_string()}")


if __name__ == "__main__":
    main()
