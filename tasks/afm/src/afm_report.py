"""Summarize new AFM reports without mixing instrument failures into accuracy."""

# ruff: noqa: T201 -- this module is a human-facing CLI.
import argparse
import gzip
import json
import math
from pathlib import Path

INFRASTRUCTURE_MARKER = "AFM_INFRASTRUCTURE_FAILURE:"


def summarize(report: dict) -> dict:
    """Classify explicit controller failures; never infer them from low scores."""
    rows = []
    scores = []
    for task_id, task in report["task_results"].items():
        if not task_id.startswith("afm_"):
            raise ValueError(f"Expected an AFM report, found {task_id}")
        for index, trial in enumerate(task["trials"]):
            # Only inspect recorded errors, not agent prose or successful outputs.
            errors = [trial.get("error")]
            errors.extend(
                call.get("error_message") for call in trial.get("tool_calls", [])
            )
            infrastructure = next(
                (
                    error
                    for error in errors
                    if isinstance(error, str) and INFRASTRUCTURE_MARKER in error
                ),
                None,
            )
            score = trial.get("score")
            valid_score = (
                isinstance(score, int | float)
                and not isinstance(score, bool)
                and math.isfinite(score)
                and 0 <= score <= 1
            )
            if infrastructure is not None:
                category = "instrument_failure"
            elif valid_score:
                category = "scored"
                scores.append(score)
            else:
                category = "unscored"
            rows.append(
                {
                    "task_id": task_id,
                    "trial_index": index,
                    "category": category,
                    "reported_score": score if valid_score else None,
                    "instrument_error": infrastructure,
                }
            )
    return {
        "classification": "Explicit managed-afm-v1 controller errors only; other infrastructure incidents require review.",
        "attempted_trials": len(rows),
        "instrument_failures": sum(
            row["category"] == "instrument_failure" for row in rows
        ),
        "unscored_trials": sum(row["category"] == "unscored" for row in rows),
        "scored_trials": len(scores),
        "mean_score_excluding_instrument_failures": sum(scores) / len(scores)
        if scores
        else None,
        "trials": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="Corral AFM report.json or report.json.gz"
    )
    args = parser.parse_args()
    opener = gzip.open if args.report.suffix == ".gz" else Path.open
    with opener(args.report, "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    print(json.dumps(summarize(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
