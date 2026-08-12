"""Deterministic reference-solution audit for the Stargazer task bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stargazer.evaluator import (
    EvaluationResult,
    SubmissionError,
    evaluate_submission,
    make_stargazer_scorer,
)
from stargazer.models import CandidateSubmission, StargazerTask, load_task

TASK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = TASK_ROOT / "data"
DEFAULT_REPORT_PATH = DEFAULT_DATA_ROOT / "reference_audit.json"
UPSTREAM_REVISION = "3f617667472061e253288c7b26f0e70f186f2dff"
FAILURE_REASONS = (
    "invalid_reference_parameters",
    "bic_gate",
    "rms_gate",
    "physical_match_gate",
    "count_gate",
)


def reference_submission(
    task: StargazerTask, raw_record: dict[str, Any]
) -> CandidateSubmission:
    """Build canonical JSON from a record's published reference system."""
    noise = raw_record.get("config", {}).get("noise", {}) or {}
    return CandidateSubmission(
        planets=[
            {
                "P_days": planet.P_days,
                "m_sin_i_mjup": planet.m_sin_i_mjup,
                "e": planet.e,
                "omega_rad": planet.omega_rad,
                "l_rad": planet.l_rad,
                "inc_rad": planet.inc_rad,
                "Omega_rad": planet.Omega_rad,
            }
            for planet in task.truth_planets
        ],
        noise_jitter_ms=float(noise.get("sigma_jitter_ms", 0.0)),
    )


def _failed_gates(result: EvaluationResult) -> list[str]:
    gates = (
        ("bic_gate", result.ok_delta_bic),
        ("rms_gate", result.ok_rms),
        ("physical_match_gate", result.ok_match),
        ("count_gate", result.ok_count),
    )
    return [name for name, passed in gates if not passed]


def _audit_record(task_file: Path, source: str) -> dict[str, Any]:
    with task_file.open(encoding="utf-8") as handle:
        raw_record = json.load(handle)
    task = load_task(task_file, source=source)
    submission = reference_submission(task, raw_record)
    final_answer = json.dumps(submission.canonical_payload(), allow_nan=False)

    try:
        result = evaluate_submission(task, submission)
    except (SubmissionError, ValueError, TypeError, OverflowError) as exc:
        score = make_stargazer_scorer(task)(final_answer)
        return {
            "task_id": task.task_id,
            "source": source,
            "difficulty": task.truth_difficulty,
            "score": score,
            "failure_reasons": ["invalid_reference_parameters"],
            "error": str(exc),
        }

    score = make_stargazer_scorer(task)(final_answer)
    if score != result.score:
        raise RuntimeError(f"Final scorer diverged from evaluator for {task.task_id}")
    return {
        "task_id": task.task_id,
        "source": source,
        "difficulty": task.truth_difficulty,
        "score": score,
        "failure_reasons": _failed_gates(result),
    }


def audit_task_bank(data_root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    """Evaluate all published systems through the final scoring contract."""
    root = Path(data_root)
    records = [
        _audit_record(task_file, source)
        for source in ("synthetic", "real")
        for task_file in sorted((root / source).glob("*.json"))
    ]
    by_source: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    for source in ("synthetic", "real"):
        source_records = [record for record in records if record["source"] == source]
        by_source[source] = {
            "passing": sum(record["score"] == 1.0 for record in source_records),
            "total": len(source_records),
        }
    for difficulty in range(1, 11):
        difficulty_records = [
            record
            for record in records
            if record["source"] == "synthetic" and record["difficulty"] == difficulty
        ]
        by_difficulty[str(difficulty)] = {
            "passing": sum(record["score"] == 1.0 for record in difficulty_records),
            "total": len(difficulty_records),
        }
    observed_reason_counts = Counter(
        reason for record in records for reason in record["failure_reasons"]
    )
    reason_counts = {
        reason: observed_reason_counts.get(reason, 0) for reason in FAILURE_REASONS
    }
    return {
        "upstream_revision": UPSTREAM_REVISION,
        "criteria": {
            "minimum_delta_bic_per_point": 0.0,
            "maximum_rms_factor": 1.5,
            "minimum_match_score": 0.8,
            "require_count_match": True,
        },
        "summary": {
            "by_source": by_source,
            "passing_synthetic_by_difficulty": by_difficulty,
            "failure_reason_counts": reason_counts,
        },
        "records": records,
    }


def selected_task_ids(
    environments_root: str | Path | None = None,
) -> dict[int, set[str]]:
    """Read the explicit official task memberships committed for Levels 1-3."""
    root = Path(environments_root or TASK_ROOT / "environments")
    selected: dict[int, set[str]] = defaultdict(set)
    for level in (1, 2, 3):
        for selector_file in sorted(
            (root / f"level_{level}" / "tasks_json").glob("*.json")
        ):
            with selector_file.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            selectors = payload if isinstance(payload, list) else [payload]
            for selector in selectors:
                selected[level].update(selector.get("task_ids", []))
    return dict(selected)


def validate_official_banks(report: dict[str, Any]) -> None:
    """Require 20 unique, reference-valid synthetic tasks in every level."""
    passing = {
        record["task_id"]
        for record in report["records"]
        if record["source"] == "synthetic" and record["score"] == 1.0
    }
    selected = selected_task_ids()
    all_ids: set[str] = set()
    for level in (1, 2, 3):
        task_ids = selected.get(level, set())
        if len(task_ids) != 20:
            raise ValueError(
                f"Level {level} selects {len(task_ids)} tasks, expected 20"
            )
        invalid = task_ids - passing
        if invalid:
            raise ValueError(
                f"Level {level} contains reference-invalid tasks: {sorted(invalid)}"
            )
        overlap = all_ids & task_ids
        if overlap:
            raise ValueError(f"Official levels overlap: {sorted(overlap)}")
        all_ids.update(task_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed report instead of rewriting it.",
    )
    args = parser.parse_args()

    report = audit_task_bank()
    validate_official_banks(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            raise SystemExit(f"Audit report is stale: {args.output}")
        return
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
