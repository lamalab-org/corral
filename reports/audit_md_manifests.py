"""Audit saved MD manifest failures without running or changing evaluations.

Run from the repository root:
    python reports/audit_md_manifests.py --output-dir analysis/md_manifest_audit

The audit compares saved validator results, terminal evaluations, and the current
answer-path parser. A rescore candidate is not an inferred scientific pass.
Only repository parser code is loaded; submitted scripts are never executed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import csv
import datetime
import gzip
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CHECKS = {
    "manifest_and_artifacts",
    "manifest_and_linked_artifacts",
    "readable_manifest",
    "submitted_manifest_exists",
}
FIELDS = [
    "level",
    "agent_model",
    "task",
    "trial",
    "execution_id",
    "score",
    "first_failed_check",
    "last_validation",
    "validated_path",
    "current_parser_status",
    "current_parser_path",
    "rescore_candidate",
    "manifest_failure",
    "manual_missing_artifacts_zero",
    "terminal_error",
    "evaluation_error",
    "feedback",
    "report_file",
    "trace_url",
]


def current_parser():
    """Load just the actual parser helper, without MD or simulator dependencies."""
    path = ROOT / "tasks/corral_md/src/corral_md/submission.py"
    source = path.read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_manifest_reference"
    )
    namespace = {"Path": Path, "re": re}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name], hashlib.sha256(source.encode()).hexdigest()


def validation_calls(trial):
    validations = []
    for position, call in enumerate(trial.get("tool_calls", [])):
        if call.get("tool_name") != "validate_submission":
            continue
        result = call.get("result")
        if isinstance(result, str):
            with contextlib.suppress(ValueError):
                result = json.loads(result)
        validations.append(
            {
                "position": position,
                "arguments": call.get("arguments") or {},
                "result": result,
            }
        )
    return validations


def parse_answer(parser, answer, validated_path):
    if not isinstance(answer, str) or not answer:
        return "no_text_answer", ""
    try:
        extracted = parser(answer)
    except ValueError as exc:
        return "ambiguous", str(exc)
    if validated_path and extracted == validated_path:
        return "matches_validated_path", extracted
    if extracted == answer and "\n" in answer:
        return "not_extracted", ""
    return "uncompared_path", extracted


def audit(reports_dir):
    parser, parser_sha = current_parser()
    index_path = reports_dir / "trials.csv"
    with index_path.open(newline="") as handle:
        traces = {
            row["execution_id"]: row["langfuse_trace_url"]
            for row in csv.DictReader(handle)
        }
    rows, details, history, sources = [], [], [], []
    normalization_count = 0
    long_answer_rescore_count = 0
    for path in sorted(reports_dir.glob("level_*/*/report.json.*")):
        opener = lzma.open if path.suffix == ".xz" else gzip.open
        with opener(path, "rt") as handle:
            report = json.load(handle)
        report_file = path.relative_to(ROOT).as_posix()
        sources.append(
            {
                "path": report_file,
                "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        events = report.get("metadata", {}).get("manifest_path_rescoring", [])
        if isinstance(events, dict):
            events = [events]
        history.extend({"report_file": report_file, **event} for event in events)
        for task, result in report["task_results"].items():
            for number, trial in enumerate(result["trials"]):
                evaluation = trial.get("evaluation") or {}
                metadata = evaluation.get("metadata") or {}
                normalization_count += "manifest_path_normalization" in metadata
                long_answer_rescore_count += (metadata.get("rescore") or {}).get(
                    "reason"
                ) == "ENAMETOOLONG submission parsing"
                workflow = metadata.get("workflow_evaluation") or {}
                failed = [
                    check
                    for check in workflow.get("checks", [])
                    if check.get("status") == "failed"
                ]
                first = failed[0]["name"] if failed else ""
                manifest_failure = first in MANIFEST_CHECKS
                validations = validation_calls(trial)
                last = validations[-1] if validations else {}
                validation = last.get("result")
                valid = (
                    validation.get("valid") if isinstance(validation, dict) else None
                )
                validated_path = last.get("arguments", {}).get("manifest_path", "")
                output = trial.get("output") or {}
                answer = output.get("answer") if isinstance(output, dict) else output
                status, extracted = parse_answer(parser, answer, validated_path)
                assignment = metadata.get("score_assignment") or {}
                row = {
                    "level": path.parent.parent.name,
                    "agent_model": path.parent.name,
                    "task": task,
                    "trial": number,
                    "execution_id": trial["trial_id"],
                    "score": trial.get("score"),
                    "first_failed_check": first,
                    "last_validation": valid if valid is not None else "not_recorded",
                    "validated_path": validated_path,
                    "current_parser_status": status,
                    "current_parser_path": extracted,
                    "rescore_candidate": manifest_failure and valid is True,
                    "manifest_failure": manifest_failure,
                    "manual_missing_artifacts_zero": assignment.get("method")
                    == "user_requested_zero_missing_artifacts",
                    "terminal_error": trial.get("error_message") or "",
                    "evaluation_error": trial.get("evaluation_error") or "",
                    "feedback": evaluation.get("feedback") or "",
                    "report_file": report_file,
                    "trace_url": traces.get(trial["trial_id"], ""),
                }
                rows.append(row)
                if manifest_failure or row["manual_missing_artifacts_zero"]:
                    calls = trial.get("tool_calls", [])
                    details.append(
                        {
                            **row,
                            "answer": answer,
                            "failed_checks": failed,
                            "validations": validations,
                            "calls_after_last_validation": calls[last["position"] + 1 :]
                            if last
                            else [],
                            "score_assignment": assignment or None,
                        }
                    )
    candidates = [row for row in rows if row["rescore_candidate"]]
    grouped = []
    for level, agent in sorted({(r["level"], r["agent_model"]) for r in rows}):
        group = [r for r in rows if (r["level"], r["agent_model"]) == (level, agent)]
        grouped.append(
            {
                "level": level,
                "agent_model": agent,
                "trials": len(group),
                "manifest_failures": sum(r["manifest_failure"] for r in group),
                "rescore_candidates": sum(r["rescore_candidate"] for r in group),
            }
        )
    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Saved final reports; no simulation, model verification, or rescoring.",
        "limitations": [
            "Validator success describes files at validation time, not scientific correctness.",
            "Parsing success alone does not establish that the final workspace is recoverable.",
            "Candidate scores remain unchanged; full retained evidence is needed for rescoring.",
            "The current parser is checked as written, without using file existence to choose a path.",
        ],
        "reports": len(sources),
        "trials": len(rows),
        "passes": sum(row["score"] == 1 for row in rows),
        "manifest_failures": sum(row["manifest_failure"] for row in rows),
        "validated_manifest_failure_candidates": len(candidates),
        "candidate_parser_status": dict(
            collections.Counter(row["current_parser_status"] for row in candidates)
        ),
        "manual_missing_artifacts_zeros": sum(
            row["manual_missing_artifacts_zero"] for row in rows
        ),
        "retained_trials_with_manifest_path_normalization": normalization_count,
        "retained_trials_with_long_answer_rescoring": long_answer_rescore_count,
        "historical_manifest_rescore_events": len(history),
        "historical_manifest_rescore_attempts": sum(
            event.get("affected_trials", 0) for event in history
        ),
        "historical_rescore_unique_execution_ids": len(
            {
                change["trial_id"]
                for event in history
                for change in event.get("changes", [])
            }
        ),
        "historical_manifest_score_increases": sum(
            change["score"] > change["previous_score"]
            for event in history
            for change in event.get("changes", [])
        ),
        "by_run": grouped,
        "submission_resolver_sha256": parser_sha,
        "sources": sources,
    }
    return rows, candidates, details, history, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, candidates, details, history, summary = audit(ROOT / "reports/corral_md")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [("trials.csv", rows), ("rescore_candidates.csv", candidates)]:
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(values)
    for name, value in [
        ("summary.json", summary),
        ("evidence.json", details),
        ("historical_rescoring.json", history),
    ]:
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "reports",
                    "trials",
                    "passes",
                    "manifest_failures",
                    "validated_manifest_failure_candidates",
                    "candidate_parser_status",
                    "manual_missing_artifacts_zeros",
                )
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
