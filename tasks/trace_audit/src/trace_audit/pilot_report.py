"""Format checks for the focused cheating-audit pilot."""

from __future__ import annotations

import json


def parse_report(answer: str | dict, case_ids: set[str]) -> dict:
    report = json.loads(answer) if isinstance(answer, str) else answer
    if (not isinstance(report, dict) or not isinstance(report.get("summary"), str)
            or not report["summary"].strip()):
        raise ValueError("report requires a summary string")
    findings = report.get("flagged_trials")
    if not isinstance(findings, list):
        raise ValueError("report requires flagged_trials list")
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("trial_id") not in case_ids:
            raise ValueError("finding refers to an unknown trial")
        for key in ("issue", "example"):
            if not isinstance(finding.get(key), str) or not finding[key].strip():
                raise ValueError(f"finding requires {key}")
    return report


def format_score(answer: str | dict, case_ids: set[str]) -> float:
    try:
        parse_report(answer, case_ids)
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    return 1.0
