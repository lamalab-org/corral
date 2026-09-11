from __future__ import annotations

import json

from stargazer.audit import (
    DEFAULT_REPORT_PATH,
    audit_task_bank,
    selected_task_ids,
    validate_official_banks,
)


def test_committed_audit_is_deterministic_and_official_references_pass():
    report = audit_task_bank()

    assert report == json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    validate_official_banks(report)

    score_by_id = {record["task_id"]: record["score"] for record in report["records"]}
    selected = selected_task_ids()
    assert {level: len(task_ids) for level, task_ids in selected.items()} == {
        1: 20,
        2: 20,
        3: 20,
    }
    assert all(
        score_by_id[task_id] == 1.0
        for task_ids in selected.values()
        for task_id in task_ids
    )


def test_audit_records_all_reference_failure_categories():
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    reasons = report["summary"]["failure_reason_counts"]

    assert set(reasons) <= {
        "invalid_reference_parameters",
        "bic_gate",
        "rms_gate",
        "physical_match_gate",
        "count_gate",
    }
    assert reasons["invalid_reference_parameters"] == 2
    assert reasons["bic_gate"] > 0
    assert reasons["rms_gate"] > 0
