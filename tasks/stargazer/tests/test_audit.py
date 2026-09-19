from __future__ import annotations

import json

import pytest
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
        1: 10,
        2: 10,
    }
    assert set(score_by_id) == set().union(*selected.values())
    assert report["summary"]["by_source"] == {"synthetic": {"passing": 20, "total": 20}}
    assert all(
        score_by_id[task_id] == 1.0
        for task_ids in selected.values()
        for task_id in task_ids
    )


def test_audit_reports_no_reference_failures():
    report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
    reasons = report["summary"]["failure_reason_counts"]

    assert set(reasons) == {
        "invalid_reference_parameters",
        "bic_gate",
        "rms_gate",
        "physical_match_gate",
        "count_gate",
    }
    assert all(count == 0 for count in reasons.values())


@pytest.fixture
def official_bank(tmp_path):
    selectors = {}
    records = []
    for level, (minimum, maximum) in {1: (5, 7), 2: (8, 10)}.items():
        task_ids = [f"level_{level}_task_{index}" for index in range(10)]
        selector = {
            "source": "synthetic",
            "difficulty_min": minimum,
            "difficulty_max": maximum,
            "task_ids": task_ids,
        }
        selector_file = tmp_path / f"level_{level}" / "tasks_json" / "tasks.json"
        selector_file.parent.mkdir(parents=True)
        selector_file.write_text(json.dumps([selector]), encoding="utf-8")
        selectors[level] = (selector_file, selector)
        records.extend(
            {
                "task_id": task_id,
                "source": "synthetic",
                "difficulty": minimum + index % (maximum - minimum + 1),
                "score": 1.0,
            }
            for index, task_id in enumerate(task_ids)
        )
    return tmp_path, selectors, {"records": records}


def test_official_bank_accepts_both_synthetic_difficulty_bands(official_bank):
    root, _, report = official_bank

    validate_official_banks(report, root)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task_ids", None, "explicit list"),
        ("source", "real", "only synthetic"),
        ("difficulty_min", 4, "difficulties 5-7"),
        ("difficulty_max", 8, "difficulties 5-7"),
    ],
)
def test_official_bank_rejects_invalid_selectors(official_bank, field, value, error):
    root, selectors, report = official_bank
    selector_file, selector = selectors[1]
    selector[field] = value
    selector_file.write_text(json.dumps([selector]), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_official_banks(report, root)


@pytest.mark.parametrize("separate_selector", [False, True])
def test_official_bank_rejects_duplicate_membership(official_bank, separate_selector):
    root, selectors, report = official_bank
    selector_file, selector = selectors[1]
    if separate_selector:
        selector_file.with_name("duplicates.json").write_text(
            json.dumps({**selector, "task_ids": [selector["task_ids"][0]]}),
            encoding="utf-8",
        )
    else:
        selector["task_ids"].append(selector["task_ids"][0])
        selector_file.write_text(json.dumps([selector]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task IDs"):
        validate_official_banks(report, root)


def test_official_bank_rejects_overlap_between_levels(official_bank):
    root, selectors, report = official_bank
    selector_file, selector = selectors[2]
    selector["task_ids"][0] = selectors[1][1]["task_ids"][0]
    selector_file.write_text(json.dumps([selector]), encoding="utf-8")

    with pytest.raises(ValueError, match="Official levels overlap"):
        validate_official_banks(report, root)


def test_official_bank_requires_ten_tasks_per_level(official_bank):
    root, selectors, report = official_bank
    selector_file, selector = selectors[1]
    selector["task_ids"].pop()
    selector_file.write_text(json.dumps([selector]), encoding="utf-8")

    with pytest.raises(ValueError, match="selects 9 tasks, expected 10"):
        validate_official_banks(report, root)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task_id", "unknown_task", "missing from audit"),
        ("source", "real", "non-synthetic tasks"),
        ("difficulty", 4, "outside difficulties 5-7"),
        ("difficulty", 8, "outside difficulties 5-7"),
        ("score", 0.0, "reference-invalid tasks"),
    ],
)
def test_official_bank_rejects_invalid_records(official_bank, field, value, error):
    root, _, report = official_bank
    report["records"][0][field] = value

    with pytest.raises(ValueError, match=error):
        validate_official_banks(report, root)


def test_official_bank_rejects_duplicate_audit_ids(official_bank):
    root, _, report = official_bank
    report["records"].append(dict(report["records"][0]))

    with pytest.raises(ValueError, match="Audit contains duplicate task ID"):
        validate_official_banks(report, root)
