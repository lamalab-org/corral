"""Shared behavioral definitions and validation of saved check labels."""

import pytest

from corral.epistemic_analysis import read_graph
from corral.epistemic_analysis.epistemic_prompts import TRACE_CHECK_EXAMPLES
from corral.epistemic_analysis.trace_checks import (
    TRACE_CHECK_DEFINITIONS,
    TRACE_CHECK_NAMES,
    validate_trace_checks,
)


def assessments(status="absent"):
    return [
        {
            "name": name,
            "status": status,
            "support": [],
            "explanation": "Fixture assessment.",
        }
        for name in TRACE_CHECK_NAMES
    ]


def test_every_trace_check_has_a_definition_and_curated_examples():
    assert len(TRACE_CHECK_NAMES) == 13
    assert "submission_with_concerns" in TRACE_CHECK_NAMES
    assert "evaluation_shortcut" in TRACE_CHECK_NAMES
    assert set(TRACE_CHECK_EXAMPLES) == set(TRACE_CHECK_DEFINITIONS)
    assert all(definition.strip() for definition in TRACE_CHECK_DEFINITIONS.values())
    assert all(
        examples.count("Verdict:") >= 3 for examples in TRACE_CHECK_EXAMPLES.values()
    )


@pytest.mark.parametrize(
    "invalid",
    ["missing", "duplicate", "unknown", "bad_quote", "bad_index", "bad_status"],
)
def test_invalid_labels_and_evidence_are_rejected(invalid):
    results = assessments()
    if invalid == "missing":
        results.pop()
    elif invalid == "duplicate":
        results[-1] = results[0]
    elif invalid == "unknown":
        results[0]["name"] = "made_up"
    elif invalid == "bad_quote":
        results[0]["support"] = [{"msg_idx": 0, "quote": "invented"}]
    elif invalid == "bad_index":
        results[0]["support"] = [{"msg_idx": True, "quote": "invented"}]
    else:
        results[0]["status"] = "not_applicable"
    with pytest.raises(ValueError):
        validate_trace_checks(
            results, [{"role": "user", "content": "Use the validation tool"}]
        )


def test_old_graphs_are_unassessed_and_complete_checks_require_labels():
    legacy = read_graph({"schema_version": 1, "nodes": [], "edges": []})
    assert legacy.trace_checks_status == "unspecified"
    assert legacy.trace_checks == []
    with pytest.raises(ValueError):
        read_graph(
            {
                "schema_version": 1,
                "nodes": [],
                "edges": [],
                "trace_checks_status": "complete",
            }
        )


def test_pre_shortcut_graphs_remain_readable_without_inventing_new_labels():
    checks = [c for c in assessments() if c["name"] != "evaluation_shortcut"]
    document = {
        "schema_version": 1,
        "nodes": [],
        "edges": [],
        "trace_checks_status": "complete",
        "trace_checks": checks,
    }
    assert read_graph(document).trace_checks == checks
    # New assessments still require the full current catalog.
    with pytest.raises(ValueError):
        validate_trace_checks(checks, [])
    # Legacy compatibility must not accept arbitrary omissions or unknown names.
    with pytest.raises(ValueError):
        read_graph({**document, "trace_checks": checks[:-1]})
    with pytest.raises(ValueError):
        read_graph(
            {
                **document,
                "trace_checks": [
                    *checks[:-1],
                    {**checks[-1], "name": "evaluation_shortcut"},
                ],
            }
        )
    with pytest.raises(ValueError):
        read_graph(
            {
                **document,
                "trace_checks": [*checks[:-1], {**checks[-1], "name": "made_up"}],
            }
        )
    current = assessments()
    assert read_graph({**document, "trace_checks": current}).trace_checks == current
