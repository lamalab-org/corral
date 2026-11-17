"""Shared behavioral definitions and validation of saved check labels."""

import json
import re
from pathlib import Path

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
    assert len(TRACE_CHECK_NAMES) == 14
    assert "submission_with_concerns" in TRACE_CHECK_NAMES
    assert "evaluation_shortcut" in TRACE_CHECK_NAMES
    assert "selective_testing_or_reporting" in TRACE_CHECK_NAMES
    assert set(TRACE_CHECK_EXAMPLES) == set(TRACE_CHECK_DEFINITIONS)
    assert all(definition.strip() for definition in TRACE_CHECK_DEFINITIONS.values())
    assert all(
        examples.count("Verdict:") >= 3 for examples in TRACE_CHECK_EXAMPLES.values()
    )


def test_selective_testing_examples_keep_their_source_quotes():
    source = json.loads(
        (
            Path(__file__).parent
            / "fixtures/selective_testing_or_reporting_sources.json"
        ).read_text()
    )
    blocks = re.split(r"\n\n(?=\d+\. )", TRACE_CHECK_EXAMPLES[source["check"]])
    assert 3 <= len(blocks) <= 5
    assert len(blocks) == len(source["examples"])
    for block, example in zip(blocks, source["examples"], strict=True):
        assert example["trace_id"] not in block
        assert f"Verdict: {example['verdict']} — {example['rationale']}" in block
        assert re.findall(r'^.*?: "(.+)"$', block, re.MULTILINE) == [
            item["quote"] for item in example["evidence"] if item["in_prompt"]
        ]


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


@pytest.mark.parametrize(
    "later_checks",
    [
        {"evaluation_shortcut", "selective_testing_or_reporting"},
        {"selective_testing_or_reporting"},
    ],
    ids=["before-evaluation-shortcut", "before-selective-testing"],
)
def test_historical_graphs_remain_readable_without_inventing_new_labels(later_checks):
    checks = [c for c in assessments() if c["name"] not in later_checks]
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
        read_graph({**document, "trace_checks": checks[1:]})
    with pytest.raises(ValueError):
        read_graph(
            {
                **document,
                "trace_checks": [
                    *checks[:-1],
                    {**checks[-1], "name": "selective_testing_or_reporting"},
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
