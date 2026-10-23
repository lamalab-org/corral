"""Legacy graph measurements and current definition snapshots; no API calls."""

import hashlib
import json
import re
from pathlib import Path

from corral.epistemic_analysis import ontology, patterns, reporting
from corral.epistemic_analysis.aggregation import compute_aggregate_stats
from corral.epistemic_analysis.analysis import summarize_trace

FIXTURES = Path(__file__).parent / "fixtures"


def test_graph_measurements_and_aggregate_match_original_scripts():
    fixture = json.loads((FIXTURES / "regression.json").read_text())
    rows = [
        summarize_trace(graph, Path(f"trace-{i}.json"))
        for i, graph in enumerate(fixture["graphs"])
    ]
    # Trace checks add columns; the historical graph measurements stay exact.
    assert [
        {key: value for key, value in row.items() if not key.startswith("trace_check_")}
        for row in rows
    ] == fixture["rows"]
    assert {
        key: value
        for key, value in compute_aggregate_stats(rows).items()
        if key != "trace_checks"
    } == fixture["aggregate"]
    # These definitions intentionally differ; neither may be derived from the other.
    assert any(
        row[f"{prefix}_local_{name}"] != row[f"{prefix}_global_{name}"]
        for row in rows
        for prefix, names in (
            ("sg", patterns.SUBGRAPH_NAMES),
            ("ap", patterns.ANTIPATTERN_NAMES),
        )
        for name in names
    )
    assert rows[0]["ap_global_fixed_belief_trace"] is True
    assert rows[0]["ap_local_fixed_belief_trace"] is True


def test_current_prompt_snapshots_and_shared_jev_criteria_match():
    fixture = json.loads((FIXTURES / "regression.json").read_text())
    for name, expected in fixture["prompt_hashes"].items():
        assert hashlib.sha256(getattr(ontology, name).encode()).hexdigest() == expected
    assert (
        dict(
            re.findall(
                r"^([HTEJCNF]) = (.+)$", ontology.PASS_A_INSTRUCTIONS, re.MULTILINE
            )
        )
        == ontology.NODE_CRITERIA
    )


def test_current_pattern_definition_tables_match_snapshots():
    assert (
        reporting.build_productive_motifs_latex()
        == (FIXTURES / "productive_motifs.tex").read_text()
    )
    assert (
        reporting.build_reasoning_breakdowns_latex()
        == (FIXTURES / "reasoning_breakdowns.tex").read_text()
    )
