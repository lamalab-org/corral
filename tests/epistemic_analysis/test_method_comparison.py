"""Scientific data checks for the paired annotation plots."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def comparison():
    for module in ("matplotlib", "numpy", "lama_aesthetics", "fire"):
        pytest.importorskip(module)
    script = (
        Path(__file__).resolve().parents[2]
        / "analysis/plot_epistemic_method_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("method_comparison", script)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(script.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


def summary(comparison, method, groups):
    configurations = {}
    for config_id, traces in groups.items():
        rows = []
        for identity, first_present in traces:
            stats = {}
            for i, pattern in enumerate(comparison.PATTERNS):
                prefix = "sg" if pattern["kind"] == "productive" else "ap"
                present = first_present if i == 0 else False
                stats[f"{prefix}_global_{pattern['name']}"] = present
                # Deliberately different: the plots must use global presence.
                stats[f"{prefix}_local_{pattern['name']}"] = not present
            rows.append(
                {
                    "source": {
                        "identity": identity,
                        "content_hash": f"hash-{identity}",
                        "normalization_version": "test-v1",
                        "details": {"session_id": "resistor-l2-test"},
                    },
                    "status": "complete",
                    "eligible": True,
                    "statistics": stats,
                }
            )
        overall = {
            "n_traces": len(rows),
            "subgraph_presence_global": {},
            "antipattern_presence_global": {},
        }
        for pattern in comparison.PATTERNS:
            productive = pattern["kind"] == "productive"
            prefix = "sg" if productive else "ap"
            section = (
                "subgraph_presence_global"
                if productive
                else "antipattern_presence_global"
            )
            count = sum(
                row["statistics"][f"{prefix}_global_{pattern['name']}"] for row in rows
            )
            overall[section][pattern["name"]] = {
                "count": count,
                "fraction": count / len(rows),
            }
        configurations[config_id] = {
            "annotation_configuration": {"backend": {"annotator": method}},
            "traces": rows,
            "overall": overall,
        }
    return {
        "configurations": configurations,
        "n_selected": sum(len(rows) for rows in groups.values()),
        "analysis_version": 2,
        "pattern_version": 1,
    }


def build(comparison, tmp_path, jev, llm):
    paths = [tmp_path / "jev.json", tmp_path / "llm.json"]
    for path, data in zip(paths, (jev, llm), strict=True):
        path.write_text(json.dumps(data))
    return comparison.build_dataset(*paths)


def test_equal_trace_weight_global_metric_and_agreement(comparison, tmp_path):
    jev = summary(
        comparison,
        "jev",
        {"small": [("a", False)], "large": [("b", True), ("c", True)]},
    )
    llm = summary(comparison, "llm", {"one": [("c", True), ("b", True), ("a", True)]})
    data = build(comparison, tmp_path, jev, llm)
    comparison.validate_dataset(data)
    row = comparison.statistics(data["traces"])[0]
    assert row["jev_percent"] == pytest.approx(200 / 3)
    assert row["llm_percent"] == 100
    assert row["jev_minus_llm_pp"] == pytest.approx(-100 / 3)
    assert row["both_present"] == 2
    assert row["llm_only"] == 1
    assert row["jev_only"] == row["both_absent"] == 0
    assert row["agreement_percent"] == pytest.approx(200 / 3)
    assert comparison.statistics(data["traces"])[1]["both_absent"] == 3
    assert comparison.scope_rows(data)[1][:3] == ("Resistor L2", "resistor", 2)
    # The exported snapshot remains independently usable after a JSON round trip.
    comparison.validate_dataset(json.loads(json.dumps(data)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_hash", "changed"),
        ("normalization_version", "changed"),
        ("session_id", "resistor-l1-test"),
    ],
)
def test_reject_different_source_snapshots(comparison, tmp_path, field, value):
    jev = summary(comparison, "jev", {"one": [("a", True)]})
    llm = summary(comparison, "llm", {"one": [("a", True)]})
    source = llm["configurations"]["one"]["traces"][0]["source"]
    target = source["details"] if field == "session_id" else source
    target[field] = value
    with pytest.raises(ValueError, match="Source snapshot mismatch"):
        build(comparison, tmp_path, jev, llm)


@pytest.mark.parametrize("duplicate", [False, True])
def test_reject_missing_or_duplicate_traces(comparison, tmp_path, duplicate):
    jev = summary(
        comparison, "jev", {"one": [("a", True), ("a" if duplicate else "b", False)]}
    )
    llm = summary(comparison, "llm", {"one": [("a", True)]})
    with pytest.raises(
        ValueError, match="Duplicate trace" if duplicate else "Trace sets differ"
    ):
        build(comparison, tmp_path, jev, llm)


@pytest.mark.parametrize("value", [None, "false", 0])
def test_missing_or_nonboolean_presence_is_not_zero(comparison, tmp_path, value):
    jev = summary(comparison, "jev", {"one": [("a", True)]})
    llm = summary(comparison, "llm", {"one": [("a", True)]})
    stats = llm["configurations"]["one"]["traces"][0]["statistics"]
    stats[f"sg_global_{comparison.PATTERNS[0]['name']}"] = value
    with pytest.raises(ValueError, match="Missing/invalid global presence"):
        build(comparison, tmp_path, jev, llm)


@pytest.mark.parametrize("problem", ["failed", "versions", "aggregate"])
def test_reject_invalid_annotation_inputs(comparison, tmp_path, problem):
    jev = summary(comparison, "jev", {"one": [("a", True)]})
    llm = summary(comparison, "llm", {"one": [("a", True)]})
    if problem == "failed":
        llm["configurations"]["one"]["traces"][0]["eligible"] = False
        message = "Ineligible trace"
    elif problem == "versions":
        llm["analysis_version"] = 999
        message = "versions differ"
    else:
        llm["configurations"]["one"]["overall"]["subgraph_presence_global"][
            comparison.PATTERNS[0]["name"]
        ]["fraction"] = 0
        message = "Inconsistent saved aggregate"
    with pytest.raises(ValueError, match=message):
        build(comparison, tmp_path, jev, llm)


def test_compact_export_checks_configuration_counts(comparison, tmp_path):
    data = build(
        comparison,
        tmp_path,
        summary(comparison, "jev", {"one": [("a", True)]}),
        summary(comparison, "llm", {"one": [("a", False)]}),
    )
    data["methods"]["jev"]["configurations"]["one"]["n_traces"] = 2
    with pytest.raises(ValueError, match="Configuration counts differ"):
        comparison.validate_dataset(data)


def test_codex_uses_same_traces_and_exports_all_pairwise_outcomes(comparison, tmp_path):
    data = build(
        comparison,
        tmp_path,
        summary(comparison, "jev", {"one": [("a", True), ("b", False)]}),
        summary(comparison, "llm", {"one": [("a", False), ("b", True)]}),
    )
    path = tmp_path / "codex.json"
    path.write_text(
        json.dumps(summary(comparison, "codex", {"full": [("b", True), ("a", True)]}))
    )
    comparison.add_method(data, path, "codex")
    comparison.validate_dataset(json.loads(json.dumps(data)))
    row = comparison.statistics(data["traces"])[0]
    reordered = json.loads(json.dumps(data, sort_keys=True))
    assert comparison.statistics(reordered["traces"])[0] == row
    assert comparison.method_order(reordered["methods"]) == ["jev", "llm", "codex"]
    assert row["codex_count"] == 2
    assert row["codex_percent"] == 100
    assert row["jev_minus_codex_pp"] == row["llm_minus_codex_pp"] == -50
    assert row["jev_vs_codex_both_present"] == row["llm_vs_codex_both_present"] == 1
    assert row["jev_vs_codex_second_only"] == 1
    assert row["jev_vs_codex_agreement_percent"] == 50
    for first, second in (("jev", "llm"), ("jev", "codex"), ("llm", "codex")):
        assert (
            sum(
                row[f"{first}_vs_{second}_{outcome}"]
                for outcome in (
                    "both_present",
                    "first_only",
                    "second_only",
                    "both_absent",
                )
            )
            == 2
        )


@pytest.mark.parametrize("problem", ["missing", "snapshot", "version"])
def test_codex_cannot_join_unmatched_comparison(comparison, tmp_path, problem):
    data = build(
        comparison,
        tmp_path,
        summary(comparison, "jev", {"one": [("a", True), ("b", False)]}),
        summary(comparison, "llm", {"one": [("a", False), ("b", True)]}),
    )
    codex = summary(
        comparison,
        "codex",
        {"one": [("a", True)] if problem == "missing" else [("a", True), ("b", True)]},
    )
    if problem == "snapshot":
        codex["configurations"]["one"]["traces"][0]["source"]["content_hash"] = "other"
    if problem == "version":
        codex["pattern_version"] = "other"
    path = tmp_path / "codex.json"
    path.write_text(json.dumps(codex))
    with pytest.raises(
        ValueError, match="Trace sets differ|Source snapshot mismatch|versions differ"
    ):
        comparison.add_method(data, path, "codex")
    assert "codex" not in data["methods"]
    assert all("codex" not in trace["annotations"] for trace in data["traces"])
