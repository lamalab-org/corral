"""Analysis for epistemic trace analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from corral.epistemic_analysis.patterns import (
    ANTIPATTERN_FAMILIES,
    ANTIPATTERN_NAMES,
    SUBGRAPH_FAMILIES,
    SUBGRAPH_NAMES,
    detect_antipatterns_global,
    detect_antipatterns_local,
    detect_subgraphs_global,
    detect_subgraphs_local,
)

if TYPE_CHECKING:
    from .schema import AnalysisResult, AnnotatedGraph


def summarize_trace(doc: dict[str, Any], file_path: Path) -> dict[str, Any]:
    nodes = doc.get("nodes", []) or []
    edges = doc.get("edges", []) or []

    row: dict[str, Any] = {
        "file": file_path.name,
        "input_file": doc.get("input_file", ""),
        "model": doc.get("provenance", {}).get("model", ""),
        "nodes_total": len(nodes),
        "edges_total": len(edges),
    }

    sg_local = detect_subgraphs_local(nodes, edges)
    sg_global = detect_subgraphs_global(nodes, edges)
    for sg_name in SUBGRAPH_NAMES:
        row[f"sg_local_{sg_name}"] = sg_local.get(sg_name, 0) > 0
        row[f"sg_local_count_{sg_name}"] = sg_local.get(sg_name, 0)
        row[f"sg_global_{sg_name}"] = sg_global.get(sg_name, 0) > 0
        row[f"sg_global_count_{sg_name}"] = sg_global.get(sg_name, 0)

    ap_local = detect_antipatterns_local(nodes, edges)
    ap_global_binary, ap_global_counts = detect_antipatterns_global(nodes, edges)
    for ap_name in ANTIPATTERN_NAMES:
        row[f"ap_local_{ap_name}"] = ap_local.get(ap_name, 0) > 0
        row[f"ap_local_count_{ap_name}"] = ap_local.get(ap_name, 0)
        row[f"ap_global_{ap_name}"] = ap_global_binary.get(ap_name, False)
        row[f"ap_global_count_{ap_name}"] = ap_global_counts.get(ap_name, 0)

    for fam_name, members in ANTIPATTERN_FAMILIES.items():
        row[f"ap_family_local_{fam_name}"] = any(
            ap_local.get(m, 0) > 0 for m in members
        )
        row[f"ap_family_local_count_{fam_name}"] = sum(
            ap_local.get(m, 0) for m in members
        )
        row[f"ap_family_global_{fam_name}"] = any(
            ap_global_binary.get(m, False) for m in members
        )
        row[f"ap_family_global_count_{fam_name}"] = sum(
            ap_global_counts.get(m, 0) for m in members
        )

    for fam_name, members in SUBGRAPH_FAMILIES.items():
        row[f"sg_family_local_{fam_name}"] = any(
            sg_local.get(m, 0) > 0 for m in members
        )
        row[f"sg_family_local_count_{fam_name}"] = sum(
            sg_local.get(m, 0) for m in members
        )
        row[f"sg_family_global_{fam_name}"] = any(
            sg_global.get(m, 0) > 0 for m in members
        )
        row[f"sg_family_global_count_{fam_name}"] = sum(
            sg_global.get(m, 0) for m in members
        )

    return row


def analyze_graph(graph: AnnotatedGraph | dict[str, Any]) -> AnalysisResult:
    """Compute the historical measurements, without I/O or provider calls.

    Empty graphs retain their historical fixed_belief_trace measurement. They
    are marked uninformative, allowing aggregation to expose their exclusion.
    """
    from .schema import AnalysisResult, AnnotatedGraph, content_hash, read_graph

    if not isinstance(graph, AnnotatedGraph):
        graph = read_graph(graph)
    status = graph.annotation_status
    if status == "complete":
        if graph.reconstruction_status != "complete":
            status = graph.reconstruction_status
        elif graph.completion_status not in {"complete", "unspecified"}:
            status = graph.completion_status
        elif not graph.nodes:
            status = "uninformative"
    configuration = {
        **graph.provenance.get("fingerprint", {}).get("configuration", {}),
        "resolved_models": graph.provenance.get("resolved_models", []),
        "effective_tool_output_max_chars": graph.provenance.get(
            "effective_tool_output_max_chars", 0
        ),
    }
    row = summarize_trace(
        graph.to_dict(),
        Path(
            (graph.source.resolved_locator or graph.source.identity)
            if graph.source
            else graph.input_file
        ),
    )
    coverage = graph.qc.get("context_coverage", {})
    row.update(
        {
            "clipped_outputs": coverage.get("clipped_outputs", 0),
            "original_tool_chars": coverage.get("original_tool_chars"),
            "retained_tool_chars": coverage.get("retained_tool_chars"),
            "omitted_tool_chars": coverage.get("omitted_tool_chars", 0),
            "effective_tool_output_max_chars": coverage.get(
                "effective_tool_output_max_chars", 0
            ),
            "reduced_context": coverage.get("reduced_context", False),
        }
    )

    return AnalysisResult(
        row=row,
        source=graph.source,
        grouping=graph.grouping,
        configuration_id=content_hash(configuration),
        annotation_configuration=configuration,
        status=status,
        eligible=status == "complete",
    )
