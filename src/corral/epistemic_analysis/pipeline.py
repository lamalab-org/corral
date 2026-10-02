"""In-memory annotation stages and configuration-aware reuse.

No source loading or report writing occurs here. Both backends receive the
corrected observation nodes before extracting any edges.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from . import ontology
from .annotators.base import AnnotationBackend, reset_models
from .budget import RequestBudgetExceeded, estimator_configuration
from .context import (
    VIEW_INSTRUCTIONS,
    VIEW_VERSION,
    annotation_context,
    build_annotation_view,
    cap_sequence,
)
from .graph import (
    _filter_invalid_edges,
    _postprocess_observation_nodes,
    _postprocess_runtime_notices,
    validate_support_quotes,
)
from .schema import (
    ANNOTATION_VERSION,
    GRAPH_SCHEMA_VERSION,
    AnnotatedGraph,
    AnnotationConfig,
    NormalizedTrace,
    content_hash,
)


def annotation_configuration(
    annotator: AnnotationBackend, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        **asdict(config),
        "backend": dict(annotator.provenance),
        "ontology_hash": content_hash(
            [ontology.NODE_CRITERIA, sorted(ontology.ALLOWED_EDGE_TYPE_COMBOS)]
        ),
        "prompt_hash": content_hash(
            [
                ontology.PASS_A_SYSTEM,
                ontology.PASS_A_INSTRUCTIONS,
                ontology.PASS_B_SYSTEM,
                ontology.PASS_B_INSTRUCTIONS,
            ]
        ),
        "implementation_version": ANNOTATION_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "annotation_view_policy_version": VIEW_VERSION,
        "annotation_view_instructions_hash": content_hash(VIEW_INSTRUCTIONS),
        "request_counter": estimator_configuration(
            annotator.provenance.get("annotator", "")
        ),
    }


def annotation_fingerprint(
    trace: NormalizedTrace, annotator: AnnotationBackend, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        "source_identity": trace.source.identity,
        "source_content_hash": trace.source.content_hash,
        "normalization_version": trace.source.normalization_version,
        "messages_hash": content_hash(trace.messages),
        "message_sources_hash": content_hash(trace.message_sources),
        "configuration": annotation_configuration(annotator, config),
    }


def can_reuse(
    graph: AnnotatedGraph,
    trace: NormalizedTrace,
    annotator: AnnotationBackend,
    config: AnnotationConfig,
) -> bool:
    requested = annotation_fingerprint(trace, annotator, config)
    saved = graph.provenance.get("fingerprint")
    try:
        view = build_annotation_view(
            trace, config, graph.provenance["effective_tool_output_max_chars"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        graph.schema_version == GRAPH_SCHEMA_VERSION
        and graph.annotation_status == "complete"
        and graph.source is not None
        and graph.source.identity == trace.source.identity
        and graph.source.content_hash == trace.source.content_hash
        and content_hash(graph.messages) == requested["messages_hash"]
        and saved == requested
        and graph.annotation_view == view.to_dict()
        and graph.provenance.get("annotation_view_hash") == view.digest
        and graph.provenance.get("fingerprint_hash") == content_hash(requested)
        and graph.provenance.get("resolved_fingerprint_hash")
        == content_hash(
            {
                **requested,
                "resolved_models": graph.provenance.get("resolved_models", []),
                "effective_tool_output_max_chars": view.effective_cap,
                "annotation_view_hash": view.digest,
            }
        )
    )


def _resolved_models(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {value["model"]} if isinstance(value.get("model"), str) else set()
        return found.union(*(_resolved_models(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_resolved_models(v) for v in value))
    return set()


async def _annotate_messages(
    messages, backend: AnnotationBackend, config: AnnotationConfig, view, originals
):
    """Extract nodes, normalize observations, then extract and validate edges."""
    model, window, overlap = config.model, config.window, config.overlap
    max_nodes_per_window, strict_support = (
        config.max_nodes_per_window,
        config.strict_support,
    )
    qc_warnings: list[str] = []

    def visible_items(items):
        visible = []
        for item in items:
            try:
                support = view.map_support(item.get("support", []), originals)
            except (ValueError, TypeError, AttributeError):
                qc_warnings.append(
                    "Removed annotation with support outside a contiguous retained range."
                )
            else:
                visible.append({**item, "support": support})
        return visible

    node_pass = backend.extract_nodes
    edge_pass = backend.extract_edges
    nodes, wA = await node_pass(
        messages=messages,
        model=model,
        window=window,
        overlap=overlap,
        max_nodes_per_window=max_nodes_per_window,
    )
    qc_warnings.extend(wA)
    node_ids = set()
    for node in nodes:
        if (
            not isinstance(node, dict)
            or node.get("type") not in ontology.NODE_TYPES_SET
        ):
            raise ValueError("Annotator returned an invalid node")
        identity = node.get("node_id")
        if not isinstance(identity, str) or not identity or identity in node_ids:
            raise ValueError("Annotator returned a missing or duplicate node ID")
        node_ids.add(identity)

    if strict_support:
        nodes = [
            n
            for n in nodes
            if validate_support_quotes(messages, n.get("support", []))[0]
        ]

    nodes, wObs = _postprocess_observation_nodes(nodes, messages)
    qc_warnings.extend(wObs)
    nodes, wRuntime = _postprocess_runtime_notices(nodes, messages)
    qc_warnings.extend(wRuntime)
    limited_view = view.coverage["reduced_context"] or len(messages) < len(originals)
    if strict_support or limited_view:
        nodes = visible_items(nodes)

    edges, wB = await edge_pass(
        messages=messages,
        nodes=nodes,
        model=model,
        window=window,
        overlap=overlap,
    )
    qc_warnings.extend(wB)
    if strict_support or limited_view:
        edges = visible_items(edges)

    if strict_support:
        node_ids = {n["node_id"] for n in nodes}
        edges = [
            e
            for e in edges
            if validate_support_quotes(messages, e.get("support", []))[0]
            and e.get("src") in node_ids
            and e.get("dst") in node_ids
        ]

    edges, wEdge = _filter_invalid_edges(edges, nodes)
    qc_warnings.extend(wEdge)

    return nodes, edges, qc_warnings


async def annotate_trace(
    trace: NormalizedTrace, annotator: AnnotationBackend, config: AnnotationConfig
) -> AnnotatedGraph:
    if annotator.provenance.get("annotator") == "jev" and config.window > 252:
        raise ValueError("Jev requires window <= 252")
    if trace.reconstruction_status != "complete" or trace.completion_status in {
        "incomplete",
        "unsupported",
        "failed",
        "missing",
    }:
        raise ValueError("Cannot annotate an incomplete or unsupported trace")
    messages = trace.messages
    diagnostics, attempts, requests = [], [], []
    caps = cap_sequence(config.tool_output_max_chars)
    failed_view_hash = None
    for index, cap in enumerate(caps):
        view = build_annotation_view(trace, config, cap)
        if content_hash(view.messages) == failed_view_hash:
            continue
        reset_models()
        review_issues = []
        try:
            with annotation_context(view, config, diagnostics, requests, review_issues):
                nodes, edges, warnings = await _annotate_messages(
                    view.messages, annotator, config, view, messages
                )
        except RequestBudgetExceeded as exc:
            attempts.append(
                {"effective_cap": cap, "status": "oversized", **exc.diagnostic}
            )
            exc.attempts = attempts
            exc.annotation_view = view.to_dict()
            exc.diagnostics = diagnostics
            exc.annotation_requests = requests
            # A cap change must alter the actual view; never retry the same input.
            failed_view_hash = content_hash(view.messages)
            if not any(
                content_hash(build_annotation_view(trace, config, remaining).messages)
                != failed_view_hash
                for remaining in caps[index + 1 :]
            ):
                raise
        except Exception as exc:
            exc.annotation_requests = requests
            raise
        else:
            attempts.append({"effective_cap": cap, "status": "complete"})
            break
    uncovered = []
    if annotator.provenance.get("annotator") == "jev":
        covered = {n["time"] for n in nodes}
        uncovered = [
            i
            for i, m in enumerate(view.messages)
            if m.get("role", "").lower() == "assistant"
            and m.get("content", "").strip()
            and i not in covered
        ]
        if uncovered:
            warnings.append(
                f"Assistant messages without nodes require review: {uncovered}."
            )
    if view.coverage["reduced_context"]:
        warnings.append(
            "Annotation used reduced tool context. Omitted middle text is unknown; absence-based measurements may change."
        )
    fingerprint = annotation_fingerprint(trace, annotator, config)
    resolved = sorted(
        _resolved_models(nodes + edges) | set(annotator.resolved_models())
    )
    return AnnotatedGraph(
        nodes=nodes,
        edges=edges,
        input_file=trace.source.locator,
        source=trace.source,
        messages=messages,
        message_sources=trace.message_sources,
        grouping=trace.grouping,
        completion_status=trace.completion_status,
        reconstruction_status=trace.reconstruction_status,
        annotation_view=view.to_dict(),
        annotation_requests=requests,
        annotation_status="incomplete" if uncovered or review_issues else "complete",
        provenance={
            **asdict(config),
            **annotator.provenance,
            "fingerprint": fingerprint,
            "fingerprint_hash": content_hash(fingerprint),
            "configuration_id": content_hash(fingerprint["configuration"]),
            "resolved_models": resolved,
            "effective_tool_output_max_chars": view.effective_cap,
            "annotation_view_hash": view.digest,
            "annotation_view_artifact": f"annotation_views/{view.digest}.json",
            "resolved_fingerprint_hash": content_hash(
                {
                    **fingerprint,
                    "resolved_models": resolved,
                    "effective_tool_output_max_chars": view.effective_cap,
                    "annotation_view_hash": view.digest,
                }
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        qc={
            "warnings": warnings,
            "context_coverage": view.coverage,
            "annotation_attempts": attempts,
            "request_diagnostics": diagnostics,
            "uncovered_messages": uncovered,
            "review_issues": review_issues,
        },
    )
