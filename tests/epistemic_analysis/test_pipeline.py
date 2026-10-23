import asyncio
import copy
import json
from dataclasses import replace

import pytest

from corral.epistemic_analysis import (
    AnnotatedGraph,
    AnnotationBackend,
    AnnotationConfig,
    GroupingMetadata,
    NormalizedTrace,
    SourceReference,
    aggregate,
    analyze_graph,
    annotate_trace,
    ontology,
    read_graph,
    schema,
)
from corral.epistemic_analysis.annotators import llm
from corral.epistemic_analysis.annotators.jev import JevAnnotator
from corral.epistemic_analysis.annotators.llm import LLMAnnotator
from corral.epistemic_analysis.pipeline import annotation_fingerprint, can_reuse


def trace():
    return NormalizedTrace(
        messages=[
            {"role": "assistant", "content": "<action>copy</action>"},
            {"role": "user", "content": "Observation: copied"},
            {"role": "assistant", "content": "<action>measure</action>"},
            {"role": "user", "content": "Observation: 4"},
        ],
        source=SourceReference("trace-one", "original.json", "source-hash"),
        grouping=GroupingMetadata(agent_model="agent-model"),
    )


def node(identity, kind, index, messages):
    return {
        "node_id": identity,
        "type": kind,
        "time": index,
        "text": messages[index]["content"],
        "support": [{"msg_idx": index, "quote": messages[index]["content"]}],
    }


@pytest.mark.parametrize("kind", ["llm", "jev"])
def test_corrected_and_added_observation_nodes_reach_both_backends(monkeypatch, kind):
    source = trace()
    checks = []
    if kind == "llm":

        async def respond(model, system, user, **kwargs):
            if system == llm.PASS_A_SYSTEM:
                return {
                    "nodes": [
                        node("ignored", "N", 0, source.messages),
                        node("ignored2", "T", 2, source.messages),
                    ]
                }
            nodes = json.loads(user.split("\n\nNodes:\n")[1])
            checks.append({n["time"]: n["type"] for n in nodes})
            return {"edges": []}

        monkeypatch.setattr(llm, "llm_json_call_async", respond)
        backend = LLMAnnotator().backend()
    else:

        class Decisions:
            async def evaluate(self, model, state, questions):
                if "nodes" in state:
                    checks.append(
                        {n["time"]: n["type"] for n in state["nodes"].values()}
                    )
                answers = {}
                for key in questions:
                    if key.startswith("edge_"):
                        answers[key] = {"choice": "unrelated", "confidence": 1.0}
                    else:
                        span_key, node_type = key.rsplit("_", 1)
                        span = state["candidates"][span_key]
                        expected = (
                            "E"
                            if span["kind"] == "observation"
                            else ("N" if span["msg_idx"] == 0 else "T")
                        )
                        choice = "present" if node_type == expected else "absent"
                        answers[key] = {"choice": choice, "confidence": 1.0}
                return answers

        backend = JevAnnotator(Decisions()).backend()
    backend = replace(
        backend, extract_trace_checks=None
    )  # Graph postprocessing fixture.
    graph = asyncio.run(
        annotate_trace(
            source, backend, AnnotationConfig("annotator-model", window=20, overlap=5)
        )
    )
    assert checks
    assert all(check == {0: "N", 1: "E", 2: "T", 3: "E"} for check in checks)
    assert len({n["node_id"] for n in graph.nodes}) == 4
    assert graph.grouping.agent_model == "agent-model"
    assert graph.provenance["model"] == "annotator-model"
    assert (
        read_graph(json.loads(json.dumps(graph.to_dict()))).to_dict() == graph.to_dict()
    )


def fake_backend():
    async def nodes(messages, **kwargs):
        return [node("N1", "T", 2, messages)], []

    async def edges(nodes, **kwargs):
        return [{"src": "N1", "dst": "absent", "relation": "tests", "support": []}], []

    return AnnotationBackend(nodes, edges, {"annotator": "fake", "threshold": 0.5})


def test_fingerprint_matches_and_invalidates_all_annotation_dependencies(monkeypatch):
    source, backend, config = trace(), fake_backend(), AnnotationConfig("model")
    graph = asyncio.run(annotate_trace(source, backend, config))
    assert graph.edges == []
    assert can_reuse(graph, source, backend, config)
    for changed in [
        replace(config, model="other"),
        replace(config, window=30),
        replace(config, overlap=0),
        replace(config, max_nodes_per_window=10),
        replace(config, strict_support=False),
    ]:
        assert not can_reuse(graph, source, backend, changed)
    for changed in [
        replace(source.source, content_hash="new"),
        replace(source.source, normalization_version="new"),
    ]:
        assert not can_reuse(graph, replace(source, source=changed), backend, config)
    assert not can_reuse(
        graph,
        source,
        replace(backend, provenance={"annotator": "fake", "threshold": 0.6}),
        config,
    )
    monkeypatch.setattr(ontology, "PASS_A_SYSTEM", ontology.PASS_A_SYSTEM + "new")
    assert not can_reuse(graph, source, backend, config)
    assert not can_reuse(replace(graph, provenance={}), source, backend, config)


def test_analysis_versions_do_not_invalidate_annotations(monkeypatch):
    source, backend, config = trace(), fake_backend(), AnnotationConfig("model")
    before = annotation_fingerprint(source, backend, config)
    monkeypatch.setattr(schema, "PATTERN_VERSION", "future")
    assert annotation_fingerprint(source, backend, config) == before


def test_status_denominators_configuration_partitioning_and_snapshot_selection():
    source, backend = trace(), fake_backend()
    graph = asyncio.run(annotate_trace(source, backend, AnnotationConfig("model")))
    empty = AnnotatedGraph(
        [],
        [],
        source=replace(source.source, identity="empty"),
        provenance=copy.deepcopy(graph.provenance),
    )
    failed = replace(
        empty,
        annotation_status="failed",
        source=replace(source.source, identity="failed"),
    )
    incomplete = replace(
        graph,
        completion_status="incomplete",
        source=replace(source.source, identity="incomplete"),
    )
    results = [analyze_graph(g) for g in (graph, empty, failed, incomplete)]
    assert results[1].row["ap_global_fixed_belief_trace"] is True
    summary = aggregate(results)
    stats = next(iter(summary.configurations.values()))["overall"]
    assert stats["n_traces"] == 1
    assert stats["n_excluded"] == 3
    assert stats["status_counts"] == {
        "complete": 1,
        "uninformative": 1,
        "failed": 1,
        "incomplete": 1,
    }
    other = asyncio.run(
        annotate_trace(source, backend, AnnotationConfig("another-model"))
    )
    assert (
        len(aggregate([analyze_graph(graph), analyze_graph(other)]).configurations) == 2
    )
    changed = replace(graph, source=replace(graph.source, content_hash="changed"))
    with pytest.raises(ValueError, match="Multiple snapshots"):
        aggregate([analyze_graph(graph), analyze_graph(changed)])
    assert aggregate([analyze_graph(graph)] * 2).n_selected == 1


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"nodes": [], "edges": []},
        {"nodes": [], "edges": [], "schema_version": 0},
        {"nodes": [], "edges": [], "schema_version": 999},
        {"nodes": [], "edges": [{"src": "N1", "dst": "N2"}]},
    ],
)
def test_invalid_saved_graphs_are_rejected(document):
    with pytest.raises(ValueError):
        read_graph(document)


@pytest.mark.parametrize("kind", ["jev", "llm"])
def test_coverage_allows_abstention_and_uses_only_the_annotation_view(kind):
    messages = [
        {"role": "user", "content": "Measure the sample."},
        {"role": "assistant", "content": "Let me think about the next step."},
        {"role": "assistant", "content": "   "},
        {
            "role": "assistant",
            "content": "<action>submit_answer</action><action_input>X</action_input>",
        },
        {"role": "assistant", "content": "Post-submission duplicate summary."},
    ]

    async def nodes(messages, **kwargs):
        assert len(messages) == 4
        return [node("N1", "F", 3, messages)], []

    async def edges(**kwargs):
        return [], []

    backend = AnnotationBackend(nodes, edges, {"annotator": kind})
    source = NormalizedTrace(
        messages, SourceReference("coverage", "coverage.json", "hash")
    )
    graph = asyncio.run(annotate_trace(source, backend, AnnotationConfig("fixture")))
    assert graph.annotation_status == "complete"
    assert graph.qc["uncovered_messages"] == [1]
    assert graph.qc["message_coverage"]["assistant_messages"] == 2
    assert graph.qc["message_coverage"]["labelled_assistant_messages"] == 1
    assert len(graph.messages) == 5  # Source evidence is retained in full.
