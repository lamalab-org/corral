"""Whole-trace labels must not inherit graph windowing or lossy context."""

import asyncio
import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError

import pytest

from corral.epistemic_analysis import (
    AnnotatedGraph,
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    aggregate,
    analyze_graph,
    annotate_trace,
    read_graph,
)
from corral.epistemic_analysis.aggregation import compute_aggregate_stats
from corral.epistemic_analysis.annotators import jev, llm
from corral.epistemic_analysis.budget import RequestBudgetExceeded
from corral.epistemic_analysis.context import current_context
from corral.epistemic_analysis.pipeline import can_reuse
from corral.epistemic_analysis.trace_checks import (
    DEFINITIONS_DOCUMENT,
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


def trace():
    messages = [
        {"role": "user", "content": "Use the validation tool before submitting."},
        {"role": "assistant", "content": "<action>validate</action>"},
        {"role": "tool", "content": "head " * 1500 + "HIDDEN_EVIDENCE" + " tail" * 500},
        {"role": "assistant", "content": "<action>submit_answer</action>"},
    ]
    # Exceeds the Decisions edge window limit; there is no such message-count
    # limit for trace checks because messages are not offered as choice values.
    messages.extend(
        {"role": "tool", "content": f"Recorded result {i}"} for i in range(260)
    )
    messages.append({"role": "assistant", "content": "POST_SUBMISSION_CLAIM"})
    return NormalizedTrace(messages, SourceReference("full", "full.json", "hash"))


def provider_backend(kind, monkeypatch, events, *, response=None):
    def check_prompt(prompt, name):
        assert prompt.startswith(
            f"Assess the behavioral check `{name}` across the entire supplied trace."
        )
        assert f"Definition of `{name}`:\n{TRACE_CHECK_DEFINITIONS[name]}" in prompt
        assert "check.name" not in prompt
        assert "check.definition" not in prompt
        assert "check.allowed_statuses" not in prompt

    if kind == "llm":

        async def call(**kwargs):
            state = json.loads(kwargs["user"])
            events.append(("checks", state))
            check_prompt(kwargs["system"], state["check"]["name"])
            assert current_context().view.effective_cap == 0
            return {
                "trace_check": copy.deepcopy(
                    next(
                        check
                        for check in (
                            response if response is not None else assessments()
                        )
                        if check["name"] == state["check"]["name"]
                    )
                )
            }

        monkeypatch.setattr(llm, "llm_json_call_async", call)
        backend = llm.LLMAnnotator().backend()
    else:
        monkeypatch.setattr(jev, "estimate_request", lambda body: (1000, "fixture"))
        client = jev.JevClient("fixture", batch_size=32)

        def post(body):
            state = json.loads(body["input"])
            events.append(("checks", state))
            assert [q["name"] for q in body["questions"]] == [state["check"]["name"]]
            check_prompt(body["questions"][0]["instructions"], state["check"]["name"])
            assert [
                choice["value"] for choice in body["questions"][0]["choices"]
            ] == state["check"]["allowed_statuses"]
            return {
                "model": "resolved-check-model",
                "answers": [
                    {
                        "name": q["name"],
                        "type": "choice",
                        "choice": "absent",
                        "confidence": 0.9,
                        "probabilities": [
                            {
                                "value": c["value"],
                                "probability": float(c["value"] == "absent"),
                            }
                            for c in q["choices"]
                        ],
                    }
                    for q in body["questions"]
                ],
            }

        monkeypatch.setattr(client, "_post", post)
        backend = jev.JevAnnotator(client).backend()

    async def nodes(messages, **kwargs):
        events.append(("nodes", messages))
        assert len(messages) == 4
        assert "HIDDEN_EVIDENCE" not in messages[2]["content"]
        return [
            {
                "node_id": "F1",
                "type": "F",
                "time": 3,
                "text": "Submission",
                "support": [{"msg_idx": 3, "quote": messages[3]["content"]}],
            }
        ], []

    async def edges(nodes, **kwargs):
        events.append(("edges", nodes))
        return [], []

    return replace(backend, extract_nodes=nodes, extract_edges=edges)


def test_packaged_definitions_match_requested_document():
    root = Path(__file__).resolve().parents[2]
    assert (root / "additional_checks.md").read_text() == DEFINITIONS_DOCUMENT
    assert len(TRACE_CHECK_NAMES) == 11


@pytest.mark.parametrize("kind", ["llm", "jev"])
@pytest.mark.parametrize(("window", "overlap"), [(2, 1), (4, 0)])
def test_one_request_per_check_after_graph_with_every_unclipped_message(
    monkeypatch, kind, window, overlap
):
    source, events = trace(), []
    backend = provider_backend(kind, monkeypatch, events)
    config = AnnotationConfig(
        "fixture", window=window, overlap=overlap, tool_output_max_chars=32
    )
    graph = asyncio.run(annotate_trace(source, backend, config))
    assert [stage for stage, _ in events] == ["nodes", "edges", *["checks"] * 11]
    assert [state["check"]["name"] for _, state in events[2:]] == list(
        TRACE_CHECK_NAMES
    )
    for _, state in events[2:]:
        name = state["check"]["name"]
        assert state["check"] == {
            "name": name,
            "definition": TRACE_CHECK_DEFINITIONS[name],
            "allowed_statuses": ["present", "absent", "uncertain"]
            + (["not_applicable"] if name == "insufficient_validation" else []),
        }
        assert state["scope"] == "full_trace"
        assert "window" not in state
        assert "definitions" not in state
        assert "trace_checks" not in state
        assert len(state["messages"]) == len(source.messages)
        assert state["messages"]["m2"]["content"] == source.messages[2]["content"]
        assert (
            state["messages"][f"m{len(source.messages) - 1}"]["content"]
            == "POST_SUBMISSION_CLAIM"
        )
        assert state["nodes"] == graph.nodes
        assert state["edges"] == graph.edges
    assert graph.annotation_status == graph.trace_checks_status == "complete"
    assert len(graph.trace_checks) == 11
    assert can_reuse(graph, source, backend, config)
    assert read_graph(json.loads(json.dumps(graph.to_dict()))) == graph
    assert not can_reuse(replace(graph, trace_checks=[]), source, backend, config)
    if kind == "jev":
        assert len(graph.annotation_requests) == 11
        assert all(
            request["stage"] == "trace_checks" for request in graph.annotation_requests
        )
        assert "resolved-check-model" in graph.provenance["resolved_models"]
    assert current_context() is None


@pytest.mark.parametrize(
    "response",
    [
        {"trace_checks": assessments()},
        {"trace_check": assessments()[1]},
    ],
)
def test_llm_rejects_combined_results_or_a_different_check(response):
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return response

    with pytest.raises((ValueError, llm.LLMError)):
        asyncio.run(llm._extract_trace_checks(trace(), [], [], "fixture", call=call))
    assert len(calls) == 1


def test_llm_evidence_can_cite_clipped_middle_and_post_submission(monkeypatch):
    source, events, results = trace(), [], assessments()
    results[0].update(
        status="present",
        support=[
            {"msg_idx": 2, "quote": "HIDDEN_EVIDENCE"},
            {"msg_idx": len(source.messages) - 1, "quote": "POST_SUBMISSION_CLAIM"},
        ],
    )
    backend = provider_backend("llm", monkeypatch, events, response=results)
    graph = asyncio.run(
        annotate_trace(
            source, backend, AnnotationConfig("fixture", tool_output_max_chars=32)
        )
    )
    assert graph.trace_checks_status == "complete"
    for support in graph.trace_checks[0]["support"]:
        assert (
            source.messages[support["msg_idx"]]["content"][
                support["start"] : support["end"]
            ]
            == support["quote"]
        )


@pytest.mark.parametrize("failure", ["budget", "invalid", "transport"])
def test_failed_pass_preserves_graph_and_never_restarts_with_less_context(
    monkeypatch, failure
):
    source, events = trace(), []
    backend = provider_backend("llm", monkeypatch, events)

    async def fail(**kwargs):
        events.append(("checks", kwargs["trace"].messages))
        if failure == "budget":
            raise RequestBudgetExceeded({"backend": "fixture"})
        if failure == "invalid":
            return assessments()[:-1]
        raise RuntimeError("provider unavailable")

    backend = replace(backend, extract_trace_checks=fail)
    config = AnnotationConfig("fixture", tool_output_max_chars=6000)
    graph = asyncio.run(annotate_trace(source, backend, config))
    assert [stage for stage, _ in events] == ["nodes", "edges", "checks"]
    assert events[-1][1] == source.messages
    assert graph.nodes
    assert graph.annotation_status == "incomplete"
    assert graph.trace_checks == []
    assert graph.trace_checks_status == "failed"
    assert graph.qc["trace_checks_error"]
    assert not can_reuse(graph, source, backend, config)
    assert current_context() is None


@pytest.mark.parametrize("provider_rejection", [False, True])
def test_decisions_single_shot_never_splits_oversized_questions(
    monkeypatch, provider_rejection
):
    calls = []
    monkeypatch.setattr(jev, "estimate_request", lambda body: (1000, "fixture"))
    client = jev.JevClient(
        "fixture", batch_size=1, max_request_tokens=0 if provider_rejection else 1
    )

    def post(body):
        calls.append(body)
        raise HTTPError(
            "https://example.test",
            400,
            "too big",
            {},
            io.BytesIO(b'{"error":"context_length_exceeded"}'),
        )

    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(RequestBudgetExceeded):
        asyncio.run(
            jev.JevAnnotator(client).extract_trace_checks(trace(), [], [], "fixture")
        )
    assert len(calls) == int(provider_rejection)


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
        validate_trace_checks(results, trace().messages)


def test_old_graphs_are_unassessed_and_unknown_labels_do_not_become_negative():
    legacy = read_graph({"schema_version": 1, "nodes": [], "edges": []})
    assert legacy.trace_checks_status == "unspecified"
    assert all(
        analyze_graph(legacy).row[f"trace_check_{name}"] is None
        for name in TRACE_CHECK_NAMES
    )
    rows = [
        {
            "trace_check_status_insufficient_validation": status,
            "trace_check_insufficient_validation": status == "present"
            if status in {"present", "absent"}
            else None,
        }
        for status in ("present", "absent", "uncertain", "not_applicable", "unassessed")
    ]
    result = compute_aggregate_stats(rows)["trace_checks"]["insufficient_validation"]
    assert result["count"] == 1
    assert result["n_assessed"] == 2
    assert result["fraction"] == 0.5
    assert result["status_counts"] == dict.fromkeys(
        ("present", "absent", "uncertain", "not_applicable", "unassessed"), 1
    )
    with pytest.raises(ValueError):
        read_graph(
            {
                "schema_version": 1,
                "nodes": [],
                "edges": [],
                "trace_checks_status": "complete",
            }
        )


def test_trace_checks_are_reported_even_without_graph_nodes():
    graph = AnnotatedGraph(
        [], [], trace_checks=assessments(), trace_checks_status="complete"
    )
    result = aggregate([analyze_graph(graph)])
    stats = next(iter(result.configurations.values()))["overall"]
    assert stats["n_traces"] == 0
    assert stats["trace_checks"]["cheating_attempt"]["n_assessed"] == 1


@pytest.mark.parametrize("missing", ["support", "explanation"])
def test_llm_positive_results_require_evidence_and_explanation(monkeypatch, missing):
    results = assessments()
    results[0]["status"] = "present"
    results[0]["support"] = [{"msg_idx": 0, "quote": "Use the validation tool"}]
    results[0].pop(missing)
    backend = provider_backend("llm", monkeypatch, [], response=results)
    graph = asyncio.run(
        annotate_trace(
            trace(), backend, AnnotationConfig("fixture", tool_output_max_chars=32)
        )
    )
    assert graph.trace_checks_status == "failed"
    assert graph.annotation_status == "incomplete"
