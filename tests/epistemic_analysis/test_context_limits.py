"""Visible evidence, bounded retries, and portable capped annotations."""

import asyncio
import copy
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from corral.cli import build_parser
from corral.epistemic_analysis import (
    AnnotationBackend,
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    analyze_graph,
    annotate_trace,
    cli,
    read_graph,
)
from corral.epistemic_analysis.annotators import jev, llm
from corral.epistemic_analysis.budget import RequestBudgetExceeded
from corral.epistemic_analysis.context import (
    annotation_context,
    build_annotation_view,
    current_context,
)
from corral.epistemic_analysis.graph import validate_support_quotes
from corral.epistemic_analysis.pipeline import can_reuse


def source():
    return NormalizedTrace(
        [
            {"role": "system", "content": "Solve the task."},
            {"role": "user", "content": "Measure X."},
            {"role": "assistant", "content": "<action>measure</action>"},
            {
                "role": "user",
                "content": "Observation: "
                + "αβ🧪" * 4000
                + "HIDDEN MIDDLE"
                + "12345" * 3000
                + " tail evidence",
            },
            {"role": "assistant", "content": "The observation supports X."},
        ],
        SourceReference("trace", "trace.json", "source-hash"),
        message_sources=[
            {"index": i, **({"tool_call_id": "call-1"} if i == 3 else {})}
            for i in range(5)
        ],
    )


def config(**kwargs):
    return AnnotationConfig(
        "jev-test",
        window=3,
        overlap=1,
        tool_output_max_chars=6000,
        max_request_tokens=24000,
        **kwargs,
    )


def test_view_is_deterministic_exact_unicode_and_preserves_short_outputs():
    trace = source()
    original = copy.deepcopy(trace.messages)
    view = build_annotation_view(trace, config())
    assert view.to_dict() == build_annotation_view(trace, config()).to_dict()
    assert trace.messages == original
    assert view.messages[:3] == original[:3]
    text = original[3]["content"]
    assert view.messages[3]["content"].startswith(text[:4500])
    assert view.messages[3]["content"].endswith(text[-1500:])
    assert "HIDDEN MIDDLE" not in view.messages[3]["content"]
    assert "original characters omitted" in view.messages[3]["content"]
    support = view.support(3)
    assert [len(s["quote"]) for s in support] == [4500, 1500]
    assert view.map_support(support, trace.messages) == support
    assert view.coverage["omitted_tool_chars"] == len(text) - 6000
    assert view.mappings[3]["tool_identity"] == {"tool_call_id": "call-1"}
    assert view.messages[-1] == original[-1]
    assert (
        build_annotation_view(
            trace, replace(config(), tool_output_max_chars=0)
        ).messages
        == original
    )


@pytest.mark.parametrize(
    ("role", "metadata"),
    [
        ("tool", {}),
        ("function", {}),
        ("user", {"original_role": "tool"}),
        ("user", {"tool_call_id": "id"}),
    ],
)
def test_tool_detection_uses_role_and_source_without_prefix(role, metadata):
    trace = NormalizedTrace(
        [{"role": role, "content": "x" * 100}],
        SourceReference("x", "x", "x"),
        [metadata],
    )
    view = build_annotation_view(
        trace, AnnotationConfig("test", tool_output_max_chars=16)
    )
    assert view.coverage["clipped_outputs"] == 1
    assert sum(len(s["quote"]) for s in view.support(0)) == 16


@pytest.mark.parametrize("action", ["submit_answer", "\n submit_answer \t"])
def test_annotation_view_ends_at_submission_inclusively(action):
    trace = source()
    trace.messages.extend(
        [
            {
                "role": "assistant",
                "content": f'<action>{action}</action><action_input>{{"answer":"X"}}</action_input>',
            },
            {"role": "tool", "content": "answer accepted"},
            {"role": "assistant", "content": "<final_answer>X</final_answer>"},
            {"role": "user", "content": "Follow-up"},
            {"role": "assistant", "content": "<action>submit_answer</action>"},
        ]
    )
    trace.message_sources.extend({"index": i} for i in range(5, 10))
    original = copy.deepcopy(trace.messages)
    for cap in (0, 16):
        view = build_annotation_view(
            trace, AnnotationConfig("test", tool_output_max_chars=cap)
        )
        assert len(view.messages) == len(view.mappings) == 6
        assert view.messages[-1] == original[5]
        assert view.support(5)[0]["quote"] == original[5]["content"]
        with pytest.raises(ValueError, match="Invalid annotation support"):
            view.map_support([{"msg_idx": 6, "quote": "answer accepted"}], original)
    assert trace.messages == original


@pytest.mark.parametrize(
    ("role", "content"),
    [
        ("system", "Use <action>submit_answer</action> to finish."),
        ("user", "Observation: <action>submit_answer</action>"),
        ("tool", "<action>submit_answer</action>"),
        ("assistant", "I will use submit_answer when done."),
        ("assistant", "<action>submit_answer_later</action>"),
    ],
)
def test_submission_mentions_do_not_end_annotation_view(role, content):
    trace = source()
    trace.messages[0] = {"role": role, "content": content}
    view = build_annotation_view(trace, config())
    assert len(view.messages) == len(trace.messages)


@pytest.mark.parametrize("kind", ["marker", "crossing", "hidden", "joined"])
def test_rejects_gap_metadata_and_invisible_evidence(kind):
    trace = source()
    view = build_annotation_view(trace, config())
    rendered = view.messages[3]["content"]
    quote = {
        "marker": "[ANNOTATION VIEW:",
        "crossing": rendered[4495:4530],
        "hidden": "HIDDEN MIDDLE",
        "joined": trace.messages[3]["content"][4495:4505],
    }[kind]
    # Use a distinctive boundary for the joined case (not repeated elsewhere).
    if kind == "joined":
        trace.messages[3]["content"] = (
            "Observation: "
            + "a" * 4486
            + "HEAD-END"
            + "MISSING" * 4000
            + "TAIL-START"
            + "z" * 1490
        )
        view = build_annotation_view(trace, config())
        quote = view.support(3)[0]["quote"][-8:] + view.support(3)[1]["quote"][:10]
    with pytest.raises(ValueError, match="retained range"):
        view.map_support([{"msg_idx": 3, "quote": quote}], trace.messages)
    with annotation_context(view, config(), []):
        assert not validate_support_quotes(
            view.messages, [{"msg_idx": 3, "quote": quote}]
        )[0]


class Decisions:
    def __init__(self):
        self.requests = []

    async def evaluate(self, model, state, questions):
        self.requests.append(copy.deepcopy(state))
        answers = {}
        for key, question in questions.items():
            if key.startswith("same_"):
                answers[key] = {"choice": "distinct", "confidence": 1}
            elif key.startswith("edge_"):
                choice = (
                    "m3"
                    if "none" in question["criteria"]
                    else (
                        "observes"
                        if "observes" in question["criteria"]
                        else ("unrelated" if key.startswith("edge_") else "T")
                    )
                )
                answers[key] = {"choice": choice, "confidence": 1}
            else:
                span_key, node_type = key.rsplit("_", 1)
                kind = state["candidates"][span_key]["kind"]
                expected = {"action": "T", "observation": "E", "prose": "J"}.get(kind)
                if (
                    "<action>submit_answer</action>"
                    in state["candidates"][span_key]["quote"]
                ):
                    expected = "F"
                answers[key] = {
                    "choice": "present" if node_type == expected else "absent",
                    "confidence": 1,
                }
        return answers


@pytest.mark.parametrize("kind", ["llm", "jev"])
@pytest.mark.parametrize("strict_support", [True, False])
def test_both_backends_stop_at_submission(monkeypatch, kind, strict_support):
    trace = source()
    trace.messages[4]["content"] = (
        '<action>submit_answer</action><action_input>{"answer":"X"}</action_input>'
    )
    trace.messages.extend(
        [
            {"role": "tool", "content": "POST_SUBMISSION_RESULT"},
            {"role": "assistant", "content": "POST_SUBMISSION_FINAL_ANSWER"},
        ]
    )
    trace.message_sources.extend([{"index": 5}, {"index": 6}])
    client = Decisions()
    requests = []

    async def respond(model, system, user, **kwargs):
        requests.append(user)
        if "Nodes:\n" in user:
            return {"edges": []}
        return {
            "nodes": [
                {
                    "type": "T",
                    "time": 4,
                    "text": "submit_answer",
                    "support": [{"msg_idx": 4, "quote": trace.messages[4]["content"]}],
                }
            ]
        }

    monkeypatch.setattr(llm, "llm_json_call_async", respond)
    backend = (
        llm.LLMAnnotator().backend()
        if kind == "llm"
        else jev.JevAnnotator(client).backend()
    )
    backend = replace(
        backend, extract_trace_checks=None
    )  # This test covers graph context only.
    settings = replace(config(), window=20, overlap=1, strict_support=strict_support)
    graph = asyncio.run(annotate_trace(trace, backend, settings))
    assert "POST_SUBMISSION" not in json.dumps(requests + client.requests)
    assert requests or client.requests
    assert graph.messages == trace.messages
    assert graph.annotation_view["messages"][-1] == trace.messages[4]
    assert max(n["time"] for n in graph.nodes) == 4
    assert all(
        s["msg_idx"] <= 4 for item in graph.nodes + graph.edges for s in item["support"]
    )
    assert graph.annotation_status == "complete"
    assert graph.qc["uncovered_messages"] == ([2] if kind == "llm" else [])
    assert can_reuse(graph, trace, backend, settings)
    old = copy.deepcopy(graph)
    old.annotation_view["policy_version"] = "head-tail-v1"
    assert not can_reuse(old, trace, backend, settings)


def test_jev_uses_same_view_everywhere_including_edge_support():
    trace, client = source(), Decisions()
    backend = replace(jev.JevAnnotator(client).backend(), extract_trace_checks=None)
    graph = asyncio.run(annotate_trace(trace, backend, config()))
    assert graph.messages == trace.messages
    assert graph.edges
    assert len([n for n in graph.nodes if n["time"] == 3]) == 1
    for request in client.requests:
        assert "HIDDEN MIDDLE" not in json.dumps(request, ensure_ascii=False)
    assert any("nodes" in r for r in client.requests)
    for item in graph.nodes + graph.edges:
        assert validate_support_quotes(trace.messages, item["support"])[0]
        for support in item["support"]:
            assert "[ANNOTATION VIEW:" not in support["quote"]
            assert (
                trace.messages[support["msg_idx"]]["content"][
                    support["start"] : support["end"]
                ]
                == support["quote"]
            )
    assert any(len(e["support"]) == 2 for e in graph.edges)
    assert analyze_graph(graph).row["reduced_context"]
    assert can_reuse(graph, trace, backend, config())
    assert not can_reuse(
        graph, trace, backend, replace(config(), tool_output_max_chars=3000)
    )
    assert not can_reuse(
        graph, trace, backend, replace(config(), tool_output_max_chars=0)
    )
    tampered = copy.deepcopy(graph)
    tampered.annotation_view["messages"][3]["content"] += "tampered"
    assert not can_reuse(tampered, trace, backend, config())


@pytest.mark.parametrize("annotator", ["llm", "jev"])
def test_runtime_postprocessing_precedes_edges_and_respects_submission(annotator):
    notice = "agent exhausted its 50 interaction budget"
    submission = "<action>submit_answer</action>"
    trace = NormalizedTrace(
        [
            {"role": "assistant", "content": notice},
            {"role": "assistant", "content": submission},
            {"role": "assistant", "content": notice},
        ],
        SourceReference("runtime", "runtime.json", "runtime"),
    )

    async def nodes(**kwargs):
        return [
            {
                "node_id": "N1",
                "type": "F",
                "time": 1,
                "text": submission,
                "support": [{"msg_idx": 1, "quote": submission}],
            }
        ], []

    async def edges(nodes, **kwargs):
        assert [(n["time"], n["type"]) for n in nodes] == [(0, "N"), (1, "F")]
        assert len({n["node_id"] for n in nodes}) == 2
        return [], []

    graph = asyncio.run(
        annotate_trace(
            trace, AnnotationBackend(nodes, edges, {"annotator": annotator}), config()
        )
    )
    assert graph.messages == trace.messages
    assert graph.annotation_status == "complete"
    assert graph.nodes[0]["support"] == [
        {"msg_idx": 0, "quote": notice, "start": 0, "end": len(notice)}
    ]


def test_runtime_neutral_fallback_preserves_existing_labels_and_ordinary_text():
    from corral.epistemic_analysis.graph import _postprocess_runtime_notices

    messages = [
        {"role": "assistant", "content": "agent exhausted its 50 interaction budget"},
        {
            "role": "assistant",
            "content": "The agent exhausted its 50 interaction budget, so X remains unknown.",
        },
        {"role": "tool", "content": "agent exhausted its 50 interaction budget"},
        {"role": "assistant", "content": "No supported reasoning."},
    ]
    nodes = [{"node_id": "N1", "type": "J", "time": 0}]
    result, warnings = _postprocess_runtime_notices(nodes, messages)
    assert result == nodes
    assert warnings == []


def test_llm_both_passes_and_automatic_observation_support(monkeypatch):
    seen = []

    async def respond(model, system, user, **kwargs):
        seen.append(user)
        assert "HIDDEN MIDDLE" not in user
        if "Nodes:\n" in user:
            return {"edges": []}
        return {
            "nodes": [
                {
                    "type": "T",
                    "time": 2,
                    "text": "measure",
                    "support": [{"msg_idx": 2, "quote": "<action>measure</action>"}],
                }
            ]
        }

    monkeypatch.setattr(llm, "llm_json_call_async", respond)
    graph = asyncio.run(
        annotate_trace(
            source(),
            replace(llm.LLMAnnotator().backend(), extract_trace_checks=None),
            config(),
        )
    )
    assert any("Nodes:\n" in request for request in seen)
    observations = [n for n in graph.nodes if n["time"] == 3]
    assert len(observations) == 1
    assert len(observations[0]["support"]) == 2
    assert all(
        validate_support_quotes(graph.messages, n["support"])[0] for n in graph.nodes
    )


def test_llm_stream_timeout_retries_even_when_chunks_keep_arriving(monkeypatch):
    calls, cancelled, delays = [], [], []
    sleep = asyncio.sleep

    async def stream(stall):
        try:
            yield "chunk"
            if stall:
                while True:
                    await sleep(0.001)
                    yield "chunk"
        finally:
            cancelled.append(stall)

    async def respond(**kwargs):
        calls.append(kwargs)
        return stream(len(calls) == 1)

    async def retry_sleep(delay):
        delays.append(delay)

    client = SimpleNamespace(
        token_counter=lambda **kwargs: 10,
        acompletion=respond,
        stream_chunk_builder=lambda chunks, **kwargs: {
            "model": "resolved",
            "choices": [{"message": {"content": '{"nodes": []}'}}],
        },
    )
    monkeypatch.setattr(llm, "_load_litellm", lambda: client)
    monkeypatch.setattr(llm.asyncio, "sleep", retry_sleep)
    response = asyncio.run(
        llm.llm_json_call_async(
            "model", "system", "user", timeout_s=0.02, max_retries=2
        )
    )
    assert response == {"nodes": []}
    assert len(calls) == 2
    assert cancelled == [True, False]
    assert delays == [2]


def test_hidden_support_rejected_even_with_strict_support_disabled():
    async def nodes(**kwargs):
        return [
            {
                "node_id": "N1",
                "type": "H",
                "time": 4,
                "text": "hidden",
                "support": [{"msg_idx": 3, "quote": "HIDDEN MIDDLE"}],
            }
        ], []

    async def edges(nodes, **kwargs):
        assert all(n["text"] != "hidden" for n in nodes)
        return [], []

    graph = asyncio.run(
        annotate_trace(
            source(), AnnotationBackend(nodes, edges, {}), config(strict_support=False)
        )
    )
    assert any("retained range" in warning for warning in graph.qc["warnings"])


def test_cap_reductions_restart_both_passes_and_cache_effective_view():
    calls = []

    async def nodes(**kwargs):
        calls.append(("nodes", current_context().view.effective_cap))
        return [], []

    async def edges(**kwargs):
        cap = current_context().view.effective_cap
        calls.append(("edges", cap))
        if cap > 1500:
            raise RequestBudgetExceeded({"backend": "mock", "budget_tokens": 24000})
        return [], []

    backend = AnnotationBackend(nodes, edges, {})
    graph = asyncio.run(annotate_trace(source(), backend, config()))
    assert calls == [
        (stage, cap) for cap in (6000, 3000, 1500) for stage in ("nodes", "edges")
    ]
    assert graph.provenance["effective_tool_output_max_chars"] == 1500
    assert len(graph.qc["annotation_attempts"]) == 3
    assert can_reuse(graph, source(), backend, config())
    assert current_context() is None


def test_oversized_non_tool_text_does_not_retry_identical_view():
    calls = []

    async def fail(**kwargs):
        calls.append(1)
        raise RequestBudgetExceeded({"backend": "mock"})

    trace = source()
    trace.messages[3] = {"role": "assistant", "content": "x" * 20000}
    trace.message_sources[3] = {"index": 3}
    with pytest.raises(RequestBudgetExceeded) as error:
        asyncio.run(annotate_trace(trace, AnnotationBackend(fail, fail, {}), config()))
    assert len(calls) == 1
    assert len(error.value.attempts) == 1


def test_fallback_skips_unchanged_cap_and_tries_next_distinct_view():
    calls = []

    async def nodes(**kwargs):
        cap = current_context().view.effective_cap
        calls.append(cap)
        if cap != 1500:
            raise RequestBudgetExceeded({"backend": "mock"})
        return [], []

    async def edges(**kwargs):
        return [], []

    trace = source()
    trace.messages[3]["content"] = "Observation: " + "x" * 2000
    graph = asyncio.run(
        annotate_trace(trace, AnnotationBackend(nodes, edges, {}), config())
    )
    assert calls == [6000, 1500]
    assert graph.provenance["effective_tool_output_max_chars"] == 1500


def test_concurrent_annotations_keep_independent_views_and_diagnostics():
    async def nodes(messages, **kwargs):
        expected = current_context().view.effective_cap
        await asyncio.sleep(0)
        assert current_context().view.effective_cap == expected
        assert len(current_context().view.support(3)[0]["quote"]) == expected * 3 // 4
        return [], []

    async def edges(**kwargs):
        await asyncio.sleep(0)
        return [], []

    async def both():
        backend = AnnotationBackend(nodes, edges, {})
        return await asyncio.gather(
            *(
                annotate_trace(
                    source(), backend, replace(config(), tool_output_max_chars=cap)
                )
                for cap in (6000, 3000)
            )
        )

    graphs = asyncio.run(both())
    assert [g.provenance["effective_tool_output_max_chars"] for g in graphs] == [
        6000,
        3000,
    ]
    assert current_context() is None


def test_unknown_jev_tokenizer_fallback_is_byte_sensitive(monkeypatch):
    from corral.epistemic_analysis import budget

    monkeypatch.setattr(budget, "_encoding", lambda: None)
    ascii_tokens, identity = budget.estimate_request({"text": "aaaa"})
    unicode_tokens, _ = budget.estimate_request({"text": "🧪🧪🧪🧪"})
    assert unicode_tokens > ascii_tokens
    assert identity == "utf8-bytes-plus-256-v1"


@pytest.mark.parametrize("provider_rejection", [False, True])
def test_jev_splits_batches_without_repeating_oversized_payload(
    monkeypatch, provider_rejection
):
    bodies = []
    monkeypatch.setattr(
        jev,
        "estimate_request",
        lambda body: (500 + len(body["questions"]) * 100, "test"),
    )

    def post(body):
        bodies.append(body)
        if provider_rejection and len(body["questions"]) > 1:
            raise HTTPError(
                "https://example.test",
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    b'{"error":{"code":"max_tokens_exceeded","message":"secret"}}'
                ),
            )
        return {
            "answers": [
                {"name": q["name"], "type": "predicate", "probability": 0.5}
                for q in body["questions"]
            ],
            "usage": {"input_tokens": 42},
        }

    client = jev.JevClient(
        "test", batch_size=8, max_request_tokens=0 if provider_rejection else 650
    )
    monkeypatch.setattr(client, "_post", post)
    questions = {f"q{i}": {"type": "noul"} for i in range(4)}
    answers = asyncio.run(client.evaluate("jev", {"content": "same"}, questions))
    assert set(answers) == set(questions)
    assert len({json.dumps(body, sort_keys=True) for body in bodies}) == len(bodies)
    assert all(json.loads(body["input"]) == {"content": "same"} for body in bodies)
    assert len(bodies) == (7 if provider_rejection else 4)


def test_decisions_honors_requested_budget_above_old_typesafe_limit(monkeypatch):
    bodies = []
    monkeypatch.setattr(jev, "estimate_request", lambda body: (40000, "test"))

    def post(body):
        bodies.append(body)
        return {"answers": [{"name": "q", "type": "predicate", "probability": 0.5}]}

    client = jev.JevClient("test", max_request_tokens=50000)
    monkeypatch.setattr(client, "_post", post)
    answers = asyncio.run(client.evaluate("gpt-6-luna", {}, {"q": {"type": "noul"}}))
    assert answers["q"]["noul"] == 0.5
    assert len(bodies) == 1


@pytest.mark.parametrize("status", [400, 401, 403])
def test_other_http_errors_never_trigger_cap_fallback(monkeypatch, status):
    calls = []

    def fail(body):
        calls.append(body)
        raise HTTPError(
            "https://example.test",
            status,
            "Bad request",
            {},
            io.BytesIO(b'{"error":"invalid_schema secret"}'),
        )

    client = jev.JevClient("secret")
    monkeypatch.setattr(client, "_post", fail)
    with pytest.raises(jev.JevError, match=f"HTTP {status}") as error:
        asyncio.run(client.evaluate("jev", {}, {"q": {"type": "noul"}}))
    assert len(calls) == 1
    assert "secret" not in str(error.value)


def test_single_question_provider_overflow_is_sanitized_and_not_retried(monkeypatch):
    calls = []

    def fail(body):
        calls.append(body)
        raise HTTPError(
            "https://example.test",
            400,
            "Bad request",
            {},
            io.BytesIO(b'{"error":{"code":"max_tokens_exceeded","message":"secret"}}'),
        )

    client = jev.JevClient("secret", max_retries=5)
    monkeypatch.setattr(client, "_post", fail)
    with pytest.raises(RequestBudgetExceeded) as error:
        asyncio.run(
            client.evaluate("jev", {"text": "private trace"}, {"q": {"type": "noul"}})
        )
    assert len(calls) == 1
    assert error.value.diagnostic["reason"] == "max_tokens_exceeded"
    assert "secret" not in str(error.value)
    assert "private trace" not in json.dumps(error.value.diagnostic)


def test_llm_reserves_output_and_counts_complete_chat_before_call(monkeypatch):
    seen = []
    client = SimpleNamespace(
        token_counter=lambda **kwargs: seen.append(kwargs) or 500,
        get_model_info=lambda model: {"max_input_tokens": 1000},
    )
    monkeypatch.setattr(llm, "_load_litellm", lambda: client)
    cfg = replace(config(), llm_output_reserve_tokens=400)
    view = build_annotation_view(source(), cfg)
    with (
        annotation_context(view, cfg, []),
        pytest.raises(RequestBudgetExceeded) as error,
    ):
        asyncio.run(
            llm.llm_json_call_async("llm", "ontology", "messages and node text")
        )
    assert error.value.diagnostic["budget_tokens"] == 600
    assert seen[0]["messages"] == [
        {"role": "system", "content": "ontology"},
        {"role": "user", "content": "messages and node text"},
    ]


def invoke(*options):
    args = build_parser().parse_args(["analyze", *map(str, options)])
    return asyncio.run(args._handler(args))


def test_cli_failure_artifacts_bounded_attempts_and_offline_reanalysis(
    tmp_path, monkeypatch
):
    calls = []

    async def fail(**kwargs):
        calls.append(current_context().view.effective_cap)
        raise RequestBudgetExceeded(
            {
                "backend": "mock",
                "estimated_tokens": 30000,
                "budget_tokens": 24000,
                "reason": "request_budget",
            }
        )

    monkeypatch.setattr(
        cli, "create_backend", lambda args: AnnotationBackend(fail, fail, {})
    )
    local, output = tmp_path / "input.json", tmp_path / "output"
    local.write_text(json.dumps({"messages": source().messages}))
    assert (
        invoke(
            "--input",
            local,
            "--output",
            output,
            "--tool-output-max-chars",
            6000,
            "--max-request-tokens",
            24000,
        )
        == 1
    )
    assert calls == [6000, 3000, 1500]
    saved = next((output / "annotations").glob("*.json"))
    graph = read_graph(json.loads(saved.read_text()))
    assert graph.annotation_status == "failed"
    assert graph.nodes == graph.edges == []
    assert len(graph.qc["annotation_attempts"]) == 3
    assert (output / graph.provenance["annotation_view_artifact"]).is_file()
    assert graph.messages == source().messages
    monkeypatch.setattr(
        cli, "create_backend", lambda args: pytest.fail("provider initialized")
    )
    assert (
        invoke(
            "--input", output, "--skip-annotate", "--output", tmp_path / "reanalysis"
        )
        == 0
    )
    report = (
        tmp_path / "reanalysis" / "analysis" / "annotation_summary.md"
    ).read_text()
    assert "Reduced-context traces: 1" in report
    assert "Eligible: 0" in report


def test_legacy_version_one_graph_still_reads():
    graph = read_graph({"schema_version": 1, "nodes": [], "edges": []})
    assert graph.annotation_view is None
    assert not analyze_graph(graph).row["reduced_context"]
