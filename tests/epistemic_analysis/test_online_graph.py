"""Prefix isolation, graph-only selection, accounting, and stage timing."""

import asyncio
import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import httpx
import pytest
from tests.epistemic_analysis.test_knowline_graph import config
from tests.epistemic_analysis.test_shared_client import chat_config, chat_response

from corral.epistemic_analysis.annotation_transcript import (
    live_annotation_trace as graph_trace,
)
from corral.epistemic_analysis.annotators.graph import (
    AnnotationBackend,
    AnnotationError,
    GraphExtractor,
)
from corral.epistemic_analysis.annotators.local import (
    GraphAnnotator,
    KnowLineClient,
    OpenAIClient,
)
from corral.epistemic_analysis.context import current_context


@pytest.fixture
def anyio_backend():
    return "asyncio"


def actor_prefix(task="Solve the task."):
    return [
        {"role": "system", "content": "Scientific assistant"},
        {"role": "user", "content": task},
    ]


def test_online_task_string_contract_does_not_require_agent_changes():
    trace = graph_trace([{"role": "assistant", "content": "Working."}], "Task.")
    assert trace.messages == [
        {"role": "user", "content": "Task."},
        {"role": "assistant", "content": "Working."},
    ]


def mock_annotator(
    monkeypatch, tmp_path, *, hypothesis=True, incomplete=False, events=None
):
    events = events if events is not None else []

    def backend(self, **_):
        async def nodes(messages, **_):
            events.append("nodes")
            await asyncio.sleep(0)
            if incomplete:
                current_context().review_issues.append("Missing node answer")
            self.client.exchanges.append(
                {"cached": False, "usage": {"input_tokens": 7, "output_tokens": 2}}
            )
            index = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
            text = messages[index]["content"]
            return (
                [
                    {
                        "node_id": "H1",
                        "type": "H",
                        "time": index,
                        "text": text,
                        "support": [{"msg_idx": index, "quote": text}],
                    }
                ]
                if hypothesis
                else []
            ), []

        async def edges(**_):
            events.append("edges")
            return [], []

        return AnnotationBackend(nodes, edges, {"annotator": "fixture"})

    monkeypatch.setattr(GraphExtractor, "backend", backend)
    monkeypatch.setattr(KnowLineClient, "preflight", AsyncMock())
    return GraphAnnotator(provider_config=asdict(config()), artifact_dir=tmp_path)


@pytest.mark.anyio
async def test_only_graph_matches_are_explained_and_timed(monkeypatch, tmp_path):
    events = []
    annotator = mock_annotator(monkeypatch, tmp_path, events=events)
    seen = []

    async def explain(motifs, graph, trajectory, task):
        events.append("explanation")
        assert all(graph["pattern_counts"][m.name] > 0 for m in motifs)
        seen.append(trajectory)
        assert task == actor_prefix("Measure resistance.")
        return {
            m.name: "The annotated hypothesis has no linked test." for m in motifs
        }, {"response": {"usage": {"input_tokens": 11, "output_tokens": 5}}}

    monkeypatch.setattr(annotator, "_explain_candidates", explain)
    messages = [
        {
            "role": "assistant",
            "content": "I hypothesize 100 ohms.",
            "reasoning_content": "PRIVATE",
        },
        {"role": "user", "name": "epistemic_report", "content": "OLD REPORT"},
    ]
    report = await annotator.annotate(
        messages, task=actor_prefix("Measure resistance.")
    )
    assert events == ["nodes", "edges", "explanation"]
    assert report.present_antipatterns
    assert "PRIVATE" not in json.dumps(seen)
    assert "OLD REPORT" not in json.dumps(seen)
    assert report.usage == {
        "input_tokens": 18,
        "output_tokens": 7,
        "reasoning_tokens": 0,
        "llm_calls": 2,
    }
    assert (
        report.timings_s["annotation"]
        >= report.timings_s["nodes"] + report.timings_s["edges"]
    )
    assert report.timings_s["total"] >= sum(
        report.timings_s[k] for k in ("annotation", "patterns", "explanation")
    )
    assert "verification" not in report.raw_response
    assert all(a.confidence is None and not a.probabilities for a in report.assessments)


@pytest.mark.anyio
async def test_empty_graph_does_not_call_explanation_model(monkeypatch, tmp_path):
    annotator = mock_annotator(monkeypatch, tmp_path, hypothesis=False)
    explain = AsyncMock(side_effect=AssertionError("No graph match"))
    monkeypatch.setattr(annotator, "_explain_candidates", explain)
    report = await annotator.annotate(
        [*actor_prefix(), {"role": "assistant", "content": "Considering."}]
    )
    assert not report.present_antipatterns
    assert not any(report.raw_response["pattern_counts"].values())
    explain.assert_not_awaited()


@pytest.mark.anyio
async def test_incomplete_graph_stops_before_explanation(monkeypatch, tmp_path):
    annotator = mock_annotator(monkeypatch, tmp_path, incomplete=True)
    explain = AsyncMock()
    monkeypatch.setattr(annotator, "_explain_candidates", explain)
    with pytest.raises(AnnotationError, match="Incomplete"):
        await annotator.annotate(
            [*actor_prefix(), {"role": "assistant", "content": "A hypothesis."}]
        )
    explain.assert_not_awaited()


@pytest.mark.anyio
async def test_concurrent_reports_have_separate_graphs_and_timings(
    monkeypatch, tmp_path
):
    annotator = mock_annotator(monkeypatch, tmp_path)
    annotator.include_explanations = False
    reports = await asyncio.gather(
        *(
            annotator.annotate(
                [*actor_prefix(), {"role": "assistant", "content": text}]
            )
            for text in ("Candidate A.", "Candidate B.")
        )
    )
    assert reports[0].graph["nodes"][0]["text"] == "Candidate A."
    assert reports[1].graph["nodes"][0]["text"] == "Candidate B."
    assert reports[0].timings_s is not reports[1].timings_s


@pytest.mark.anyio
async def test_native_annotation_and_chat_explanation_use_separate_endpoints(
    monkeypatch, tmp_path
):
    calls = []

    def handle_native(request):
        calls.append(request.url.path)
        if request.url.path == "/metadata":
            return httpx.Response(
                200,
                json={"revision": "revision", "scoring": {}, "context_tokens": 131072},
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 30})
        body = json.loads(request.content)
        answers = {}
        for name, question in body["questions"].items():
            assert name not in {"untested_claim", "fixed_belief_trace"}
            choice = "present" if name.endswith("_H") else "absent"
            answers[name] = {
                "type": "choice",
                "choice": choice,
                "confidence": 0.2,
                "probabilities": {k: float(k == choice) for k in question["criteria"]},
            }
        return httpx.Response(
            200,
            json={"answers": answers, "usage": {"input_tokens": 9, "output_tokens": 2}},
        )

    def handle_chat(request):
        calls.append(request.url.path)
        body = json.loads(request.content)
        names = body["response_format"]["json_schema"]["schema"]["required"]
        return httpx.Response(
            200, json=chat_response(json.dumps(dict.fromkeys(names, "A graph match.")))
        )

    monkeypatch.setattr(
        "corral.epistemic_analysis.annotators.local.create_provider",
        lambda cfg: KnowLineClient(cfg, transport=httpx.MockTransport(handle_native)),
    )
    monkeypatch.setattr(
        "corral.epistemic_analysis.annotators.local.OpenAIClient",
        lambda cfg: OpenAIClient(cfg, transport=httpx.MockTransport(handle_chat)),
    )
    annotator = GraphAnnotator(
        provider_config=asdict(config()),
        explanation_config=asdict(chat_config()),
        artifact_dir=tmp_path,
    )
    reports = [
        await annotator.annotate(
            [
                *actor_prefix(),
                {"role": "assistant", "content": "The resistance is 100 ohms."},
            ]
        )
        for _ in range(2)
    ]
    assert calls.count("/infer") == 2
    assert calls.count("/v1/chat/completions") == 2
    assert reports[0].usage["llm_calls"] == 2
    assert reports[0].usage["input_tokens"] == 19
    assert (
        reports[0].graph["annotation_requests"][0]["artifact_root"]
        != reports[1].graph["annotation_requests"][0]["artifact_root"]
    )


def test_live_trace_preserves_task_and_tool_dispatch():
    messages = [
        {
            "role": "assistant",
            "content": "Measure.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "measure", "arguments": '{"voltage":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "0.01 amperes"},
    ]
    trace = graph_trace(messages, actor_prefix("The voltage is 1V."))
    saved = json.dumps(trace.messages)
    assert "The voltage is 1V" in saved
    assert "measure" in saved
    assert "0.01 amperes" in saved
    assert trace.message_sources[-1]["tool_call_id"] == "c1"


def test_proposed_answer_has_final_claim_spans_without_a_fake_dispatch():
    from corral.epistemic_analysis.annotators.graph import candidate_spans

    trace = graph_trace(
        [
            {
                "role": "assistant",
                "name": "submission_proposal",
                "content": "Based on the measurement.\n\nProposed final answer (not yet submitted):\n100 ohms.",
            }
        ],
        actor_prefix("Find the resistance."),
    )
    spans = candidate_spans(trace.messages, 1200)
    assert any(s.kind == "final" and s.quote == "100 ohms." for s in spans)
    assert all(s.kind != "action" for s in spans)
