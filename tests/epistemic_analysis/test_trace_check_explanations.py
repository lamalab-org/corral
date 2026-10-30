"""Exercise whole-trace follow-ups through the SDK without provider calls."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI
from tests.epistemic_analysis.test_decisions import explanation_payload
from tests.epistemic_analysis.test_trace_checks import trace

from corral.epistemic_analysis import AnnotationConfig, annotate_trace, read_graph
from corral.epistemic_analysis import trace_check_explanations as explanations
from corral.epistemic_analysis.annotators import jev
from corral.epistemic_analysis.pipeline import can_reuse
from corral.epistemic_analysis.trace_checks import TRACE_CHECK_NAMES, trace_check_prompt


@pytest.fixture
def anyio_backend():
    return "asyncio"


def configure(monkeypatch, respond, choices, *, concurrency=8):
    client = jev.JevClient(
        "fixture-key",
        endpoint="https://trace.test/custom/v1/decisions",
        timeout_s=10,
        max_retries=1,
    )
    decisions, options = [], []

    def post(body):
        decisions.append(body)
        question = body["questions"][0]
        name = question["name"]
        status, confidence = choices.get(name, ("absent", 0.9))
        return {
            "model": "resolved-decision-model",
            "answers": [
                {
                    "name": name,
                    "type": "choice",
                    "choice": status,
                    "confidence": confidence,
                    "probabilities": [
                        {
                            "value": c["value"],
                            "probability": float(c["value"] == status),
                        }
                        for c in question["choices"]
                    ],
                }
            ],
        }

    def sdk(**kwargs):
        options.append(kwargs)
        return AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(client, "_post", post)
    monkeypatch.setattr(explanations, "AsyncOpenAI", sdk)
    # Keep these tests independent of tokenizer cache/network availability.
    monkeypatch.setattr(jev, "estimate_request", lambda body: (100, "fixture"))
    monkeypatch.setattr(explanations, "estimate_request", lambda body: (100, "fixture"))
    return (
        jev.JevAnnotator(client, explanation_concurrency=concurrency),
        decisions,
        options,
    )


def response(name, *, supported=True):
    payload = explanation_payload(
        {"supported": supported, "explanation": f"  Assessment for {name}.  "}
    )
    payload["id"] = name
    payload["model"] = "resolved-explanation-model"
    return httpx.Response(200, json=payload)


@pytest.mark.anyio
@pytest.mark.parametrize("concurrency", [1, 2, 8])
async def test_same_model_full_trace_parallel_calls_preserve_order(
    monkeypatch, concurrency
):
    names = TRACE_CHECK_NAMES[: concurrency + 1]
    releases = {name: asyncio.Event() for name in names}
    started = asyncio.Queue()
    active, finished = set(), []
    source = trace()

    async def respond(request):
        assert len(decisions) == len(TRACE_CHECK_NAMES)
        assert str(request.url) == "https://trace.test/custom/v1/responses"
        body = json.loads(request.content)
        state = json.loads(body["input"])
        name = state["check"]["name"]
        assert body["model"] == "same-model"
        assert body["store"] is False
        assert body["instructions"].startswith(trace_check_prompt(name))
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["required"] == [
            "supported",
            "explanation",
        ]
        assert [m["content"] for m in state["messages"].values()] == [
            m["content"] for m in source.messages
        ]
        assert state["nodes"] == [{"node_id": "fixture"}]
        active.add(name)
        started.put_nowait(name)
        try:
            await releases[name].wait()
            finished.append(name)
            return response(name, supported=name != names[0])
        finally:
            active.remove(name)

    annotator, decisions, options = configure(
        monkeypatch,
        respond,
        dict.fromkeys(names, ("present", 0.9)),
        concurrency=concurrency,
    )
    run = asyncio.create_task(
        annotator.extract_trace_checks(
            source, [{"node_id": "fixture"}], [], "same-model"
        )
    )
    try:
        in_flight = [
            await asyncio.wait_for(started.get(), 2) for _ in range(concurrency)
        ]
        assert len(active) == concurrency
        assert started.empty()
        releases[in_flight.pop()].set()
        in_flight.append(await asyncio.wait_for(started.get(), 2))
        assert len(active) == concurrency
        for name in reversed(in_flight):
            releases[name].set()
        checks = await asyncio.wait_for(run, 2)
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
    assert not active
    assert set(finished) == set(names)
    if concurrency > 1:
        assert finished[0] != names[0]
    assert [c["name"] for c in checks] == list(TRACE_CHECK_NAMES)
    for check in checks[: len(names)]:
        assert check["status"] == "present"
        assert check["jev"]["choice"] == "present"
        assert check["explanation"] == f"Assessment for {check['name']}."
        assert check["explanation_supported"] is (check["name"] != names[0])
    assert all("explanation" not in c for c in checks[len(names) :])
    assert options == [
        {
            "api_key": "fixture-key",
            "base_url": "https://trace.test/custom/v1/",
            "timeout": 10,
            "max_retries": 0,
        }
    ]
    assert all(body["model"] == "same-model" for body in decisions)


@pytest.mark.anyio
async def test_unflagged_and_low_confidence_checks_do_not_open_responses_client(
    monkeypatch,
):
    async def respond(request):
        pytest.fail("Unflagged checks must not request explanations")

    annotator, _, options = configure(
        monkeypatch,
        respond,
        {
            TRACE_CHECK_NAMES[0]: ("present", 0.1),
            TRACE_CHECK_NAMES[1]: ("uncertain", 0.9),
            "insufficient_validation": ("not_applicable", 0.9),
        },
    )
    checks = await annotator.extract_trace_checks(trace(), [], [], "same-model")
    assert not options
    assert all("explanation" not in check for check in checks)


def graph_backend(annotator):
    async def nodes(messages, **kwargs):
        return [
            {
                "node_id": "F1",
                "type": "F",
                "time": 3,
                "text": "Submission",
                "support": [{"msg_idx": 3, "quote": messages[3]["content"]}],
            }
        ], []

    async def edges(*args, **kwargs):
        return [], []

    return replace(annotator.backend(), extract_nodes=nodes, extract_edges=edges)


@pytest.mark.anyio
async def test_explanations_usage_and_provenance_survive_saved_graph(monkeypatch):
    async def respond(request):
        name = json.loads(json.loads(request.content)["input"])["check"]["name"]
        return response(name)

    names = TRACE_CHECK_NAMES[:2]
    annotator, _, _ = configure(
        monkeypatch, respond, dict.fromkeys(names, ("present", 0.9))
    )
    backend, source = graph_backend(annotator), trace()
    config = AnnotationConfig("same-model", tool_output_max_chars=32)
    graph = await annotate_trace(source, backend, config)
    assert graph.annotation_status == graph.trace_checks_status == "complete"
    assert can_reuse(graph, source, backend, config)
    saved = read_graph(json.loads(json.dumps(graph.to_dict())))
    assert saved == graph
    assert saved.trace_checks[0]["explanation_supported"] is True
    calls = [r for r in saved.annotation_requests if r.get("api") == "openai-responses"]
    assert len(calls) == 2
    assert all(r["stage"] == "trace_checks" for r in calls)
    assert all(r["effective_tool_output_max_chars"] == 0 for r in calls)
    assert {r["response"]["id"] for r in calls} == set(names)
    usage = [
        r for r in saved.qc["request_diagnostics"] if r.get("api") == "openai-responses"
    ]
    assert sum(r["usage"]["input_tokens"] for r in usage) == 38
    assert "resolved-explanation-model" in saved.provenance["resolved_models"]
    old_backend = replace(
        backend,
        provenance={
            k: v
            for k, v in backend.provenance.items()
            if k != "trace_check_explanations"
        },
    )
    assert not can_reuse(graph, source, old_backend, config)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    [
        "blank",
        "missing",
        "type",
        "incomplete",
        "refusal",
        "json",
        "http400",
        "budget",
        "context",
    ],
)
async def test_invalid_explanations_fail_pass_and_preserve_graph(monkeypatch, failure):
    calls = []

    async def respond(request):
        calls.append(request)
        if failure in {"http400", "context"}:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "context_length_exceeded"
                        if failure == "context"
                        else "invalid request"
                    }
                },
            )
        item = {"supported": True, "explanation": "Evidence."}
        if failure == "blank":
            item["explanation"] = " "
        elif failure == "missing":
            item.pop("supported")
        elif failure == "type":
            item["supported"] = "true"
        payload = explanation_payload(
            item, status="incomplete" if failure == "incomplete" else "completed"
        )
        if failure == "refusal":
            payload["output"][0]["content"] = [
                {"type": "refusal", "refusal": "Refused"}
            ]
        elif failure == "json":
            payload["output"][0]["content"][0]["text"] = "not JSON"
        return httpx.Response(200, json=payload)

    annotator, _, _ = configure(
        monkeypatch, respond, {TRACE_CHECK_NAMES[0]: ("present", 0.9)}
    )
    if failure == "budget":
        monkeypatch.setattr(
            explanations, "estimate_request", lambda body: (1001, "fixture")
        )
    graph = await annotate_trace(
        trace(),
        graph_backend(annotator),
        AnnotationConfig(
            "same-model", tool_output_max_chars=32, max_request_tokens=1000
        ),
    )
    assert graph.nodes
    assert graph.trace_checks_status == "failed"
    assert graph.annotation_status == "incomplete"
    assert graph.trace_checks == []
    assert len(calls) == (0 if failure == "budget" else 1)
    if failure in {"budget", "context"}:
        assert graph.qc["trace_checks_error"]["type"] == "RequestBudgetExceeded"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["cancel", "invalid", "timeout"])
async def test_failure_or_cancellation_drains_other_explanations(monkeypatch, failure):
    started = asyncio.Queue()
    release = asyncio.Event()
    active, cancelled = set(), set()
    names = TRACE_CHECK_NAMES[:3]

    async def respond(request):
        name = json.loads(json.loads(request.content)["input"])["check"]["name"]
        active.add(name)
        started.put_nowait(name)
        try:
            if failure == "invalid" and name == names[0]:
                await release.wait()
                return httpx.Response(
                    200, json=explanation_payload({"explanation": "invalid"})
                )
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(name)
            raise
        finally:
            active.remove(name)

    annotator, _, _ = configure(
        monkeypatch, respond, dict.fromkeys(names, ("present", 0.9)), concurrency=2
    )
    if failure == "timeout":
        annotator.client.timeout_s = 0.1
    run = asyncio.create_task(
        annotator.extract_trace_checks(trace(), [], [], "same-model")
    )
    try:
        for _ in range(2):
            await asyncio.wait_for(started.get(), 2)
        if failure == "cancel":
            run.cancel()
        elif failure == "invalid":
            release.set()
        error = {
            "cancel": asyncio.CancelledError,
            "invalid": ValueError,
            "timeout": asyncio.TimeoutError,
        }[failure]
        with pytest.raises(error):
            await asyncio.wait_for(run, 2)
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
    assert not active
    assert names[1] in cancelled


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None])
def test_invalid_explanation_concurrency(value):
    with pytest.raises(
        ValueError, match="explanation_concurrency must be a positive integer"
    ):
        jev.JevAnnotator(None, explanation_concurrency=value)
