"""Exercise explanation overlap, bounded fan-out, and cleanup through the SDK."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from tests.epistemic_analysis.test_decisions import (
    decision_payload,
    explanation_payload,
)
from tests.epistemic_analysis.test_explanation_streaming import InterruptedStream

from corral.epistemic_analysis.decisions import (
    DecisionsAnnotationError,
    DecisionsAnnotator,
)
from corral.epistemic_analysis.motifs import MOTIFS


@pytest.fixture
def anyio_backend():
    return "asyncio"


def classification(count):
    payload = decision_payload()
    payload["answers"][:count] = decision_payload("active")["answers"][:count]
    return httpx.Response(200, json=payload)


def explanation(name, *, stream, valid=True):
    payload = explanation_payload(
        {
            name: {
                "supported": True,
                "explanation": f"Evidence for {name}." if valid else " ",
            }
        }
    )
    payload["id"] = name
    if not stream:
        return httpx.Response(200, json=payload)
    event = {"type": "response.completed", "sequence_number": 1, "response": payload}
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"event: response.completed\ndata: {json.dumps(event)}\n\n".encode(),
    )


MESSAGES = [
    {"role": "assistant", "content": "An observed result changed the hypothesis."}
]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("concurrency", [1, 2, 8])
async def test_explanations_start_after_decisions_overlap_and_preserve_order(
    stream, concurrency
):
    names = [m.name for m in MOTIFS[: concurrency + 2]]
    releases = {name: asyncio.Event() for name in names}
    started = asyncio.Queue()
    decisions_started = asyncio.Event()
    release_decisions = asyncio.Event()
    active = set()
    completed = []
    peak = 0
    classifications = 0

    async def respond(request):
        nonlocal peak, classifications
        if request.url.path.endswith("/decisions"):
            classifications += 1
            decisions_started.set()
            await release_decisions.wait()
            return classification(len(names))
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-luna"
        assert body.get("stream", False) is stream
        requested = body["text"]["format"]["schema"]["required"]
        assert len(requested) == 1
        name = requested[0]
        active.add(name)
        peak = max(peak, len(active))
        started.put_nowait(name)
        try:
            await releases[name].wait()
            completed.append(name)
            return explanation(name, stream=stream)
        finally:
            active.remove(name)

    async with AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        annotator = DecisionsAnnotator(
            client=client,
            stream_explanations=stream,
            explanation_concurrency=concurrency,
        )
        run = asyncio.create_task(annotator.annotate(MESSAGES))
        try:
            await asyncio.wait_for(decisions_started.wait(), timeout=2)
            assert started.empty()
            release_decisions.set()
            in_flight = [
                await asyncio.wait_for(started.get(), timeout=2)
                for _ in range(concurrency)
            ]
            assert peak == concurrency
            assert started.empty()
            # A finished request frees its slot immediately, without waiting for
            # the slower candidates in the same assessment.
            for _ in range(len(names) - concurrency):
                releases[in_flight.pop()].set()
                in_flight.append(await asyncio.wait_for(started.get(), timeout=2))
            assert not run.done()
            for name in reversed(in_flight):
                releases[name].set()
            report = await asyncio.wait_for(run, timeout=2)
        finally:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert not active
    assert classifications == 1
    assert peak == concurrency
    assert set(completed) == set(names)
    if concurrency > 1:
        assert completed[0] != names[0]
    assert [r["id"] for r in report.raw_explanation_responses] == names
    assert report.raw_explanation_response is None
    for assessment in report.assessments[: len(names)]:
        assert assessment.explanation == f"Evidence for {assessment.name}."
        assert assessment.explanation_supported is True
        assert assessment.evidence_message_ids == ()
    assert all(a.explanation is None for a in report.assessments[len(names) :])
    assert report.usage == {
        "input_tokens": 31 + 19 * len(names),
        "output_tokens": 2 + 11 * len(names),
        "reasoning_tokens": 1 + 3 * len(names),
        "llm_calls": 1 + len(names),
    }
    saved = json.loads(json.dumps(report.to_dict()))
    assert [r["id"] for r in saved["raw_explanation_responses"]] == names


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "4", None])
def test_explanation_concurrency_requires_positive_integer(value):
    with pytest.raises(
        ValueError, match="explanation_concurrency must be a positive integer"
    ):
        DecisionsAnnotator(explanation_concurrency=value)


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "invalid", "http400"])
async def test_failure_or_cancellation_drains_other_explanations(stream, failure):
    started = asyncio.Queue()
    fail = asyncio.Event()
    active = set()
    cancelled = set()

    async def respond(request):
        if request.url.path.endswith("/decisions"):
            return classification(5)
        name = json.loads(request.content)["text"]["format"]["schema"]["required"][0]
        active.add(name)
        started.put_nowait(name)
        try:
            if name == MOTIFS[0].name and failure != "cancel":
                await fail.wait()
                if failure == "http400":
                    return httpx.Response(
                        400, json={"error": {"message": "bad request"}}
                    )
                return explanation(name, stream=stream, valid=False)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(name)
            raise
        finally:
            active.remove(name)

    async with AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        run = asyncio.create_task(
            DecisionsAnnotator(
                client=client,
                stream_explanations=stream,
                explanation_concurrency=2,
            ).annotate(MESSAGES)
        )
        try:
            first = await asyncio.wait_for(started.get(), timeout=2)
            second = await asyncio.wait_for(started.get(), timeout=2)
            assert {first, second} == {MOTIFS[0].name, MOTIFS[1].name}
            if failure == "cancel":
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, timeout=2)
            else:
                fail.set()
                with pytest.raises(DecisionsAnnotationError):
                    await asyncio.wait_for(run, timeout=2)
            # Cleanup completes before annotate raises and before client closure.
            assert not active
            assert second in cancelled
        finally:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)


@pytest.mark.anyio
async def test_parallel_stream_retry_only_repeats_failed_pattern():
    attempts = {}
    classifications = 0

    def respond(request):
        nonlocal classifications
        if request.url.path.endswith("/decisions"):
            classifications += 1
            return classification(2)
        body = json.loads(request.content)
        name = body["text"]["format"]["schema"]["required"][0]
        attempts[name] = attempts.get(name, 0) + 1
        if name == MOTIFS[0].name and attempts[name] == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=InterruptedStream(),
            )
        return explanation(name, stream=True)

    async with AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        report = await DecisionsAnnotator(
            client=client,
            stream_explanations=True,
            max_retries=1,
        ).annotate(MESSAGES)
    assert classifications == 1
    assert attempts == {MOTIFS[0].name: 2, MOTIFS[1].name: 1}
    assert report.usage["llm_calls"] == 3
    assert report.usage["input_tokens"] == 69
    assert [r["id"] for r in report.raw_explanation_responses] == [
        m.name for m in MOTIFS[:2]
    ]
