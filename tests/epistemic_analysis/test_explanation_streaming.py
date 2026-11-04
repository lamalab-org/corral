"""Exercise real SDK SSE transport, retries, and terminal response validation."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from tests.epistemic_analysis.test_decisions import (
    decision_payload,
    explanation_payload,
)

from corral.epistemic_analysis.decisions import (
    DecisionsAnnotationError,
    DecisionsAnnotator,
)
from corral.epistemic_analysis.motifs import MOTIFS


@pytest.fixture
def anyio_backend():
    return "asyncio"


def completed_response(*, status="completed", refusal=False):
    payload = explanation_payload(
        {
            MOTIFS[0].name: {
                "supported": True,
                "explanation": "The observation supports revising the hypothesis.",
            }
        },
        status=status,
    )
    if refusal:
        payload["output"][0]["content"] = [
            {"type": "refusal", "refusal": "Cannot assess"}
        ]
    event = {"type": f"response.{status}", "sequence_number": 1, "response": payload}
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode(),
    )


class InterruptedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b": keepalive\n\n"
        raise httpx.ReadTimeout("stream interrupted")


class HangingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        await asyncio.sleep(30)
        yield b": keepalive\n\n"


async def annotate(respond, **kwargs):
    async with AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        return await DecisionsAnnotator(
            client=client, stream_explanations=True, **kwargs
        ).annotate(
            [{"role": "assistant", "content": "Observed a contradiction; revise H."}]
        )


def classification():
    payload = decision_payload()
    payload["answers"][0] = decision_payload("active")["answers"][0]
    return httpx.Response(200, json=payload)


@pytest.mark.anyio
async def test_stream_retries_interruption_without_reclassifying_and_keeps_usage():
    requests = []
    classifications = []

    def respond(request):
        if request.url.path.endswith("/decisions"):
            classifications.append(request)
            return classification()
        requests.append(json.loads(request.content))
        assert requests[-1]["stream"] is True
        assert request.extensions["timeout"]["read"] == 600
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=InterruptedStream(),
            )
        return completed_response()

    report = await annotate(respond, timeout_s=600)
    assert len(classifications) == 1
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert report.assessments[0].evidence_message_ids == ()
    assert report.usage == {
        "input_tokens": 50,
        "output_tokens": 13,
        "reasoning_tokens": 4,
        "llm_calls": 2,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["eof", "timeout", "http400", "incomplete", "refusal"])
async def test_stream_errors_are_bounded_and_never_become_absent_feedback(kind):
    attempts = []

    def respond(request):
        if request.url.path.endswith("/decisions"):
            return classification()
        attempts.append(request)
        if kind == "eof":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b": keepalive\n\n",
            )
        if kind == "timeout":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=HangingStream(),
            )
        if kind == "http400":
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        return completed_response(
            status="incomplete" if kind == "incomplete" else "completed",
            refusal=kind == "refusal",
        )

    with pytest.raises(DecisionsAnnotationError):
        await annotate(
            respond, max_retries=1, timeout_s=0.05 if kind == "timeout" else 600
        )
    assert len(attempts) == (2 if kind in {"eof", "timeout"} else 1)


@pytest.mark.anyio
async def test_classification_failure_identifies_stage_and_timeout():
    def respond(request):
        raise httpx.ReadTimeout("provider did not respond")

    with pytest.raises(
        DecisionsAnnotationError,
        match=r"classification request failed: APITimeoutError .*timeout_s=600",
    ):
        await annotate(respond, timeout_s=600)


@pytest.mark.anyio
async def test_stream_cancellation_is_not_retried():
    attempts = []

    def respond(request):
        if request.url.path.endswith("/decisions"):
            return classification()
        attempts.append(request)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await annotate(respond)
    assert len(attempts) == 1
