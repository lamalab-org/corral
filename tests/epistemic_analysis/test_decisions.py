"""Decisions wire contract and independent motif reporting; no live API calls."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from corral.epistemic_analysis.decisions import (
    REPORT_MESSAGE_NAME,
    DecisionsAnnotationError,
    DecisionsAnnotator,
)
from corral.epistemic_analysis.motifs import (
    MOTIFS,
    RELATION_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def decision_payload(status="absent"):
    return {
        "model": "gpt-6-luna",
        "answers": [
            {
                "type": "choice",
                "name": motif.name,
                "choice": status,
                "confidence": 0.8,
                "probabilities": [
                    {"value": option, "probability": 0.8 if option == status else 0.05}
                    for option in STATUS_DESCRIPTIONS
                ],
            }
            for motif in MOTIFS
        ],
        "usage": {
            "input_tokens": 31,
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def explanation_payload(explanations, *, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "status": status,
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(explanations),
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 19,
            "output_tokens": 11,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def test_motif_registry_matches_original_final_graph_aggregation():
    # Read literals without importing the old analysis script or loading .env.
    source = Path(__file__).parents[2] / "reasoning_reports" / "analyze.py"
    definitions = {
        node.target.id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"SUBGRAPH_DESCRIPTIONS", "ANTIPATTERN_DESCRIPTIONS"}
    }
    assert {motif.name: motif.description for motif in MOTIFS} == {
        **definitions["SUBGRAPH_DESCRIPTIONS"],
        **definitions["ANTIPATTERN_DESCRIPTIONS"],
    }
    assert len(MOTIFS) == 18


@pytest.mark.anyio
async def test_real_sdk_transport_batches_motifs_and_preserves_observations():
    requests = []

    def respond(request):
        assert request.url.path == "/v1/decisions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=decision_payload("emerging"))

    messages = [
        {"role": "system", "content": "private scaffold"},
        {
            "role": "assistant",
            "content": "Test H",
            "tool_calls": [
                {
                    "id": "test-1",
                    "type": "function",
                    "function": {"name": "measure", "arguments": '{"sample":"a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "test-1", "content": "result: 17"},
        {"role": "user", "name": REPORT_MESSAGE_NAME, "content": "old assessment"},
    ]
    original = copy.deepcopy(messages)
    async with AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        report = await DecisionsAnnotator(client=client).annotate(
            messages, task="solve X"
        )
        assert not client.is_closed()
    assert messages == original
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gpt-6-luna"
    assert [question["name"] for question in request["questions"]] == [
        m.name for m in MOTIFS
    ]
    assert all(question["type"] == "choice" for question in request["questions"])
    rendered_report = report.format_for_agent()
    for motif, question in zip(MOTIFS, request["questions"], strict=True):
        assert motif.graph_description in question["instructions"]
        assert motif.description in question["instructions"]
        assert motif.criteria in question["instructions"]
        assert motif.name not in rendered_report
    for description in RELATION_DESCRIPTIONS.values():
        assert description in request["input"]
    assert "You do not\nreceive previous classifier results" in request["input"]
    assert "Reassess every motif from the supplied\ntrajectory" in request["input"]
    convergence = next(
        q for q in request["questions"] if q["name"] == "convergent_multi_test_evidence"
    )
    assert "at least three distinct tests" in convergence["instructions"]
    assert "result: 17" in request["input"]
    assert "test-1" in request["input"]
    assert "solve X" in request["input"]
    assert "old assessment" not in request["input"]
    assert "private scaffold" not in request["input"]
    assert report.message_count == 2
    assert report.usage["reasoning_tokens"] == 1
    assert all(a.status == "emerging" for a in report.assessments)
    assert rendered_report == "No present epistemic patterns detected."
    assert report.usage["llm_calls"] == 1
    assert report.raw_explanation_response is None
    assert json.loads(json.dumps(report.to_dict()))["raw_response"] == decision_payload(
        "emerging"
    )


@pytest.mark.anyio
async def test_native_sdk_helper_and_refusal_are_supported():
    payload = decision_payload()
    payload["answers"][0] = {"type": "refusal", "name": MOTIFS[0].name}
    create = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda **kwargs: payload)
    )
    client = SimpleNamespace(decisions=SimpleNamespace(create=create))
    report = await DecisionsAnnotator(client=client).annotate([])
    create.assert_awaited_once()
    assert report.assessments[0].status == "uncertain"
    assert report.assessments[0].refused is True
    assert report.assessments[0].confidence is None
    assert report.format_for_agent() == "No present epistemic patterns detected."
    assert report.assessments[1].status == "absent"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid",
    ["missing", "reordered", "type", "choice", "confidence", "sum", "duplicate"],
)
async def test_invalid_decisions_are_never_reported_as_absent(invalid):
    payload = decision_payload()
    answer = payload["answers"][0]
    if invalid == "missing":
        payload["answers"].pop()
    elif invalid == "reordered":
        payload["answers"].reverse()
    elif invalid == "type":
        answer["type"] = "predicate"
    elif invalid == "choice":
        answer["choice"] = "invented"
    elif invalid == "confidence":
        answer["confidence"] = float("nan")
    elif invalid == "sum":
        answer["probabilities"][0]["probability"] = 0.2
    elif invalid == "duplicate":
        answer["probabilities"][1]["value"] = "absent"
    client = SimpleNamespace(post=AsyncMock(return_value=payload))
    with pytest.raises(DecisionsAnnotationError):
        await DecisionsAnnotator(client=client).annotate([])


@pytest.mark.anyio
async def test_api_failure_propagates_without_a_fake_report():
    client = SimpleNamespace(
        post=AsyncMock(side_effect=RuntimeError("service unavailable"))
    )
    with pytest.raises(RuntimeError, match="service unavailable"):
        await DecisionsAnnotator(client=client).annotate([])


@pytest.mark.anyio
async def test_only_active_evidenced_patterns_reach_agent_with_reasons():
    requests = []
    payload = decision_payload()
    # Exercise every status in one report, plus an active false positive.
    for i, status in enumerate(
        ["active", "emerging", "resolved", "uncertain", "active"]
    ):
        payload["answers"][i] = decision_payload(status)["answers"][i]
    reason = (
        "The solubility test returned an insoluble sample, contradicting H1; "
        "the agent acknowledged that result and revised H1 to H2."
    )
    explanations = {
        MOTIFS[0].name: {
            "supported": True,
            "explanation": reason,
            "evidence_message_ids": [1, 2, 3],
        },
        MOTIFS[4].name: {
            "supported": False,
            "explanation": "No subsequent test of H2 occurred.",
            "evidence_message_ids": [3],
        },
    }

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/decisions":
            return httpx.Response(200, json=payload)
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json=explanation_payload(explanations))

    messages = [
        {"role": "assistant", "content": "H1 predicts a soluble sample; test it."},
        {"role": "tool", "content": "Solubility test: insoluble."},
        {"role": "user", "name": REPORT_MESSAGE_NAME, "content": "STALE_REPORT"},
        {"role": "assistant", "content": "That contradicts H1; revise to H2."},
    ]
    async with AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        report = await DecisionsAnnotator(
            client=client, explanation_model="explanation-model"
        ).annotate(messages, task="Test solubility")
    assert len(requests) == 2
    body = requests[1][1]
    assert body["model"] == "explanation-model"
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    data = json.loads(body["input"])
    assert [m["message_id"] for m in data["messages"]] == [1, 2, 3]
    assert {m["name"] for m in data["candidate_patterns"]} == set(explanations)
    assert all("graph_description" in m for m in data["candidate_patterns"])
    assert all("STALE_REPORT" not in json.dumps(body) for _, body in requests)
    rendered = report.format_for_agent()
    assert MOTIFS[0].name in rendered
    assert reason in rendered
    assert all(m.name not in rendered for m in MOTIFS[1:])
    assert "confidence" not in rendered
    assert "probabilit" not in rendered
    assert all(m.graph_description not in rendered for m in MOTIFS)
    assert report.assessments[0].evidence_message_ids == (1, 2, 3)
    assert report.assessments[4].status == "active"
    assert report.assessments[4].explanation_supported is False
    assert len(report.to_dict()["assessments"]) == 18
    assert report.raw_response == payload
    assert report.raw_explanation_response["usage"]["input_tokens"] == 19
    assert report.usage == {
        "input_tokens": 50,
        "output_tokens": 13,
        "reasoning_tokens": 4,
        "llm_calls": 2,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "extra",
        "blank",
        "no_evidence",
        "bad_id",
        "bool_id",
        "duplicate_id",
        "refusal",
        "incomplete",
    ],
)
async def test_invalid_explanations_are_not_shown_as_evidence(invalid):
    payload = decision_payload()
    payload["answers"][0] = decision_payload("active")["answers"][0]
    explanations = {
        MOTIFS[0].name: {
            "supported": True,
            "explanation": "Observation supports the revision.",
            "evidence_message_ids": [1],
        }
    }
    item = explanations[MOTIFS[0].name]
    if invalid == "missing":
        explanations.clear()
    elif invalid == "extra":
        explanations["invented_pattern"] = dict(item)
    elif invalid == "blank":
        item["explanation"] = " "
    elif invalid in {"no_evidence", "bad_id", "bool_id", "duplicate_id"}:
        item["evidence_message_ids"] = {
            "no_evidence": [],
            "bad_id": [2],
            "bool_id": [True],
            "duplicate_id": [1, 1],
        }[invalid]
    output = json.dumps(explanations) if invalid != "refusal" else ""
    create = AsyncMock(
        return_value=SimpleNamespace(
            status="incomplete" if invalid == "incomplete" else "completed",
            output_text=output,
            model_dump=lambda **kwargs: explanation_payload(explanations),
        )
    )
    client = SimpleNamespace(
        post=AsyncMock(return_value=payload), responses=SimpleNamespace(create=create)
    )
    with pytest.raises(DecisionsAnnotationError):
        await DecisionsAnnotator(client=client).annotate(
            [{"role": "assistant", "content": "A scientific observation."}]
        )
