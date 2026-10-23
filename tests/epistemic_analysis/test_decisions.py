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
    EpistemicReport,
    MotifAssessment,
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


def test_motif_registry_preserves_original_pattern_names_and_kinds():
    # Read literals without importing the old analysis script or loading .env.
    source = Path(__file__).parents[2] / "reasoning_reports" / "analyze.py"
    definitions = {
        node.target.id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"SUBGRAPH_DESCRIPTIONS", "ANTIPATTERN_DESCRIPTIONS"}
    }
    # Definitions may become sharper without breaking stored pattern identities.
    assert {motif.name: motif.kind for motif in MOTIFS} == {
        **dict.fromkeys(definitions["SUBGRAPH_DESCRIPTIONS"], "productive"),
        **dict.fromkeys(definitions["ANTIPATTERN_DESCRIPTIONS"], "antipattern"),
    }
    assert len(MOTIFS) == 18


@pytest.mark.parametrize("status", STATUS_DESCRIPTIONS)
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("evidence_ids", [(), (1, 2)])
def test_historical_selection_does_not_override_current_status(
    status, selected, evidence_ids
):
    assessment = MotifAssessment(
        name="unsupported_judgment",
        kind="antipattern",
        family="evidence_handling",
        status=status,
        confidence=0.8,
        probabilities={status: 1.0},
        selected=selected,
        explanation_supported=True,
        explanation=(
            "The current proposal still claims five DBE against the formula."
            if status == "active"
            else "The earlier proposal claimed five DBE; it was corrected to four."
        ),
        evidence_message_ids=evidence_ids,
    )
    report = EpistemicReport(
        model="mock",
        assessed_at="",
        message_count=2,
        trajectory_sha256="",
        assessments=(assessment,),
        usage={},
        raw_response={},
    )
    # Even a supported explanation of an earlier error cannot make a nonactive
    # assessment an outstanding problem. Active findings still reach the actor.
    actionable = status == "active" and selected
    assert report.present_antipatterns == ((assessment,) if actionable else ())
    assert ("unsupported_judgment" in report.format_for_agent()) == actionable
    assert report.to_dict()["assessments"][0]["explanation_supported"] is True


def test_descriptive_fixed_belief_does_not_hide_a_real_active_problem():
    def assessment(name):
        return MotifAssessment(
            name=name,
            kind="antipattern",
            family="experimental_strategy",
            status="active",
            confidence=0.9,
            probabilities={"active": 1.0},
            selected=True,
            explanation_supported=True,
            explanation="The observations support this classification.",
            evidence_message_ids=(1,),
        )

    fixed = assessment("fixed_belief_trace")
    contradiction = assessment("contradiction_without_repair")
    report = EpistemicReport(
        model="mock",
        assessed_at="",
        message_count=1,
        trajectory_sha256="",
        assessments=(fixed, contradiction),
        usage={},
        raw_response={},
    )
    assert report.present_antipatterns == (contradiction,)
    assert "fixed_belief_trace" not in report.format_for_agent()
    assert "contradiction_without_repair" in report.format_for_agent()
    assert report.to_dict()["assessments"][0]["name"] == "fixed_belief_trace"


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
    assert rendered_report == "No present unconstructive epistemic patterns detected."
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
    assert (
        report.format_for_agent()
        == "No present unconstructive epistemic patterns detected."
    )
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
    for i in (7, 8):
        payload["answers"][i] = decision_payload("active")["answers"][i]
    reason = (
        "The solubility test returned an insoluble sample, contradicting H1; "
        "the agent acknowledged that result and revised H1 to H2."
    )
    explanations = {
        MOTIFS[0].name: {
            "supported": True,
            "explanation": reason,
        },
        MOTIFS[4].name: {
            "supported": False,
            "explanation": "No subsequent test of H2 occurred.",
        },
        MOTIFS[7].name: {
            "supported": True,
            "explanation": "The revised H2 is being treated as settled without a test.",
        },
        MOTIFS[8].name: {
            "supported": False,
            "explanation": "The agent interpreted the insoluble result.",
        },
    }

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/decisions":
            return httpx.Response(200, json=payload)
        assert request.url.path == "/v1/responses"
        names = body["text"]["format"]["schema"]["required"]
        return httpx.Response(
            200, json=explanation_payload({name: explanations[name] for name in names})
        )

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
    assert len(requests) == 1 + len(explanations)
    requested_names = []
    for _, body in requests[1:]:
        assert body["model"] == "explanation-model"
        assert body["store"] is False
        assert body["text"]["format"]["strict"] is True
        data = json.loads(body["input"])
        assert data["messages"] == [messages[0], messages[1], messages[3]]
        schema = body["text"]["format"]["schema"]["properties"]
        for item in schema.values():
            assert set(item["properties"]) == {"supported", "explanation"}
            assert set(item["required"]) == {"supported", "explanation"}
        assert len(data["candidate_patterns"]) == 1
        assert "graph_description" in data["candidate_patterns"][0]
        requested_names.append(data["candidate_patterns"][0]["name"])
    assert set(requested_names) == set(explanations)
    assert all("STALE_REPORT" not in json.dumps(body) for _, body in requests)
    rendered = report.format_for_agent()
    assert MOTIFS[7].name in rendered
    assert explanations[MOTIFS[7].name]["explanation"] in rendered
    assert all(m.name not in rendered for m in MOTIFS if m != MOTIFS[7])
    assert reason not in rendered
    assert report.assessments[0].explanation == reason  # Productive record retained.
    assert "confidence" not in rendered
    assert "probabilit" not in rendered
    assert all(m.graph_description not in rendered for m in MOTIFS)
    assert report.assessments[0].evidence_message_ids == ()
    assert report.assessments[4].status == "active"
    assert report.assessments[4].explanation_supported is False
    assert len(report.to_dict()["assessments"]) == 18
    assert report.raw_response == payload
    assert report.raw_explanation_response is None
    assert len(report.raw_explanation_responses) == 4
    assert all(
        r["usage"]["input_tokens"] == 19 for r in report.raw_explanation_responses
    )
    assert report.usage == {
        "input_tokens": 107,
        "output_tokens": 46,
        "reasoning_tokens": 13,
        "llm_calls": 5,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "extra",
        "blank",
        "bad_supported",
        "bad_explanation",
        "missing_supported",
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
        }
    }
    item = explanations[MOTIFS[0].name]
    if invalid == "missing":
        explanations.clear()
    elif invalid == "extra":
        explanations["invented_pattern"] = dict(item)
    elif invalid == "blank":
        item["explanation"] = " "
    elif invalid == "bad_supported":
        item["supported"] = "true"
    elif invalid == "bad_explanation":
        item["explanation"] = 123
    elif invalid == "missing_supported":
        item.pop("supported")
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
