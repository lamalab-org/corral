"""End-to-end submission review with real commit-backed sessions."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.agents.test_action_interface import make_session, tool_call
from tests.agents.test_epistemic_agent import annotator_with_transport, response
from tests.epistemic_analysis.test_decisions import (
    decision_payload,
    explanation_payload,
)

from corral.agents.epistemic import EpistemicAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.transition import ToolRecoveryPending
from corral.epistemic_analysis.decision_thresholds import CALIBRATED_THRESHOLD_POLICY
from corral.epistemic_analysis.decisions import (
    REPORT_MESSAGE_NAME,
    SUBMISSION_REVIEW_MESSAGE_NAME,
    scientific_messages,
)

pytestmark = pytest.mark.usefixtures("session_stores")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def trajectory(body):
    return json.loads(
        body["input"].split("Trajectory data (chronological JSON):\n")[1]
    )["messages"]


def selected_payload(name, status="active"):
    payload = decision_payload()
    for i, answer in enumerate(payload["answers"]):
        if answer["name"] == name:
            payload["answers"][i] = decision_payload(status)["answers"][i]
    return payload


async def explain(**kwargs):
    explanations = {
        name: {
            "supported": True,
            "explanation": "The proposed final answer has not been validated.",
        }
        for name in kwargs["text"]["format"]["schema"]["required"]
    }
    return SimpleNamespace(
        status="completed",
        output_text=json.dumps(explanations),
        model_dump=lambda **_: explanation_payload(explanations),
    )


@pytest.mark.anyio
async def test_submission_is_held_then_validated_and_reassessed(monkeypatch):
    calls, observations, sent = [], [], []
    session = await make_session(calls, max_iterations=3)

    async def decide(*_, **kwargs):
        messages = trajectory(kwargs["body"])
        observations.append(deepcopy(messages))
        assert not session.state.is_terminal
        assert all(
            m.get("name") not in {REPORT_MESSAGE_NAME, SUBMISSION_REVIEW_MESSAGE_NAME}
            for m in messages
        )
        if any(m.get("content") == "measured:validation" for m in messages):
            return decision_payload()
        return selected_payload("untested_claim")

    replies = iter(
        [
            response(
                tool_call("submit_answer", '{"answer":"42"}', "unvalidated"),
                tool_call("measure", '{"sample":"must-not-run"}', "late"),
                content="I think the answer is 42, without testing it.",
            ),
            response(
                tool_call("measure", '{"sample":"validation"}', "validate"),
                content="Check the proposed answer before committing.",
            ),
            response(
                tool_call("submit_answer", '{"answer":"43"}', "validated"),
                content="The validation contradicts 42 and supports 43.",
            ),
        ]
    )

    async def actor(**kwargs):
        assert session.state.submission is None
        sent.append(deepcopy(kwargs["messages"]))
        return next(replies)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    agent = EpistemicAgent(annotator=annotator_with_transport(decide, explain))
    result = await agent.run_session(session)
    assert result.status == "completed", result.error
    assert session.state.submission == "43"
    assert calls == ["validation"]
    assert session.state.tool_statistics == {"measure": 1, "submit_answer": 1}
    assert "unvalidated" not in session.state.actions
    assert observations[0][-1]["content"].endswith("not yet submitted):\n42")
    assert "without testing" in observations[0][-1]["content"]
    assert observations[-1][-1]["content"].endswith("not yet submitted):\n43")
    assert "validation contradicts 42" in observations[-1][-1]["content"]
    assert len(observations) == 3  # Two submission reviews + post-validation report.
    assert sent[1][-1]["name"] == SUBMISSION_REVIEW_MESSAGE_NAME
    assert "Submission held" in sent[1][-1]["content"]
    assert "untested_claim" in sent[1][-2]["content"]
    assert "The proposed final answer has not been validated." in sent[1][-2]["content"]
    # Deferred tool calls are preserved as proposals, never dangling provider calls.
    assert not any(
        call["id"] in {"unvalidated", "late"}
        for m in sent[1]
        for call in m.get("tool_calls", [])
    )
    assert all("metadata" not in m for messages in sent for m in messages)
    prefix = []
    for message in session.messages:
        report = message.get("metadata", {}).get("epistemic_report")
        if report:
            assert report["message_count"] == len(scientific_messages(prefix))
            assert len(report["assessments"]) == 18
        prefix.append(message)
    assert result.usage.llm_calls == session.state.usage.llm_calls == 7
    assert result.usage.input_tokens == session.state.usage.input_tokens
    assert result.usage.output_tokens == session.state.usage.output_tokens


@pytest.mark.anyio
async def test_repeated_unvalidated_submission_cannot_bypass_gate_or_budget(
    monkeypatch,
):
    session = await make_session([], max_iterations=3)
    model = AsyncMock(
        side_effect=[
            response(tool_call("submit_answer", '{"answer":"42"}', f"attempt-{i}"))
            for i in range(3)
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    post = AsyncMock(return_value=selected_payload("untested_claim"))
    result = await EpistemicAgent(
        annotator=annotator_with_transport(post, explain)
    ).run_session(session)
    assert result.status == "iteration_limit"
    assert session.state.submission is None
    assert not session.state.actions
    assert model.await_count == post.await_count == 3
    assert result.usage.llm_calls == session.state.usage.llm_calls == 9
    assert sum(m.get("name") == "submission_proposal" for m in session.messages) == 3
    assert sum(m.get("name") == REPORT_MESSAGE_NAME for m in session.messages) == 3


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("pattern", "supported"),
    [
        ("refutation_driven_belief_revision", True),
        ("fixed_belief_trace", True),
        ("untested_claim", False),
    ],
)
async def test_productive_descriptive_or_unsupported_patterns_do_not_hold_submission(
    monkeypatch, pattern, supported
):
    async def explanation(**kwargs):
        items = {
            name: {
                "supported": supported,
                "explanation": "This is a saved explanation.",
            }
            for name in kwargs["text"]["format"]["schema"]["required"]
        }
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(items),
            model_dump=lambda **_: explanation_payload(items),
        )

    model = AsyncMock(
        return_value=response(tool_call("submit_answer", '{"answer":"42"}', "done"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    session = await make_session([], max_iterations=1)
    result = await EpistemicAgent(
        annotator=annotator_with_transport(
            AsyncMock(return_value=selected_payload(pattern)), explanation
        )
    ).run_session(session)
    assert result.status == "completed", result.error
    assert session.state.submission == "42"
    report = next(m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME)
    assert pattern not in report["content"]
    saved = next(
        a
        for a in report["metadata"]["epistemic_report"]["assessments"]
        if a["name"] == pattern
    )
    assert saved["selected"] is True
    assert saved["explanation_supported"] is supported
    assert not any(
        m.get("name") == SUBMISSION_REVIEW_MESSAGE_NAME for m in session.messages
    )
    assert result.usage.llm_calls == session.state.usage.llm_calls == 3


@pytest.mark.anyio
async def test_final_review_failure_preserves_proposal_without_submitting(monkeypatch):
    session = await make_session([], max_iterations=1)
    model = AsyncMock(
        return_value=response(
            tool_call("submit_answer", '{"answer":"42"}', "done"),
            content="Final justification.",
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    result = await EpistemicAgent(
        annotator=SimpleNamespace(
            annotate=AsyncMock(side_effect=ValueError("bad evidence"))
        )
    ).run_session(session)
    assert result.status == "agent_failure"
    assert result.error == "bad evidence"
    assert session.state.submission is None
    assert not session.state.actions
    proposal = next(
        m for m in session.messages if m.get("name") == "submission_proposal"
    )
    assert proposal["metadata"]["submission_proposal"]["id"] == "done"
    assert "Final justification." in proposal["content"]
    assert result.usage.llm_calls == session.state.usage.llm_calls == 1


@pytest.mark.anyio
async def test_tools_before_submission_are_observed_and_usage_counted_once(monkeypatch):
    calls = []
    session = await make_session(calls, max_iterations=1)
    post = AsyncMock(return_value=decision_payload())
    model = AsyncMock(
        return_value=response(
            tool_call("measure", '{"sample":"a"}', "measure"),
            tool_call("submit_answer", '{"answer":"42"}', "submit"),
            tool_call("measure", '{"sample":"must-not-run"}', "late"),
            content="Test a and propose 42.",
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    result = await EpistemicAgent(annotator=annotator_with_transport(post)).run_session(
        session
    )
    assert result.status == "completed", result.error
    assert calls == ["a"]
    observed = trajectory(post.await_args.kwargs["body"])
    assert any(m.get("content") == "measured:a" for m in observed)
    assert not any(c["id"] == "late" for m in observed for c in m.get("tool_calls", []))
    assert observed[-1]["content"].endswith("not yet submitted):\n42")
    assert result.usage.llm_calls == session.state.usage.llm_calls == 2
    assert result.usage.input_tokens == session.state.usage.input_tokens == 38


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["absent", "resolved", "emerging", "uncertain"])
async def test_default_threshold_nonactive_candidate_cannot_hold_submission(
    monkeypatch, status
):
    target = "uninformative_test"  # Default threshold is zero.
    explained_names = []

    async def check(**kwargs):
        candidates = kwargs["text"]["format"]["schema"]["required"]
        assert len(candidates) == 1
        explained_names.extend(candidates)
        explanations = {
            name: {
                "supported": name == target,
                "explanation": "An earlier IR call failed and was excluded from inference.",
            }
            for name in candidates
        }
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(explanations),
            model_dump=lambda **_: explanation_payload(explanations),
        )

    # Use the agent's default annotator configuration; supply only mock transport.
    agent = EpistemicAgent(epistemic_api_key="test-key")
    agent.annotator.client = annotator_with_transport(
        AsyncMock(return_value=selected_payload(target, status)), check
    ).client
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=response(tool_call("submit_answer", '{"answer":"42"}', "done"))
        ),
    )
    session = await make_session([], max_iterations=1)
    result = await agent.run_session(session)
    assert result.status == "completed", result.error
    assert target in explained_names
    assert session.state.submission == "42"
    report = next(m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME)
    assert target not in report["content"]
    saved = report["metadata"]["epistemic_report"]
    assert saved["selection_policy"]["name"] == CALIBRATED_THRESHOLD_POLICY
    assessment = next(a for a in saved["assessments"] if a["name"] == target)
    assert assessment["status"] == status
    assert assessment["selection_threshold"] == 0
    assert assessment["selected"] is True
    assert assessment["explanation_supported"] is True
    assert not any(
        m.get("name") == SUBMISSION_REVIEW_MESSAGE_NAME for m in session.messages
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("pattern", "repair"),
    [
        (
            "unsupported_judgment",
            "I correct the earlier arithmetic: the formula has four DBE, not five.",
        ),
        (
            "contradiction_without_repair",
            "I withdraw the contradicted peak assignment and narrow the claim accordingly.",
        ),
    ],
)
async def test_active_error_holds_until_repaired_even_when_history_remains_selected(
    monkeypatch, pattern, repair
):
    session = await make_session([], max_iterations=2)
    model = AsyncMock(
        side_effect=[
            response(
                tool_call("submit_answer", '{"answer":"42"}', "unrepaired"),
                content="The proposed answer still relies on the disputed claim.",
            ),
            response(
                tool_call("submit_answer", '{"answer":"42"}', "repaired"),
                content=repair,
            ),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    post = AsyncMock(
        side_effect=[
            selected_payload(pattern, "active"),
            selected_payload(pattern, "resolved"),
        ]
    )
    # The historical threshold policy deliberately selects both assessments.
    # A stale positive explanation on the repaired issue must not reopen it.
    agent = EpistemicAgent(epistemic_api_key="test-key")
    agent.annotator.client = annotator_with_transport(post, explain).client
    result = await agent.run_session(session)
    assert result.status == "completed", result.error
    assert session.state.submission == "42"
    assert "unrepaired" not in session.state.actions
    assert "repaired" in session.state.actions
    assert session.state.tool_statistics == {"submit_answer": 1}
    holds = [
        m for m in session.messages if m.get("name") == SUBMISSION_REVIEW_MESSAGE_NAME
    ]
    assert len(holds) == 1
    reports = [m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert "The proposed final answer has not been validated." in reports[0]["content"]
    assert pattern in reports[0]["content"]
    assert pattern not in reports[1]["content"]
    assert (
        reports[1]["content"]
        == "No present unconstructive epistemic patterns detected."
    )
    for report, status in zip(reports, ("active", "resolved"), strict=True):
        assessment = next(
            a
            for a in report["metadata"]["epistemic_report"]["assessments"]
            if a["name"] == pattern
        )
        assert assessment["status"] == status
        assert assessment["selected"] is True
        assert assessment["explanation_supported"] is True
    assert model.await_count == post.await_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("epistemic", [False, True])
async def test_native_submission_recovery_still_propagates(monkeypatch, epistemic):
    agent = (
        EpistemicAgent(
            annotator=annotator_with_transport(
                AsyncMock(return_value=decision_payload())
            )
        )
        if epistemic
        else ToolCallingAgent()
    )
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=response(
                tool_call("submit_answer", '{"answer":"42"}', "pending")
            )
        ),
    )
    session = await make_session([], max_iterations=1)
    monkeypatch.setattr(
        session, "execute", AsyncMock(side_effect=ToolRecoveryPending("Resume tool"))
    )
    with pytest.raises(ToolRecoveryPending, match="Resume tool"):
        await agent.run_session(session)
    assert session.state.submission is None
