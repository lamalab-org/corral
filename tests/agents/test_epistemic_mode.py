"""Opt-in feedback and submission review across both actor protocols."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from promptstore import PromptStore
from tests.agents.test_action_interface import make_session, tool_call
from tests.agents.test_epistemic_agent import annotator_with_transport, response
from tests.agents.test_epistemic_submission import explain, selected_payload, trajectory
from tests.agents.test_tool_calling_agent import Session
from tests.epistemic_analysis.test_decisions import decision_payload

from corral.agents.react import ReActAgent
from corral.agents.schema import SURRENDER_SENTINEL, BudgetExhaustedError
from corral.agents.tool_calling import ToolCallingAgent
from corral.cli import create_agent
from corral.epistemic_analysis.decisions import (
    REPORT_MESSAGE_NAME,
    SUBMISSION_REVIEW_MESSAGE_NAME,
    scientific_messages,
)

pytestmark = pytest.mark.usefixtures("session_stores")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(params=[ToolCallingAgent, ReActAgent], ids=["tool-calling", "react"])
def agent_type(request):
    return request.param


def reply(agent_type, *calls, content="Explain the observations.", **fields):
    result = response(*calls, content=content)
    if agent_type is ReActAgent:
        result.tool_calls = []
        result.content = (f"<thought>{content}</thought>" if content else "") + "".join(
            f"<action>{call.function.name}</action>"
            f"<action_input>{call.function.arguments}</action_input>"
            for call in calls
        )
    for key, value in fields.items():
        setattr(result, key, value)
    return result


@pytest.mark.parametrize("custom_prompts", [False, True])
def test_mode_preserves_actor_prompts_and_tool_descriptions(
    monkeypatch, agent_type, custom_prompts
):
    monkeypatch.setattr("corral.agents.base_agent.PromptStore", PromptStore)
    prompts = (
        {
            "system_prompt": "Follow the supplied task instructions.",
            "user_prompt": (
                "Solve this task: {{task_guide}}\n"
                "{{examples}}\n{{surrender_instructions}}"
            ),
        }
        if custom_prompts
        else {}
    )
    base = agent_type(**prompts)
    enabled = agent_type(epistemic=True, annotator=AsyncMock(), **prompts)
    session = Session()
    original_tools = deepcopy(session.tools)

    assert enabled.system_prompt == base.system_prompt
    assert enabled._initial_messages(session) == base._initial_messages(session)
    assert enabled._tools(session) == base._tools(session)
    assert enabled._text_only_feedback() == base._text_only_feedback()
    assert session.tools == original_tools


@pytest.mark.anyio
async def test_review_observes_completed_tools_and_holds_until_validated(
    monkeypatch, agent_type
):
    calls, assessed, sent = [], [], []
    session = await make_session(calls, max_iterations=3)

    async def decide(*_, **kwargs):
        observed = trajectory(kwargs["body"])
        assessed.append(deepcopy(observed))
        assert session.state.submission is None
        assert not any(
            m.get("name") in {REPORT_MESSAGE_NAME, SUBMISSION_REVIEW_MESSAGE_NAME}
            for m in observed
        )
        assert "must-not-run" not in json.dumps(observed)
        return (
            decision_payload()
            if any(m.get("content") == "measured:validation" for m in observed)
            else selected_payload("untested_claim")
        )

    replies = iter(
        [
            reply(
                agent_type,
                tool_call("measure", '{"sample":"a"}', "measure"),
                tool_call("submit_answer", '{"answer":"42"}', "held"),
                tool_call("measure", '{"sample":"must-not-run"}', "late"),
                content="Test sample a; my unvalidated answer is 42.",
            ),
            reply(
                agent_type,
                tool_call("measure", '{"sample":"validation"}', "validate"),
            ),
            reply(
                agent_type,
                tool_call("submit_answer", '{"answer":"43"}', "accepted"),
                content="Validation supports 43.",
            ),
        ]
    )

    async def actor(**kwargs):
        if not sent:
            assert assessed == []  # No annotation before the initial actor turn.
        sent.append(deepcopy(kwargs["messages"]))
        return next(replies)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(decide, explain)
    ).run_session(session)

    assert outcome.status == "completed", outcome.error
    assert outcome.answer == session.state.submission == "43"
    assert calls == ["a", "validation"]
    assert session.state.tool_statistics == {"measure": 2, "submit_answer": 1}
    assert len(assessed) == 3  # Two submissions plus the post-validation feedback.
    assert any(m.get("content") == "measured:a" for m in assessed[0])
    assert "unvalidated answer" in json.dumps(assessed[0])
    assert assessed[0][-1]["content"].endswith("not yet submitted):\n42")
    assert "Validation supports 43" in assessed[-1][-1]["content"]
    assert sent[1][-1]["name"] == SUBMISSION_REVIEW_MESSAGE_NAME
    assert "must-not-run" not in json.dumps(sent[1])
    assert all("metadata" not in m for turn in sent for m in turn)
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 7
    assert outcome.usage.input_tokens == session.state.usage.input_tokens
    assert outcome.usage.output_tokens == session.state.usage.output_tokens
    prefix = []
    for message in session.messages:
        report = message.get("metadata", {}).get("epistemic_report")
        if report:
            assert report["message_count"] == len(scientific_messages(prefix))
        prefix.append(message)


@pytest.mark.anyio
async def test_repeated_submissions_cannot_bypass_review(monkeypatch, agent_type):
    session = await make_session([], max_iterations=3)
    actor = AsyncMock(
        side_effect=[
            reply(agent_type, tool_call("submit_answer", '{"answer":"42"}', str(i)))
            for i in range(3)
        ]
    )
    post = AsyncMock(return_value=selected_payload("untested_claim"))
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(post, explain)
    ).run_session(session)
    assert outcome.status == "iteration_limit"
    assert session.state.submission is None
    assert not session.state.actions
    assert actor.await_count == post.await_count == 3
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 9


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["absent", "resolved", "emerging", "uncertain"])
async def test_nonactive_patterns_allow_immediate_submission(
    monkeypatch, agent_type, status
):
    session = await make_session([], max_iterations=1)
    post = AsyncMock(return_value=selected_payload("untested_claim", status))
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=reply(
                agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")
            )
        ),
    )
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(post, explain)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert session.state.submission == "42"
    post.assert_awaited_once()
    assert outcome.usage.llm_calls == session.state.usage.llm_calls


@pytest.mark.anyio
async def test_mode_is_disabled_by_default(monkeypatch, agent_type):
    session = await make_session([], max_iterations=1)
    annotator = AsyncMock()
    agent = agent_type(annotator=annotator)
    assert agent._call_kwargs() == {}
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=reply(
                agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")
            )
        ),
    )
    outcome = await agent.run_session(session)
    assert outcome.status == "completed", outcome.error
    annotator.annotate.assert_not_awaited()
    assert not any(m.get("name") == REPORT_MESSAGE_NAME for m in session.messages)
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize(
    "error",
    [RuntimeError("annotation unavailable"), BudgetExhaustedError("budget exhausted")],
)
async def test_annotation_failure_never_submits(
    monkeypatch, agent_type, resumed, error
):
    session = await make_session([], max_iterations=1)
    if resumed:
        await session.record_message(
            {"role": "assistant", "content": "Previous hypothesis."}
        )
    actor = AsyncMock(
        return_value=reply(
            agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    outcome = await agent_type(
        epistemic=True, annotator=AsyncMock(annotate=AsyncMock(side_effect=error))
    ).run_session(session)
    assert outcome.status == (
        "budget_exhausted"
        if isinstance(error, BudgetExhaustedError)
        else "agent_failure"
    )
    assert session.state.submission is None
    assert not session.state.actions
    assert actor.await_count == (0 if resumed else 1)
    assert (
        any(m.get("name") == "submission_proposal" for m in session.messages) != resumed
    )


@pytest.mark.anyio
async def test_resume_keeps_reports_out_of_annotation(monkeypatch, agent_type):
    session = await make_session([])
    await session.record_message(
        {"role": "assistant", "content": "Previous hypothesis."}
    )
    await session.record_message(
        {"role": "user", "name": REPORT_MESSAGE_NAME, "content": "OLD_REPORT"}
    )
    actor = AsyncMock(
        return_value=reply(
            agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    post = AsyncMock(return_value=decision_payload())
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert post.await_count == 2
    assert "OLD_REPORT" in json.dumps(actor.await_args.kwargs["messages"])
    for request in post.await_args_list:
        assert "Previous hypothesis" in request.kwargs["body"]["input"]
        assert "OLD_REPORT" not in request.kwargs["body"]["input"]


@pytest.mark.anyio
async def test_thought_only_turn_is_reviewed(monkeypatch, agent_type):
    session = await make_session([])
    actor = AsyncMock(
        side_effect=[
            reply(agent_type, content="Compare hypotheses before experimenting."),
            reply(agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    post = AsyncMock(return_value=decision_payload())
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert post.await_count == 2
    assert "Compare hypotheses" in post.await_args_list[0].kwargs["body"]["input"]
    assert "No valid action" not in json.dumps(
        actor.await_args_list[1].kwargs["messages"]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("interruption", ["empty", "reasoning", "refusal"])
async def test_retry_does_not_reannotate_unchanged_history(
    monkeypatch, agent_type, interruption
):
    session = await make_session([], max_iterations=3)
    await session.record_message(
        {"role": "assistant", "content": "Previous hypothesis."}
    )
    fields = {"finish_reason": "refusal"} if interruption == "refusal" else {}
    if interruption == "reasoning":
        fields["content_source"] = "reasoning_summary"
    actor = AsyncMock(
        side_effect=[
            reply(
                agent_type,
                content=None if interruption == "empty" else "PRIVATE_OR_REFUSED",
                **fields,
            ),
            reply(agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    post = AsyncMock(return_value=decision_payload())
    outcome = await agent_type(
        epistemic=True, annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert post.await_count == 2  # Resume feedback and final review only.
    assert "PRIVATE_OR_REFUSED" not in json.dumps(actor.await_args.kwargs["messages"])
    assert "PRIVATE_OR_REFUSED" not in json.dumps(list(session.messages), default=str)
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 4


@pytest.mark.anyio
async def test_shared_instance_keeps_sessions_separate(monkeypatch, agent_type):
    sessions = [await make_session([]), await make_session([])]
    for index, session in enumerate(sessions):
        await session.record_message(
            {"role": "assistant", "content": f"HYPOTHESIS_{index}"}
        )
    inputs = []

    async def decide(*_, **kwargs):
        inputs.append(kwargs["body"]["input"])
        await asyncio.sleep(0)
        return decision_payload()

    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=reply(
                agent_type, tool_call("submit_answer", '{"answer":"42"}', "done")
            )
        ),
    )
    agent = agent_type(epistemic=True, annotator=annotator_with_transport(decide))
    outcomes = await asyncio.gather(*(agent.run_session(s) for s in sessions))
    assert all(o.status == "completed" for o in outcomes)
    assert len(inputs) == 4
    assert all(
        ("HYPOTHESIS_0" in value) != ("HYPOTHESIS_1" in value) for value in inputs
    )


def test_configuration_is_not_forwarded_to_actor(agent_type):
    agent = create_agent(
        "react" if agent_type is ReActAgent else "tool-calling",
        agent_kwargs={
            "epistemic": True,
            "epistemic_model": "classifier-model",
            "epistemic_explanation_model": "explanation-model",
            "epistemic_timeout_s": 42,
            "epistemic_max_retries": 0,
            "epistemic_pattern_thresholds": None,
            "actor_refusal_max_retries": 0,
            "reasoning_effort": "low",
        },
    )
    assert agent.annotator.model == "classifier-model"
    assert agent.annotator.explanation_model == "explanation-model"
    assert agent.annotator.timeout_s == 42
    assert agent.annotator.max_retries == 0
    assert agent.annotator.pattern_thresholds is None
    assert agent._call_kwargs() == {"reasoning_effort": "low"}


@pytest.mark.anyio
async def test_allowed_surrender_bypasses_review(monkeypatch, agent_type):
    session = Session()
    session.surrender_allowed = True
    annotator = AsyncMock()
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=reply(
                agent_type,
                tool_call(
                    "submit_answer", json.dumps({"answer": SURRENDER_SENTINEL}), "done"
                ),
            )
        ),
    )
    outcome = await agent_type(epistemic=True, annotator=annotator).run_session(session)
    assert outcome.status == "surrendered"
    annotator.annotate.assert_not_awaited()
    assert len(session.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("refused", [False, True])
async def test_terminal_empty_or_refused_response_never_executes_tools(
    monkeypatch, agent_type, refused
):
    session = await make_session([], max_iterations=5)
    await session.record_message(
        {"role": "assistant", "content": "Previous hypothesis."}
    )
    actor = AsyncMock(
        return_value=(
            reply(
                agent_type,
                tool_call("submit_answer", '{"answer":"42"}', "refused"),
                finish_reason="refusal",
            )
            if refused
            else reply(agent_type, content=None)
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    post = AsyncMock(return_value=decision_payload())
    outcome = await agent_type(
        epistemic=True,
        annotator=annotator_with_transport(post),
        actor_refusal_max_retries=0,
    ).run_session(session)
    assert outcome.status == ("agent_failure" if refused else "protocol_failure")
    assert session.state.submission is None
    assert not session.state.actions
    assert actor.await_count == (1 if refused else 3)
    post.assert_awaited_once()
    assert (
        outcome.usage.llm_calls
        == session.state.usage.llm_calls
        == actor.await_count + 1
    )
