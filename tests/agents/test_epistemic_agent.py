"""Online annotation order, isolation, accounting, and native submission."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from promptstore import PromptStore
from tests.agents.test_action_interface import make_session, tool_call
from tests.agents.test_tool_calling_agent import Session
from tests.epistemic_analysis.test_decisions import (
    decision_payload,
    explanation_payload,
)

from corral.agents.epistemic import EPISTEMIC_PROMPT, EpistemicAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.action import submit_answer_tool
from corral.epistemic_analysis.decision_thresholds import (
    CALIBRATED_PATTERN_THRESHOLDS,
    CALIBRATED_THRESHOLD_POLICY,
)
from corral.epistemic_analysis.decisions import REPORT_MESSAGE_NAME, DecisionsAnnotator
from corral.epistemic_analysis.motifs import MOTIFS

pytestmark = pytest.mark.usefixtures("session_stores")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def response(*calls, content="Explain observations and the next experiment."):
    return SimpleNamespace(
        content=content,
        tool_calls=list(calls),
        id=None,
        usage={"prompt_tokens": 7, "completion_tokens": 3},
    )


def annotator_with_transport(post, explain=None):
    async def default_explanation(**kwargs):
        explanations = {
            name: {
                "supported": True,
                "explanation": "Measurements measured:a and measured:b support the claim.",
            }
            for name in kwargs["text"]["format"]["schema"]["required"]
        }
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(explanations),
            model_dump=lambda **kwargs: explanation_payload(explanations),
        )

    return DecisionsAnnotator(
        client=SimpleNamespace(
            post=post, responses=SimpleNamespace(create=explain or default_explanation)
        )
    )


@pytest.mark.anyio
async def test_epistemic_prompt_overrides_leave_base_agent_unchanged(monkeypatch):
    # This isolation check needs the real templates, not the suite's default
    # PromptStore mock that returns the same user prompt for every agent.
    monkeypatch.setattr("corral.agents.base_agent.PromptStore", PromptStore)
    base = ToolCallingAgent(model="test-model")
    epistemic = EpistemicAgent(
        model="test-model",
        annotator=annotator_with_transport(AsyncMock(return_value=decision_payload())),
    )
    catalog = (*Session.tools, submit_answer_tool())
    original_catalog = deepcopy(catalog)
    model = AsyncMock()
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)

    # Reuse the same catalog and base agent after the epistemic run to catch
    # prompt or tool-description mutations leaking between agents.
    for agent in (base, epistemic, base):
        model.reset_mock()
        model.side_effect = [
            response(content="Compare the hypotheses before choosing an action."),
            response(tool_call("submit_answer", '{"answer":"X"}', "done")),
        ]
        session = Session()
        session.tools = catalog
        outcome = await agent.run_session(session)
        assert outcome.status == "completed", outcome.error
        initial = model.await_args_list[0].kwargs
        user_prompt = initial["messages"][1]["content"]
        description = next(
            t["function"]["description"]
            for t in initial["tools"]
            if t["function"]["name"] == "submit_answer"
        )
        reminder = next(
            m["content"]
            for m in session.messages
            if m["role"] == "user" and m.get("name") != REPORT_MESSAGE_NAME
        )
        assert session.prompt in user_prompt
        if agent is epistemic:
            assert "Before every tool call, including `submit_answer`" in user_prompt
            assert "A tool call is not required on every turn" in user_prompt
            assert "Do not return a final answer as plain text" not in user_prompt
            assert "a tool call is not required on every turn" in reminder
            assert "visible assistant text" in description
            assert (
                "will only interact with the environment"
                not in initial["messages"][0]["content"]
            )
        else:
            assert "You must think what to do next" in user_prompt
            assert "Do not return a final answer as plain text" in user_prompt
            assert "Before every tool call" not in user_prompt
            assert reminder == (
                "A plain-text answer does not finish the task. Call "
                "submit_answer with the complete answer, or call an "
                "available environment tool."
            )
            assert description == submit_answer_tool()["function"]["description"]
            assert (
                "will only interact with the environment"
                in initial["messages"][0]["content"]
            )
        assert catalog == original_catalog


@pytest.mark.anyio
@pytest.mark.parametrize("with_task_context", [False, True])
async def test_initial_turn_has_no_annotation_or_report(monkeypatch, with_task_context):
    session = await make_session([], max_iterations=1)
    if with_task_context:
        await session.record_message(
            {
                "role": "user",
                "content": "Additional task facts.",
                "metadata": {"internal": "PRIVATE_TASK_METADATA"},
            }
        )
    annotate = AsyncMock(side_effect=AssertionError("No initial assessment allowed"))
    model = AsyncMock(
        return_value=response(tool_call("measure", '{"sample":"a"}', "measure"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert outcome.status == "iteration_limit", outcome.error
    annotate.assert_not_awaited()
    model.assert_awaited_once()
    assert outcome.usage.llm_calls == 1
    assert session.state.usage.llm_calls == 1
    assert all(m.get("name") != REPORT_MESSAGE_NAME for m in session.messages)
    sent = model.await_args.kwargs["messages"]
    assert all(m.get("name") != REPORT_MESSAGE_NAME for m in sent)
    assert all("metadata" not in m for m in sent)
    assert "PRIVATE_TASK_METADATA" not in json.dumps(sent)


@pytest.mark.anyio
async def test_reports_start_after_first_turn_and_observe_all_completed_tools(
    monkeypatch,
):
    events = []
    requests = []
    explanation_requests = []

    async def decide(*args, **kwargs):
        events.append("annotate")
        requests.append(kwargs["body"])
        return decision_payload(
            {1: "emerging", 2: "active"}.get(len(requests), "absent")
        )

    responses = iter(
        [
            response(
                tool_call("measure", '{"sample":"a"}', "a"),
                tool_call("measure", '{"sample":"b"}', "b"),
            ),
            response(tool_call("measure", '{"sample":"c"}', "c")),
            response(
                tool_call("submit_answer", '{"answer":"42"}', "finish"),
                tool_call("measure", '{"sample":"must-not-run"}', "late"),
            ),
        ]
    )
    sent = []

    async def actor(**kwargs):
        events.append("actor")
        sent.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    calls = []
    session = await make_session(calls, max_iterations=3)
    default_explain = annotator_with_transport(decide).client.responses.create

    async def explain(**kwargs):
        events.append("explain")
        explanation_requests.append(kwargs)
        return await default_explain(**kwargs)

    annotator = annotator_with_transport(decide, explain)
    annotator.annotate = AsyncMock(wraps=annotator.annotate)
    agent = EpistemicAgent(model="test-model", annotator=annotator)
    outcome = await agent.run_session(session)

    assert outcome.status == "completed", outcome.error
    assert session.state.submission == "42"
    assert calls == ["a", "b", "c"]
    assert events == (
        ["actor", "annotate", "actor", "annotate"]
        + ["explain"] * len(MOTIFS)
        + ["actor", "annotate"]
    )
    assert "measured:a" in requests[0]["input"]
    assert "measured:b" in requests[0]["input"]
    assert "measured:c" not in requests[0]["input"]
    assert "measured:a" in requests[1]["input"]
    assert "measured:b" in requests[1]["input"]
    assert "measured:c" in requests[1]["input"]
    assert "Explain observations" in requests[1]["input"]
    for request in requests:
        assert REPORT_MESSAGE_NAME not in request["input"]
        assert EPISTEMIC_PROMPT not in request["input"]
    assert len(explanation_requests) == len(MOTIFS)
    assert (
        "No present unconstructive epistemic patterns"
        not in explanation_requests[0]["input"]
    )
    assert REPORT_MESSAGE_NAME not in explanation_requests[0]["input"]
    assert "measured:a" in explanation_requests[0]["input"]
    assert "measured:b" in explanation_requests[0]["input"]
    for call in annotator.annotate.await_args_list:
        assert all(m.get("name") != REPORT_MESSAGE_NAME for m in call.args[0])
    for turn, messages in enumerate(sent):
        assert (
            len([m for m in messages if m.get("name") == REPORT_MESSAGE_NAME]) == turn
        )
        if turn:
            assert messages[-1]["name"] == REPORT_MESSAGE_NAME
        assert (
            "On the first turn, state your initial working hypotheses"
            in messages[0]["content"]
        )
        assert "do not give judgments about them" in messages[0]["content"]
        assert "1. Interpret and judge" in messages[0]["content"]
        assert "2. Assess whether" in messages[0]["content"]
        assert "3. Justify the next" in messages[0]["content"]
        assert all(m.graph_description not in messages[-1]["content"] for m in MOTIFS)
        assert "confidence" not in messages[-1]["content"]
        assert all("metadata" not in message for message in messages)
    assert (
        sent[1][-1]["content"]
        == "No present unconstructive epistemic patterns detected."
    )
    assert "Measurements measured:a and measured:b" in sent[2][-1]["content"]
    previous_report_index = sent[2].index(sent[1][-1])
    assert sent[2][previous_report_index + 1]["role"] == "assistant"
    assert "tool_calls" in sent[2][previous_report_index + 1]
    reports = [m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert len(reports) == 3
    assert len(reports[1]["metadata"]["epistemic_report"]["assessments"]) == 18
    assert outcome.usage.llm_calls == 6 + len(MOTIFS)
    assert outcome.usage.input_tokens == 114 + 19 * len(MOTIFS)
    assert outcome.usage.output_tokens == 15 + 11 * len(MOTIFS)
    assert outcome.usage.reasoning_tokens == 3 + 3 * len(MOTIFS)
    assert session.state.usage.llm_calls == outcome.usage.llm_calls
    assert session.state.usage.input_tokens == outcome.usage.input_tokens


@pytest.mark.anyio
@pytest.mark.parametrize("with_tool", [True, False])
async def test_reasoning_summary_is_excluded_from_public_history(
    monkeypatch, with_tool
):
    session = await make_session([], max_iterations=2)
    first = response(
        *([tool_call("measure", '{"sample":"a"}', "a")] if with_tool else []),
        content="Provider-returned summary of the experiment rationale.",
    )
    first.content_source = "reasoning_summary"
    model = AsyncMock(
        side_effect=[
            first,
            response(
                tool_call("submit_answer", '{"answer":"42"}', "done"),
                content="The measurements support 42.",
            ),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(AsyncMock(return_value=decision_payload()))
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert all(m.get("content") != first.content for m in session.messages)
    sent = model.await_args.kwargs["messages"]
    assert all(m.get("content") != first.content for m in sent)
    assert all("metadata" not in m for m in sent)
    submission = next(
        m
        for m in session.messages
        if (m.get("content") or "").startswith("The measurements support 42.")
    )
    assert submission.get("metadata", {}).get("content_source") != "reasoning_summary"


@pytest.mark.anyio
@pytest.mark.parametrize("empty_content", [None, "", " \n\t"])
async def test_empty_replies_do_not_reassess_unchanged_evidence(
    monkeypatch, empty_content
):
    post = AsyncMock(return_value=decision_payload("emerging"))
    model = AsyncMock(
        side_effect=[
            response(content="Measure sample a to test the hypothesis."),
            response(content=empty_content),
            response(content=empty_content),
            response(tool_call("submit_answer", '{"answer":"42"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    session = await make_session([], max_iterations=4)
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert post.await_count == 2  # One ordinary report and the final review.
    assert model.await_count == 4
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 6
    assert outcome.usage.input_tokens == session.state.usage.input_tokens
    for request in model.await_args_list:
        assert not any(
            m["role"] == "assistant"
            and not m.get("content")
            and not m.get("tool_calls")
            for m in request.kwargs["messages"]
        )
    assert sum(m.get("name") == "empty_response_retry" for m in session.messages) == 1
    assert not any(
        m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls")
        for m in session.messages
    )


@pytest.mark.anyio
async def test_repeated_empty_replies_stop_without_inventing_history(monkeypatch):
    model = AsyncMock(return_value=response(content=""))
    model.return_value.finish_reason = "stop"
    annotate = AsyncMock(side_effect=AssertionError("No public actor evidence"))
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    session = await make_session([], max_iterations=100)
    outcome = await EpistemicAgent(
        annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert outcome.status == "protocol_failure"
    assert "3 consecutive responses" in outcome.error
    assert "finish_reason='stop'" in outcome.error
    assert model.await_count == 3
    annotate.assert_not_awaited()
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 3
    assert outcome.usage.input_tokens == session.state.usage.input_tokens == 21
    assert not any(m["role"] == "assistant" for m in session.messages)


@pytest.mark.anyio
@pytest.mark.parametrize("agent_kind", ["tool_calling", "epistemic_disabled"])
@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
@pytest.mark.parametrize("content", ["", "I cannot provide that response."])
async def test_provider_filter_without_retries_keeps_only_public_text(
    monkeypatch, agent_kind, finish_reason, content
):
    result = response(content=content)
    result.finish_reason = finish_reason
    model = AsyncMock(return_value=result)
    annotate = AsyncMock(side_effect=AssertionError("No prior public evidence"))
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    session = await make_session([], max_iterations=100)
    agent = (
        ToolCallingAgent()
        if agent_kind == "tool_calling"
        else EpistemicAgent(
            annotator=SimpleNamespace(annotate=annotate),
            actor_refusal_max_retries=0,
        )
    )
    outcome = await agent.run_session(session)
    assert outcome.status == "agent_failure"
    assert finish_reason in outcome.error
    model.assert_awaited_once()
    annotate.assert_not_awaited()
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 1
    assert [m["content"] for m in session.messages] == ([content] if content else [])


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
async def test_refusal_retry_preserves_request_report_and_usage(
    monkeypatch, finish_reason
):
    calls = []
    session = await make_session(calls, max_iterations=3)
    post = AsyncMock(return_value=decision_payload("emerging"))
    rejected = response(
        tool_call("measure", '{"sample":"must-not-run"}', "rejected"),
        content="Partial text from a refused response.",
    )
    rejected.finish_reason = finish_reason
    rejected.usage = {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "reasoning_tokens": 2,
    }
    replies = iter(
        [
            response(tool_call("measure", '{"sample":"a"}', "first")),
            rejected,
            response(tool_call("submit_answer", '{"answer":"42"}', "done")),
        ]
    )
    requests = []

    async def actor(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(replies)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    agent = EpistemicAgent(
        model="test-model",
        reasoning_effort="medium",
        temperature=1,
        annotator=annotator_with_transport(post),
    )
    outcome = await agent.run_session(session)
    assert outcome.status == "completed", outcome.error
    assert session.state.submission == "42"
    assert calls == ["a"]
    assert session.state.tool_statistics == {"measure": 1, "submit_answer": 1}
    assert len(requests) == 3
    assert requests[1] == requests[2]
    assert requests[1]["messages"][-1]["name"] == REPORT_MESSAGE_NAME
    assert post.await_count == 2
    reports = [
        m["metadata"]["epistemic_report"]
        for m in session.messages
        if m.get("name") == REPORT_MESSAGE_NAME
    ]
    assert len(reports) == 2
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 5
    assert (
        outcome.usage.input_tokens
        == session.state.usage.input_tokens
        == (25 + sum(r["usage"]["input_tokens"] for r in reports))
    )
    assert (
        outcome.usage.output_tokens
        == session.state.usage.output_tokens
        == (11 + sum(r["usage"]["output_tokens"] for r in reports))
    )
    assert (
        outcome.usage.reasoning_tokens
        == session.state.usage.reasoning_tokens
        == (2 + sum(r["usage"]["reasoning_tokens"] for r in reports))
    )
    assert all(m.get("content") != rejected.content for m in session.messages)
    assert not any(m.get("name") == "empty_response_retry" for m in session.messages)
    assert not any(
        m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls")
        for m in session.messages
    )


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
@pytest.mark.parametrize("max_iterations", [1, 100])
async def test_repeated_refusals_respect_default_retry_and_iteration_limits(
    monkeypatch, finish_reason, max_iterations
):
    calls = []
    session = await make_session(calls, max_iterations=max_iterations)
    rejected = response(
        tool_call("submit_answer", '{"answer":"must-not-submit"}', "rejected"),
        content="I cannot provide that response.",
    )
    rejected.finish_reason = finish_reason
    model = AsyncMock(return_value=rejected)
    annotate = AsyncMock(side_effect=AssertionError("No public actor evidence"))
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    expected_calls = min(2, max_iterations)
    assert outcome.status == "agent_failure"
    assert finish_reason in outcome.error
    assert model.await_count == expected_calls
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == expected_calls
    assert (
        outcome.usage.input_tokens
        == session.state.usage.input_tokens
        == (7 * expected_calls)
    )
    assert session.state.submission is None
    assert calls == []
    assert [m["content"] for m in session.messages] == [rejected.content]
    annotate.assert_not_awaited()


@pytest.mark.anyio
async def test_refusal_retry_allowance_resets_after_accepted_response(monkeypatch):
    session = await make_session([], max_iterations=4)
    rejected = response(content="")
    rejected.finish_reason = "content_filter"
    model = AsyncMock(
        side_effect=[
            rejected,
            response(content="The observations support hypothesis X."),
            rejected,
            response(tool_call("submit_answer", '{"answer":"X"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    post = AsyncMock(return_value=decision_payload("emerging"))
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert model.await_count == 4
    assert post.await_count == 2
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 6
    assert not any(
        m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls")
        for m in session.messages
    )


@pytest.mark.anyio
async def test_custom_actor_refusal_retry_limit(monkeypatch):
    session = await make_session([], max_iterations=100)
    rejected = response(content="")
    rejected.finish_reason = "refusal"
    model = AsyncMock(return_value=rejected)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    agent = EpistemicAgent(actor_refusal_max_retries=2)
    outcome = await agent.run_session(session)
    assert outcome.status == "agent_failure"
    assert model.await_count == 3
    assert outcome.usage.llm_calls == session.state.usage.llm_calls == 3
    assert session.messages == ()
    assert "actor_refusal_max_retries" not in model.await_args.kwargs


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_invalid_actor_refusal_retry_limit(value):
    with pytest.raises(ValueError, match="actor_refusal_max_retries"):
        EpistemicAgent(actor_refusal_max_retries=value)


@pytest.mark.anyio
async def test_resume_filters_empty_history_without_losing_trajectory_boundary(
    monkeypatch,
):
    session = await make_session([], max_iterations=1)
    await session.record_message(
        {"role": "assistant", "content": "Prior public hypothesis."}
    )
    await session.record_message({"role": "assistant", "content": ""})
    await session.record_message(
        {
            "role": "assistant",
            "content": "Private reasoning summary.",
            "metadata": {"content_source": "reasoning_summary"},
        }
    )
    post = AsyncMock(return_value=decision_payload("emerging"))
    model = AsyncMock(
        return_value=response(tool_call("submit_answer", '{"answer":"42"}', "done"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert post.await_count == 2
    annotation_input = post.await_args.kwargs["body"]["input"]
    assert "Prior public hypothesis." in annotation_input
    assert "Private reasoning summary." not in annotation_input
    assert "Private reasoning summary." not in json.dumps(
        model.await_args.kwargs["messages"]
    )


@pytest.mark.anyio
async def test_resume_sends_old_reports_only_to_actor(monkeypatch):
    session = await make_session([])
    await session.record_message(
        {"role": "assistant", "content": "Previous hypothesis: X"}
    )
    await session.record_message(
        {
            "role": "user",
            "name": REPORT_MESSAGE_NAME,
            "content": "STALE_REPORT",
            "metadata": {"epistemic_report": {"raw_response": "PRIVATE_METADATA"}},
        }
    )
    post = AsyncMock(return_value=decision_payload())
    annotator = annotator_with_transport(post)
    annotator.annotate = AsyncMock(wraps=annotator.annotate)
    model = AsyncMock(
        return_value=response(tool_call("submit_answer", '{"answer":"X"}', "done"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(annotator=annotator).run_session(session)
    assert outcome.status == "completed", outcome.error
    assert "Previous hypothesis: X" in post.await_args.kwargs["body"]["input"]
    assert "STALE_REPORT" not in post.await_args.kwargs["body"]["input"]
    assert "STALE_REPORT" not in json.dumps(annotator.annotate.await_args.args[0])
    assert "PRIVATE_METADATA" not in post.await_args.kwargs["body"]["input"]
    actor_messages = model.await_args.kwargs["messages"]
    assert "STALE_REPORT" in json.dumps(actor_messages)
    assert "PRIVATE_METADATA" not in json.dumps(actor_messages)
    assert all("metadata" not in message for message in actor_messages)
    assert any(
        "PRIVATE_METADATA" in json.dumps(message) for message in session.messages
    )
    reports = [m for m in actor_messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert len(reports) == 2
    assert reports[0]["content"] == "STALE_REPORT"
    assert reports[1] == actor_messages[-1]


@pytest.mark.anyio
async def test_reasoning_only_turns_are_reannotated_and_do_not_finish(monkeypatch):
    environment_calls = []
    session = await make_session(environment_calls, max_iterations=3)
    post = AsyncMock(return_value=decision_payload())
    model = AsyncMock(
        side_effect=[
            response(content="The supplied facts favor hypothesis X."),
            response(content="Hypothesis Y conflicts with the supplied facts."),
            response(tool_call("submit_answer", '{"answer":"X"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "completed"
    assert post.await_count == 3
    assert environment_calls == []
    assert session.state.tool_statistics == {"submit_answer": 1}
    assert session.state.submission == "X"
    assert session.state.usage.llm_calls == 6
    assert [
        m["content"]
        for m in session.messages
        if m["role"] == "assistant" and m.get("content")
    ] == [
        "The supplied facts favor hypothesis X.",
        "Hypothesis Y conflicts with the supplied facts.",
        "Explain observations and the next experiment.\n\n"
        "Proposed final answer (not yet submitted):\nX",
    ]
    first_input, second_input = [
        request.kwargs["body"]["input"] for request in post.await_args_list[:2]
    ]
    assert "The supplied facts favor hypothesis X." in first_input
    assert "Hypothesis Y conflicts" not in first_input
    assert "The supplied facts favor hypothesis X." in second_input
    assert "Hypothesis Y conflicts" in second_input
    assert REPORT_MESSAGE_NAME not in second_input
    assert "No present unconstructive epistemic patterns detected." not in second_input
    assert (
        "a tool call is not required on every turn"
        in model.await_args.kwargs["messages"][-2]["content"]
    )
    assert (
        len(
            [
                m
                for m in model.await_args.kwargs["messages"]
                if m.get("name") == REPORT_MESSAGE_NAME
            ]
        )
        == 2
    )


@pytest.mark.anyio
async def test_annotation_failure_stops_before_actor_and_tools(monkeypatch):
    model = AsyncMock()
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    post = AsyncMock(side_effect=RuntimeError("Decisions unavailable"))
    session = Session()
    session.messages.append({"role": "assistant", "content": "Previous hypothesis."})
    outcome = await EpistemicAgent(
        annotator=annotator_with_transport(post)
    ).run_session(session)
    assert outcome.status == "agent_failure"
    assert "Decisions unavailable" in outcome.error
    model.assert_not_awaited()
    assert session.calls == []


@pytest.mark.anyio
async def test_surrender_and_iteration_budget_follow_tool_calling_protocol(monkeypatch):
    post = AsyncMock(return_value=decision_payload())
    session = Session()
    session.surrender_allowed = True
    model = AsyncMock(
        return_value=response(
            tool_call("submit_answer", '{"answer":"surrender"}', "done")
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    agent = EpistemicAgent(annotator=annotator_with_transport(post))
    outcome = await agent.run_session(session)
    assert outcome.status == "surrendered"
    session = Session()
    session.iteration_limit = 1
    model.return_value = response(content="Still thinking")
    outcome = await agent.run_session(session)
    assert outcome.status == "iteration_limit"
    assert session.calls == []


@pytest.mark.anyio
async def test_shared_agent_keeps_parallel_trials_separate(monkeypatch):
    inputs = []

    async def decide(*args, **kwargs):
        inputs.append(kwargs["body"]["input"])
        await asyncio.sleep(0)
        return decision_payload()

    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=response(
                tool_call("submit_answer", '{"answer":"done"}', "done")
            )
        ),
    )
    agent = EpistemicAgent(annotator=annotator_with_transport(decide))
    sessions = [Session(), Session()]
    sessions[0].prompt = "TASK_ALPHA"
    sessions[1].prompt = "TASK_BETA"
    for session in sessions:
        session.messages.append(
            {"role": "assistant", "content": "Previous hypothesis."}
        )
    outcomes = await asyncio.gather(*(agent.run_session(s) for s in sessions))
    assert all(o.status == "completed" for o in outcomes)
    assert len(inputs) == 4
    assert all(("TASK_ALPHA" in text) != ("TASK_BETA" in text) for text in inputs)


def test_epistemic_settings_are_not_forwarded_to_actor():
    agent = EpistemicAgent(
        actor_refusal_max_retries=2,
        epistemic_model="classifier-model",
        epistemic_explanation_model="explanation-model",
        epistemic_timeout_s=42,
        epistemic_max_retries=0,
        epistemic_stream_explanations=True,
        epistemic_explanation_concurrency=3,
        reasoning_effort="low",
    )
    assert agent.annotator.model == "classifier-model"
    assert agent.annotator.explanation_model == "explanation-model"
    assert agent.annotator.timeout_s == 42
    assert agent.annotator.max_retries == 0
    assert agent.actor_refusal_max_retries == 2
    assert agent.annotator.stream_explanations is True
    assert agent.annotator.explanation_concurrency == 3
    assert agent.annotator.pattern_thresholds == CALIBRATED_PATTERN_THRESHOLDS
    assert agent.annotator.selection_policy["name"] == CALIBRATED_THRESHOLD_POLICY
    assert agent._call_kwargs() == {"reasoning_effort": "low"}


def test_agent_threshold_policy_can_be_disabled_or_overridden():
    legacy = EpistemicAgent(epistemic_pattern_thresholds=None)
    assert legacy.annotator.pattern_thresholds is None
    thresholds = dict.fromkeys((m.name for m in MOTIFS), 0.6)
    custom = EpistemicAgent(epistemic_pattern_thresholds=thresholds)
    thresholds[MOTIFS[0].name] = 0.1
    assert custom.annotator.pattern_thresholds[MOTIFS[0].name] == 0.6
    assert custom._call_kwargs() == {}
    supplied = DecisionsAnnotator()
    assert EpistemicAgent(annotator=supplied).annotator is supplied
    assert supplied.pattern_thresholds is None


@pytest.mark.anyio
async def test_default_agent_uses_thresholds_and_persists_policy(monkeypatch):
    payload = decision_payload()
    target = "evidence_led_hypothesis_generation"  # P(active)+P(resolved)=.10 >= .04.
    requests = []

    async def decide(self, client, body):
        requests.append(body)
        return payload

    explained_names = []

    async def explain(client, *, model, motifs, trajectory, task):
        names = {m.name for m in motifs}
        assert len(names) == 1
        explained_names.extend(names)
        assert "refutation_driven_belief_revision" not in names  # .10 < .73.
        assert all(m.get("name") != REPORT_MESSAGE_NAME for m in trajectory)
        explanations = {
            m.name: {
                "supported": m.name == target,
                "explanation": "The measured result supports this hypothesis.",
            }
            for m in motifs
        }
        return explanations, explanation_payload(explanations)

    monkeypatch.setattr(DecisionsAnnotator, "_request", decide)
    monkeypatch.setattr("corral.epistemic_analysis.decisions.explain_patterns", explain)
    model = AsyncMock(
        side_effect=[
            response(tool_call("measure", '{"sample":"a"}', "a")),
            response(tool_call("submit_answer", '{"answer":"42"}', "done")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    session = await make_session([], max_iterations=2)
    agent = EpistemicAgent(epistemic_api_key="test-key")
    outcome = await agent.run_session(session)
    assert outcome.status == "completed", outcome.error
    assert len(requests) == 2
    assert explained_names.count(target) == 2
    reports = [m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert len(reports) == 2
    assert target not in reports[0]["content"]  # Productive patterns are saved only.
    saved = reports[0]["metadata"]["epistemic_report"]
    assert saved["selection_policy"]["name"] == CALIBRATED_THRESHOLD_POLICY
    selected = next(a for a in saved["assessments"] if a["name"] == target)
    assert selected["status"] == "absent"
    assert selected["selected"] is True
    assert selected["selection_probability"] == 0.1
    assert selected["selection_threshold"] == 0.04
    assert selected["explanation_supported"] is True
    actor_input = model.await_args_list[-1].kwargs["messages"]
    assert target not in actor_input[-1]["content"]
    assert all("metadata" not in message for message in actor_input)
