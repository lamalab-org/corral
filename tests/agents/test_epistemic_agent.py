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
                "evidence_message_ids": [1, 2],
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
        return_value=response(tool_call("submit_answer", '{"answer":"42"}', "finish"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)
    outcome = await EpistemicAgent(
        annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert outcome.status == "completed", outcome.error
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
        return decision_payload("emerging" if len(requests) == 1 else "active")

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
    assert events == ["actor", "annotate", "actor", "annotate", "explain", "actor"]
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
    assert len(explanation_requests) == 1
    assert "No present epistemic patterns" not in explanation_requests[0]["input"]
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
    assert sent[1][-1]["content"] == "No present epistemic patterns detected."
    assert "Measurements measured:a and measured:b" in sent[2][-1]["content"]
    previous_report_index = sent[2].index(sent[1][-1])
    assert sent[2][previous_report_index + 1]["role"] == "assistant"
    assert "tool_calls" in sent[2][previous_report_index + 1]
    reports = [m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert len(reports) == 2
    assert len(reports[1]["metadata"]["epistemic_report"]["assessments"]) == 18
    assert outcome.usage.llm_calls == 6
    assert outcome.usage.input_tokens == 102
    assert outcome.usage.output_tokens == 24
    assert outcome.usage.reasoning_tokens == 5
    assert session.state.usage.llm_calls == 6
    assert session.state.usage.input_tokens == 102


@pytest.mark.anyio
@pytest.mark.parametrize("with_tool", [True, False])
async def test_reasoning_summary_source_is_saved_but_not_sent_as_metadata(
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
    saved = next(m for m in session.messages if m.get("content") == first.content)
    assert saved["metadata"]["content_source"] == "reasoning_summary"
    sent = model.await_args.kwargs["messages"]
    assert any(m.get("content") == first.content for m in sent)
    assert all("metadata" not in m for m in sent)
    submission = next(
        m
        for m in session.messages
        if m.get("content") == "The measurements support 42."
    )
    assert submission.get("metadata", {}).get("content_source") != "reasoning_summary"


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
    assert post.await_count == 2
    assert environment_calls == []
    assert session.state.tool_statistics == {"submit_answer": 1}
    assert session.state.submission == "X"
    assert session.state.usage.llm_calls == 5
    assert [m["content"] for m in session.messages if m["role"] == "assistant"] == [
        "The supplied facts favor hypothesis X.",
        "Hypothesis Y conflicts with the supplied facts.",
        "Explain observations and the next experiment.",
    ]
    first_input, second_input = [
        request.kwargs["body"]["input"] for request in post.await_args_list
    ]
    assert "The supplied facts favor hypothesis X." in first_input
    assert "Hypothesis Y conflicts" not in first_input
    assert "The supplied facts favor hypothesis X." in second_input
    assert "Hypothesis Y conflicts" in second_input
    assert REPORT_MESSAGE_NAME not in second_input
    assert "No present epistemic patterns detected." not in second_input
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
    assert len(inputs) == 2
    assert all(("TASK_ALPHA" in text) != ("TASK_BETA" in text) for text in inputs)


def test_epistemic_settings_are_not_forwarded_to_actor():
    agent = EpistemicAgent(
        epistemic_model="classifier-model",
        epistemic_explanation_model="explanation-model",
        epistemic_timeout_s=42,
        epistemic_max_retries=0,
        reasoning_effort="low",
    )
    assert agent.annotator.model == "classifier-model"
    assert agent.annotator.explanation_model == "explanation-model"
    assert agent.annotator.timeout_s == 42
    assert agent.annotator.max_retries == 0
    assert agent._call_kwargs() == {"reasoning_effort": "low"}
