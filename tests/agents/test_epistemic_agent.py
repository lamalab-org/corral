"""Graph-only epistemic defaults and feedback behavior in both agent loops."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.agents.test_action_interface import make_session
from tests.agents.test_tool_calling_agent import Session, call

from corral.agents.epistemic import EpistemicAgent
from corral.agents.react import ReActAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.epistemic_analysis.annotators.local import (
    DEFAULT_CHAT_MODEL,
    GraphAnnotator,
)
from corral.epistemic_analysis.schema import (
    REPORT_MESSAGE_NAME,
    EpistemicReport,
    MotifAssessment,
)

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


def reply(agent_type, *calls, content="Explain the observations.", **fields):
    result = response(*calls, content=content)
    if agent_type is ReActAgent:
        result.tool_calls = []
        result.content = (f"<thought>{content}</thought>" if content else "") + "".join(
            f"<action>{call.function.name}</action><action_input>{call.function.arguments}</action_input>"
            for call in calls
        )
    for key, value in fields.items():
        setattr(result, key, value)
    return result


def report(*, matched=False, name="untested_claim", kind="antipattern"):
    return EpistemicReport(
        model="local",
        assessed_at="test",
        message_count=1,
        trajectory_sha256="test",
        assessments=(
            MotifAssessment(
                name,
                kind,
                "test",
                "active" if matched else "absent",
                None,
                {},
                selected=matched,
                explanation_supported=matched,
                explanation="Annotated claim has no test." if matched else None,
            ),
        ),
        usage={"input_tokens": 4, "output_tokens": 2, "llm_calls": 1},
        raw_response={},
        graph={"nodes": [], "edges": []},
        timings_s={"total": 0.01},
    )


def make_agent(agent_type, **kwargs):
    return agent_type(
        **kwargs, **({"epistemic": True} if agent_type is not EpistemicAgent else {})
    )


@pytest.mark.parametrize("agent_type", [EpistemicAgent, ToolCallingAgent, ReActAgent])
def test_local_actor_default_and_explicit_actor_override(agent_type):
    agent = make_agent(agent_type)
    assert agent.model == f"openai/{DEFAULT_CHAT_MODEL}"
    assert agent.api_endpoint == "http://127.0.0.1:8790/v1"
    assert isinstance(agent.annotator, GraphAnnotator)
    explicit = make_agent(
        agent_type,
        model="openai/custom",
        api_endpoint="https://example.test/v1",
        api_key="custom",
    )
    assert explicit.model == "openai/custom"
    assert explicit.api_endpoint == "https://example.test/v1"
    assert explicit.kwargs["api_key"] == "custom"


def test_removed_direct_options_fail_before_reaching_actor():
    with pytest.raises(TypeError, match="Direct pattern annotation was removed"):
        EpistemicAgent(epistemic_backend="direct")


def test_annotation_and_explanation_overrides_are_independent():
    agent = EpistemicAgent(
        model="test-actor",
        epistemic_model="annotator-model",
        epistemic_base_url="http://annotation.test/v1",
        epistemic_explanation_model="explainer-model",
        epistemic_explanation_base_url="http://explanation.test/v1",
    )
    assert agent.annotator.provider_config.model == "annotator-model"
    assert (
        agent.annotator.provider_config.endpoint
        == "http://annotation.test/v1/chat/completions"
    )
    assert agent.annotator.explanation_config.model == "explainer-model"
    assert (
        agent.annotator.explanation_config.endpoint
        == "http://explanation.test/v1/chat/completions"
    )
    assert agent._call_kwargs() == {}


@pytest.mark.anyio
@pytest.mark.parametrize("agent_type", [EpistemicAgent, ToolCallingAgent, ReActAgent])
async def test_first_turn_skips_annotation_then_reviews_completed_tools(
    monkeypatch, agent_type
):
    prefixes = []
    events = []

    async def annotate(messages, **kwargs):
        events.append("annotate")
        prefixes.append(deepcopy(messages))
        assert kwargs["task"] == session.prompt
        return report()

    responses = iter(
        [
            reply(
                agent_type,
                call("measure", '{"sample":"a"}', "p1"),
                call("measure", '{"sample":"b"}', "p2"),
            ),
            reply(agent_type, call("submit_answer", '{"answer":"done"}', "submit")),
        ]
    )

    async def actor(**kwargs):
        events.append("actor")
        return next(responses)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    calls = []
    session = await make_session(calls)
    result = await make_agent(
        agent_type, model="actor", annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert result.status == "completed", result.error
    assert events[0] == "actor"
    assert events[1] == "annotate"
    assert len(prefixes) == 2  # Pre-turn feedback and proposed final answer.
    assert sum(m["role"] == "tool" for m in prefixes[0]) == 2
    assert all(
        m.get("name") != REPORT_MESSAGE_NAME for prefix in prefixes for m in prefix
    )
    assert calls == ["a", "b"]
    assert result.usage.llm_calls == 4


@pytest.mark.anyio
@pytest.mark.parametrize("agent_type", [EpistemicAgent, ToolCallingAgent, ReActAgent])
async def test_repeated_graph_matches_hold_submissions_within_budget(
    monkeypatch, agent_type
):
    annotate = AsyncMock(return_value=report(matched=True))
    actor = AsyncMock(
        return_value=reply(agent_type, call("submit_answer", '{"answer":"100"}', "s"))
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    session = Session()
    session.iteration_limit = 2
    result = await make_agent(
        agent_type, model="actor", annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert result.status == "iteration_limit"
    assert session.calls == []
    assert actor.await_count == 2
    assert all(
        "epistemic_report" not in str(c.args[0]) for c in annotate.await_args_list
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "kind"),
    [("fixed_belief_trace", "antipattern"), ("traceable_conclusion", "productive")],
)
async def test_descriptive_and_productive_matches_do_not_hold_submission(
    monkeypatch, name, kind
):
    annotate = AsyncMock(return_value=report(matched=True, name=name, kind=kind))
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=response(call("submit_answer", '{"answer":"100"}', "s"))
        ),
    )
    session = Session()
    result = await EpistemicAgent(
        model="actor", annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert result.status == "completed"
    assert len(session.calls) == 1


@pytest.mark.anyio
async def test_refusal_retry_does_not_reannotate_same_prefix(monkeypatch):
    annotate = AsyncMock(return_value=report())
    actor = AsyncMock(
        side_effect=[
            reply(
                ToolCallingAgent,
                content="",
                refusal="Cannot comply",
                finish_reason="refusal",
            ),
            response(call("submit_answer", '{"answer":"done"}', "s")),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    session = Session()
    session.record_messages = AsyncMock()
    session.messages.append({"role": "assistant", "content": "Observed evidence."})
    result = await EpistemicAgent(
        model="actor", annotator=SimpleNamespace(annotate=annotate)
    ).run_session(session)
    assert result.status == "completed", result.error
    assert annotate.await_count == 2
    assert actor.await_count == 2
