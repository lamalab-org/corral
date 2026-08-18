"""Session-boundary tests for AI Scientist."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import anyio
import pytest
from anyio.from_thread import BlockingPortal

from corral.agents import Agent, AIScientistAgent
from corral.agents.ai_scientist import agent as agent_module
from corral.agents.ai_scientist.agent import (
    _BranchSessionRegistry,
    _call_llm_from_harness,
)
from corral.agents.ai_scientist.manager import ExperimentManager
from corral.agents.schema import AgentUsage
from corral.agents.session import AgentSession
from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def make_session(calls, *, max_iterations=64):
    def measure(value: int) -> str:
        """Return a measurement."""
        calls.append(value)
        return str(value)

    task = TaskDefinition(
        name="science",
        description="measure",
        tools=["measure"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "science",
        task,
        toolset=Toolset(
            pool={"measure": tool(measure)},
            workspace_factory=None,
        ),
    )
    return AgentSession(
        environment,
        environment.initial_state(started_at=datetime.now(timezone.utc)),
        max_iterations=max_iterations,
    )


def test_ai_scientist_exposes_only_the_session_entrypoint():
    agent = AIScientistAgent(model="test-model")

    assert isinstance(agent, Agent)
    assert callable(agent.run_session)
    assert not hasattr(agent, "run")
    assert not hasattr(agent, "arun")
    assert not hasattr(agent, "run_agent")
    assert not hasattr(agent, "arun_agent")
    assert not hasattr(agent, "step")
    assert not hasattr(agent, "last_state")
    assert not hasattr(agent, "messages")
    assert not hasattr(agent, "_available_tools")
    assert not hasattr(agent, "_branch_tool_statistics")
    assert not hasattr(AgentSession, "_adopt_state")


def test_experiment_manager_is_not_a_second_agent_entrypoint():
    assert callable(ExperimentManager.execute_search)
    assert not hasattr(ExperimentManager, "run")
    assert not hasattr(ExperimentManager, "run_session")


def test_ai_scientist_uses_the_session_model_call_budget():
    agent = AIScientistAgent(model="test-model")
    session = make_session([], max_iterations=7)

    assert not hasattr(agent.config, "max_llm_calls")
    assert session.iteration_limit == 7


@pytest.mark.anyio()
async def test_scientist_harness_uses_native_async_llm_call(monkeypatch):
    response = object()
    async_call = AsyncMock(return_value=response)
    monkeypatch.setattr(agent_module, "llm_call", async_call)

    async with BlockingPortal() as portal:
        result = await anyio.to_thread.run_sync(
            lambda: _call_llm_from_harness(portal, model="test-model")
        )

    assert result is response
    async_call.assert_awaited_once_with(model="test-model")


@pytest.mark.anyio()
async def test_scientist_branches_execute_through_agent_sessions():
    calls = []
    parent = make_session(calls)

    async with BlockingPortal() as portal:
        sessions = _BranchSessionRegistry(parent, portal)
        first = sessions.create_branch()
        second = sessions.create_branch()

        first_response = await anyio.to_thread.run_sync(
            lambda: first.execute(Action(name="measure", arguments={"value": 1}))
        )
        second_response = await anyio.to_thread.run_sync(
            lambda: second.execute(Action(name="measure", arguments={"value": 2}))
        )

        first_session = sessions.session(first.execution_id)
        second_session = sessions.session(second.execution_id)
        assert first_response.success is True
        assert second_response.success is True
        assert first_session.state.tool_statistics == {"measure": 1}
        assert second_session.state.tool_statistics == {"measure": 1}
        assert first_session.state.actions[0].arguments == {"value": 1}
        assert second_session.state.actions[0].arguments == {"value": 2}
        assert first_session.state.actions[0].actor_id == "agent_0"
        assert second_session.state.actions[0].actor_id == "agent_0"

    assert calls == [1, 2]


@pytest.mark.anyio()
async def test_scientist_branches_inherit_current_hook_state():
    calls = []
    parent = make_session(calls)
    await parent.record_message(
        {"role": "assistant", "content": "hook-provided context"}
    )
    hook_response = await parent.execute(
        Action(
            name="measure",
            arguments={"value": 0},
            metadata={"source": "hook-intervention"},
        )
    )
    assert hook_response.success

    async with BlockingPortal() as portal:
        sessions = _BranchSessionRegistry(parent, portal)
        branch = sessions.create_branch()
        response = await anyio.to_thread.run_sync(
            lambda: branch.execute(Action(name="measure", arguments={"value": 1}))
        )
        branch_state = sessions.session(branch.execution_id).state

    assert response.success
    assert calls == [0, 1]
    assert branch_state.tool_statistics == {"measure": 2}
    assert branch_state.actions[0].metadata == {"source": "hook-intervention"}
    assert any(
        message.get("content") == "hook-provided context"
        for message in branch_state.messages
    )


@pytest.mark.anyio()
async def test_ai_scientist_maps_harness_result_to_outcome(monkeypatch):
    agent = AIScientistAgent(model="test-model")
    session = make_session([])

    monkeypatch.setattr(
        agent,
        "_execute_session",
        lambda _session, _owner, _portal: (
            "42",
            AgentUsage(input_tokens=4, output_tokens=1, llm_calls=2),
            {"graph": {}},
        ),
    )
    outcome = await agent.run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.llm_calls == 2
    assert session.state.submission == "42"
    assert session.state.tool_statistics == {"submit_answer": 1}
    scientist_state = session.get_agent_state("ai_scientist")
    assert scientist_state is not None
    assert scientist_state["status"] == "completed"


@pytest.mark.anyio()
async def test_ai_scientist_instance_is_reentrant_across_sessions(monkeypatch):
    agent = AIScientistAgent(model="test-model")
    first = make_session([])
    second = make_session([])
    first.execution_id = "first"
    second.execution_id = "second"

    monkeypatch.setattr(
        agent,
        "_execute_session",
        lambda session, _owner, _portal: (
            session.execution_id,
            AgentUsage(llm_calls=1),
            {"graph": {"execution_id": session.execution_id}},
        ),
    )

    first_outcome, second_outcome = await asyncio.gather(
        agent.run_session(first),
        agent.run_session(second),
    )

    assert first_outcome.answer == "first"
    assert second_outcome.answer == "second"
    assert first.state.submission == "first"
    assert second.state.submission == "second"
    assert first.get_agent_state("ai_scientist")["status"] == "completed"
    assert second.get_agent_state("ai_scientist")["status"] == "completed"
