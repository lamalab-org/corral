"""Tests for commit-backed intervention hooks."""

import json
from pathlib import Path

import pytest
from tests.agents.commit_session import start_session

from corral.agents.hooks import AgentHooks, CriticalHookError, HookPoint
from corral.agents.hooks.intervention import (
    _parse_react_actions,
    create_intervention_hook,
    create_trace_intervention_hook,
)
from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSession, run_agent_session
from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool

pytestmark = pytest.mark.usefixtures("session_stores")


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class RecordingAgent:
    model = "test-model"

    def __init__(self, hooks: AgentHooks) -> None:
        self.hooks = hooks
        self.seen_messages: tuple = ()

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        self.seen_messages = session.messages
        response = await session.execute(
            Action(name="submit_answer", arguments={"answer": "42"})
        )
        assert response.success
        return AgentOutcome(status="completed", answer="42")


class ReActAgent(RecordingAgent):
    pass


def make_environment(calls: list[str], *, fail: bool = False) -> Environment:
    def search(query: str) -> str:
        """Search for a query."""
        calls.append(query)
        if fail:
            raise RuntimeError("tool failed")
        return f"result:{query}"

    task = TaskDefinition(
        name="task1",
        description="search and answer",
        tools=["search"],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    return Environment(
        "task1",
        task,
        toolset=Toolset(
            pool={"search": tool(search)},
            workspace_factory=None,
        ),
    )


async def run_test_agent(agent, environment: Environment):
    session = await start_session(environment, max_iterations=10, agent=agent)
    return await run_agent_session(
        agent,
        environment,
        session.state,
        actor=session.actor,
        runtime_actor=session.runtime_actor,
        state_store=session.state_store,
        max_iterations=10,
    )


def test_parse_react_actions_uses_current_action_type():
    actions = _parse_react_actions(
        """
        <action>configure</action>
        <action_input>{"enabled": True, "disabled": False}</action_input>
        <action>search</action>
        <action_input>{"query": "test"}</action_input>
        """
    )

    assert [action.name for action in actions] == ["configure", "search"]
    assert actions[0].arguments == {"enabled": True, "disabled": False}
    assert all(isinstance(action, Action) for action in actions)
    assert _parse_react_actions("<action>missing-input</action>") == []


@pytest.mark.anyio()
async def test_text_intervention_is_recorded_in_canonical_state():
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        create_intervention_hook({"task1": "use the calibration result"}),
    )
    agent = RecordingAgent(hooks)
    environment = make_environment([])

    result = await run_test_agent(agent, environment)

    assert any(
        message.get("content") == "use the calibration result"
        for message in agent.seen_messages
    )
    assert (
        result.state.agent_runs[result.final_commit.author.run_id].algorithm_state[
            "hooks"
        ]["metadata"]["intervention_applied"]
        is True
    )


@pytest.mark.anyio()
async def test_react_intervention_without_execution_strips_actions():
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        create_intervention_hook(
            {
                "task1": (
                    "inspect first <action>search</action>"
                    '<action_input>{"query":"sample"}</action_input>'
                )
            },
            execute_tools=False,
        ),
    )
    agent = ReActAgent(hooks)
    environment = make_environment([])

    await run_test_agent(agent, environment)

    injected = str(agent.seen_messages[0]["content"])
    assert injected == "<thought>inspect first</thought>"
    assert "<action>" not in injected


@pytest.mark.anyio()
async def test_react_intervention_executes_through_agent_session():
    calls: list[str] = []
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        create_intervention_hook(
            {
                "task1": (
                    "<thought>search first</thought>"
                    "<action>search</action>"
                    '<action_input>{"query":"sample"}</action_input>'
                )
            },
            execute_tools=True,
        ),
    )
    agent = ReActAgent(hooks)
    environment = make_environment(calls)

    result = await run_test_agent(agent, environment)

    assert calls == ["sample"]
    assert result.state.tool_statistics == {"search": 1, "submit_answer": 1}
    intervention_action = next(iter(result.state.actions.values())).action
    assert intervention_action.name == "search"
    assert intervention_action.metadata == {"source": "hook-intervention"}


@pytest.mark.anyio()
async def test_intervention_tool_failure_is_critical():
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        create_intervention_hook(
            {
                "task1": (
                    "<action>search</action>"
                    '<action_input>{"query":"sample"}</action_input>'
                )
            },
            execute_tools=True,
        ),
    )
    agent = ReActAgent(hooks)
    environment = make_environment([], fail=True)

    with pytest.raises(CriticalHookError, match="execution failed"):
        await run_test_agent(agent, environment)


@pytest.mark.anyio()
async def test_tool_call_trace_replay_uses_new_actions(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "searching",
                        "tool_calls": [
                            {
                                "id": "old-trace-id",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"trace"}',
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        create_trace_intervention_hook(
            {"task1": [str(trace)]},
            num_steps=1,
            execute_tools=True,
        ),
    )
    agent = RecordingAgent(hooks)
    environment = make_environment(calls)

    result = await run_test_agent(agent, environment)

    assert calls == ["trace"]
    replayed_action = next(iter(result.state.actions.values())).action
    assert replayed_action.name == "search"
    assert replayed_action.id != "old-trace-id"
