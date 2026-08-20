"""Architecture tests for the agent-owned session boundary."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.agents.commit_session import start_session

from corral.agents import (
    Agent,
    LLMPlanner,
    ReActAgent,
    TerminusAgent,
    ToolCallingAgent,
)
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.session import AgentSession
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool


@pytest.fixture()
def anyio_backend():
    return "asyncio"


async def make_session(
    calls: list[str],
    *,
    max_iterations: int = 10,
) -> AgentSession:
    def measure(sample: str) -> str:
        """Measure a named sample."""
        calls.append(sample)
        return f"measured:{sample}"

    task = TaskDefinition(
        name="session-task",
        description="measure sample a and answer",
        tools=["measure"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "session-task",
        task,
        toolset=Toolset(
            pool={"measure": tool(measure)},
            workspace_factory=None,
        ),
    )
    return await start_session(environment, max_iterations=max_iterations)


def tool_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _submission_conformance_cases():
    common = {
        "model": "test-model",
        "system_prompt": "system",
    }
    return (
        pytest.param(
            ReActAgent(**common),
            SimpleNamespace(
                content=(
                    "<thought>done</thought>"
                    "<action>submit_answer</action>"
                    '<action_input>{"answer":"42"}</action_input>'
                    "<action>measure</action>"
                    '<action_input>{"sample":"must-not-run"}</action_input>'
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2},
                id="react-submit",
            ),
            id="react",
        ),
        pytest.param(
            ToolCallingAgent(**common),
            SimpleNamespace(
                content="done",
                tool_calls=[
                    tool_call("submit_answer", '{"answer":"42"}', "submit"),
                    tool_call("measure", '{"sample":"must-not-run"}', "late"),
                ],
                usage={"prompt_tokens": 3, "completion_tokens": 2},
                id="tool-submit",
            ),
            id="tool-calling",
        ),
        pytest.param(
            TerminusAgent(
                **common,
                confirmations_required=0,
                max_actions_per_turn=2,
            ),
            SimpleNamespace(
                content=json.dumps(
                    {
                        "analysis": "done",
                        "plan": "finish",
                        "tool_calls": [
                            {
                                "name": "submit_answer",
                                "arguments": {"answer": "42"},
                            },
                            {
                                "name": "measure",
                                "arguments": {"sample": "must-not-run"},
                            },
                        ],
                    }
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2},
                id="terminus-submit",
            ),
            id="terminus",
        ),
        pytest.param(
            LLMPlanner(**common),
            SimpleNamespace(
                content="Final Answer: 42",
                usage={"prompt_tokens": 3, "completion_tokens": 2},
                id="planner-submit",
            ),
            id="planner",
        ),
    )


@pytest.mark.anyio()
async def test_all_local_agents_expose_the_session_contract_only():
    agents = [
        ReActAgent(system_prompt="system", user_prompt="Task: {{task_guide}}"),
        ToolCallingAgent(system_prompt="system", user_prompt="Task: {{task_guide}}"),
    ]

    for agent in agents:
        assert isinstance(agent, Agent)
        assert callable(agent.run_session)
        assert not hasattr(agent, "execution_mode")
        assert not hasattr(agent, "step")
        assert not hasattr(agent, "run")
        assert not hasattr(agent, "run_agent")


@pytest.mark.anyio()
@pytest.mark.parametrize(("agent", "response"), _submission_conformance_cases())
async def test_successful_submission_stops_the_local_agent_loop(
    monkeypatch,
    agent,
    response,
):
    model_call = AsyncMock(return_value=response)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    environment_calls: list[str] = []
    session = await make_session(environment_calls)

    outcome = await agent.run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.llm_calls == 1
    model_call.assert_awaited_once()
    assert environment_calls == []
    assert session.state.submission == "42"
    assert session.state.tool_statistics == {"submit_answer": 1}


@pytest.mark.anyio()
async def test_tool_calling_agent_executes_submission_in_its_session(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(
                content="measuring",
                tool_calls=[tool_call("measure", '{"sample":"a"}', "call-1")],
                usage={"prompt_tokens": 7, "completion_tokens": 2},
                id="response-1",
            ),
            SimpleNamespace(
                content="done",
                tool_calls=[tool_call("submit_answer", '{"answer":"42"}', "call-2")],
                usage={"prompt_tokens": 9, "completion_tokens": 1},
                id="response-2",
            ),
        ]
    )
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call", AsyncMock(side_effect=responses)
    )
    calls: list[str] = []
    session = await make_session(calls)
    outcome = await ToolCallingAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    ).run_session(session)

    assert outcome == AgentOutcome(
        status="completed",
        answer="42",
        usage=AgentUsage(input_tokens=16, output_tokens=3, llm_calls=2),
    )
    assert calls == ["a"]
    assert session.state.tool_statistics == {"measure": 1, "submit_answer": 1}
    assert session.state.submission == "42"


@pytest.mark.anyio()
async def test_react_agent_executes_tools_then_returns_typed_outcome(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(
                content=(
                    "<thought>measure</thought><action>measure</action>"
                    '<action_input>{"sample":"a"}</action_input>'
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2},
                id="react-1",
            ),
            SimpleNamespace(
                content=(
                    "<thought>done</thought><action>submit_answer</action>"
                    '<action_input>{"answer":"42"}</action_input>'
                ),
                usage={"prompt_tokens": 4, "completion_tokens": 2},
                id="react-2",
            ),
        ]
    )
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call", AsyncMock(side_effect=responses)
    )
    calls: list[str] = []
    session = await make_session(calls)
    outcome = await ReActAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    ).run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.llm_calls == 2
    assert calls == ["a"]
    assert session.state.tool_statistics == {"measure": 1, "submit_answer": 1}
    assert session.state.submission == "42"


@pytest.mark.anyio()
async def test_planner_uses_submit_tool_when_planning_already_solves_task(monkeypatch):
    model_call = AsyncMock(
        return_value=SimpleNamespace(
            content="Final Answer: 42",
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            id="planner-1",
        )
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    session = await make_session([])
    await session.record_message(
        {"role": "user", "content": "resume from canonical state"}
    )

    outcome = await LLMPlanner(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    ).run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert session.state.submission == "42"
    assert session.state.tool_statistics == {"submit_answer": 1}
    sent = model_call.await_args.kwargs["messages"]
    assert {"role": "user", "content": "resume from canonical state"} in sent


@pytest.mark.anyio()
async def test_planner_and_executor_share_one_task_interaction_budget(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(
                content="Measure first, then answer.",
                usage={"prompt_tokens": 2, "completion_tokens": 1},
                id="planner",
            ),
            SimpleNamespace(
                content="still thinking",
                usage={"prompt_tokens": 3, "completion_tokens": 1},
                id="executor-1",
            ),
            SimpleNamespace(
                content="still checking",
                usage={"prompt_tokens": 4, "completion_tokens": 1},
                id="executor-2",
            ),
        ]
    )
    model_call = AsyncMock(side_effect=responses)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    session = await make_session([], max_iterations=3)

    outcome = await LLMPlanner(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    ).run_session(session)

    assert outcome.status == "iteration_limit"
    assert outcome.usage.llm_calls == 3
    assert model_call.await_count == 3
