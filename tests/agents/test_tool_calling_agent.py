"""Focused tests for the provider-native session loop."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corral.agents.session import ToolResponse
from corral.agents.tool_calling import ToolCallingAgent


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class Session:
    prompt = "solve"
    surrender_allowed = False
    examples = ()
    iteration_limit = 10
    tools = (
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )
    initial_state = SimpleNamespace(messages=(), metadata=SimpleNamespace(scaffold={}))

    def __init__(self):
        self.calls = []
        self.messages = []

    def final_messages(self):
        return tuple(self.messages)

    async def record_message(self, message, **_kwargs):
        self.messages.append(message)

    async def execute(self, action, **_kwargs):
        self.calls.append(action)
        return ToolResponse(success=True, result="pong", error=None)


@pytest.mark.anyio()
async def test_multiple_provider_calls_are_serialized_before_submission(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(
                content=None,
                tool_calls=[call("ping", "{}", "one"), call("ping", "{}", "two")],
                usage=None,
            ),
            SimpleNamespace(
                content=None,
                tool_calls=[call("submit_answer", '{"answer":"done"}', "three")],
                usage=None,
            ),
        ]
    )
    model_call = AsyncMock(side_effect=responses)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    session = Session()
    session.messages.append({"role": "user", "content": "resume from canonical state"})
    outcome = await ToolCallingAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    ).run_session(session)

    assert outcome.status == "completed"
    assert [action.id for action in session.calls] == ["one", "two", "three"]
    sent = model_call.await_args_list[0].kwargs["messages"]
    assert {"role": "user", "content": "resume from canonical state"} in sent


@pytest.mark.anyio()
async def test_plain_text_never_bypasses_runtime_submission(monkeypatch):
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        AsyncMock(
            return_value=SimpleNamespace(
                content="Final Answer: 42", tool_calls=[], usage=None, id=None
            )
        ),
    )
    session = Session()
    session.iteration_limit = 1
    agent = ToolCallingAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    )
    outcome = await agent.run_session(session)

    assert outcome.status == "iteration_limit"
    assert session.calls == []
