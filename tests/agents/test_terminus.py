"""Tests for the session-owned Terminus loop."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corral.agents.session import ToolResponse
from corral.agents.terminus import TerminusAgent, TerminusResponse
from corral.core.action import submit_answer_tool


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class FakeSession:
    prompt = "measure sample a"
    surrender_allowed = False
    examples = ()
    iteration_limit = 10
    execution_id = "execution-1"
    tools = (
        {
            "type": "function",
            "function": {
                "name": "measure",
                "description": "measure a sample",
                "parameters": {
                    "type": "object",
                    "properties": {"sample": {"type": "string"}},
                    "required": ["sample"],
                },
            },
        },
        submit_answer_tool(),
    )

    def __init__(self):
        self.initial_state = SimpleNamespace(
            messages=(),
            metadata=SimpleNamespace(scaffold={}, task={"id": "task-1"}),
        )
        self.calls = []
        self.messages = []

    async def execute(self, action, **_kwargs):
        self.calls.append(action)
        result = "answer accepted" if action.name == "submit_answer" else "measured:a"
        return ToolResponse(success=True, result=result, error=None)

    async def record_message(self, message, **_kwargs):
        self.messages.append(message)

    def final_messages(self):
        return tuple(self.messages)


def response(payload, response_id):
    return SimpleNamespace(
        content=json.dumps(payload),
        id=response_id,
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )


def test_terminus_response_requires_exactly_one_terminal_choice():
    with pytest.raises(ValueError):
        TerminusResponse(
            analysis="done",
            plan="finish",
            tool_calls=[{"name": "submit_answer", "arguments": {"answer": "42"}}],
            surrender=True,
        )


@pytest.mark.anyio()
async def test_terminus_executes_through_session_and_returns_outcome(monkeypatch):
    responses = iter(
        [
            response(
                {
                    "analysis": "need a measurement",
                    "plan": "measure",
                    "tool_calls": [{"name": "measure", "arguments": '{"sample":"a"}'}],
                },
                "turn-1",
            ),
            response(
                {
                    "analysis": "done",
                    "plan": "finish",
                    "tool_calls": [
                        {"name": "submit_answer", "arguments": {"answer": "42"}}
                    ],
                },
                "turn-2",
            ),
        ]
    )
    model_call = AsyncMock(side_effect=responses)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    session = FakeSession()
    session.messages.append({"role": "user", "content": "resume from canonical state"})
    outcome = await TerminusAgent(
        model="test-model",
        system_prompt="system",
        confirmations_required=0,
    ).run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.llm_calls == 2
    assert [(call.name, dict(call.arguments)) for call in session.calls] == [
        ("measure", {"sample": "a"}),
        ("submit_answer", {"answer": "42"}),
    ]
    sent = model_call.await_args_list[0].kwargs["messages"]
    assert {"role": "user", "content": "resume from canonical state"} in sent
