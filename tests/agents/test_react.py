"""Focused tests for the ReAct session agent."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corral.agents.react import ReActAgent


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def agent():
    return ReActAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}",
    )


def test_parse_llm_response_returns_actions(agent):
    thoughts, actions = agent.parse_llm_response(
        "<thought>inspect</thought><action>measure</action>"
        '<action_input>{"sample":"a"}</action_input>'
    )

    assert thoughts[0].content == "inspect"
    assert actions[0].name == "measure"
    assert actions[0].arguments == {"sample": "a"}


def test_python_literals_inside_strings_are_left_alone(agent):
    """Code written through a tool must keep Python's True/False/None."""
    code = 'MANIFEST = {"concurrent": True, "x": False, "y": None}\n'
    # Bare True in the same input forces the Python-literal fallback path.
    raw = (
        '{"path": "policy.py", "overwrite": True, "content": ' + json.dumps(code) + "}"
    )
    _thoughts, actions = agent.parse_llm_response(
        f"<action>write_file</action><action_input>{raw}</action_input>"
    )

    assert actions[0].arguments["content"] == code
    assert actions[0].arguments["overwrite"] is True


def test_bare_python_literals_are_accepted_as_json(agent):
    _thoughts, actions = agent.parse_llm_response(
        "<action>run</action>"
        '<action_input>{"dry": True, "cmd": "echo True", "limit": None}</action_input>'
    )

    assert actions[0].arguments == {"dry": True, "cmd": "echo True", "limit": None}


def test_parse_llm_response_rejects_missing_action_input(agent):
    _thoughts, actions = agent.parse_llm_response(
        "<thought>inspect</thought><action>measure</action>"
    )

    assert actions is None


@pytest.mark.anyio
async def test_iteration_limit_is_a_typed_outcome(monkeypatch, agent):
    model_call = AsyncMock(
        return_value=SimpleNamespace(content="not an action", usage=None, id=None)
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)

    class Session:
        prompt = "solve"
        surrender_allowed = False
        tools = ()
        examples = ()
        iteration_limit = 2
        initial_state = SimpleNamespace(
            messages=(), metadata=SimpleNamespace(scaffold={})
        )

        def __init__(self):
            self.messages = [{"role": "user", "content": "resume from canonical state"}]

        def final_messages(self):
            return tuple(self.messages)

        async def record_message(self, message, **_kwargs):
            self.messages.append(message)

    outcome = await agent.run_session(Session())

    assert outcome.status == "iteration_limit"
    assert outcome.error == "agent exhausted its 2 interaction budget"
    sent = model_call.await_args_list[0].kwargs["messages"]
    assert {"role": "user", "content": "resume from canonical state"} in sent


@pytest.mark.anyio
async def test_model_failure_keeps_only_the_final_error_message(monkeypatch, agent):
    async def fail_model(**_kwargs):
        try:
            raise ValueError("Required field 'answer' is missing.")
        except ValueError as cause:
            raise RuntimeError("provider adapter failed") from cause

    monkeypatch.setattr("corral.agents.base_agent.llm_call", fail_model)

    class Session:
        prompt = "solve"
        surrender_allowed = False
        tools = ()
        examples = ()
        iteration_limit = 1
        initial_state = SimpleNamespace(
            messages=(), metadata=SimpleNamespace(scaffold={})
        )

        def __init__(self):
            self.messages = []

    outcome = await agent.run_session(Session())

    assert outcome.status == "agent_failure"
    assert outcome.error == "Required field 'answer' is missing."


@pytest.mark.anyio
async def test_malformed_action_name_is_not_sent_as_a_message_name(monkeypatch, agent):
    garbled = "` in final, got output. Let's do plain text final tags " * 4
    model_call = AsyncMock(
        side_effect=[
            SimpleNamespace(
                content=f"<action>{garbled}</action><action_input>{{}}</action_input>",
                usage=None,
                id=None,
            ),
            SimpleNamespace(
                content="<action>PythonREPL</action><action_input>{}</action_input>",
                usage=None,
                id=None,
            ),
            SimpleNamespace(content="not an action", usage=None, id=None),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)

    class Session:
        prompt = "solve"
        surrender_allowed = False
        tools = ()
        examples = ()
        iteration_limit = 3
        initial_state = SimpleNamespace(
            messages=(), metadata=SimpleNamespace(scaffold={})
        )

        def __init__(self):
            self.messages = []

        def final_messages(self):
            return tuple(self.messages)

        async def record_message(self, message, **_kwargs):
            self.messages.append(message)

        async def execute(self, action, **_kwargs):
            return SimpleNamespace(success=False, result=None, error="unknown tool")

    await agent.run_session(Session())

    sent = model_call.await_args_list[2].kwargs["messages"]
    observations = [m for m in sent if m.get("content") == "Error: unknown tool"]
    assert len(observations) == 2
    assert "name" not in observations[0]
    assert observations[1]["name"] == "PythonREPL"
