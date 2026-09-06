"""Tests for the first-class Claude Code session agent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from corral.agents import ClaudeCodeAgent
from corral.agents import claude_code as claude_code_module
from corral.agents.claude_code import _to_sdk_content_blocks
from corral.agents.schema import AgentOutcome
from corral.core.action import submit_answer_tool
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import ToolConnection
from corral.persistence import SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class FakeAssistantMessage:
    def __init__(self, content, model=None):
        self.content = content
        self.model = model


class FakeUserMessage:
    def __init__(self, content):
        self.content = content


class FakeTextBlock:
    def __init__(self, text):
        self.text = text


class FakeThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class FakeToolUseBlock:
    def __init__(self, id, name, input):  # noqa: A002
        self.id = id
        self.name = name
        self.input = input


class FakeToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class FakeResultMessage:
    def __init__(
        self,
        result,
        *,
        usage=None,
        subtype="success",
        is_error=False,
        num_turns=1,
        session_id="claude-session-1",
    ):
        self.result = result
        self.usage = usage or {}
        self.subtype = subtype
        self.is_error = is_error
        self.stop_reason = None
        self.num_turns = num_turns
        self.session_id = session_id


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _default_mcp_status(options):
    exposed = [
        {"name": name.removeprefix("mcp__corral__")}
        for name in getattr(options, "allowed_tools", [])
        if name.startswith("mcp__corral__")
    ]
    return {"mcpServers": [{"name": "corral", "status": "connected", "tools": exposed}]}


def _install_fake_sdk(monkeypatch, message_factory, mcp_status_factory=None):
    captured: dict = {}
    status_factory = mcp_status_factory or _default_mcp_status

    class FakeClaudeSDKClient:
        def __init__(self, options=None):
            self.options = options
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_mcp_status(self):
            return status_factory(self.options)

        async def query(self, prompt):
            captured["prompt"] = prompt
            if hasattr(prompt, "__aiter__"):
                captured["prompt_messages"] = [item async for item in prompt]

        async def receive_response(self):
            for message in await message_factory(self.options, captured):
                yield message

    monkeypatch.setattr(claude_code_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(claude_code_module, "UserMessage", FakeUserMessage)
    monkeypatch.setattr(claude_code_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(claude_code_module, "ThinkingBlock", FakeThinkingBlock)
    monkeypatch.setattr(claude_code_module, "ToolUseBlock", FakeToolUseBlock)
    monkeypatch.setattr(claude_code_module, "ToolResultBlock", FakeToolResultBlock)
    monkeypatch.setattr(claude_code_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(
        claude_code_module,
        "ClaudeAgentOptions",
        FakeClaudeAgentOptions,
    )
    monkeypatch.setattr(claude_code_module, "ClaudeSDKClient", FakeClaudeSDKClient)
    return captured


class FakeSession:
    def __init__(
        self,
        *,
        prompt="Test task",
        surrender_allowed=False,
        submission=None,
        submission_status=None,
    ):
        self.prompt = prompt
        self.tools = (
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            submit_answer_tool(),
        )
        self.workspace = None
        self.surrender_allowed = surrender_allowed
        self.iteration_limit = 7
        self.execution_id = "execution-1"
        self.messages: list[dict] = []
        self.tool_connection = ToolConnection("mcp", "http://127.0.0.1:8765/mcp")
        self.submission = submission
        self.submission_status = submission_status

    async def record_message(self, message):
        self.messages.append(dict(message))


def test_claude_is_only_a_new_session_agent():
    agent = ClaudeCodeAgent(model="claude-opus-4-8")

    assert agent.model == "claude-opus-4-8"
    assert not hasattr(agent, "max_turns")
    assert not hasattr(agent, "run")
    assert not hasattr(agent, "arun")
    assert not hasattr(agent, "step")
    assert not hasattr(agent, "arun_agent")
    assert not hasattr(agent, "messages")
    assert not hasattr(agent, "harness_result")


def test_claude_usage_includes_cached_and_cache_created_input_tokens():
    usage = ClaudeCodeAgent()._usage(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 7,
        },
        llm_calls=3,
    )

    assert usage.input_tokens == 112
    assert usage.output_tokens == 25
    assert usage.llm_calls == 3


@pytest.mark.anyio()
async def test_run_session_returns_typed_outcome_and_preserves_harness(
    monkeypatch,
):
    async def factory(options, captured):
        captured["cwd_exists_during_run"] = Path(options.cwd).is_dir()
        return [
            FakeAssistantMessage(
                [
                    FakeThinkingBlock("let me reason"),
                    FakeTextBlock("working"),
                    FakeToolUseBlock("tu-1", "test_tool", {"query": "hi"}),
                ],
                model="claude-opus-4-8",
            ),
            FakeUserMessage([FakeToolResultBlock("tu-1", "tool-ok")]),
            FakeResultMessage(
                "42",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "cache_read_input_tokens": 5,
                },
                num_turns=3,
            ),
        ]

    captured = _install_fake_sdk(monkeypatch, factory)
    session = FakeSession(submission="42", submission_status="submitted")
    outcome = await ClaudeCodeAgent().run_session(session)

    assert isinstance(outcome, AgentOutcome)
    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.error is None
    assert outcome.usage.input_tokens == 105
    assert outcome.usage.output_tokens == 25
    assert outcome.usage.llm_calls == 3
    assert outcome.usage.reasoning_tokens == 0
    assert outcome.metadata["session_id"] == "claude-session-1"
    assert outcome.metadata["mcp_tools_exposed"] == ["submit_answer", "test_tool"]

    options = captured["options"]
    assert captured["cwd_exists_during_run"] is True
    assert not Path(options.cwd).exists()
    assert options.model == "claude-opus-4-8"
    assert options.max_turns == 7
    assert options.tools == {"type": "preset", "preset": "claude_code"}
    assert options.allowed_tools == [
        "mcp__corral__test_tool",
        "mcp__corral__submit_answer",
    ]
    assert options.mcp_servers == {
        "corral": {"type": "http", "url": "http://127.0.0.1:8765/mcp"}
    }
    assert options.strict_mcp_config is True
    assert options.permission_mode == "auto"
    assert options.sandbox == {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
    }
    assert options.setting_sources == []
    assert options.skills == []
    assert options.plugins == []
    assert options.include_partial_messages is True
    assert options.system_prompt["preset"] == "claude_code"
    assert session.messages[0] == {"role": "user", "content": "Test task"}
    assert {"role": "assistant", "content": "42"} not in session.messages


@pytest.mark.anyio()
async def test_claude_hydrates_query_and_limit_from_session_state(monkeypatch):
    async def factory(_options, _captured):
        return [FakeResultMessage("42", num_turns=1)]

    captured = _install_fake_sdk(monkeypatch, factory)
    session = FakeSession(submission="42", submission_status="submitted")
    session.messages.append({"role": "user", "content": "resume from canonical state"})
    session.iteration_limit = 2

    outcome = await ClaudeCodeAgent().run_session(session)

    assert outcome.status == "completed"
    assert "resume from canonical state" in captured["prompt"]
    assert captured["options"].max_turns == 2


@pytest.mark.anyio()
@pytest.mark.parametrize(
    ("subtype", "expected_status"),
    [
        ("error_max_turns", "iteration_limit"),
        ("error_mcp_connection", "tool_failure"),
        ("error_mcp_tool_mismatch", "protocol_failure"),
        ("error_unknown", "harness_failure"),
    ],
)
async def test_sdk_failures_are_typed_not_error_answers(
    monkeypatch,
    subtype,
    expected_status,
):
    async def factory(options, captured):
        return [FakeResultMessage("partial", subtype=subtype, is_error=True)]

    _install_fake_sdk(monkeypatch, factory)
    outcome = await ClaudeCodeAgent().run_session(FakeSession())

    assert outcome.status == expected_status
    assert outcome.answer is None
    assert outcome.error


@pytest.mark.anyio()
async def test_wall_clock_timeout_is_a_typed_failure(monkeypatch):
    async def factory(options, captured):
        await asyncio.sleep(1)
        return [FakeResultMessage("too late")]

    _install_fake_sdk(monkeypatch, factory)
    outcome = await ClaudeCodeAgent(wall_clock_timeout_s=0.01).run_session(
        FakeSession()
    )

    assert outcome.status == "timeout"
    assert outcome.answer is None
    assert "timed out" in outcome.error


@pytest.mark.anyio()
async def test_surrender_is_status_not_a_magic_answer(monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("SURRENDER")]

    _install_fake_sdk(monkeypatch, factory)
    outcome = await ClaudeCodeAgent().run_session(
        FakeSession(
            surrender_allowed=True,
            submission="SURRENDER",
            submission_status="surrendered",
        )
    )

    assert outcome.status == "surrendered"
    assert outcome.answer is None
    assert outcome.error is None


@pytest.mark.anyio()
async def test_multimodal_prompt_uses_sdk_streaming_input(monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("a cat")]

    captured = _install_fake_sdk(monkeypatch, factory)
    prompt = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    outcome = await ClaudeCodeAgent().run_session(
        FakeSession(
            prompt=prompt,
            submission="a cat",
            submission_status="submitted",
        )
    )

    assert outcome.answer == "a cat"
    content = captured["prompt_messages"][0]["message"]["content"]
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


@pytest.mark.anyio()
async def test_mcp_tool_mismatch_is_a_protocol_failure(monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("wrong")]

    def mismatch(_options):
        return {
            "mcpServers": [
                {
                    "name": "corral",
                    "status": "connected",
                    "tools": [{"name": "other_tool"}],
                }
            ]
        }

    _install_fake_sdk(monkeypatch, factory, mismatch)
    outcome = await ClaudeCodeAgent().run_session(FakeSession())

    assert outcome.status == "protocol_failure"
    assert "tool mismatch" in outcome.error


@pytest.mark.anyio()
async def test_plain_text_result_is_not_a_submission(monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("42")]

    _install_fake_sdk(monkeypatch, factory)
    outcome = await ClaudeCodeAgent().run_session(FakeSession())

    assert outcome.status == "protocol_failure"
    assert "without calling submit_answer" in outcome.error


def test_data_uri_converted_to_sdk_image_block():
    assert _to_sdk_content_blocks(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    ) == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        }
    ]


@pytest.mark.anyio()
async def test_claude_submits_through_the_session_mcp_tool(monkeypatch, tmp_path):
    async def factory(options, captured):
        url = options.mcp_servers["corral"]["url"]
        async with (
            streamablehttp_client(url) as streams,
            ClientSession(streams[0], streams[1]) as mcp_session,
        ):
            await mcp_session.initialize()
            result = await mcp_session.call_tool("submit_answer", {"answer": "42"})
            assert result.isError is False
        return [
            FakeResultMessage(
                "42",
                usage={"input_tokens": 5, "output_tokens": 2},
                num_turns=1,
            )
        ]

    _install_fake_sdk(monkeypatch, factory)
    task = TaskDefinition(
        name="task",
        description="answer 42",
        tools=[],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={}, workspace_factory=None),
    )

    async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
        final = await TaskRuntime(store).run(
            ClaudeCodeAgent(),
            environment,
            execution_id="claude-task",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.runtime.status == "submitted"
    assert final.submission == "42"
    assert final.usage.input_tokens == 5
    assert final.usage.output_tokens == 2
    assert final.usage.llm_calls == 1
    assert final.usage.tool_calls == 1
    assert final.usage.agent_steps == 1
    assert final.tool_statistics == {"submit_answer": 1}
