"""Tests for the ClaudeCodeAgent (Claude Code harness wrapper)."""

import asyncio

from corral.agents import ClaudeCodeAgent
from corral.agents import claude_code as claude_code_module
from corral.agents.claude_code import _to_sdk_content_blocks


class FakeAssistantMessage:
    def __init__(self, content):
        self.content = content


class FakeUserMessage:
    def __init__(self, content):
        self.content = content


class FakeTextBlock:
    def __init__(self, text):
        self.text = text


class FakeThinkingBlock:
    def __init__(self, thinking, signature=""):
        self.thinking = thinking
        self.signature = signature


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
    def __init__(self, result, usage=None, subtype="success", is_error=False):
        self.result = result
        self.usage = usage or {}
        self.subtype = subtype
        self.is_error = is_error
        self.stop_reason = None


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _default_mcp_status(options):
    """A healthy MCP status exposing exactly the run's allow-listed tools.

    The agent's preflight compares the tools exposed by the (faked) MCP endpoint
    against the REST allowlist, so by default we echo the allow-listed tools back
    as bare (un-namespaced) names — the shape the real tools/list returns.
    """
    exposed = [
        {"name": name.removeprefix("mcp__corral__")}
        for name in getattr(options, "allowed_tools", [])
    ]
    return {"mcpServers": [{"name": "corral", "status": "connected", "tools": exposed}]}


def _install_fake_sdk(monkeypatch, message_factory, mcp_status_factory=None):
    """Patch the SDK names in claude_code with fakes.

    The agent imports the Claude Agent SDK symbols at the top of
    corral.agents.claude_code, so the fakes are patched onto that module's
    namespace. message_factory receives the ClaudeAgentOptions and
    returns the list of messages the fake client's receive_response should
    yield. It also gets a captured dict to record what the harness saw.
    mcp_status_factory (given the options) returns the value the fake
    get_mcp_status reports; it defaults to a healthy, matching status.
    """
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
        claude_code_module, "ClaudeAgentOptions", FakeClaudeAgentOptions
    )
    monkeypatch.setattr(claude_code_module, "ClaudeSDKClient", FakeClaudeSDKClient)
    return captured


def test_model_is_split_between_harness_and_extractor():
    agent = ClaudeCodeAgent(model="claude-opus-4-8")
    assert agent.harness_model == "claude-opus-4-8"
    # Base machinery gets a LiteLLM-routable model derived from the harness one.
    assert agent.model == "anthropic/claude-opus-4-8"


def test_explicit_extractor_model_and_provider_prefixed_model():
    agent = ClaudeCodeAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.harness_model == "anthropic/claude-sonnet-4-5"
    assert agent.model == "anthropic/claude-sonnet-4-5"

    agent2 = ClaudeCodeAgent(model="claude-opus-4-8", extractor_model="openai/gpt-4o")
    assert agent2.model == "openai/gpt-4o"


def test_run_returns_final_answer_and_points_at_task_mcp(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [
            FakeAssistantMessage(
                [
                    FakeThinkingBlock("let me reason"),
                    FakeTextBlock("thinking out loud"),
                    FakeToolUseBlock("tu-1", "test_tool", {"query": "hi"}),
                ]
            ),
            FakeUserMessage([FakeToolResultBlock("tu-1", "tool-ok", is_error=False)]),
            FakeResultMessage(
                "42",
                usage={"input_tokens": 100, "output_tokens": 25},
            ),
        ]

    captured = _install_fake_sdk(monkeypatch, factory)

    agent = ClaudeCodeAgent(model="claude-opus-4-8", max_iterations=7)
    answer = agent.run(mock_interface, "task-1")

    # Answer extraction is off by default: the harness answer is submitted
    # verbatim and appears in the transcript exactly once (no duplicate append).
    assert answer == "42"
    assert [m.get("content") for m in agent.messages].count("42") == 1

    options = captured["options"]
    # The harness connects directly to the environment server's task-scoped MCP
    # endpoint (Streamable HTTP); the agent no longer bridges tools in-process.
    # The REST tool verbosity (defaulting to "brief") is forwarded to the MCP
    # endpoint so the harness sees the same tool descriptions the allowlist used.
    assert options.mcp_servers["corral"] == {
        "type": "http",
        "url": "http://test-server:8000/tasks/task-1/mcp/?verbosity=brief",
    }
    assert agent.harness_result.metadata["mcp_url"] == (
        "http://test-server:8000/tasks/task-1/mcp/?verbosity=brief"
    )
    assert agent.harness_result.metadata["tool_verbosity"] == "brief"
    # Only the corral tool is allowed; built-ins are disabled by fail-closed
    # config (empty base tool set + dontAsk), not a stale denylist.
    assert options.allowed_tools == ["mcp__corral__test_tool"]
    assert options.tools == []
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 7
    assert options.model == "claude-opus-4-8"
    # Fully isolated config: no filesystem settings/skills/plugins are loaded
    # and only the corral MCP server is used (strict_mcp_config).
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.skills == []
    assert options.plugins == []
    # A fresh, per-episode working directory is used (cleaned up after the run).
    assert isinstance(options.cwd, str)
    assert "corral-claude-" in options.cwd
    # Reasoning is configured explicitly, not left to a version-dependent default.
    assert options.effort == "high"
    assert options.thinking == {"type": "adaptive"}
    # The system prompt uses the real Claude Code preset (not a bare string).
    assert options.system_prompt["type"] == "preset"
    assert options.system_prompt["preset"] == "claude_code"

    # Thinking, tool-use, and tool-result blocks are recorded in the transcript.
    roles = [(m.get("role"), m.get("name")) for m in agent.messages]
    assert ("assistant", "thinking") in roles
    assert ("tool", "tool_result") in roles
    tool_call_msg = next(m for m in agent.messages if m.get("tool_calls"))
    assert tool_call_msg["tool_calls"][0]["function"]["name"] == "test_tool"

    # Token usage is mapped from the harness usage dict.
    usage = agent.get_total_token_usage()
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }
    # Cache token components are preserved separately for cost accounting.
    assert agent.token_usage["cache_read_input_tokens"] == 0
    assert agent.token_usage["cache_creation_input_tokens"] == 0
    assert agent.token_usage["input_tokens"] == 100

    # A structured run result records the outcome + provenance for the record.
    assert agent.harness_result is not None
    assert agent.harness_result.status == "success"
    assert agent.harness_result.answer == "42"
    meta = agent.harness_result.metadata
    assert meta["model_requested"] == "claude-opus-4-8"
    assert meta["reasoning_effort"] == "high"
    assert meta["max_turns_configured"] == 7
    assert "system_prompt_append_sha256" in meta
    assert "rest_tool_schema_sha256" in meta

    # First message is the task prompt; last message is the final answer.
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[-1]["content"] == "42"


def test_run_falls_back_to_assistant_text_without_result(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeAssistantMessage([FakeTextBlock("just the answer")])]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer == "just the answer"


def test_surrender_returns_give_up(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("SURRENDER")]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1", enable_surrender=True)
    assert answer == "SURRENDER"


def test_harness_error_is_caught(mock_interface, monkeypatch):
    async def factory(options, captured):
        raise RuntimeError("boom")

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    # Infrastructure failures are tagged so the benchmark does not score them
    # as a wrong model answer.
    assert agent.harness_result.status == "sdk_failure"


def test_error_result_message_is_reported_as_failure(mock_interface, monkeypatch):
    # An error ResultMessage (e.g. hitting the turn limit) must not be accepted
    # as a valid answer.
    async def factory(options, captured):
        return [FakeResultMessage("partial", subtype="error_max_turns", is_error=True)]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    # Budget exhaustion is distinguished from an ordinary failure/wrong answer.
    assert agent.harness_result.status == "max_turns"


def test_wall_clock_timeout_is_reported(mock_interface, monkeypatch):
    async def factory(options, captured):
        await asyncio.sleep(1.0)
        return [FakeResultMessage("too late")]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent(wall_clock_timeout_s=0.05)
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    assert agent.harness_result.status == "timeout"


def test_surrender_records_status(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("SURRENDER")]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1", enable_surrender=True)
    assert answer == "SURRENDER"
    assert agent.harness_result.status == "surrender"


def test_multimodal_prompt_is_forwarded(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("a cat")]

    captured = _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    task_prompt = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    answer = agent.run(mock_interface, "task-1", task_prompt=task_prompt)
    assert answer == "a cat"

    # Images go through the SDK streaming-input interface (an async iterable),
    # not the plain-string prompt.
    assert hasattr(captured["prompt"], "__aiter__")
    # The structured content is preserved in the transcript, image included.
    first = agent.messages[0]
    assert first["role"] == "user"
    assert any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in first["content"]
    )


def test_claude_code_bypasses_answer_extraction():
    # The harness returns a submit-ready answer, so the base class must not make
    # a second answer-extraction model call.
    agent = ClaudeCodeAgent(model="claude-opus-4-8")
    assert agent.requires_answer_extraction is False


def test_preflight_records_exposed_mcp_tool_metadata(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("ok")]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    agent.run(mock_interface, "task-1")

    meta = agent.harness_result.metadata
    # Provenance reflects the tools the harness actually sees via `tools/list`.
    assert meta["mcp_tools_exposed"] == ["test_tool"]
    # `get_mcp_status` exposes tool metadata, not input schemas — named honestly.
    assert "mcp_tool_metadata_sha256" in meta


def test_mcp_connection_failure_is_reported(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("unreachable")]

    def bad_status(options):
        return {
            "mcpServers": [
                {"name": "corral", "status": "failed", "error": "connection refused"}
            ]
        }

    _install_fake_sdk(monkeypatch, factory, mcp_status_factory=bad_status)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    # A dead MCP transport is an infrastructure/tool failure, not a wrong answer.
    assert agent.harness_result.status == "tool_failure"


def test_mcp_tool_mismatch_is_reported(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [FakeResultMessage("wrong tools")]

    def mismatched_status(options):
        return {
            "mcpServers": [
                {
                    "name": "corral",
                    "status": "connected",
                    "tools": [{"name": "some_other_tool"}],
                }
            ]
        }

    _install_fake_sdk(monkeypatch, factory, mcp_status_factory=mismatched_status)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    assert agent.harness_result.status == "sdk_failure"


def test_mcp_pending_state_is_retried(mock_interface, monkeypatch):
    # A transient `pending` MCP state must be retried, not treated as a failure.
    async def factory(options, captured):
        return [FakeResultMessage("eventually")]

    calls = {"n": 0}

    def flaky_status(options):
        calls["n"] += 1
        state = "pending" if calls["n"] == 1 else "connected"
        exposed = [
            {"name": name.removeprefix("mcp__corral__")}
            for name in options.allowed_tools
        ]
        return {"mcpServers": [{"name": "corral", "status": state, "tools": exposed}]}

    _install_fake_sdk(monkeypatch, factory, mcp_status_factory=flaky_status)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    assert answer == "eventually"
    assert calls["n"] >= 2  # polled again after the initial pending status


def test_tool_errors_are_counted(mock_interface, monkeypatch):
    async def factory(options, captured):
        return [
            FakeAssistantMessage([FakeToolUseBlock("tu-1", "test_tool", {})]),
            FakeUserMessage([FakeToolResultBlock("tu-1", "boom", is_error=True)]),
            FakeResultMessage("recovered"),
        ]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    answer = agent.run(mock_interface, "task-1")
    # A recoverable tool error does not fail the run, but is recorded.
    assert answer == "recovered"
    assert agent.harness_result.status == "success"
    assert agent.harness_result.metadata["tool_errors"] == 1


def test_default_system_prompt_omits_final_answer_marker():
    """With extraction off (default) the harness is not asked for the marker."""
    agent = ClaudeCodeAgent()
    assert agent.requires_answer_extraction is False
    append = agent._build_system_prompt(enable_surrender=False)["append"]
    assert "Final Answer:" not in append
    assert "reply with your final answer and nothing else" in append


def test_extraction_on_strips_marker_and_avoids_duplicate(mock_interface, monkeypatch):
    """When extraction is enabled, the `Final Answer:` marker is stripped once."""
    monkeypatch.setattr(
        ClaudeCodeAgent, "requires_answer_extraction", property(lambda self: True)
    )

    async def factory(options, captured):
        return [
            FakeAssistantMessage([FakeTextBlock("Final Answer: 42")]),
            FakeResultMessage("Final Answer: 42"),
        ]

    _install_fake_sdk(monkeypatch, factory)
    agent = ClaudeCodeAgent()
    # The harness is asked for the marker precisely because extraction will strip it.
    append = agent._build_system_prompt(enable_surrender=False)["append"]
    assert "Final Answer:" in append

    answer = agent.run(mock_interface, "task-1")
    assert answer == "42"
    contents = [m.get("content") for m in agent.messages]
    assert "Final Answer: 42" in contents
    assert contents.count("42") == 1


def test_data_uri_converted_to_sdk_image_block():
    blocks = _to_sdk_content_blocks(
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
