"""Tests for the CodexAgent (OpenAI Codex harness wrapper).

The Codex SDK is an optional dependency and is not installed in CI, so these
tests patch the SDK names on the corral.agents.codex module with fakes that
mimic the SDK's Codex / Thread / turn-stream shapes.
"""

import pytest

from corral.agents import CodexAgent
from corral.agents import codex as codex_module


@pytest.fixture(autouse=True)
def _require_api_key(monkeypatch):
    """CodexAgent.run now requires OPENAI_API_KEY (isolated CODEX_HOME)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Fakes mirroring the openai_codex SDK surface the agent depends on.
# ---------------------------------------------------------------------------


class FakeApprovalMode:
    deny_all = "deny_all"
    auto_review = "auto_review"


class FakeSandbox:
    read_only = "read-only"


class FakeCodexConfig:
    def __init__(self, *, cwd=None, env=None):
        self.cwd = cwd
        self.env = env


class _Enum:
    """Minimal stand-in for a pydantic enum exposing .value."""

    def __init__(self, value):
        self.value = value


class FakeItem:
    """A completed thread item (ItemCompletedNotification.item.root)."""

    def __init__(self, type, **kwargs):  # noqa: A002
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTurn:
    def __init__(self, status="completed", error=None, duration_ms=42):
        self.status = _Enum(status)
        self.error = error
        self.duration_ms = duration_ms


class _Payload:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeEvent:
    def __init__(self, method, payload):
        self.method = method
        self.payload = payload


class FakeTurnHandle:
    def __init__(self, events, interrupts):
        self._events = events
        self._interrupts = interrupts

    def interrupt(self):
        self._interrupts.append(True)

    def stream(self):
        return _ClosableIter(self._events)


class _ClosableIter:
    def __init__(self, events):
        self._it = iter(events)

    def __iter__(self):
        return self._it

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


class FakeThread:
    def __init__(self, events, captured, interrupts):
        self._events = events
        self._captured = captured
        self._interrupts = interrupts

    def turn(self, prompt, **kwargs):
        self._captured["turn_prompt"] = prompt
        self._captured["turn_kwargs"] = kwargs
        return FakeTurnHandle(self._events, self._interrupts)


class FakeCodex:
    def __init__(self, events, captured, interrupts):
        self._events = events
        self._captured = captured
        self._interrupts = interrupts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login_api_key(self, api_key):
        self._captured["login_api_key"] = api_key

    def thread_start(self, **kwargs):
        self._captured["thread_start_kwargs"] = kwargs
        return FakeThread(self._events, self._captured, self._interrupts)


def _install_fake_sdk(monkeypatch, events):
    """Patch the SDK names on codex_module and capture what Codex saw."""
    captured: dict = {"interrupts": []}

    def _codex_factory(config=None):
        captured["config"] = config
        return FakeCodex(events, captured, captured["interrupts"])

    monkeypatch.setattr(codex_module, "Codex", _codex_factory)
    monkeypatch.setattr(codex_module, "CodexConfig", FakeCodexConfig)
    monkeypatch.setattr(codex_module, "ApprovalMode", FakeApprovalMode)
    monkeypatch.setattr(codex_module, "Sandbox", FakeSandbox)
    return captured


def _usage_event():
    total = _Payload(
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=25,
        total_tokens=125,
    )
    return FakeEvent(
        "thread/tokenUsage/updated",
        _Payload(token_usage=_Payload(total=total), turn_id="t1"),
    )


def _agent_message_event(text, phase="final_answer"):
    item = FakeItem("agentMessage", text=text, phase=_Enum(phase) if phase else None)
    return FakeEvent("item/completed", _Payload(item=_Payload(root=item), turn_id="t1"))


def _tool_call_event(tool, arguments, result=None, error=None):
    item = FakeItem(
        "mcpToolCall",
        id="tc-1",
        tool=tool,
        server="corral",
        arguments=arguments,
        result=result,
        error=error,
    )
    return FakeEvent("item/completed", _Payload(item=_Payload(root=item), turn_id="t1"))


def _completed_event(status="completed", error=None):
    return FakeEvent(
        "turn/completed", _Payload(turn=FakeTurn(status=status, error=error))
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_split_between_harness_and_extractor():
    agent = CodexAgent(model="gpt-5.4")
    assert agent.harness_model == "gpt-5.4"
    # Base machinery gets a LiteLLM-routable model derived from the harness one.
    assert agent.model == "openai/gpt-5.4"

    agent2 = CodexAgent(model="gpt-5.4", extractor_model="openai/gpt-4o")
    assert agent2.model == "openai/gpt-4o"
    # No second (extractor) model call: the harness answer is submitted verbatim.
    assert agent2.requires_answer_extraction is False


def test_run_returns_final_answer_and_reuses_task_mcp(mock_interface, monkeypatch):
    events = [
        FakeEvent(
            "item/completed",
            _Payload(
                item=_Payload(root=FakeItem("reasoning", content=["let me think"])),
                turn_id="t1",
            ),
        ),
        _tool_call_event("test_tool", {"query": "hi"}, result="tool-ok"),
        _usage_event(),
        _agent_message_event("42"),
        _completed_event(),
    ]
    captured = _install_fake_sdk(monkeypatch, events)

    agent = CodexAgent(model="gpt-5.4", reasoning_effort="high")
    answer = agent.run(mock_interface, "task-1")

    # Answer extraction is off by default: the harness answer is submitted
    # verbatim and appears in the transcript exactly once (no duplicate append).
    assert answer == "42"
    assert [m.get("content") for m in agent.messages].count("42") == 1

    # The harness connects to the environment server's task-scoped MCP endpoint,
    # with the REST tool verbosity (default "brief") forwarded as a query param.
    meta = agent.harness_result.metadata
    assert meta["mcp_url"] == (
        "http://test-server:8000/tasks/task-1/mcp?verbosity=brief"
    )
    assert meta["tool_verbosity"] == "brief"
    assert meta["mcp_tools_enabled"] == ["test_tool"]
    assert meta["model_requested"] == "gpt-5.4"
    # The schema hash reflects the MCP schema Codex actually sees.
    assert meta["mcp_tool_schema_sha256"] == "deadbeef"

    # Read-only sandbox + deny-all approvals + isolated CODEX_HOME.
    ts_kwargs = captured["thread_start_kwargs"]
    assert ts_kwargs["sandbox"] == FakeSandbox.read_only
    assert ts_kwargs["approval_mode"] == FakeApprovalMode.deny_all
    assert ts_kwargs["ephemeral"] is True
    assert ts_kwargs["model"] == "gpt-5.4"
    config = captured["config"]
    assert "CODEX_HOME" in config.env
    assert "corral-codex-" in config.env["CODEX_HOME"]
    # The isolated CODEX_HOME has no auth.json, so the API key is logged in
    # explicitly (otherwise the harness hits /v1/responses with no bearer token).
    assert captured["login_api_key"] == "test-key"
    # Effort is configured explicitly on the turn.
    assert captured["turn_kwargs"]["effort"] == "high"

    # Reasoning, tool-use, and tool-result are recorded in the transcript. The
    # tool-result message is named after the actual tool, not a generic label.
    roles = [(m.get("role"), m.get("name")) for m in agent.messages]
    assert ("assistant", "thinking") in roles
    assert ("tool", "test_tool") in roles
    tool_call_msg = next(m for m in agent.messages if m.get("tool_calls"))
    assert tool_call_msg["tool_calls"][0]["function"]["name"] == "test_tool"

    # Token usage mapped from the Codex thread usage.
    usage = agent.get_total_token_usage()
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 125
    assert agent.token_usage["cached_input_tokens"] == 10

    assert agent.harness_result.status == "success"
    assert agent.harness_result.answer == "42"
    assert agent.harness_result.num_tool_calls == 1
    assert agent.harness_result.duration_ms == 42
    # First message is the task prompt; last message is the final answer.
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[-1]["content"] == "42"


def test_config_toml_is_isolated_and_task_scoped(monkeypatch, mock_interface, tmp_path):
    """The generated config disables built-ins and allowlists only task tools."""
    agent = CodexAgent(model="gpt-5.4")
    toml = agent._render_config_toml(
        "http://test-server:8000/tasks/task-1/mcp?verbosity=brief", ["add"]
    )
    assert 'web_search = "disabled"' in toml
    assert "shell_tool = false" in toml
    assert "unified_exec = false" in toml
    assert "multi_agent = false" in toml
    assert "[mcp_servers.corral]" in toml
    assert 'url = "http://test-server:8000/tasks/task-1/mcp?verbosity=brief"' in toml
    assert "required = true" in toml
    assert 'enabled_tools = ["add"]' in toml
    # Allowlisted corral tools run without interactive approval (non-interactive
    # benchmark); this does not widen access beyond enabled_tools.
    assert 'default_tools_approval_mode = "approve"' in toml


def test_unexpected_interrupt_is_infra_failure(mock_interface, monkeypatch):
    """An interrupted turn with no local timeout cause is a failure."""
    events = [
        _agent_message_event("partial", phase=None),
        _completed_event(status="interrupted"),
    ]
    _install_fake_sdk(monkeypatch, events)

    agent = CodexAgent(model="gpt-5.4")
    answer = agent.run(mock_interface, "task-1")

    assert "Error solving the task" in answer
    assert agent.harness_result.status == "sdk_failure"


def test_missing_api_key_raises(mock_interface, monkeypatch):
    """Without OPENAI_API_KEY the agent fails fast (isolated CODEX_HOME)."""
    _install_fake_sdk(monkeypatch, [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = CodexAgent(model="gpt-5.4")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent.run(mock_interface, "task-1")


def test_failed_turn_is_infra_failure(mock_interface, monkeypatch):
    events = [_completed_event(status="failed", error=_Payload(message="boom"))]
    _install_fake_sdk(monkeypatch, events)

    agent = CodexAgent(model="gpt-5.4")
    answer = agent.run(mock_interface, "task-1")

    assert "Error solving the task" in answer
    assert agent.harness_result.status == "sdk_failure"
    assert "boom" in agent.harness_result.error


def test_default_instructions_omit_final_answer_marker():
    """With extraction off (default) the harness is not asked for the marker."""
    agent = CodexAgent(model="gpt-5.4")
    assert agent.requires_answer_extraction is False
    instructions = agent._developer_instructions(enable_surrender=False)
    assert "Final Answer:" not in instructions
    assert "reply with your final answer and nothing else" in instructions


def test_extraction_on_strips_marker_and_avoids_duplicate(mock_interface, monkeypatch):
    """When extraction is enabled, the `Final Answer:` marker is stripped once."""
    # Enable extraction so the deterministic marker-stripping path runs.
    monkeypatch.setattr(
        CodexAgent, "requires_answer_extraction", property(lambda self: True)
    )
    events = [
        _agent_message_event("Final Answer: 42"),
        _completed_event(),
    ]
    _install_fake_sdk(monkeypatch, events)

    agent = CodexAgent(model="gpt-5.4")
    # The harness is asked for the marker precisely because extraction will strip it.
    assert "Final Answer:" in agent._developer_instructions(enable_surrender=False)

    answer = agent.run(mock_interface, "task-1")
    assert answer == "42"
    # The raw marked message and the normalized answer are both present, but the
    # normalized answer is appended only once.
    contents = [m.get("content") for m in agent.messages]
    assert "Final Answer: 42" in contents
    assert contents.count("42") == 1
