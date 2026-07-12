"""Tests for the OpenHandsAgent (OpenHands harness wrapper).

The OpenHands SDK is an optional dependency; these tests patch the SDK names on
the ``corral.agents.openhands`` module with fakes that mimic the SDK's Agent /
Conversation / event shapes. The real ``ConversationExecutionStatus`` enum is
reused because it is a stable, dependency-free enum.
"""

import threading
import time

from openhands.sdk.conversation.state import ConversationExecutionStatus

from corral.agents import OpenHandsAgent
from corral.agents import openhands as openhands_module
from corral.agents.schema import SURRENDER_SENTINEL

# ---------------------------------------------------------------------------
# Fakes mirroring the openhands-sdk surface the agent depends on.
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeLLMConvertibleEvent:
    """Base marker class so `isinstance(event, LLMConvertibleEvent)` works."""


class FakeMessageEvent(FakeLLMConvertibleEvent):
    def __init__(self, role, content):
        self._message = FakeMessage(role, content)

    def to_llm_message(self):
        return self._message


class FakeFinishAction:
    def __init__(self, message):
        self.message = message


class FakeToolAction:
    """A non-finish action carrying structured tool arguments."""

    def __init__(self, **arguments):
        self._arguments = arguments

    def model_dump(self, mode=None):
        return dict(self._arguments)


class FakeActionEvent(FakeLLMConvertibleEvent):
    def __init__(self, action, tool_name="", tool_call_id="", thought=""):
        self.action = action
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.thought = thought

    def to_llm_message(self):
        message = getattr(self.action, "message", "")
        return FakeMessage("assistant", message)


class FakeObservationEvent(FakeLLMConvertibleEvent):
    def __init__(self, tool_name, tool_call_id, result, error=None):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.observation = result
        self.error = error

    def to_llm_message(self):
        return FakeMessage("tool", self.observation)


class FakeConversationErrorEvent:
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail


class FakeTokenUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMetrics:
    def __init__(self, prompt_tokens=100, completion_tokens=25, cost=0.5):
        self.accumulated_token_usage = FakeTokenUsage(prompt_tokens, completion_tokens)
        self.accumulated_cost = cost


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metrics = FakeMetrics()


class FakeAgentContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeMCPServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeStateAgent:
    def __init__(self, tools_map):
        self._tools_map = tools_map

    @property
    def tools_map(self):
        return self._tools_map


class FakeState:
    def __init__(self, execution_status, agent):
        self.execution_status = execution_status
        self.agent = agent


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConversation:
    def __init__(self, captured, events, block_seconds, run_error, state, **kwargs):
        self._captured = captured
        self._events = events
        self._block_seconds = block_seconds
        self._run_error = run_error
        self.state = state
        self.kwargs = kwargs
        captured["conversation_kwargs"] = kwargs
        self.callbacks = kwargs.get("callbacks", [])
        self._interrupted = threading.Event()

    def send_message(self, prompt):
        self._captured["sent_message"] = prompt

    def _emit(self):
        if self._run_error is not None:
            raise self._run_error
        if self._block_seconds:
            # Cooperative block so an interrupt can unwind the run promptly.
            waited = 0.0
            while waited < self._block_seconds and not self._interrupted.is_set():
                time.sleep(0.02)
                waited += 0.02
            if self._interrupted.is_set():
                return
        for event in self._events:
            for cb in self.callbacks:
                cb(event)

    def run(self):
        self._captured["ran"] = True
        self._emit()

    async def arun(self):
        self._captured["aran"] = True
        self._emit()

    def interrupt(self):
        self._captured["interrupted"] = True
        self._interrupted.set()

    def pause(self):
        self._captured["paused"] = True

    def close(self):
        self._captured["closed"] = self._captured.get("closed", 0) + 1


def _default_state(execution_status, runtime_tools):
    if runtime_tools is None:
        runtime_tools = {"finish": object(), "think": object(), "test_tool": object()}
    else:
        runtime_tools = {name: object() for name in runtime_tools}
    return FakeState(execution_status, FakeStateAgent(runtime_tools))


def _install_fake_sdk(
    monkeypatch,
    events,
    *,
    block_seconds=0.0,
    run_error=None,
    execution_status=ConversationExecutionStatus.FINISHED,
    runtime_tools=None,
):
    """Patch the SDK names on openhands_module and capture what the agent saw."""
    captured: dict = {}
    state = _default_state(execution_status, runtime_tools)

    def _agent_factory(**kwargs):
        captured["agent_kwargs"] = kwargs
        return FakeAgent(**kwargs)

    def _conversation_factory(**kwargs):
        return FakeConversation(
            captured, events, block_seconds, run_error, state, **kwargs
        )

    def _llm_factory(**kwargs):
        llm = FakeLLM(**kwargs)
        captured["llm"] = llm
        return llm

    monkeypatch.setattr(openhands_module, "Agent", _agent_factory)
    monkeypatch.setattr(openhands_module, "Conversation", _conversation_factory)
    monkeypatch.setattr(openhands_module, "LLM", _llm_factory)
    monkeypatch.setattr(openhands_module, "AgentContext", FakeAgentContext)
    monkeypatch.setattr(openhands_module, "MCPServer", FakeMCPServer)
    monkeypatch.setattr(
        openhands_module, "LLMConvertibleEvent", FakeLLMConvertibleEvent
    )
    monkeypatch.setattr(openhands_module, "ActionEvent", FakeActionEvent)
    monkeypatch.setattr(openhands_module, "ObservationEvent", FakeObservationEvent)
    monkeypatch.setattr(openhands_module, "FinishAction", FakeFinishAction)
    monkeypatch.setattr(
        openhands_module, "ConversationErrorEvent", FakeConversationErrorEvent
    )
    return captured


def _finish(message):
    return FakeActionEvent(FakeFinishAction(message))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_split_between_harness_and_extractor():
    agent = OpenHandsAgent(model="openai/gpt-5.6")
    assert agent.harness_model == "openai/gpt-5.6"
    # An already-routable model is reused verbatim for the base machinery.
    assert agent.model == "openai/gpt-5.6"
    # No second (extractor) model call: the harness answer is submitted verbatim.
    assert agent.requires_answer_extraction is False

    agent2 = OpenHandsAgent(model="gpt-5.6", extractor_model="openai/gpt-4o")
    assert agent2.harness_model == "gpt-5.6"
    assert agent2.model == "openai/gpt-4o"


def test_run_returns_final_answer_and_reuses_task_mcp(mock_interface, monkeypatch):
    events = [
        FakeMessageEvent("assistant", "let me think"),
        _finish("42"),
    ]
    captured = _install_fake_sdk(monkeypatch, events)

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")

    # Answer extraction is off by default: the harness answer is submitted
    # verbatim and appears in the transcript exactly once (no duplicate append).
    assert answer == "42"
    assert [m.get("content") for m in agent.messages].count("42") == 1

    # The harness connects to the environment server's task-scoped MCP endpoint,
    # with the REST tool verbosity (default "brief") forwarded as a query param.
    meta = agent.harness_result.metadata
    assert meta["mcp_url"] == "http://test-server:8000/tasks/task-1/mcp?verbosity=brief"
    assert meta["tool_verbosity"] == "brief"
    assert meta["mcp_tools_enabled"] == ["test_tool"]
    assert meta["model_requested"] == "openai/gpt-5.6"
    # The schema hash reflects the MCP schema OpenHands actually sees.
    assert meta["mcp_tool_schema_sha256"] == "deadbeef"
    # The runtime tool set is recorded and only the expected tools are present.
    assert meta["runtime_tools"] == ["finish", "test_tool", "think"]
    assert meta["system_prompt_identity_pinned"] is True

    # Token usage mapped from the OpenHands LLM metrics.
    usage = agent.get_total_token_usage()
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 125

    assert agent.harness_result.status == "success"
    assert agent.harness_result.answer == "42"
    assert agent.harness_result.total_cost_usd == 0.5
    # First message is the task prompt; last message is the final answer.
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[-1]["content"] == "42"
    # The conversation is always closed, and the default visualizer is disabled.
    assert captured["closed"] == 1
    assert captured["conversation_kwargs"]["visualizer"] is None


def test_tool_isolation(mock_interface, monkeypatch):
    """OpenHands receives no built-in acting tools; only the corral MCP endpoint."""
    captured = _install_fake_sdk(monkeypatch, [_finish("ok")])

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    agent.run(mock_interface, "task-1")

    agent_kwargs = captured["agent_kwargs"]
    assert agent_kwargs["tools"] == []
    assert agent_kwargs["include_default_tools"] == ["FinishTool", "ThinkTool"]
    # The agent identity is pinned so it cannot inherit a machine-local SOUL.md.
    assert agent_kwargs["system_prompt_kwargs"]["soul_content"]
    assert len(agent_kwargs["mcp_config"]) == 1
    assert "corral" in agent_kwargs["mcp_config"]
    server = agent_kwargs["mcp_config"]["corral"]
    assert server.kwargs["url"] == (
        "http://test-server:8000/tasks/task-1/mcp?verbosity=brief"
    )
    assert server.kwargs["transport"] == "streamable-http"


def test_unexpected_runtime_tool_is_sdk_failure(mock_interface, monkeypatch):
    """A tool OpenHands adds outside the allowlist fails the run (capability leak)."""
    _install_fake_sdk(
        monkeypatch,
        [_finish("42")],
        runtime_tools={"finish", "think", "test_tool", "browser_tool_set"},
    )

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "sdk_failure"
    assert "browser_tool_set" in agent.harness_result.error


def test_structured_tool_call_is_recorded(mock_interface, monkeypatch):
    """Tool calls/results are recorded structurally (name, id, args, result)."""
    events = [
        FakeActionEvent(
            FakeToolAction(query="q"),
            tool_name="test_tool",
            tool_call_id="call_1",
            thought="let me use the tool",
        ),
        FakeObservationEvent("test_tool", "call_1", "tool output"),
        _finish("42"),
    ]
    _install_fake_sdk(monkeypatch, events)

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    agent.run(mock_interface, "task-1")

    tool_calls = [m for m in agent.messages if m.get("tool_calls")]
    assert len(tool_calls) == 1
    fn = tool_calls[0]["tool_calls"][0]["function"]
    assert fn["name"] == "test_tool"
    assert '"query": "q"' in fn["arguments"]
    assert tool_calls[0]["tool_calls"][0]["id"] == "call_1"

    tool_results = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0]["name"] == "test_tool"
    assert tool_results[0]["tool_call_id"] == "call_1"
    assert tool_results[0]["content"] == "tool output"

    # The model's reasoning is preserved as a thinking message.
    assert any(m.get("name") == "thinking" for m in agent.messages)


def test_mcp_url_encodes_task_id_and_verbosity(mock_interface, monkeypatch):
    _install_fake_sdk(monkeypatch, [_finish("ok")])
    mock_interface.current_verbosity = "full"

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    url = agent._mcp_url(mock_interface, "task/with space", "full")
    assert url == (
        "http://test-server:8000/tasks/task%2Fwith%20space/mcp?verbosity=full"
    )


def test_finish_action_returned_verbatim_without_extraction(
    mock_interface, monkeypatch
):
    """A FinishAction message is returned exactly, with no answer-extractor call."""
    _install_fake_sdk(monkeypatch, [_finish("42")])

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    # Guard: any extractor call would flip this contract.
    assert agent.requires_answer_extraction is False
    assert agent.run(mock_interface, "task-1") == "42"


def test_surrender_only_on_exact_sentinel(mock_interface, monkeypatch):
    _install_fake_sdk(monkeypatch, [_finish(SURRENDER_SENTINEL)])

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1", enable_surrender=True)
    assert answer == SURRENDER_SENTINEL
    assert agent.harness_result.status == "surrender"


def test_sentinel_inside_longer_answer_is_not_surrender(mock_interface, monkeypatch):
    longer = f"The answer mentions {SURRENDER_SENTINEL} but is not a surrender."
    _install_fake_sdk(monkeypatch, [_finish(longer)])

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1", enable_surrender=True)
    assert answer == longer
    assert agent.harness_result.status == "success"


def test_missing_finish_action_is_sdk_failure(mock_interface, monkeypatch):
    """A run that never emits a FinishAction is an infra failure, not an answer."""
    _install_fake_sdk(monkeypatch, [FakeMessageEvent("assistant", "thinking...")])

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "sdk_failure"


def test_max_iterations_error_event_is_reported(mock_interface, monkeypatch):
    """Reaching max iterations (no exception) is reported as `max_iterations`."""
    _install_fake_sdk(
        monkeypatch,
        [FakeConversationErrorEvent("MaxIterationsReached", "reached max iterations")],
        execution_status=ConversationExecutionStatus.ERROR,
    )

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "max_iterations"


def test_stuck_detection_is_reported(mock_interface, monkeypatch):
    """A STUCK terminal status (no exception) is reported as `stuck`."""
    _install_fake_sdk(
        monkeypatch,
        [FakeMessageEvent("assistant", "looping")],
        execution_status=ConversationExecutionStatus.STUCK,
    )

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "stuck"


def test_mcp_connection_failure_is_tool_failure(mock_interface, monkeypatch):
    captured = _install_fake_sdk(
        monkeypatch,
        [],
        run_error=RuntimeError("MCP server failed to connect"),
    )

    agent = OpenHandsAgent(model="openai/gpt-5.6", api_key="test-key")
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "tool_failure"
    # The conversation is closed even when run() raises.
    assert captured["closed"] == 1


def test_wall_clock_timeout_interrupts_and_closes(mock_interface, monkeypatch):
    """An overrunning run is interrupted + closed and reported as a timeout."""
    captured = _install_fake_sdk(monkeypatch, [_finish("42")], block_seconds=2.0)

    agent = OpenHandsAgent(
        model="openai/gpt-5.6", api_key="test-key", wall_clock_timeout_s=0.1
    )
    answer = agent.run(mock_interface, "task-1")
    assert "Error solving the task" in answer
    assert agent.harness_result.status == "timeout"
    # The async entrypoint is driven and cancelled via interrupt(), not pause().
    assert captured.get("aran") is True
    assert captured.get("interrupted") is True
    assert captured["closed"] == 1
