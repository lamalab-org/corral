"""Real-SDK boundary tests for :class:`OpenHandsAgent`.

Unlike `test_openhands.py` — which fakes the SDK to exercise the agent's
control flow — these tests construct *real* `openhands-sdk` objects. They pin
down the exact SDK shapes the transcript/serialization code depends on
(`Observation.is_error`, `MCPToolAction.data`, `AgentContext` datetime,
the MCP executor timeout), which fakes cannot catch: a fake that mirrors a buggy
implementation would agree with it. If the SDK moves these, these tests fail
instead of the trace silently going wrong.

The `openhands` extra is gated to Python >= 3.12, so the whole module is
skipped when the SDK is not importable.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import anyio
import httpx
import pytest
from litellm import ModelResponse
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    Timeout,
)
from pydantic import ValidationError

pytest.importorskip("openhands.sdk")

from openhands.sdk import AgentContext, LocalConversation
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.llm.exceptions import LLMNoResponseError
from openhands.sdk.mcp.definition import (
    MCPToolAction,
    MCPToolObservation,
)
from openhands.sdk.mcp.tool import MCP_TOOL_TIMEOUT_SECONDS

from corral.agents.openhands import (
    _OPENHANDS_DEFAULT_EXECUTOR_TIMEOUT_S,
    OpenHandsAgent,
    _TimeoutMCPToolProvider,
)
from corral.backend.mcp import open_mcp_host
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot
from corral.orchestration.parameters import model_parameter_metadata


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
def test_generic_provider_error_retries_same_request(monkeypatch, asynchronous, status):
    """A transient generic APIError recovers inside the existing conversation."""
    error = APIError(status, "provider unavailable", "openai/gpt-4o", "openrouter")
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "Recovered"}}]
    )
    transport = (AsyncMock if asynchronous else Mock)(side_effect=[error, response])
    method = "_atransport_call" if asynchronous else "_transport_call"
    monkeypatch.setattr(LLM, method, transport)
    warnings = []
    monkeypatch.setattr("corral.agents.utils.logger.warning", warnings.append)
    llm = OpenHandsAgent(model="openai/gpt-4o", api_key="offline-test")._make_llm()
    llm.num_retries = 2
    llm.retry_min_wait = llm.retry_max_wait = 0
    listener = Mock()
    llm.retry_listener = listener
    messages = [Message(role="user", content=[TextContent(text="Hello")])]
    on_token = Mock()

    result = (
        asyncio.run(llm.acompletion(messages, on_token=on_token))
        if asynchronous
        else llm.completion(messages, on_token=on_token)
    )

    assert result.message.content[0].text == "Recovered"
    assert transport.call_count == 2
    assert transport.call_args_list[0] == transport.call_args_list[1]
    assert transport.call_args.kwargs["enable_streaming"] is True
    listener.assert_called_once_with(1, 2, error)
    assert len(warnings) == 1
    assert "1/2" in warnings[0]
    assert "APIError" in warnings[0]
    assert "provider unavailable" in warnings[0]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (APIError(400, "invalid request", "test", "openrouter"), 1),
        (APIError(401, "invalid key", "test", "openrouter"), 1),
        (APIError(402, "payment required", "test", "openrouter"), 1),
        (APIError(403, "forbidden", "test", "openrouter"), 1),
        (APIError(429, "insufficient_quota", "test", "openrouter"), 1),
        (AuthenticationError("invalid key", "test", "openrouter"), 1),
        (RateLimitError("insufficient_quota", "openrouter", "test"), 1),
        (APIError(503, "provider unavailable", "test", "openrouter"), 2),
        (RateLimitError("too many requests", "openrouter", "test"), 2),
        (APIConnectionError("connection lost", "openrouter", "test"), 2),
        (Timeout("request timed out", "test", "openrouter"), 2),
        (LLMNoResponseError("empty response"), 2),
    ],
)
def test_provider_retry_stops_and_preserves_error(
    monkeypatch, asynchronous, error, attempts
):
    """Permanent errors fail once; transient errors stop at the SDK limit."""
    transport = (AsyncMock if asynchronous else Mock)(side_effect=error)
    method = "_atransport_call" if asynchronous else "_transport_call"
    monkeypatch.setattr(LLM, method, transport)
    llm = OpenHandsAgent(model="openai/gpt-4o", api_key="offline-test")._make_llm()
    llm.num_retries = 2
    llm.retry_min_wait = llm.retry_max_wait = 0
    messages = [Message(role="user", content=[TextContent(text="Hello")])]

    with pytest.raises(Exception) as raised:
        if asynchronous:
            asyncio.run(llm.acompletion(messages))
        else:
            llm.completion(messages)

    assert transport.call_count == attempts
    assert str(error) in str(raised.value)


def test_action_arguments_reads_real_mcp_tool_action_data():
    """The recorder surfaces `MCPToolAction.data`, the real per-call arguments."""

    class _Event:
        action = MCPToolAction(data={"query": "benzene", "n": 3})

    assert OpenHandsAgent._action_arguments(_Event()) == {"query": "benzene", "n": 3}


def test_real_mcp_observation_error_flag_is_is_error_not_error():
    """A failed MCP observation exposes `is_error` (and no `error` attribute)."""
    obs = MCPToolObservation.from_text(text="boom", is_error=True, tool_name="t")

    # This is exactly the attribute `_record_observation_event` reads.
    assert getattr(obs, "is_error", None) is True
    # The pre-fix code read `.error`, which does not exist on real observations,
    # so it silently recorded every failure as a success.
    assert not hasattr(obs, "error")

    ok = MCPToolObservation.from_text(text="fine", is_error=False, tool_name="t")
    assert ok.is_error is False


def test_agent_context_datetime_none_omits_datetime_block():
    """`current_datetime=None` removes the machine-local datetime from the prompt."""
    assert AgentContext(current_datetime=None).get_formatted_datetime() is None
    # The default captures wall-clock time, i.e. it is non-reproducible.
    assert AgentContext().get_formatted_datetime() is not None


def test_executor_timeout_constant_tracks_installed_sdk():
    """Retain the SDK default separately from the configured execution limit."""
    assert float(MCP_TOOL_TIMEOUT_SECONDS) == _OPENHANDS_DEFAULT_EXECUTOR_TIMEOUT_S


def test_default_tool_timeout_is_enforceable_end_to_end():
    """Use twice the longest complete MD call, rounded up to whole seconds."""
    agent = OpenHandsAgent(model="openai/gpt-5.6")
    assert agent.tool_timeout_s == 1202
    assert agent.effective_tool_timeout_s == 1202


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_tool_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        OpenHandsAgent(tool_timeout_s=timeout)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("tool_timeout", [1202, 1800])
async def test_real_mcp_call_uses_configured_transport_and_executor_timeouts(
    monkeypatch, tmp_path, tool_timeout
):
    """Exercise the SDK provider hook and inspect actual HTTP request deadlines."""
    read_timeouts = []
    original_send = httpx.AsyncClient.send

    async def capture_send(self, request, **kwargs):
        read_timeouts.append(request.extensions.get("timeout", {}).get("read"))
        return await original_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", capture_send)
    # Native tools are irrelevant to a timeout test and require Chromium.
    monkeypatch.setattr("corral.agents.openhands.get_default_tools", list)
    schema = {
        "type": "function",
        "function": {
            "name": "measure",
            "description": "A slow local measurement",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    async def execute(action):
        assert action.name == "measure"
        await asyncio.sleep(0.02)
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture((schema,)), execute=execute
        ) as endpoint,
    ):
        adapter = OpenHandsAgent(api_key="offline-test", tool_timeout_s=tool_timeout)
        agent = adapter._build_agent(adapter._make_llm(), endpoint.url, False)

        def call_tool():
            conversation = LocalConversation(
                agent=agent,
                workspace=str(tmp_path),
                visualizer=None,
                persistence_dir=None,
                mcp_tool_provider=_TimeoutMCPToolProvider(adapter.tool_timeout_s),
            )
            try:
                conversation.send_message("Initialize tools without calling a model")
                tool = conversation.state.agent.tools_map["measure"]
                assert tool.executor.timeout == tool_timeout
                observation = tool(MCPToolAction(data={}))
                assert not observation.is_error
                assert observation.content[-1].text == "measured"
            finally:
                conversation.close()

        await anyio.to_thread.run_sync(call_tool)
    assert read_timeouts
    assert all(value == tool_timeout for value in read_timeouts)


def test_reasoning_effort_is_applied_to_real_llm():
    """A requested reasoning_effort reaches the real SDK `LLM`; unset -> default."""
    llm = OpenHandsAgent(
        model="openai/gpt-5.6", api_key="x", reasoning_effort="low"
    )._make_llm()
    assert llm.reasoning_effort == "low"
    assert llm.stream is True

    # Unset defers to the SDK's own default rather than pinning a value here.
    default_llm = OpenHandsAgent(model="openai/gpt-5.6", api_key="x")._make_llm()
    assert default_llm.reasoning_effort == "high"
    metadata = model_parameter_metadata(
        OpenHandsAgent(model="openai/gpt-5.6"), model="openai/gpt-5.6"
    )
    assert metadata["reasoning_effort"] == default_llm.reasoning_effort
    assert metadata["temperature"] == default_llm.temperature
    assert metadata["parameter_sources"] == {
        "temperature": "agent",
        "reasoning_effort": "sdk_default",
    }


def test_invalid_reasoning_effort_is_rejected_by_sdk():
    """An unsupported reasoning_effort fails at LLM construction (SDK-validated)."""
    with pytest.raises(ValidationError):
        OpenHandsAgent(
            model="openai/gpt-5.6", api_key="x", reasoning_effort="bogus"
        )._make_llm()


def test_agent_uses_openhands_native_default_tools():
    """The adapter follows the OpenHands-owned preset instead of an empty list."""
    agent_adapter = OpenHandsAgent(model="openai/gpt-5.6", api_key="x")
    agent = agent_adapter._build_agent(
        agent_adapter._make_llm(),
        "http://127.0.0.1:1234/mcp",
        enable_surrender=False,
    )

    assert [tool.name for tool in agent.tools] == [
        "terminal",
        "file_editor",
        "task_tracker",
        "browser_tool_set",
    ]
    assert agent.include_default_tools == ["FinishTool", "ThinkTool"]


def test_task_terminal_does_not_collide_with_openhands_native_terminal():
    """Docker's MCP terminal must not collide with the SDK terminal preset."""
    adapter = OpenHandsAgent(model="openai/gpt-5.6", api_key="x")
    agent = adapter._build_agent(
        adapter._make_llm(),
        "http://127.0.0.1:1234/mcp",
        enable_surrender=False,
        mcp_tool_names={"terminal", "submit_answer"},
    )
    assert [tool.name for tool in agent.tools] == [
        "file_editor",
        "task_tracker",
        "browser_tool_set",
    ]
    assert agent.mcp_config["corral"].url == "http://127.0.0.1:1234/mcp"
    assert agent.include_default_tools == ["FinishTool", "ThinkTool"]
