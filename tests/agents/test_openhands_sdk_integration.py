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

import anyio
import httpx
import pytest
from pydantic import ValidationError

pytest.importorskip("openhands.sdk")

from openhands.sdk import AgentContext, LocalConversation
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


@pytest.mark.anyio()
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
