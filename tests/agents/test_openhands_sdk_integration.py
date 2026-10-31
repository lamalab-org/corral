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
import json
from concurrent.futures import ThreadPoolExecutor

import anyio
import httpx
import pytest
from mcp import ClientSession
from pydantic import ValidationError

pytest.importorskip("openhands.sdk")

from openhands.sdk import AgentContext, LocalConversation
from openhands.sdk.mcp import MCPServer
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.definition import (
    MCPToolAction,
    MCPToolObservation,
)
from openhands.sdk.mcp.tool import MCP_TOOL_TIMEOUT_SECONDS

from corral.agents.openhands import (
    _OPENHANDS_DEFAULT_EXECUTOR_TIMEOUT_S,
    OpenHandsAgent,
    _mcp_call_timeout_s,
    _TimeoutMCPToolProvider,
)
from corral.backend.mcp import open_mcp_host
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot
from corral.orchestration.parameters import model_parameter_metadata


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


def test_tool_timeout_metadata_describes_per_call_policy(monkeypatch):
    """The run records a fallback and grace, not a fixed effective deadline."""
    monkeypatch.setattr("corral.agents.openhands.get_default_tools", list)
    agent = OpenHandsAgent(model="openai/gpt-5.6")
    assert agent.tool_timeout_s == 100_000
    metadata = agent._harness_metadata("http://localhost/mcp", "brief", [], None, 10)
    assert metadata["configured_mcp_timeout_s"] == 100_000
    assert metadata["effective_tool_timeout_s"] is None
    assert metadata["openhands_executor_timeout_s"] is None
    assert metadata["tool_timeout_policy"] == "argument_or_schema_default_plus_grace"
    assert metadata["tool_timeout_grace_s"] == 10


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_tool_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        OpenHandsAgent(tool_timeout_s=timeout)


@pytest.mark.parametrize(
    "value", [None, True, "600", 0, -1, float("nan"), float("inf"), 10**400]
)
def test_invalid_call_timeout_keeps_fallback(value):
    schema = {"properties": {"timeout": {"type": "number", "default": 900}}}
    assert _mcp_call_timeout_s({"timeout": value}, schema, 1202) == 1202


def test_undeclared_timeout_argument_does_not_change_deadline():
    assert _mcp_call_timeout_s({"timeout": 3600}, {"properties": {}}, 1202) == 1202


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("with_transcript_flush", [False, True])
@pytest.mark.parametrize(
    ("parameter", "field", "arguments", "fallback", "expected"),
    [
        (None, {}, {}, None, 100_000),
        (None, {}, {}, 1202, 1202),
        (None, {}, {}, 1800, 1800),
        ("timeout", {"default": 900}, {"timeout": 30}, None, 40),
        ("timeout", {"default": 900}, {}, None, 910),
        ("timeout", {"default": 900}, {"timeout": 3600}, 1202, 3610),
        ("timeout", {"default": 900}, {"timeout": 30}, 1202, 40),
        ("timeout", {"default": 900}, {}, 1202, 910),
        ("timeout", {}, {}, 1202, 1202),
        ("timeout_seconds", {"default": 120}, {}, 1202, 130),
        ("timeout_seconds", {}, {"timeout_seconds": 60}, 1202, 70),
        ("timeout_s", {}, {"timeout_s": 0.5}, 1202, 10.5),
    ],
)
async def test_real_mcp_call_uses_per_call_deadlines_at_every_layer(
    monkeypatch,
    tmp_path,
    parameter,
    field,
    arguments,
    fallback,
    expected,
    with_transcript_flush,
):
    """Inspect executor, MCP session, and actual HTTP request deadlines."""
    read_timeouts = []
    session_timeouts = []
    executor_timeouts = []
    call_clients = []
    call_order = []
    original_send = httpx.AsyncClient.send
    original_call_tool = ClientSession.call_tool
    original_bridge = MCPClient.call_async_from_sync

    async def capture_send(self, request, **kwargs):
        if (
            request.method == "POST"
            and json.loads(request.content).get("method") == "tools/call"
        ):
            read_timeouts.append(request.extensions["timeout"]["read"])
        return await original_send(self, request, **kwargs)

    async def capture_session(self, *args, **kwargs):
        timeout = (
            kwargs.get("read_timeout_seconds") or self._session_read_timeout_seconds
        )
        session_timeouts.append(timeout.total_seconds())
        return await original_call_tool(self, *args, **kwargs)

    def capture_executor(self, *args, **kwargs):
        if "action" in kwargs:
            executor_timeouts.append(kwargs["timeout"])
            call_clients.append(self)
        return original_bridge(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", capture_send)
    monkeypatch.setattr(ClientSession, "call_tool", capture_session)
    monkeypatch.setattr(MCPClient, "call_async_from_sync", capture_executor)
    # Native tools are irrelevant to a timeout test and require Chromium.
    monkeypatch.setattr("corral.agents.openhands.get_default_tools", list)
    # The pinned SDK caches validated action types by tool name.
    name = f"measure_{parameter}_{expected}"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "A slow local measurement",
            "parameters": {
                "type": "object",
                "properties": {parameter: {"type": "number", **field}}
                if parameter
                else {},
            },
        },
    }

    async def flush_messages():
        call_order.append("flush")

    async def execute(action):
        call_order.append("execute")
        assert action.name == name
        assert action.arguments == arguments
        await asyncio.sleep(0.02)
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture((schema,)), execute=execute
        ) as endpoint,
    ):
        adapter = OpenHandsAgent(
            api_key="offline-test",
            **({"tool_timeout_s": fallback} if fallback is not None else {}),
        )
        agent = adapter._build_agent(adapter._make_llm(), endpoint.url, False)

        def call_tool():
            conversation = LocalConversation(
                agent=agent,
                workspace=str(tmp_path),
                visualizer=None,
                persistence_dir=None,
                mcp_tool_provider=_TimeoutMCPToolProvider(
                    adapter.tool_timeout_s,
                    before_tool=flush_messages if with_transcript_flush else None,
                ),
            )
            try:
                conversation.send_message("Initialize tools without calling a model")
                tool = conversation.state.agent.tools_map[name]
                observation = tool(MCPToolAction(data=arguments))
                assert not observation.is_error
                assert observation.content[-1].text == "measured"
            finally:
                conversation.close()

        await anyio.to_thread.run_sync(call_tool)
    assert read_timeouts == session_timeouts == executor_timeouts == [expected]
    assert all(client._closed for client in call_clients)
    assert call_order == (
        ["flush", "execute"] if with_transcript_flush else ["execute"]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("with_transcript_flush", [False, True])
async def test_mcp_native_name_collision_keeps_alias_with_transcript_flush(
    with_transcript_flush,
):
    """A transcript barrier must preserve the MCP alias and original wire name."""
    name = "native_collision_measure"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "A measurement sharing a native tool name",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    call_order = []

    async def flush_messages():
        call_order.append("flush")

    async def execute(action):
        assert action.name == name
        call_order.append("execute")
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture((schema,)), execute=execute
        ) as endpoint,
    ):

        def call_tool():
            provider = _TimeoutMCPToolProvider(
                100_000,
                native_tool_names={name},
                before_tool=flush_messages if with_transcript_flush else None,
            )
            with provider.create_tools(
                {"corral": MCPServer(url=endpoint.url)}
            ) as client:
                registered = {tool.name: tool for tool in client.tools}
                assert set(registered) == {f"mcp__corral__{name}"}
                observation = registered[f"mcp__corral__{name}"](MCPToolAction(data={}))
                assert not observation.is_error
                assert observation.content[-1].text == "measured"

        await anyio.to_thread.run_sync(call_tool)
    assert call_order == (
        ["flush", "execute"] if with_transcript_flush else ["execute"]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_concurrent_calls_keep_independent_timeouts_and_recover(monkeypatch):
    """A short timed-out call must not cut off a longer call or poison retries."""
    # Keep the real timeout/cleanup test fast; the tests above pin the 10s grace.
    monkeypatch.setattr("corral.agents.openhands._TOOL_TIMEOUT_GRACE_S", 0.1)
    schema = {
        "type": "function",
        "function": {
            "name": "concurrent_measure",
            "description": "Measurement with caller-controlled duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "number"},
                    "duration": {"type": "number"},
                },
            },
        },
    }
    started = asyncio.Event()
    call_clients = []
    original_bridge = MCPClient.call_async_from_sync

    def capture_executor(self, *args, **kwargs):
        if "action" in kwargs:
            call_clients.append(self)
        return original_bridge(self, *args, **kwargs)

    monkeypatch.setattr(MCPClient, "call_async_from_sync", capture_executor)

    async def execute(action):
        if action.arguments["duration"]:
            started.set()
        await asyncio.sleep(action.arguments["duration"])
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture((schema,)), execute=execute
        ) as endpoint,
    ):

        def call_tools():
            with _TimeoutMCPToolProvider(2).create_tools(
                {"corral": MCPServer(url=endpoint.url)}
            ) as client:
                tool = client.tools[0]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    long = pool.submit(
                        tool, MCPToolAction(data={"timeout": 3, "duration": 2.2})
                    )
                    anyio.from_thread.run(started.wait)
                    short = pool.submit(
                        tool, MCPToolAction(data={"timeout": 0.1, "duration": 0.5})
                    )
                    assert short.result().is_error
                    assert not long.result().is_error
                assert not tool(
                    MCPToolAction(data={"timeout": 1, "duration": 0})
                ).is_error

        await anyio.to_thread.run_sync(call_tools)
    assert len(call_clients) == 3
    assert all(client._closed for client in call_clients)


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
