"""Real MCP coverage for Codex's argument-dependent tool deadlines."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import anyio
import httpx
import pytest
from mcp import ClientSession

pytest.importorskip("openai_codex")

from openai_codex import CodexConfig
from openai_codex.client import CodexClient

from corral.agents import _codex_mcp as bridge_module
from corral.agents._codex_mcp import (
    CodexWithMCPTools,
    MCPToolBridge,
    _MCPClient,
    timeout_tools,
)
from corral.agents._mcp_timeout import mcp_call_timeout_s
from corral.agents.codex import CodexAgent, _RunState
from corral.backend.mcp import open_mcp_host
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot


@pytest.fixture
def anyio_backend():
    return "asyncio"


def timed_tool(parameter="timeout_seconds", default=900):
    return {
        "type": "function",
        "function": {
            "name": "measure",
            "description": "Measure a sample",
            "parameters": {
                "type": "object",
                "properties": {parameter: {"type": "number", "default": default}},
            },
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("parameter", "arguments", "fallback", "expected"),
    [
        ("timeout", {"timeout": 3600}, 1202, 3610),
        ("timeout_seconds", {"timeout_seconds": 30}, None, 40),
        ("timeout_s", {}, None, 910),
        ("timeout", {"timeout": False}, None, 100_000),
        ("timeout", {"timeout": False}, 1202, 1202),
    ],
)
async def test_real_mcp_call_uses_argument_deadline_at_every_layer(
    monkeypatch, parameter, arguments, fallback, expected
):
    http_deadlines, session_deadlines, executor_deadlines = [], [], []
    original_send = httpx.AsyncClient.send
    original_call = ClientSession.call_tool
    original_fail_after = anyio.fail_after

    async def send(self, request, **kwargs):
        if (
            request.method == "POST"
            and json.loads(request.content).get("method") == "tools/call"
        ):
            http_deadlines.append(request.extensions["timeout"]["read"])
        return await original_send(self, request, **kwargs)

    async def call(self, *args, **kwargs):
        session_deadlines.append(kwargs["read_timeout_seconds"].total_seconds())
        return await original_call(self, *args, **kwargs)

    def fail_after(delay, *args, **kwargs):
        if delay == expected:
            executor_deadlines.append(delay)
        return original_fail_after(delay, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(ClientSession, "call_tool", call)
    monkeypatch.setattr(anyio, "fail_after", fail_after)
    schema = timed_tool(parameter)

    async def execute(action):
        assert action.name == "measure"
        assert action.arguments == arguments
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture([schema]), execute=execute
        ) as endpoint,
    ):
        adapter = CodexAgent(
            system_prompt="system",
            **({"tool_timeout_s": fallback} if fallback is not None else {}),
        )
        with MCPToolBridge(
            endpoint.url, timeout_tools([schema]), adapter.tool_timeout_s, 5
        ) as bridge:
            result = await bridge.call("corral__measure", arguments)
            if arguments.get(parameter) is False:
                assert result.isError
                assert "Input validation error" in result.content[0].text
            else:
                assert not result.isError
                assert result.content[0].text == "measured"
    assert http_deadlines == session_deadlines == [expected]
    assert executor_deadlines
    assert all(value == expected for value in executor_deadlines)


@pytest.mark.anyio
async def test_concurrent_calls_have_independent_deadlines_and_recover(monkeypatch):
    monkeypatch.setattr(
        bridge_module,
        "mcp_call_timeout_s",
        lambda *args: mcp_call_timeout_s(*args, grace_s=0.02),
    )
    schema = timed_tool()
    schema["function"]["parameters"]["properties"]["duration"] = {"type": "number"}
    results = {}

    async def execute(action):
        await anyio.sleep(action.arguments["duration"])
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture([schema]), execute=execute
        ) as endpoint,
    ):
        with MCPToolBridge(endpoint.url, timeout_tools([schema]), 0.01, 5) as bridge:

            async def invoke(name, limit, duration):
                try:
                    results[name] = await bridge.call(
                        "corral__measure",
                        {"timeout_seconds": limit, "duration": duration},
                    )
                except Exception as exc:
                    results[name] = exc

            async with anyio.create_task_group() as group:
                group.start_soon(invoke, "long", 3, 0.4)
                group.start_soon(invoke, "short", 0.05, 0.3)
            await invoke("recovery", 1, 0)
    assert isinstance(results["short"], Exception)
    assert results["long"].content[0].text == "measured"
    assert results["recovery"].content[0].text == "measured"


def test_reader_dispatches_tools_without_blocking_responses_or_approvals(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    responses = []
    schema = timed_tool()
    with MCPToolBridge("unused", timeout_tools([schema]), 1202, 5) as bridge:

        def respond(request, write_message):
            started.set()
            assert release.wait(2)
            write_message({"id": request["id"], "result": {"success": True}})

        monkeypatch.setattr(bridge, "_respond", respond)
        messages = iter(
            [
                {"id": "tool", "method": "item/tool/call", "params": {}},
                {"id": "interrupt", "result": {}},
                {"id": "approval", "method": "item/commandExecution/requestApproval"},
            ]
        )
        monkeypatch.setattr(CodexClient, "_read_message", lambda _: next(messages))
        client = _MCPClient(None, bridge)
        monkeypatch.setattr(client, "_write_message", responses.append)
        try:
            assert client._read_message() == {"id": "interrupt", "result": {}}
            assert started.wait(1)
            assert client._read_message()["id"] == "approval"
        finally:
            release.set()
    assert responses == [{"id": "tool", "result": {"success": True}}]


def test_failed_response_write_wakes_pending_sdk_requests(monkeypatch):
    with MCPToolBridge("unused", [], 1202, 5) as bridge:
        client = _MCPClient(None, bridge)
        error = BrokenPipeError("app-server closed stdin")
        monkeypatch.setattr(client, "_write_message", Mock(side_effect=error))
        failed = Mock()
        monkeypatch.setattr(client._router, "fail_all", failed)
        client._write_tool_response({"id": "tool", "result": {}})
    failed.assert_called_once_with(error)


@pytest.mark.parametrize("arguments", [[], None, "invalid"])
def test_invalid_dynamic_arguments_return_a_failed_tool_result(arguments):
    responses = []
    with MCPToolBridge("unused", timeout_tools([timed_tool()]), 1202, 5) as bridge:
        bridge._respond(
            {
                "id": "bad-input",
                "params": {"tool": "corral__measure", "arguments": arguments},
            },
            responses.append,
        )
    assert responses[0]["result"]["success"] is False
    assert "invalid tool arguments" in responses[0]["result"]["contentItems"][0]["text"]


def test_pinned_appserver_accepts_dynamic_tool_registration_without_model_call(
    tmp_path,
):
    # Exercise the actual SDK/binary contract: mcp__ names are reserved, and
    # the pinned Python ThreadStartParams does not expose dynamicTools.
    (tmp_path / "config.toml").write_text("")
    with (
        MCPToolBridge("unused", timeout_tools([timed_tool()]), 1202, 5) as bridge,
        CodexWithMCPTools(
            CodexConfig(cwd=str(tmp_path), env={"CODEX_HOME": str(tmp_path)}), bridge
        ) as codex,
    ):
        codex.login_api_key("offline-test-key")
        started = codex._client.thread_start(
            {
                "model": "gpt-5.4",
                "cwd": str(tmp_path),
                "ephemeral": True,
                "dynamicTools": bridge.specifications,
            }
        )
        assert started.thread.id


@pytest.mark.anyio
async def test_owning_turn_cancels_pending_bridge_call():
    started = anyio.Event()
    schema = timed_tool()
    errors = []

    async def execute(action):
        started.set()
        await anyio.sleep(0.2)
        return ToolResponse(success=True, result="finished", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture([schema]), execute=execute
        ) as endpoint,
    ):
        with MCPToolBridge(endpoint.url, timeout_tools([schema]), 1202, 5) as bridge:

            async def invoke():
                try:
                    await bridge.call("corral__measure", {"timeout_seconds": 900})
                except RuntimeError as exc:
                    errors.append(str(exc))

            with anyio.fail_after(2):
                async with anyio.create_task_group() as group:
                    group.start_soon(invoke)
                    await started.wait()
                    bridge.cancelled.set()
    assert errors == ["Codex tool call cancelled with its owning turn"]


def test_bridged_mcp_events_are_not_duplicated_as_native_calls():
    from openai_codex.generated.v2_all import ThreadItem

    run = _RunState(bridged_tool_names={"corral__measure"})
    agent = CodexAgent(system_prompt="system")
    for name in ["corral__measure", "other_native_tool"]:
        item = ThreadItem.model_validate(
            {
                "type": "dynamicToolCall",
                "id": name,
                "tool": name,
                "arguments": {"timeout_seconds": 900},
                "status": "completed",
                "success": True,
            }
        ).root
        agent._record_item(item, run)
    assert len(run.messages) == 2
    assert run.messages[1]["name"] == "other_native_tool"
    assert run.messages[0]["sdk_native"] is True


def test_failed_bridge_call_retains_sdk_diagnostic_without_duplicate_native_call():
    from openai_codex.generated.v2_all import ThreadItem

    run = _RunState(bridged_tool_names={"corral__measure"})
    item = ThreadItem.model_validate(
        {
            "type": "dynamicToolCall",
            "id": "failed-call",
            "tool": "corral__measure",
            "arguments": {"timeout_seconds": 1},
            "status": "failed",
            "success": False,
            "contentItems": [
                {"type": "inputText", "text": "MCP call failed: TimeoutError"}
            ],
        }
    ).root
    CodexAgent(system_prompt="system")._record_item(item, run)
    assert run.metadata["tool_errors"] == 1
    assert len(run.messages) == 1
    assert run.messages[0]["name"] == "codex-mcp-error"
    assert "TimeoutError" in run.messages[0]["content"]
    assert "tool_calls" not in run.messages[0]
    assert not run.messages[0].get("sdk_native")


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_fallback_timeout_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        CodexAgent(tool_timeout_s=value)


def test_thread_registers_bridged_tools_without_exposing_duplicate_mcp_tools(
    monkeypatch, tmp_path
):
    captured = {}
    schema = timed_tool()

    class FakeCodex:
        def __init__(self, config, bridge):
            self._client = self
            captured["bridge"] = bridge

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def login_api_key(self, _key):
            pass

        def thread_start(self, params):
            captured["params"] = params
            return SimpleNamespace(thread=SimpleNamespace(id="test"))

    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setattr("corral.agents.codex.CodexWithMCPTools", FakeCodex)
    monkeypatch.setattr(CodexAgent, "_drive_turn", lambda *_args: "done")
    run = _RunState(available_tools=[schema])
    result = CodexAgent(system_prompt="system")._execute_codex_turn(
        codex_home=tmp_path,
        workspace=tmp_path,
        mcp_url="http://unused/mcp",
        tool_names=["measure", "submit_answer"],
        developer_instructions="system",
        prompt="solve",
        run=run,
    )
    assert result == "done"
    config = (tmp_path / "config.toml").read_text()
    assert 'enabled_tools = ["submit_answer"]' in config
    assert "tool_timeout_sec = 100000" in config
    assert captured["params"]["dynamicTools"] == captured["bridge"].specifications
    assert run.bridged_tool_names == {"corral__measure"}
