"""Per-call MCP execution behind Codex's client-executed tool interface.

The pinned app-server only supports a fixed timeout for its own MCP client.
Tools with timeout arguments therefore use its dynamic-tool interface and a
call-owned Python MCP client. Tools without such arguments can retain the
app-server's MCP client and configured fallback deadline.
"""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import anyio
import httpx
from exceptiongroup import ExceptionGroup
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai_codex import Codex
from openai_codex._initialize_metadata import validate_initialize_metadata
from openai_codex.client import CodexClient

from corral.agents._mcp_timeout import TIMEOUT_ARGUMENTS, mcp_call_timeout_s


def timeout_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        tool["function"]
        for tool in tools
        if set(TIMEOUT_ARGUMENTS)
        & tool["function"].get("parameters", {}).get("properties", {}).keys()
    ]


def _error_details(exc: Exception) -> str:
    if isinstance(exc, ExceptionGroup):
        return "; ".join(_error_details(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


class MCPToolBridge:
    def __init__(
        self,
        url: str,
        tools: list[dict[str, Any]],
        fallback_s: float,
        startup_timeout_s: float,
    ) -> None:
        self.url = url
        self.fallback_s = fallback_s
        self.startup_timeout_s = startup_timeout_s
        self.cancelled = threading.Event()
        self.tools = {}
        for tool in tools:
            name = f"corral__{tool['name']}"
            if len(name) > 64:
                name = name[:55] + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]
            self.tools[name] = tool
        self._workers = ThreadPoolExecutor(thread_name_prefix="corral-codex-mcp")

    @property
    def specifications(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": tool.get("description", ""),
                "inputSchema": tool.get("parameters", {}),
            }
            for name, tool in self.tools.items()
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.cancelled.set()
        self._workers.shutdown(wait=True, cancel_futures=True)

    async def call(self, name: str, arguments: dict[str, Any]):
        tool = self.tools[name]
        timeout = mcp_call_timeout_s(
            arguments, tool.get("parameters", {}), self.fallback_s
        )
        result = None
        async with anyio.create_task_group() as group:

            async def watch_cancellation():
                # Each concurrent call owns its loop; cancellation is shared
                # with the synchronous SDK owner via a threading.Event.
                while not self.cancelled.is_set():  # noqa: ASYNC110
                    await anyio.sleep(0.05)
                group.cancel_scope.cancel()

            group.start_soon(watch_cancellation)
            async with (
                httpx.AsyncClient(
                    timeout=httpx.Timeout(self.startup_timeout_s)
                ) as http,
                streamable_http_client(self.url, http_client=http) as streams,
                ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=self.startup_timeout_s),
                ) as session,
            ):
                with anyio.fail_after(self.startup_timeout_s):
                    await session.initialize()
                http.timeout = httpx.Timeout(timeout, connect=self.startup_timeout_s)
                with anyio.fail_after(timeout):
                    result = await session.call_tool(
                        tool["name"],
                        arguments,
                        read_timeout_seconds=timedelta(seconds=timeout),
                    )
            group.cancel_scope.cancel()
        if result is None:
            raise RuntimeError("Codex tool call cancelled with its owning turn")
        return result

    def dispatch(self, request: dict[str, Any], write_message) -> None:
        """Keep the SDK reader free to handle concurrent calls and interrupts."""
        self._workers.submit(self._respond, request, write_message)

    def _respond(self, request: dict[str, Any], write_message) -> None:
        try:
            params = request.get("params") or {}
            name = params.get("tool")
            arguments = params.get("arguments", {})
            if name not in self.tools or not isinstance(arguments, dict):
                raise ValueError("Unknown Corral tool or invalid tool arguments")
            result = anyio.run(self.call, name, arguments)
            # Corral's MCP binding returns text, including serialized rich results.
            content = [
                {"type": "inputText", "text": item.text} for item in result.content
            ]
            response = {"contentItems": content, "success": not result.isError}
        except Exception as exc:
            response = {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": f"MCP call failed: {_error_details(exc)}",
                    }
                ],
                "success": False,
            }
        if not self.cancelled.is_set():
            write_message({"id": request["id"], "result": response})


class _MCPClient(CodexClient):
    def __init__(self, config, bridge: MCPToolBridge):
        super().__init__(config=config)
        self._bridge = bridge

    def _read_message(self):
        # SDK 0.1.0b3's default reader executes callbacks inline. Intercept only
        # our tool requests; preserve its notification routing and approvals.
        while True:
            message = super()._read_message()
            if message.get("method") != "item/tool/call" or "id" not in message:
                return message
            self._bridge.dispatch(message, self._write_tool_response)

    def _write_tool_response(self, response):
        try:
            self._write_message(response)
        except Exception as exc:
            # Worker exceptions must wake SDK waiters, just as reader errors do.
            # Otherwise a failed response write can leave the turn blocked.
            self._router.fail_all(exc)


class CodexWithMCPTools(Codex):
    def __init__(self, config, bridge: MCPToolBridge):
        # Retain the pinned SDK's initialization/metadata validation with a
        # client whose reader does not block on a long-running MCP operation.
        self._client = _MCPClient(config, bridge)
        try:
            self._client.start()
            self._init = validate_initialize_metadata(self._client.initialize())
        except Exception:
            self._client.close()
            raise
