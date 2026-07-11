"""Task-scoped MCP transport mounted inside the Corral benchmark server.

This exposes each task's tools over the Model Context Protocol (Streamable
HTTP) *in the same FastAPI application* that already serves the REST API, so
generic MCP clients (Claude Code, Codex, OpenHands, ...) can connect directly
to a task without a local proxy:

    [mcp_servers.corral]
    url = "http://localhost:8000/tasks/<task_id>/mcp"

Design properties:

* **Single source of truth for schemas.** `tools/list` is served straight
  from :meth:`corral.backend.tool.Tool.to_mcp`, so there is no second
  hand-written OpenAI/MCP schema conversion to keep in sync.
* **Reuses tool execution.** `tools/call` routes through the existing
  :meth:`Environment.call_tool`, so recording, hidden-argument merging, and
  validation behave identically to the REST `/tools/execute` path.
* **Task-scoped by construction.** Each task gets its own MCP server closing
  over its own :class:`Environment`, mounted at its own URL. The task boundary
  is structural, not a runtime filter: a server for task A can only ever see
  task A's tools. A defensive membership check is kept as a second line of
  defence.
"""

import contextlib
import json
from collections.abc import AsyncIterator, Mapping
from contextvars import ContextVar
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI
from loguru import logger
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from starlette.types import Receive, Scope, Send

from corral.backend.env import Environment
from corral.backend.schema import ToolCall, ToolCallStatus
from corral.router.verbosity import ToolVerbosity

# Per-request override for the tool-description verbosity, taken from the MCP
# endpoint's `?verbosity=` query parameter. It is set in the ASGI endpoint just
# before the request is handled and read by `tools/list`, so a client (e.g. the
# Claude Code agent) can request the same verbosity condition the REST allowlist
# was built at. Falls back to the server's mount-time verbosity when unset.
_request_verbosity: ContextVar[ToolVerbosity | None] = ContextVar(
    "corral_mcp_request_verbosity", default=None
)


def _verbosity_from_scope(scope: Scope) -> ToolVerbosity | None:
    """Parse a `?verbosity=` value out of an ASGI scope, if present and valid."""
    raw = scope.get("query_string", b"") if isinstance(scope, Mapping) else b""
    if not raw:
        return None
    values = parse_qs(raw.decode("latin-1")).get("verbosity")
    if not values:
        return None
    try:
        return ToolVerbosity(values[0])
    except ValueError:
        logger.warning(f"Ignoring unknown MCP verbosity {values[0]!r}")
        return None


def task_mcp_tools(env: Environment, verbosity: ToolVerbosity) -> list[MCPTool]:
    """Return the task's tools as MCP tool definitions.

    Schemas come directly from :meth:`Tool.to_mcp`, which already handles
    verbosity filtering and marking defaulted parameters as optional.
    """
    return [MCPTool(**tool.to_mcp(verbosity=verbosity)) for tool in env.tools.values()]


def _render_tool_call(tool_call: ToolCall) -> list[TextContent] | CallToolResult:
    """Convert a Corral :class:`ToolCall` record into an MCP tool result.

    Successful calls return their result as text (JSON-encoded when the result
    is not already a string). Failed calls return an isError result carrying
    the environment's error message so the model can react to it.
    """
    if tool_call.status == ToolCallStatus.SUCCESS:
        result = tool_call.result
        text = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        return [TextContent(type="text", text=text)]

    message = tool_call.error_message or f"Tool {tool_call.tool_name!r} failed."
    return CallToolResult(
        content=[TextContent(type="text", text=message)], isError=True
    )


def execute_task_tool(
    env: Environment, task_id: str, name: str, arguments: dict[str, Any] | None
) -> list[TextContent] | CallToolResult:
    """Execute one tool call against the task environment.

    Enforces the task boundary (`name` must belong to `env`) before delegating
    to :meth:`Environment.call_tool`.
    """
    if name not in env.tools:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Tool {name!r} is not available for task {task_id!r}.",
                )
            ],
            isError=True,
        )
    tool_call = env.call_tool(name, arguments or {})
    return _render_tool_call(tool_call)


def build_task_mcp_server(
    task_id: str, env: Environment, verbosity: ToolVerbosity
) -> Server:
    """Build a low-level MCP :class:`Server` scoped to a single task."""
    server: Server = Server(f"corral-{task_id}")

    @server.list_tools()
    async def _list_tools() -> list[MCPTool]:
        # Honor a per-request `?verbosity=` override, else the mount default.
        effective = _request_verbosity.get() or verbosity
        return task_mcp_tools(env, effective)

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent] | CallToolResult:
        return execute_task_tool(env, task_id, name, arguments)

    return server


def _build_task_manager(
    task_id: str, env: Environment, verbosity: ToolVerbosity
) -> StreamableHTTPSessionManager:
    """Wrap a task's MCP server in a stateless Streamable-HTTP session manager.

    Stateless mode keeps each request self-contained, which suits ephemeral
    benchmark episodes: there is no cross-request session state to leak or clean
    up between trials, and clients can connect without a session handshake.
    """
    server = build_task_mcp_server(task_id, env, verbosity)
    return StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)


def attach_task_mcp_servers(
    app: FastAPI,
    environments: Mapping[str, Environment],
    verbosity: ToolVerbosity = ToolVerbosity.FULL,
) -> None:
    """Mount a task-scoped MCP endpoint for every environment on `app`.

    Each task is served at /tasks/{task_id}/mcp alongside the existing REST
    routes. The session managers are driven by an application lifespan so their
    internal task groups start and stop with the server.
    """
    managers: dict[str, StreamableHTTPSessionManager] = {
        task_id: _build_task_manager(task_id, env, verbosity)
        for task_id, env in environments.items()
    }
    if not managers:
        return

    for task_id, manager in managers.items():

        def _make_endpoint(mgr: StreamableHTTPSessionManager):
            async def _endpoint(scope: Scope, receive: Receive, send: Send) -> None:
                # Bind the request's verbosity for the duration of this request
                # so `tools/list` can pick it up, then restore it afterwards.
                token = _request_verbosity.set(_verbosity_from_scope(scope))
                try:
                    await mgr.handle_request(scope, receive, send)
                finally:
                    _request_verbosity.reset(token)

            return _endpoint

        app.mount(f"/tasks/{task_id}/mcp", _make_endpoint(manager))

    _install_lifespan(app, managers)
    logger.info(f"Mounted MCP endpoints for {len(managers)} task(s) at /tasks/*/mcp")


def _install_lifespan(
    app: FastAPI, managers: Mapping[str, StreamableHTTPSessionManager]
) -> None:
    """Chain a lifespan onto `app` that runs every MCP session manager.

    A `StreamableHTTPSessionManager` can only serve requests while its `run()`
    context is active (it owns an anyio task group). We enter all of them on
    startup and exit them on shutdown, composing with any pre-existing lifespan.
    """
    previous = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for manager in managers.values():
                await stack.enter_async_context(manager.run())
            async with previous(app):
                yield

    app.router.lifespan_context = lifespan
