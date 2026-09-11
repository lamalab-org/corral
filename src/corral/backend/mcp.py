"""A lazy HTTP host per execution, with independently revocable tool bindings."""

from __future__ import annotations

import asyncio
import secrets
import socket
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

import anyio
import uvicorn
from mcp.server.lowlevel import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from corral.core.action import Action
from corral.core.tool import ToolConnection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from starlette.types import Receive, Scope, Send

    from corral.core.tool import ToolResponse
    from corral.core.tool_catalog import ToolCatalogSnapshot


class _EmbeddedServer(uvicorn.Server):
    """Serve on the caller's loop without replacing its signal handlers."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = asyncio.Event()

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self.ready.set()


class _Binding:
    def __init__(
        self,
        catalog: ToolCatalogSnapshot,
        execute: Callable[[Action], Awaitable[ToolResponse]],
        name: str,
    ) -> None:
        self._context = copy_context()
        self._requests: set[asyncio.Task[None]] = set()
        self._scopes: set[anyio.CancelScope] = set()
        self._cancelled = False
        self._server: MCPServer = MCPServer(name)

        @self._server.list_tools()
        async def list_tools() -> list[MCPTool]:
            return [MCPTool(**tool) for tool in catalog.mcp_tools()]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            response = await execute(Action(name=name, arguments=arguments or {}))
            content = response.result if response.success else response.error
            return CallToolResult(
                content=[TextContent(type="text", text=str(content or ""))],
                isError=not response.success,
            )

    async def _handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        with anyio.CancelScope() as cancel_scope:
            self._scopes.add(cancel_scope)
            try:
                if self._cancelled:
                    cancel_scope.cancel()
                # Stateless MCP already creates a protocol session per request.
                # Keep its task group in that request's cancellation scope, so
                # cancellation also waits for non-abandoning synchronous tools.
                manager = StreamableHTTPSessionManager(
                    app=self._server, json_response=True, stateless=True
                )
                async with manager.run():
                    await manager.handle_request(scope, receive, send)
            finally:
                self._scopes.remove(cancel_scope)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Tasks inherit the invocation's context, including delegate budgets,
        # rather than the server startup context. Each request gets a copy.
        task = self._context.run(
            asyncio.create_task,
            self._handle(scope, receive, send),
            name="corral-mcp-request",
        )
        self._requests.add(task)
        try:
            await asyncio.shield(task)
        finally:
            # A disconnected/cancelled HTTP caller must not abandon its tool.
            if task.done():
                self._requests.discard(task)

    async def close(self, *, cancel: bool) -> None:
        if cancel:
            self._cancelled = True
            for scope in self._scopes:
                scope.cancel()
        # No timeout: a tool in a Python thread can still modify execution data.
        if self._requests:
            # Only cancel via the request scopes: cancelling the gather would
            # inject raw asyncio cancellation through AnyIO's thread shielding.
            await asyncio.shield(
                asyncio.gather(*self._requests, return_exceptions=True)
            )
            self._requests.clear()


class MCPHost:
    """Execution-owned host whose first binding starts the shared listener."""

    def __init__(self) -> None:
        self.port: int | None = None
        self._bindings: dict[str, _Binding] = {}
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()
        self._resources = AsyncExitStack()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        binding = self._bindings.get(scope["path_params"]["capability"])
        if binding is None:
            await Response(status_code=404)(scope, receive, send)
        else:
            await binding(scope, receive, send)

    @asynccontextmanager
    async def bind(
        self,
        *,
        catalog: ToolCatalogSnapshot,
        execute: Callable[[Action], Awaitable[ToolResponse]],
        name: str = "corral-tools",
    ) -> AsyncIterator[ToolConnection]:
        # Delegates may request their first connections concurrently. Publish
        # the port only after startup succeeds, so failures can be retried.
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("execution MCP host is closed")
            if self.port is None:
                self.port = await self._resources.enter_async_context(
                    _serve_mcp_host(self)
                )
        capability = secrets.token_urlsafe(32)
        binding = _Binding(catalog, execute, name)
        self._bindings[capability] = binding
        cancel = False
        try:
            yield ToolConnection(
                transport="mcp",
                url=f"http://127.0.0.1:{self.port}/{capability}/mcp",
            )
        except BaseException:
            cancel = True
            raise
        finally:
            self._bindings.pop(capability, None)
            try:
                if not cancel:
                    await binding.close(cancel=False)
            finally:
                # Cancellation can also arrive while a normal exit is draining.
                with anyio.CancelScope(shield=True):
                    await binding.close(cancel=True)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            bindings = tuple(self._bindings.values())
            self._bindings.clear()
            try:
                for binding in bindings:
                    await binding.close(cancel=True)
            finally:
                await self._resources.aclose()


@asynccontextmanager
async def open_mcp_host() -> AsyncIterator[MCPHost]:
    """Own an execution's host; allocate a listener only when a binding needs it."""
    host = MCPHost()
    try:
        yield host
    finally:
        with anyio.CancelScope(shield=True):
            await host.close()


@asynccontextmanager
async def _serve_mcp_host(host: MCPHost) -> AsyncIterator[int]:
    """Start the listener on the current loop and release it on host shutdown."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
        app = Starlette(routes=[Route("/{capability}/mcp", endpoint=host)])
        server = _EmbeddedServer(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
                lifespan="off",
            )
        )

        async def serve() -> None:
            try:
                await server.serve(sockets=[listener])
            finally:
                # Wake startup waiters on failure as well as on readiness.
                server.ready.set()

        task = asyncio.create_task(serve(), name=f"corral-mcp-host-{port}")
        try:
            with anyio.fail_after(10):
                await server.ready.wait()
            if task.done():
                await task
                raise RuntimeError("tool MCP server failed to start")
            yield port
        finally:
            with anyio.CancelScope(shield=True):
                server.should_exit = True
                if not server.started:
                    task.cancel()
                try:
                    await asyncio.gather(task, return_exceptions=True)
                finally:
                    # Also release a listener created during partial startup.
                    for http_server in getattr(server, "servers", ()):
                        http_server.close()
                        await http_server.wait_closed()


__all__ = ["MCPHost", "open_mcp_host"]
