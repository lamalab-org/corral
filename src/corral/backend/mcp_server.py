import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextvars import ContextVar
from functools import partial
from typing import Any
from urllib.parse import parse_qs

import anyio
from fastapi import FastAPI
from loguru import logger
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, CreateTaskResult, TextContent
from mcp.types import Tool as MCPTool
from starlette.types import Receive, Scope, Send

from corral.backend.env import Environment
from corral.backend.mcp_tasks import enable_job_task_surface, maybe_start_task
from corral.backend.schema import ToolCall, ToolCallStatus
from corral.backend.trial_runtime import InProcessTrialRuntime, TrialRuntime
from corral.backend.trial_worker import WorkerTrialRuntime
from corral.router.verbosity import ToolVerbosity

# Per-request override for the tool-description verbosity, taken from the MCP
# endpoint's `?verbosity=` query parameter. It is set in the ASGI endpoint just
# before the request is handled and read by `tools/list`, so a client (e.g. the
# Claude Code agent) can request the same verbosity condition the REST allowlist
# was built at. Falls back to the server's mount-time verbosity when unset.
_request_verbosity: ContextVar[ToolVerbosity | None] = ContextVar(
    "corral_mcp_request_verbosity", default=None
)

# Per-request trial-runtime id for the shared `/trials` MCP mount. A single
# endpoint serves every runtime; the id is parsed from the request path
# (`/trials/{trial_runtime_id}/mcp`) and read by `tools/list` / `tools/call` to
# resolve the right runtime from the registry. This avoids mounting (and
# tearing down) a separate MCP endpoint per trial.
_request_trial_id: ContextVar[str | None] = ContextVar(
    "corral_mcp_request_trial_id", default=None
)


# Default ceiling for the server process's anyio worker-thread pool. The MCP
# tool-call handlers offload each synchronous `Environment.call_tool` to this
# pool (Section 9), so its size caps how many tool calls can run at once across
# *every* concurrent trial the server drives. anyio's own default is 40, which
# bottlenecks under very high concurrency (many trials each firing parallel tool
# calls); this raises the ceiling well above that. Threads are still created
# lazily — the ceiling only permits growth, it does not pre-spawn — so a small
# run costs nothing while a large one no longer serialises on a starved pool.
DEFAULT_MCP_THREAD_POOL_SIZE = 128


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


def _trial_id_from_scope(scope: Scope) -> str | None:
    """Parse the `trial_runtime_id` out of a `/trials/{id}/mcp` request scope.

    The endpoint is mounted at `/trials`. Depending on the Starlette version the
    scope path may arrive with the mount prefix stripped (`/{id}/mcp`) or intact
    (`/trials/{id}/mcp`), so anchor on the `trials` segment and take the segment
    after it, falling back to the first segment when the prefix is already gone.
    """
    raw = scope.get("path", "") if isinstance(scope, Mapping) else ""
    parts = [segment for segment in raw.split("/") if segment]
    if "trials" in parts:
        idx = parts.index("trials")
        return parts[idx + 1] if idx + 1 < len(parts) else None
    return parts[0] if parts else None


def _quiet_mcp_transport_logs() -> None:
    """Silence the low-level MCP transport's per-request INFO chatter.

    The Streamable-HTTP session manager and the low-level server each emit an
    INFO line for *every* request ("Processing request of type ...",
    "Terminating session: ..."). For our stateless, per-request benchmark
    episodes that is pure noise, so raise those two loggers to WARNING while
    leaving genuine warnings and errors intact.
    """
    for name in ("mcp.server.streamable_http", "mcp.server.lowlevel.server"):
        logging.getLogger(name).setLevel(logging.WARNING)


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


def _current_experimental(server: Server) -> Any | None:
    """Return the request's experimental (task) context, if any.

    The low-level server populates `request_context.experimental` with the
    task-augmentation metadata parsed from the request. It is only bound while a
    request is in flight, so a missing context resolves to `None`.
    """
    try:
        return server.request_context.experimental
    except LookupError:
        return None


def build_task_mcp_server(
    task_id: str, env: Environment, verbosity: ToolVerbosity
) -> Server:
    """Build a low-level MCP :class:`Server` scoped to a single task."""
    server: Server = Server(f"corral-{task_id}")
    # Serve the MCP Tasks surface (tasks/get|result|cancel|list) off this task's
    # JobManager so a task-augmented call has somewhere to be polled (Section 7).
    enable_job_task_surface(server, lambda: env)

    @server.list_tools()
    async def _list_tools() -> list[MCPTool]:
        # Honor a per-request `?verbosity=` override, else the mount default.
        effective = _request_verbosity.get() or verbosity
        return task_mcp_tools(env, effective)

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent] | CallToolResult | CreateTaskResult:
        # A task-augmented call of a background-capable tool returns a durable
        # task handle immediately instead of blocking (Section 7); everything
        # else falls through to the synchronous path.
        task_result = await maybe_start_task(
            env, name, arguments, _current_experimental(server)
        )
        if task_result is not None:
            return task_result
        # `Environment.call_tool` is synchronous and may run for a while, so
        # offload it to a worker thread rather than blocking the event loop that
        # drives every concurrent trial (Section 9). The environment's own locks
        # keep concurrent calls to one runtime safe.
        return await anyio.to_thread.run_sync(
            partial(execute_task_tool, env, task_id, name, arguments)
        )

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

    _quiet_mcp_transport_logs()

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


def _current_trial_env(
    trial_registry: Mapping[str, TrialRuntime],
) -> Environment | None:
    """Resolve the in-process environment for the trial bound to the request.

    The MCP surface (tool rendering plus the background-job task-polling
    surface) is defined against a live `Environment`, which only the in-process
    backend exposes. A runtime executing in a worker process (Phase 2) has no
    in-process env to hand back here, so callers get `None` and treat it as an
    inactive runtime — its MCP surface is served over RPC instead.
    """
    trial_id = _request_trial_id.get()
    if trial_id is None:
        return None
    runtime = trial_registry.get(trial_id)
    if isinstance(runtime, InProcessTrialRuntime):
        return runtime.env
    return None


def _current_worker_runtime(
    trial_registry: Mapping[str, TrialRuntime],
) -> WorkerTrialRuntime | None:
    """Resolve the worker-backed runtime for the request, if that's the backend.

    The in-process path (`_current_trial_env`) has a live `Environment`; a
    `"process"` env runs its trial in a worker instead, so its MCP tool
    surface is served by proxying `tools/list` and `tools/call` over the
    worker's pipe here. Returns `None` for in-process, unknown, or closed
    runtimes.
    """
    trial_id = _request_trial_id.get()
    if trial_id is None:
        return None
    runtime = trial_registry.get(trial_id)
    return runtime if isinstance(runtime, WorkerTrialRuntime) else None


def _worker_mcp_tools(
    runtime: WorkerTrialRuntime, verbosity: ToolVerbosity
) -> list[MCPTool]:
    """Fetch a worker runtime's MCP tool list over the pipe (blocking RPC)."""
    payload = runtime.mcp_tools_payload(verbosity)
    return [MCPTool(**tool) for tool in payload["tools"]]


def build_trial_mcp_server(
    trial_registry: Mapping[str, TrialRuntime], verbosity: ToolVerbosity
) -> Server:
    """Build one low-level MCP :class:`Server` shared by *all* trial runtimes.

    Its handlers resolve the target runtime from `trial_registry` using the
    per-request `_request_trial_id` set by the ASGI endpoint, so a single mount
    serves every runtime — including ones created long after startup, and
    including worker-backed ones whose tool surface is proxied over their pipe.
    """
    server: Server = Server("corral-trials")
    # One MCP Tasks surface shared by every runtime: the handlers resolve the
    # per-request runtime the same way the tool handlers do, then dispatch to its
    # JobManager (Section 7). Only the in-process backend exposes a live env with
    # a JobManager; worker-backed runtimes have none (wetlab has no
    # background-capable tools), so the task surface is inactive for them.
    enable_job_task_surface(server, lambda: _current_trial_env(trial_registry))

    @server.list_tools()
    async def _list_tools() -> list[MCPTool]:
        effective = _request_verbosity.get() or verbosity
        env = _current_trial_env(trial_registry)
        if env is not None:
            return task_mcp_tools(env, effective)
        # Worker-backed runtime: proxy the tool list over its pipe (offloaded so
        # the blocking RPC never stalls the loop serving other trials).
        worker = _current_worker_runtime(trial_registry)
        if worker is not None:
            return await anyio.to_thread.run_sync(
                partial(_worker_mcp_tools, worker, effective)
            )
        # Unknown/closed runtime: expose no tools rather than error out.
        return []

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent] | CallToolResult | CreateTaskResult:
        env = _current_trial_env(trial_registry)
        if env is not None:
            # A task-augmented call of a background-capable tool returns a
            # durable task handle immediately instead of blocking (Section 7);
            # resolve the runtime on the loop first (both this and the fallback
            # read a ContextVar).
            task_result = await maybe_start_task(
                env, name, arguments, _current_experimental(server)
            )
            if task_result is not None:
                return task_result
            # Offload the synchronous call so one runtime's (possibly slow) tool
            # never blocks the loop serving every other concurrent trial. The
            # runtime's locks guard its state (Section 9).
            return await anyio.to_thread.run_sync(
                partial(execute_task_tool, env, env.task_id, name, arguments)
            )

        # Worker-backed runtime: proxy the call over its pipe. The worker runs
        # the same `Environment.call_tool`, so the rendered result is identical;
        # offload the blocking RPC so it doesn't stall the loop.
        worker = _current_worker_runtime(trial_registry)
        if worker is not None:
            tool_call = await anyio.to_thread.run_sync(
                partial(worker.call_tool, name, arguments or {})
            )
            return _render_tool_call(tool_call)

        trial_id = _request_trial_id.get()
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Trial runtime {trial_id!r} is not active.",
                )
            ],
            isError=True,
        )

    return server


def attach_trial_mcp_server(
    app: FastAPI,
    trial_registry: Mapping[str, TrialRuntime],
    verbosity: ToolVerbosity = ToolVerbosity.FULL,
) -> None:
    """Mount one MCP endpoint serving every trial runtime at `/trials`.

    A request to `/trials/{trial_runtime_id}/mcp` is dispatched to the runtime
    named in its path (resolved live against `trial_registry`), so runtimes
    created after startup need no per-trial mount. The REST `/trials/{id}/...`
    routes are registered before this mount, so they still win for their exact
    paths; only `.../mcp` falls through to here.
    """
    # The manager wraps a server that resolves the target env per request from
    # the registry, rather than binding a single environment at mount time.
    manager = StreamableHTTPSessionManager(
        app=build_trial_mcp_server(trial_registry, verbosity),
        json_response=True,
        stateless=True,
    )

    _quiet_mcp_transport_logs()

    async def _endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        # Bind both the request verbosity and the trial-runtime id for the
        # duration of the request so `tools/list` and `tools/call` can read them.
        v_token = _request_verbosity.set(_verbosity_from_scope(scope))
        t_token = _request_trial_id.set(_trial_id_from_scope(scope))
        try:
            await manager.handle_request(scope, receive, send)
        finally:
            _request_verbosity.reset(v_token)
            _request_trial_id.reset(t_token)

    app.mount("/trials", _endpoint)
    _install_lifespan(app, {"__trials__": manager})
    logger.info("Mounted trial MCP endpoint at /trials/{trial_runtime_id}/mcp")


def install_mcp_thread_capacity(app: FastAPI, min_threads: int) -> None:
    """Raise the server's anyio worker-thread pool at startup (Section 9).

    The MCP tool-call handlers offload each synchronous
    :meth:`Environment.call_tool` to anyio's default worker-thread pool so one
    runtime's slow tool never blocks the loop serving every other concurrent
    trial. That pool defaults to 40 threads, which caps how many tool calls run
    at once across the *whole* server — a bottleneck when many trials each fire
    parallel tool calls. This chains a startup step onto `app`'s lifespan that
    raises the pool ceiling to `min_threads`. It only ever *raises* the ceiling
    (never shrinks a pool another component sized up), and it runs inside the
    event loop where `current_default_thread_limiter` is valid.
    """
    previous = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        limiter = anyio.to_thread.current_default_thread_limiter()
        if limiter.total_tokens < min_threads:
            limiter.total_tokens = min_threads
        async with previous(app):
            yield

    app.router.lifespan_context = lifespan


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
