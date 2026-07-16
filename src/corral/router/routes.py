import copy
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Self
from urllib.parse import quote, urlencode

import anyio
import httpx
import requests  # type: ignore[import-untyped]
from loguru import logger

from corral.backend.schema import TrialCompletionResponse
from corral.report.results import TaskTrialResult
from corral.router.verbosity import ToolVerbosity
from corral.types import ToolResponse


async def acall(method: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Invoke a benchmark-interface method, awaiting async, offloading sync.

    The async scheduler and the natively-async agents drive the router through
    this one seam so the *same* call site works whether `interface` is an
    :class:`AsyncCorralRouter` or the synchronous :class:`CorralRouter`:

    * an `async def` method (the async router) is **awaited directly**, so the
      event loop stays free while the request is in flight and no worker thread
      is used — this is exactly the "native path never touches a worker thread
      for HTTP" property the concurrency plan targets;
    * a plain method (the sync router, or a fake) has its blocking `requests`
      call **offloaded to a worker thread** so it never stalls the loop.

    Detection is per-method via :func:`inspect.iscoroutinefunction`, so a
    partially-async interface still routes each call correctly, and the method is
    invoked **exactly once** on either branch (important — these calls create
    trials, submit answers, and configure apps, so a double invocation would be a
    real side effect).
    """
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await anyio.to_thread.run_sync(partial(method, *args, **kwargs))


def as_sync_interface(interface: Any) -> Any:
    """Return a synchronous view of `interface` for thread-offloaded code.

    A non-native agent runs its blocking, synchronous :meth:`BaseAgent.run` in a
    worker thread that has no event loop, so it cannot drive an
    :class:`AsyncCorralRouter` (its methods return un-awaited coroutines). When
    handed one, this returns the equivalent synchronous :class:`CorralRouter`
    (via `interface.to_sync()`) for that thread to use. An already-synchronous
    interface — or any object without `to_sync` — is returned unchanged, so the
    historical single-router path is byte-for-byte the same. Native agents await
    the async router directly and never need this conversion.
    """
    to_sync = getattr(interface, "to_sync", None)
    if callable(to_sync):
        return to_sync()
    return interface


def _parse_trial_completion(task_id: str, response_data: dict) -> TaskTrialResult:
    """
    Convert TrialCompletionResponse JSON to TaskTrialResult.

    This helper validates the server response using the Pydantic model and
    converts it to the TaskTrialResult format used for reporting.

    Args:
        task_id: The task identifier
        response_data: JSON response data from server

    Returns:
        TaskTrialResult with data from the completion response
    """
    completion = TrialCompletionResponse(**response_data)
    return TaskTrialResult(
        task_id=task_id,
        trial_id=completion.trial_id,
        score=completion.score,
        state=completion.state,
        tool_statistics=completion.state["tool_statistics"],
        surrendered=completion.surrendered,
        duration=completion.state.get("duration"),
    )


class _RouterURLs:
    """URL-building and verbosity helpers shared by the sync and async routers.

    Both :class:`CorralRouter` (top-level `requests`) and
    :class:`AsyncCorralRouter` (a shared `httpx.AsyncClient`) must build the
    exact same endpoint URLs and resolve tool verbosity the same way. Keeping
    that logic in one mixin means the two transports can never drift on the URL
    convention or the verbosity fallback. Every helper here is a pure
    string/attribute operation with no I/O, so it stays synchronous on both.

    Subclasses are expected to define `base_url` and `current_verbosity` in
    their own `__init__` (mirroring the historical :class:`CorralRouter`), so
    this mixin has no `__init__` of its own.
    """

    base_url: str
    current_verbosity: str | None

    def set_verbosity(self, verbosity: str) -> None:
        """Set the verbosity level for subsequent requests.

        Mutates shared state, so it is only safe on a router that is *not* shared
        across concurrent trials (the serial path). When a router may be shared,
        bind verbosity into the trial/benchmark context instead — via
        :meth:`with_verbosity` or `for_trial(..., verbosity=...)` — so no
        concurrent caller observes another's mutation.
        """
        self.current_verbosity = verbosity
        logger.info(f"Set tool verbosity to: {verbosity}")

    def with_verbosity(self, verbosity: str | None) -> Self:
        """Return a shallow view of this router bound to a fixed `verbosity`.

        The view shares this router's transport (the `requests` module for the
        sync router, or the `httpx.AsyncClient` for the async one) but carries
        its **own** `current_verbosity`, so binding a benchmark- or trial-level
        verbosity never mutates a router other concurrent trials still hold. This
        is the "bind verbosity into the context" seam from the concurrency plan:
        prefer it (or `for_trial(..., verbosity=...)`) over
        :meth:`set_verbosity` whenever a router is shared.
        """
        clone = copy.copy(self)
        clone.current_verbosity = verbosity
        return clone

    def _resolve_verbosity(self, verbosity: str | None) -> str | None:
        """Fall back to the router's bound verbosity when a call omits one."""
        return verbosity or self.current_verbosity

    def _task_url(self, task_id: str, suffix: str = "") -> str:
        """Build a task-scoped endpoint URL: `{base}/tasks/{task_id}{suffix}`.

        Every mutable, task-scoped REST call goes through this one seam so a
        trial-scoped router can retarget them all at `/trials/{id}/...` by
        overriding just this method (and :meth:`mcp_url`).
        """
        return f"{self.base_url}/tasks/{task_id}{suffix}"

    def mcp_url(self, task_id: str, verbosity: str | None = None) -> str:
        """Build the MCP endpoint URL an MCP client should connect to for a task.

        Centralises the URL convention every MCP harness (Claude Code, Codex,
        OpenHands) needs, so a per-trial router can override just this method to
        point at `/trials/{id}/mcp` instead of `/tasks/{id}/mcp`.

        The `verbosity` is forwarded as a query parameter so the MCP tool
        descriptions match exactly the verbosity the REST allowlist was fetched
        at. The trailing slash on `/mcp/` hits the canonical path directly and
        avoids a 307 redirect (the endpoint is a Starlette mount).
        """
        verbosity = verbosity or self.current_verbosity or ToolVerbosity.BRIEF.value
        base_url = self.base_url.rstrip("/")
        encoded_task_id = quote(str(task_id), safe="")
        query = urlencode({"verbosity": verbosity})
        return f"{base_url}/tasks/{encoded_task_id}/mcp/?{query}"


class CorralRouter(_RouterURLs):
    """General interface for interacting with benchmark server"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_verbosity: str | None = ToolVerbosity.BRIEF.value,
    ):
        self.base_url = base_url
        self.current_verbosity = default_verbosity

    def create_trial(
        self,
        task_id: str,
        benchmark_run_id: str | None = None,
        episode_id: str | None = None,
        trial_index: int | None = None,
        tool_jobs_per_trial: int | None = None,
    ) -> dict[str, Any]:
        """Open a fresh, isolated runtime for one trial of `task_id`.

        Returns the server's runtime descriptor
        (`trial_runtime_id`, `task_id`, `workspace`, `mcp_url`). Pair with
        :meth:`for_trial` to get a router scoped to the returned runtime, and
        :meth:`close_trial` to free it when done.

        `tool_jobs_per_trial` (from `ConcurrencyConfig.tool_jobs_per_trial`)
        sizes the new runtime's background-job pool; `None` keeps the server's
        own default.
        """
        payload = {
            "benchmark_run_id": benchmark_run_id,
            "episode_id": episode_id,
            "trial_index": trial_index,
            "tool_jobs_per_trial": tool_jobs_per_trial,
        }
        response = requests.post(
            f"{self.base_url}/tasks/{task_id}/trials", json=payload
        )
        response.raise_for_status()
        return response.json()

    def close_trial(self, trial_runtime_id: str) -> None:
        """Discard a trial runtime created with :meth:`create_trial`."""
        response = requests.delete(f"{self.base_url}/trials/{trial_runtime_id}")
        response.raise_for_status()

    def close_episode(self, episode_id: str) -> None:
        """Free an episode's shared dependency-output store on the server.

        An *episode* is one trial round of a dependency chain: every trial
        runtime created with the same `episode_id` shares one output store so
        chained siblings see each other's results. Call this once the whole
        round has finished so those outputs never leak into the next round.
        """
        response = requests.delete(f"{self.base_url}/episodes/{episode_id}")
        response.raise_for_status()

    def for_trial(
        self,
        trial_runtime_id: str,
        task_id: str,
        *,
        verbosity: str | None = None,
        workspace: str | None = None,
    ) -> "TrialScopedRouter":
        """Return a router whose mutable calls address one trial runtime.

        The returned router exposes the same method surface agents already use
        (`get_task_prompt`, `execute_tool`, `submit_answer`, ...) but routes
        every call to `/trials/{trial_runtime_id}/...` and points `mcp_url` at
        that runtime's MCP mount, so concurrent trials never share state.

        `verbosity`, when given, is **bound** into the returned router's
        `current_verbosity`, so the trial's verbosity travels with its own router
        rather than being read from this (potentially shared) parent's mutable
        field. This is the per-trial half of the plan's "bind verbosity into the
        context" rule; omitting it falls back to the parent's current verbosity,
        preserving the historical behaviour.

        `workspace` is the runtime's server-side trial workspace (from
        `create_trial`). It is carried on the returned router as
        `trial_workspace` so a sandbox-running agent (e.g. Codex) can point its
        own working directory at the *scored* workspace instead of a throwaway
        temp dir, ensuring files the model writes there survive to scoring.
        """
        return TrialScopedRouter(
            self, trial_runtime_id, task_id, verbosity=verbosity, workspace=workspace
        )

    def get_available_tasks(self) -> list[str]:
        """Get list of available task IDs"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def supports_dependency_chain(self) -> bool:
        """Check if the environment supports dependency chaining"""
        try:
            response = requests.get(f"{self.base_url}/dependency_chain")
            response.raise_for_status()
            return response.json()["dependency_chain"]
        except Exception:
            return False

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Fetch the task dependency graph as a `{task_id: [deps]}` dict."""
        response = requests.get(f"{self.base_url}/dependency_graph")
        response.raise_for_status()
        return response.json()

    def get_concurrency_modes(self) -> dict[str, str]:
        """Fetch each task's env concurrency mode as `{task_id: mode}`.

        Lets the concurrent scheduler serialise `"process"`/`"serial"` envs (see
        `Environment.concurrency`). A server predating the `/concurrency`
        endpoint 404s; the scheduler treats a missing map as all-`"thread"`.
        """
        response = requests.get(f"{self.base_url}/concurrency")
        response.raise_for_status()
        return response.json()

    def get_trial_worker_active(self) -> bool:
        """Whether the server has a live process-worker pool (Phase 3).

        Tells the concurrent scheduler whether independent `"process"` trials can
        be routed to isolated worker processes and thus overlap, instead of being
        serialised. A server predating the `/trial_worker` endpoint 404s; the
        scheduler then treats the backend as inactive and keeps serialising.
        """
        response = requests.get(f"{self.base_url}/trial_worker")
        response.raise_for_status()
        return bool(response.json().get("active", False))

    def get_available_tools_for_task(
        self, task_id: str, verbosity: str | None = None
    ) -> dict[str, Any]:
        """Get list of available tools for a task with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(self._task_url(task_id, "/tools"), params=params)
        response.raise_for_status()
        return response.json()

    def get_mcp_tool_schema(
        self, task_id: str, verbosity: str | None = None
    ) -> dict[str, Any]:
        """Get the task's tools in MCP representation plus their schema digest.

        Returns a dict with tools (the MCP tools/list payload) and
        mcp_schema_sha256 (a hash of exactly what an MCP client sees).
        """
        verbosity = verbosity or self.current_verbosity
        params = {"verbosity": verbosity}
        response = requests.get(self._task_url(task_id, "/tools/mcp"), params=params)
        response.raise_for_status()
        return response.json()

    def get_task_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get complete guide for task including tools with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(self._task_url(task_id, "/guide"), params=params)
        response.raise_for_status()
        return response.json()["prompt"]

    def get_tools_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get tools guide for task with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(self._task_url(task_id, "/tools/guide"), params=params)
        response.raise_for_status()
        return response.json()["prompt"]

    def get_task_prompt(self, task_id: str) -> str | list[dict]:
        """Get task prompt without tools description"""
        response = requests.get(self._task_url(task_id, "/prompt"))
        response.raise_for_status()
        return response.json()["prompt"]

    def execute_tool(
        self, task_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResponse:
        """Execute a tool and get result"""
        try:
            logger.info(f"Agent calling tool {tool_name} with args {arguments}")
            response = requests.post(
                self._task_url(task_id, "/tools/execute"),
                json={"tool_name": tool_name, "arguments": arguments},
            )
            response.raise_for_status()
            data = response.json()
            return ToolResponse(success=True, result=data["result"], error=None)
        except Exception as e:
            return ToolResponse(success=False, result=None, error=str(e))

    def submit_answer(self, task_id: str, answer: str) -> TaskTrialResult:
        """Submit final answer for a task"""
        logger.info(f"Agent submitting answer {answer} for task {task_id}")
        response = requests.post(
            self._task_url(task_id, "/submit"), json={"answer": answer}
        )
        response.raise_for_status()
        return _parse_trial_completion(task_id, response.json())

    def surrender_task(self, task_id: str) -> TaskTrialResult:
        """Surrender from a task without submitting an answer"""
        logger.info(f"Agent retiring from task {task_id}")
        response = requests.post(self._task_url(task_id, "/surrender"))
        response.raise_for_status()
        return _parse_trial_completion(task_id, response.json())

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get current status of a task"""
        response = requests.get(self._task_url(task_id, "/status"))
        response.raise_for_status()
        return response.json()

    def get_last_score(self, task_id: str) -> dict[str, Any]:
        """Get the score from the most recent trial submission"""
        response = requests.get(self._task_url(task_id, "/last_score"))
        response.raise_for_status()
        return response.json()

    def get_trial_state(self, task_id: str, trial_id: str) -> dict[str, Any]:
        """Get specific trial state"""
        response = requests.get(self._task_url(task_id, f"/trials/{trial_id}"))
        response.raise_for_status()
        return response.json()["trial_state"]

    def configure_additional_apps(
        self, task_id: str, timeout: float | None = None
    ) -> dict:
        """Configure additional apps/services for specific task

        Args:
            task_id: The task identifier to configure.
            timeout: Optional timeout in seconds for the HTTP request.
        """
        url = self._task_url(task_id, "/configure")
        if timeout is None:
            response = requests.post(url)
        else:
            response = requests.post(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def generate_latex(
        self,
        task_id: str,
        output_dir: str,
        level: int | str,
        env_name: str | None = None,
        task_name: str | None = None,
        verbosity: str | None = None,
    ) -> dict[str, str]:
        """Generate LaTeX documentation for a task.

        Args:
            task_id: The task identifier.
            output_dir: Directory for output .tex files.
            level: Task level identifier (e.g., 1, 2, "advanced").
            env_name: Environment name (e.g., "afm", "catalyst").
            task_name: Optional custom name for the task.
            verbosity: Tool verbosity level used to filter descriptions and
                       return sections (e.g. `"brief"`, `"detailed"`). Defaults
                       to `"detailed"` when not provided.

        Returns:
            Dictionary with 'output_path' (task .tex), 'tools_output_path' (tools .tex),
            and 'scoring_output_path' (scoring functions .tex, may be None).
        """
        payload = {
            "output_dir": output_dir,
            "level": level,
            "env_name": env_name,
            "task_name": task_name,
            "verbosity": verbosity,
        }
        response = requests.post(self._task_url(task_id, "/latex"), json=payload)
        response.raise_for_status()
        return response.json()


class TrialScopedRouter(CorralRouter):
    """A :class:`CorralRouter` view bound to one trial runtime.

    Because the base router funnels every mutable, task-scoped call through
    :meth:`CorralRouter._task_url` (and MCP through :meth:`mcp_url`), retargeting
    an entire agent run at an isolated `/trials/{trial_runtime_id}/...` runtime
    only takes overriding those two seams — every inherited method (prompt,
    tools, execute, submit, surrender, status, configure) then addresses the
    runtime automatically. The `task_id` an agent passes is ignored for routing
    (the bound runtime already identifies the task) but still flows through to
    result provenance via the inherited methods.
    """

    def __init__(
        self,
        parent: CorralRouter,
        trial_runtime_id: str,
        task_id: str,
        *,
        verbosity: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self.base_url = parent.base_url
        # Bind the trial's verbosity into this router so it no longer depends on
        # a mutable field the shared parent may change under a concurrent trial;
        # fall back to the parent's current verbosity when none is bound.
        self.current_verbosity = verbosity or parent.current_verbosity
        self.trial_runtime_id = trial_runtime_id
        self.task_id = task_id
        # The runtime's server-side trial workspace (may be None for template
        # envs / non-runtime execution). Carried so a sandbox-running agent can
        # write into the scored workspace. See `for_trial`.
        self.trial_workspace = workspace

    def _task_url(self, task_id: str, suffix: str = "") -> str:  # noqa: ARG002
        # Routing is by the bound runtime id, so the caller's task_id is ignored.
        return f"{self.base_url}/trials/{self.trial_runtime_id}{suffix}"

    def mcp_url(self, task_id: str | None = None, verbosity: str | None = None) -> str:  # noqa: ARG002
        """Point the MCP client at this runtime's endpoint (`/trials/{id}/mcp`)."""
        verbosity = verbosity or self.current_verbosity or ToolVerbosity.BRIEF.value
        base_url = self.base_url.rstrip("/")
        encoded = quote(str(self.trial_runtime_id), safe="")
        query = urlencode({"verbosity": verbosity})
        return f"{base_url}/trials/{encoded}/mcp/?{query}"


class AsyncCorralRouter(_RouterURLs):
    """Async twin of :class:`CorralRouter` over one long-lived `httpx.AsyncClient`.

    :class:`CorralRouter` makes a blocking top-level `requests` call per method,
    which stalls the event loop when a concurrent scheduler drives many trials on
    it. This router issues the same requests through a **shared**
    `httpx.AsyncClient` instead, so its methods are `await`-able and the loop
    stays free while HTTP is in flight. A single reused client also pools
    connections across concurrent tasks — HTTPX recommends against constructing a
    client per call in a hot loop.

    URL building and verbosity resolution are inherited verbatim from
    :class:`_RouterURLs`, so this router and the synchronous one can never drift
    on the endpoint convention. The HTTP-issuing methods keep the **same names**
    as :class:`CorralRouter` (they are just `async def`), matching the async
    API the concurrency plan targets — `result = await router.submit_answer(...)`.

    The client may be supplied (so callers can share one pool across many
    routers) or created internally. A router that owns its client closes it on
    :meth:`aclose` / async-context exit; one handed an external client leaves it
    open for the owner to close. Pure, non-I/O helpers (`mcp_url`,
    `set_verbosity`, `with_verbosity`, `for_trial`) stay synchronous.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        client: httpx.AsyncClient | None = None,
        default_verbosity: str | None = ToolVerbosity.BRIEF.value,
    ):
        self.base_url = base_url
        self.current_verbosity = default_verbosity
        if client is None:
            # No default timeout, matching `requests`' behaviour on the sync
            # router (app configuration and agent submits can run long); callers
            # that need a deadline pass one per call or supply their own client.
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
            self._owns_client = True
        else:
            self.client = client
            self._owns_client = False

    def with_verbosity(self, verbosity: str | None) -> Self:
        """A verbosity-bound view that shares (but never owns) the client."""
        clone = super().with_verbosity(verbosity)
        # A view must never close the shared client out from under the original.
        clone._owns_client = False
        return clone

    async def aclose(self) -> None:
        """Close the underlying client if this router created it."""
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def to_sync(self) -> CorralRouter:
        """Return a synchronous :class:`CorralRouter` over the same server.

        A non-native agent's blocking :meth:`BaseAgent.run` runs in a worker
        thread with no event loop and so cannot drive this async router; the
        default :meth:`BaseAgent.arun` hands it this equivalent synchronous
        router instead (same `base_url` and current verbosity, top-level
        `requests` with no shared client). The native path awaits this router
        directly and never calls this — see :func:`as_sync_interface`.
        """
        return CorralRouter(self.base_url, default_verbosity=self.current_verbosity)

    async def create_trial(
        self,
        task_id: str,
        benchmark_run_id: str | None = None,
        episode_id: str | None = None,
        trial_index: int | None = None,
        tool_jobs_per_trial: int | None = None,
    ) -> dict[str, Any]:
        """Open a fresh, isolated runtime for one trial of `task_id`.

        `tool_jobs_per_trial` (from `ConcurrencyConfig.tool_jobs_per_trial`)
        sizes the new runtime's background-job pool; `None` keeps the server's
        own default.
        """
        payload = {
            "benchmark_run_id": benchmark_run_id,
            "episode_id": episode_id,
            "trial_index": trial_index,
            "tool_jobs_per_trial": tool_jobs_per_trial,
        }
        response = await self.client.post(
            f"{self.base_url}/tasks/{task_id}/trials", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def close_trial(self, trial_runtime_id: str) -> None:
        """Discard a trial runtime created with :meth:`create_trial`."""
        response = await self.client.delete(
            f"{self.base_url}/trials/{trial_runtime_id}"
        )
        response.raise_for_status()

    async def close_episode(self, episode_id: str) -> None:
        """Free an episode's shared dependency-output store on the server."""
        response = await self.client.delete(f"{self.base_url}/episodes/{episode_id}")
        response.raise_for_status()

    def for_trial(
        self,
        trial_runtime_id: str,
        task_id: str,
        *,
        verbosity: str | None = None,
        workspace: str | None = None,
    ) -> "AsyncTrialScopedRouter":
        """Return an async router whose mutable calls address one trial runtime.

        Mirrors :meth:`CorralRouter.for_trial`: the returned router shares this
        one's client (and never closes it), routes every call at
        `/trials/{trial_runtime_id}/...`, and binds `verbosity` into its own
        `current_verbosity` (falling back to this router's) so a shared parent's
        mutations never leak into the trial. `workspace` is carried as
        `trial_workspace` for sandbox-running agents (see the sync twin).
        """
        return AsyncTrialScopedRouter(
            self, trial_runtime_id, task_id, verbosity=verbosity, workspace=workspace
        )

    async def get_available_tasks(self) -> list[str]:
        """Get list of available task IDs"""
        response = await self.client.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    async def supports_dependency_chain(self) -> bool:
        """Check if the environment supports dependency chaining"""
        try:
            response = await self.client.get(f"{self.base_url}/dependency_chain")
            response.raise_for_status()
            return response.json()["dependency_chain"]
        except Exception:
            return False

    async def get_dependency_graph(self) -> dict[str, list[str]]:
        """Fetch the task dependency graph as a `{task_id: [deps]}` dict."""
        response = await self.client.get(f"{self.base_url}/dependency_graph")
        response.raise_for_status()
        return response.json()

    async def get_concurrency_modes(self) -> dict[str, str]:
        """Fetch each task's env concurrency mode as `{task_id: mode}`.

        Lets the concurrent scheduler serialise `"process"`/`"serial"` envs (see
        `Environment.concurrency`). A server predating the `/concurrency`
        endpoint 404s; the scheduler treats a missing map as all-`"thread"`.
        """
        response = await self.client.get(f"{self.base_url}/concurrency")
        response.raise_for_status()
        return response.json()

    async def get_trial_worker_active(self) -> bool:
        """Whether the server has a live process-worker pool (Phase 3).

        Async twin of :meth:`CorralRouter.get_trial_worker_active`. A server
        predating the `/trial_worker` endpoint 404s; the scheduler then treats
        the backend as inactive and keeps serialising `"process"` envs.
        """
        response = await self.client.get(f"{self.base_url}/trial_worker")
        response.raise_for_status()
        return bool(response.json().get("active", False))

    async def get_available_tools_for_task(
        self, task_id: str, verbosity: str | None = None
    ) -> dict[str, Any]:
        """Get list of available tools for a task with specified verbosity"""
        params = {"verbosity": self._resolve_verbosity(verbosity)}
        response = await self.client.get(
            self._task_url(task_id, "/tools"), params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_mcp_tool_schema(
        self, task_id: str, verbosity: str | None = None
    ) -> dict[str, Any]:
        """Get the task's tools in MCP representation plus their schema digest."""
        params = {"verbosity": self._resolve_verbosity(verbosity)}
        response = await self.client.get(
            self._task_url(task_id, "/tools/mcp"), params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_task_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get complete guide for task including tools with specified verbosity"""
        params = {"verbosity": self._resolve_verbosity(verbosity)}
        response = await self.client.get(
            self._task_url(task_id, "/guide"), params=params
        )
        response.raise_for_status()
        return response.json()["prompt"]

    async def get_tools_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get tools guide for task with specified verbosity"""
        params = {"verbosity": self._resolve_verbosity(verbosity)}
        response = await self.client.get(
            self._task_url(task_id, "/tools/guide"), params=params
        )
        response.raise_for_status()
        return response.json()["prompt"]

    async def get_task_prompt(self, task_id: str) -> str | list[dict]:
        """Get task prompt without tools description"""
        response = await self.client.get(self._task_url(task_id, "/prompt"))
        response.raise_for_status()
        return response.json()["prompt"]

    async def execute_tool(
        self, task_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResponse:
        """Execute a tool and get result"""
        try:
            logger.info(f"Agent calling tool {tool_name} with args {arguments}")
            response = await self.client.post(
                self._task_url(task_id, "/tools/execute"),
                json={"tool_name": tool_name, "arguments": arguments},
            )
            response.raise_for_status()
            data = response.json()
            return ToolResponse(success=True, result=data["result"], error=None)
        except Exception as e:
            return ToolResponse(success=False, result=None, error=str(e))

    async def submit_answer(self, task_id: str, answer: str) -> TaskTrialResult:
        """Submit final answer for a task"""
        logger.info(f"Agent submitting answer {answer} for task {task_id}")
        response = await self.client.post(
            self._task_url(task_id, "/submit"), json={"answer": answer}
        )
        response.raise_for_status()
        return _parse_trial_completion(task_id, response.json())

    async def surrender_task(self, task_id: str) -> TaskTrialResult:
        """Surrender from a task without submitting an answer"""
        logger.info(f"Agent retiring from task {task_id}")
        response = await self.client.post(self._task_url(task_id, "/surrender"))
        response.raise_for_status()
        return _parse_trial_completion(task_id, response.json())

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get current status of a task"""
        response = await self.client.get(self._task_url(task_id, "/status"))
        response.raise_for_status()
        return response.json()

    async def get_last_score(self, task_id: str) -> dict[str, Any]:
        """Get the score from the most recent trial submission"""
        response = await self.client.get(self._task_url(task_id, "/last_score"))
        response.raise_for_status()
        return response.json()

    async def get_trial_state(self, task_id: str, trial_id: str) -> dict[str, Any]:
        """Get specific trial state"""
        response = await self.client.get(self._task_url(task_id, f"/trials/{trial_id}"))
        response.raise_for_status()
        return response.json()["trial_state"]

    async def configure_additional_apps(
        self, task_id: str, timeout: float | None = None
    ) -> dict:
        """Configure additional apps/services for specific task."""
        url = self._task_url(task_id, "/configure")
        if timeout is None:
            response = await self.client.post(url)
        else:
            response = await self.client.post(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def generate_latex(
        self,
        task_id: str,
        output_dir: str,
        level: int | str,
        env_name: str | None = None,
        task_name: str | None = None,
        verbosity: str | None = None,
    ) -> dict[str, str]:
        """Generate LaTeX documentation for a task."""
        payload = {
            "output_dir": output_dir,
            "level": level,
            "env_name": env_name,
            "task_name": task_name,
            "verbosity": verbosity,
        }
        response = await self.client.post(
            self._task_url(task_id, "/latex"), json=payload
        )
        response.raise_for_status()
        return response.json()


class AsyncTrialScopedRouter(AsyncCorralRouter):
    """An :class:`AsyncCorralRouter` view bound to one trial runtime.

    The async counterpart of :class:`TrialScopedRouter`: it overrides the same
    two seams (`_task_url` + `mcp_url`) so every inherited async method addresses
    `/trials/{trial_runtime_id}/...`, and it **shares the parent's client without
    owning it** (so closing the view never disturbs the shared pool). Building it
    does no I/O, so :meth:`AsyncCorralRouter.for_trial` stays synchronous.
    """

    def __init__(
        self,
        parent: AsyncCorralRouter,
        trial_runtime_id: str,
        task_id: str,
        *,
        verbosity: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self.base_url = parent.base_url
        self.client = parent.client
        # A view is never the client's owner; only the router that created the
        # client (or the external owner) closes it.
        self._owns_client = False
        # Bind the trial's verbosity so it no longer depends on a mutable field
        # the shared parent may change under a concurrent trial.
        self.current_verbosity = verbosity or parent.current_verbosity
        self.trial_runtime_id = trial_runtime_id
        self.task_id = task_id
        # The runtime's server-side trial workspace (see the sync twin).
        self.trial_workspace = workspace

    def to_sync(self) -> TrialScopedRouter:
        """Return a synchronous trial-scoped router bound to the same runtime.

        Mirrors :meth:`AsyncCorralRouter.to_sync` for a trial view: a non-native
        agent offloaded onto a worker thread gets a :class:`TrialScopedRouter`
        that still addresses `/trials/{trial_runtime_id}/...` (same runtime,
        same bound verbosity, same trial workspace) over the synchronous
        `requests` transport.
        """
        parent = CorralRouter(self.base_url, default_verbosity=self.current_verbosity)
        return TrialScopedRouter(
            parent,
            self.trial_runtime_id,
            self.task_id,
            verbosity=self.current_verbosity,
            workspace=self.trial_workspace,
        )

    def _task_url(self, task_id: str, suffix: str = "") -> str:  # noqa: ARG002
        # Routing is by the bound runtime id, so the caller's task_id is ignored.
        return f"{self.base_url}/trials/{self.trial_runtime_id}{suffix}"

    def mcp_url(self, task_id: str | None = None, verbosity: str | None = None) -> str:  # noqa: ARG002
        """Point the MCP client at this runtime's endpoint (`/trials/{id}/mcp`)."""
        verbosity = verbosity or self.current_verbosity or ToolVerbosity.BRIEF.value
        base_url = self.base_url.rstrip("/")
        encoded = quote(str(self.trial_runtime_id), safe="")
        query = urlencode({"verbosity": verbosity})
        return f"{base_url}/trials/{encoded}/mcp/?{query}"
