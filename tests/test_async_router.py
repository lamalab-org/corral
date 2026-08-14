"""Tests for the router changes in Section 10 of `make_efficiency.md`.

Two deliverables are covered:

* :class:`AsyncCorralRouter` — the async twin of :class:`CorralRouter` over one
  long-lived `httpx.AsyncClient`. It is driven end-to-end against a real
  benchmark server through `httpx.ASGITransport` (no network), exercising the
  task surface, the trial-runtime lifecycle, and client ownership.
* Verbosity binding — `for_trial(..., verbosity=...)` and `with_verbosity`
  carry a fixed verbosity on their own router instead of reading a shared
  mutable `current_verbosity`, so a shared router is safe under concurrency.

The async flows are driven with `anyio.run` inside synchronous test
functions, so no async test runner/plugin is required.
"""

import json
import threading
from typing import Literal

import anyio
import httpx
import pytest
from pydantic import Field

from corral.backend.env import Toolset, build_environments
from corral.backend.server import create_benchmark_server
from corral.backend.task import TaskDefinition
from corral.backend.tool import tool
from corral.core.action import submit_answer_action
from corral.router.routes import (
    AsyncCorralRouter,
    AsyncTrialScopedRouter,
    CorralRouter,
    TrialScopedRouter,
    acall,
    as_sync_interface,
)
from corral.run import CorralRunner


def calc(
    operation: Literal["add", "mul"] = Field(description="operation to perform"),
    x: float = Field(description="first operand"),
    y: float = Field(default=1.0, description="second operand"),
) -> str:
    """Perform a basic math operation."""
    return str(x + y if operation == "add" else x * y)


def _make_task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        name=task_id,
        description="a task",
        tools=["calc"],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
    )


def _build_environments(*task_ids: str) -> dict:
    tasks = {task_id: _make_task(task_id) for task_id in task_ids}
    toolset = Toolset(pool={"calc": tool(calc)}, workspace_factory=None)
    return build_environments(tasks, toolset=toolset)


def _server(*task_ids: str):
    return create_benchmark_server(_build_environments(*task_ids))


def test_async_router_drives_task_and_trial_surface():
    app = _server("task_a")

    async def _drive() -> None:
        # ASGITransport routes every request into the app in-process; enter the
        # app lifespan so the server's startup/shutdown (MCP mounts) runs.
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                router = AsyncCorralRouter("http://test", client=client)

                # Task-scoped surface.
                assert await router.get_available_tasks() == ["task_a"]
                assert isinstance(await router.supports_dependency_chain(), bool)
                tools = await router.get_available_tools_for_task("task_a")
                assert "calc" in json.dumps(tools)
                prompt = await router.get_task_prompt("task_a")
                assert prompt  # non-empty prompt

                # Trial-runtime lifecycle: create -> scope -> execute -> submit.
                created = await router.create_trial("task_a", trial_index=0)
                rid = created["trial_runtime_id"]
                assert rid.startswith("tr_")

                trial = router.for_trial(rid, "task_a")
                assert isinstance(trial, AsyncTrialScopedRouter)

                exec_resp = await trial.execute_tool(
                    "task_a", "calc", {"operation": "add", "x": 2, "y": 5}
                )
                assert exec_resp.success
                assert exec_resp.result["result"] == "7"

                result = await trial.submit_answer("task_a", "7")
                assert result.score == 1.0
                assert result.surrendered is False

                # Freeing the runtime removes it: a later scoped call 404s.
                await router.close_trial(rid)
                with pytest.raises(httpx.HTTPStatusError):
                    await trial.get_task_status("task_a")

    anyio.run(_drive)


def test_async_router_surrender_via_trial_runtime():
    app = _server("task_a")

    async def _drive() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                router = AsyncCorralRouter("http://test", client=client)
                rid = (await router.create_trial("task_a", trial_index=0))[
                    "trial_runtime_id"
                ]
                trial = router.for_trial(rid, "task_a")
                result = await trial.surrender_task("task_a")
                assert result.surrendered is True
                await router.close_trial(rid)

    anyio.run(_drive)


def test_async_router_create_trial_accepts_tool_jobs_per_trial():
    """The router serialises `tool_jobs_per_trial` into the create_trial body.

    Over the real ASGI transport the server's pydantic schema accepts the field
    (no 422) — proving the Section 3 wiring reaches the HTTP boundary.
    """
    app = _server("task_a")

    async def _drive() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                router = AsyncCorralRouter("http://test", client=client)
                created = await router.create_trial(
                    "task_a", trial_index=0, tool_jobs_per_trial=3
                )
                assert created["trial_runtime_id"].startswith("tr_")
                await router.close_trial(created["trial_runtime_id"])

    anyio.run(_drive)


def test_async_trial_view_shares_pool_and_targets_trial_endpoints():
    async def _drive() -> None:
        async with httpx.AsyncClient(base_url="http://test") as client:
            router = AsyncCorralRouter("http://test", client=client)
            assert router._owns_client is False

            trial = router.for_trial("tr_9", "task_a", verbosity="detailed")
            # The view reuses the parent's single AsyncClient (pooled), and must
            # never own it (closing a view must not disturb the shared pool).
            assert trial.client is client
            assert trial._owns_client is False
            # And it retargets every seam at the runtime endpoints.
            assert (
                trial._task_url("ignored", "/submit")
                == "http://test/trials/tr_9/submit"
            )
            assert "/trials/tr_9/mcp/" in trial.mcp_url()

    anyio.run(_drive)


def test_async_router_closes_only_the_client_it_created():
    async def _drive() -> None:
        # A router that created its own client closes it on context exit.
        router = AsyncCorralRouter("http://test")
        assert router._owns_client is True
        owned = router.client
        async with router:
            assert not owned.is_closed
        assert owned.is_closed

        # A router handed an external client (and its views) never closes it.
        external = httpx.AsyncClient(base_url="http://test")
        borrower = AsyncCorralRouter("http://test", client=external)
        view = borrower.for_trial("tr_1", "t")
        await borrower.aclose()
        await view.aclose()
        assert not external.is_closed
        await external.aclose()

    anyio.run(_drive)


def test_async_with_verbosity_shares_but_does_not_own_client():
    async def _drive() -> None:
        async with httpx.AsyncClient(base_url="http://test") as client:
            router = AsyncCorralRouter(
                "http://test", client=client, default_verbosity="brief"
            )
            view = router.with_verbosity("full")
            assert view.current_verbosity == "full"
            assert router.current_verbosity == "brief"  # parent not mutated
            assert view.client is client  # shared pool
            assert view._owns_client is False  # never closes the shared pool

    anyio.run(_drive)


def test_sync_for_trial_binds_verbosity_without_mutating_parent():
    router = CorralRouter("http://x", default_verbosity="brief")

    bound = router.for_trial("tr_1", "task", verbosity="detailed")
    assert isinstance(bound, TrialScopedRouter)
    assert bound.current_verbosity == "detailed"
    assert "verbosity=detailed" in bound.mcp_url()

    # No explicit verbosity falls back to the parent's current one.
    fallback = router.for_trial("tr_2", "task")
    assert fallback.current_verbosity == "brief"

    # Binding a trial's verbosity never mutates the shared parent.
    assert router.current_verbosity == "brief"


def test_with_verbosity_is_an_independent_view():
    router = CorralRouter("http://x", default_verbosity="brief")

    view = router.with_verbosity("full")
    assert view.current_verbosity == "full"
    assert "verbosity=full" in view.mcp_url("task_a")
    # The original is untouched, so it stays safe to share across trials.
    assert router.current_verbosity == "brief"


def test_async_and_sync_routers_build_identical_urls():
    """The shared mixin keeps both transports on one URL convention."""

    async def _drive() -> None:
        sync = CorralRouter("http://host:9000", default_verbosity="brief")
        async with AsyncCorralRouter(
            "http://host:9000", default_verbosity="brief"
        ) as async_router:
            assert sync._task_url("t", "/tools") == async_router._task_url(
                "t", "/tools"
            )
            assert sync.mcp_url("t") == async_router.mcp_url("t")

            sync_trial = sync.for_trial("tr_1", "t", verbosity="full")
            async_trial = async_router.for_trial("tr_1", "t", verbosity="full")
            assert sync_trial._task_url("t", "/submit") == async_trial._task_url(
                "t", "/submit"
            )
            assert sync_trial.mcp_url() == async_trial.mcp_url()

    anyio.run(_drive)


def test_acall_awaits_async_and_offloads_sync():
    main = threading.current_thread().name
    seen: dict[str, str] = {}

    async def async_method(x: int) -> int:
        seen["async"] = threading.current_thread().name
        return x + 1

    def sync_method(x: int) -> int:
        seen["sync"] = threading.current_thread().name
        return x * 2

    async def _drive() -> None:
        # An async method is awaited on the loop; a sync one is offloaded.
        assert await acall(async_method, 1) == 2
        assert await acall(sync_method, 3) == 6

    anyio.run(_drive)

    assert seen["async"] == main  # awaited directly on the event-loop thread
    assert seen["sync"] != main  # offloaded to a worker thread


def test_acall_invokes_a_sync_method_exactly_once():
    """A sync method is offloaded once — never invoked to *detect* async first.

    These calls create trials and submit answers, so a double invocation would
    be a real, duplicated side effect.
    """
    calls: list[int] = []

    def method() -> str:
        calls.append(1)
        return "ok"

    async def _drive() -> str:
        return await acall(method)

    assert anyio.run(_drive) == "ok"
    assert len(calls) == 1


def test_async_router_to_sync_returns_equivalent_sync_router():
    router = AsyncCorralRouter("http://host:9000", default_verbosity="detailed")
    sync = router.to_sync()
    assert isinstance(sync, CorralRouter)
    assert not isinstance(sync, AsyncCorralRouter)
    assert sync.base_url == "http://host:9000"
    assert sync.current_verbosity == "detailed"
    # Same URL convention as the async router it was derived from.
    assert sync._task_url("t", "/submit") == router._task_url("t", "/submit")


def test_async_trial_router_to_sync_keeps_the_runtime_scope():
    router = AsyncCorralRouter("http://host:9000", default_verbosity="brief")
    sync_trial = router.for_trial("tr_7", "task_a", verbosity="full").to_sync()
    assert isinstance(sync_trial, TrialScopedRouter)
    # Still addresses the same runtime, with the same bound verbosity.
    assert (
        sync_trial._task_url("ignored", "/submit")
        == "http://host:9000/trials/tr_7/submit"
    )
    assert sync_trial.current_verbosity == "full"
    assert "/trials/tr_7/mcp/" in sync_trial.mcp_url()


def test_as_sync_interface_passes_sync_through_and_converts_async():
    sync = CorralRouter("http://x")
    # An already-synchronous interface is returned unchanged (identity).
    assert as_sync_interface(sync) is sync

    converted = as_sync_interface(AsyncCorralRouter("http://x"))
    assert isinstance(converted, CorralRouter)
    assert not isinstance(converted, AsyncCorralRouter)


class _NativeFetchAgent:
    """A minimal action agent recording the event-loop thread."""

    model = "test-model"
    max_iterations = 1

    def __init__(self) -> None:
        self.thread_name: str | None = None

    async def step(self, state):
        self.thread_name = threading.current_thread().name
        assert state.metadata.task["prompt"]
        return submit_answer_action("7")


def _run_abench_over_async_router(app, tmp_path, factory, **abench_kwargs):
    """Drive `runner.abench` against `app` via an in-process ASGI transport."""

    async def _drive():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                router = AsyncCorralRouter("http://test", client=client)
                runner = CorralRunner(
                    router, agent_factory=factory, checkpoint_dir=str(tmp_path)
                )
                return await runner.abench(**abench_kwargs)

    return anyio.run(_drive)


def test_async_router_is_the_runner_interface_end_to_end(tmp_path, monkeypatch):
    """`abench` runs a full benchmark with an `AsyncCorralRouter` interface.

    Configure, prompt/tool fetch, and submit all flow through the async router
    against a real server, and the native agent runs on the event-loop thread
    (no worker thread for HTTP).
    """
    monkeypatch.chdir(tmp_path)
    app = _server("task_a", "task_b")
    main_thread = threading.current_thread().name
    agents: list[_NativeFetchAgent] = []

    def factory(ctx) -> _NativeFetchAgent:
        agent = _NativeFetchAgent()
        agents.append(agent)
        return agent

    result = _run_abench_over_async_router(
        app,
        tmp_path,
        factory,
        task_ids=["task_a", "task_b"],
        trials_per_task=2,
        max_concurrency=2,
        run_name="report",
    )

    total = sum(len(r.trials) for r in result.task_results.values())
    assert total == 4
    assert all(t.score == 1.0 for r in result.task_results.values() for t in r.trials)
    # Every native agent ran on the loop thread: the async router awaited its
    # HTTP directly rather than offloading to a worker thread.
    assert agents
    assert all(a.thread_name == main_thread for a in agents)


def test_async_router_runner_end_to_end_with_trial_runtimes(tmp_path, monkeypatch):
    """`abench` drives the per-trial runtime lifecycle through the async router.

    With `max_concurrency_per_task > 1` each trial goes through a server-side
    trial runtime: create -> scoped async router -> submit -> close, plus the
    episode store free — all awaited on the async client end-to-end.
    """
    monkeypatch.chdir(tmp_path)
    app = _server("task_a")

    result = _run_abench_over_async_router(
        app,
        tmp_path,
        lambda ctx: _NativeFetchAgent(),
        task_ids=["task_a"],
        trials_per_task=2,
        max_concurrency=2,
        max_concurrency_per_task=2,
        run_name="report",
    )

    assert len(result.task_results["task_a"].trials) == 2
    assert all(t.score == 1.0 for t in result.task_results["task_a"].trials)
