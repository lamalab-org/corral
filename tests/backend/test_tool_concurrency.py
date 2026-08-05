"""Tests for concurrent-tool-call protection (Section 9 of `make_efficiency.md`).

Claude Code and Codex both issue tool calls in parallel, and the MCP transport
now offloads each synchronous `Environment.call_tool` to a worker thread rather
than running it on the event loop. These tests pin the resulting safety
guarantees:

* a tool's :class:`ToolConcurrency` mode governs whether its executions may
  overlap within one runtime — `SERIAL` runs alone, `READ_ONLY` overlaps,
  and `CONCURRENT` overlaps only across distinct `concurrency_key` s;
* the read/write split within one key: same-key `CONCURRENT_READ` readers
  overlap each other but stay exclusive with same-key `CONCURRENT` writers and
  background jobs;
* the single `CorralState` mutation (recording a call) is never lost under
  concurrent calls;
* two *different* runtimes never serialise against each other (isolated locks);
* a `concurrency_key` serialises a foreground call against a background job on
  the same key (the JobManager shares the runtime's resource locks);
* the MCP endpoint really offloads, so a slow tool on one runtime does not block
  a concurrent call on another;
* the server raises its anyio worker-thread pool at startup so offloaded tool
  calls do not serialise on anyio's default pool under high concurrency.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anyio
import httpx
from pydantic import Field

from corral.backend.env import Toolset, build_environments
from corral.backend.jobs import JobStatus
from corral.backend.mcp_server import DEFAULT_MCP_THREAD_POOL_SIZE
from corral.backend.server import create_benchmark_server
from corral.backend.task import TaskDefinition
from corral.backend.tool import Tool, ToolConcurrency, tool

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


class _Recorder:
    """Tracks the peak number of tool executions running at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


class _ProbeTool(Tool):
    """A tool that marks an execution window on a shared :class:`_Recorder`."""

    def __init__(
        self,
        name: str,
        recorder: _Recorder,
        *,
        hold: float = 0.15,
        concurrency: ToolConcurrency = ToolConcurrency.SERIAL,
        concurrency_key: str | None = None,
        background_capable: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            description="probe",
            params_json_schema=_EMPTY_SCHEMA,
            concurrency=concurrency,
            concurrency_key=concurrency_key,
            background_capable=background_capable,
        )
        self._recorder = recorder
        self._hold = hold

    def execute(self, **kwargs) -> str:
        self._recorder.enter()
        try:
            time.sleep(self._hold)
            return self.name
        finally:
            self._recorder.exit()


def _trial_runtime(tools: dict[str, Tool], runtime_id: str = "tr"):
    """Build one isolated trial runtime whose pool is exactly `tools`."""
    task = TaskDefinition(
        name="t",
        description="d",
        tools=list(tools),
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments(
        {"t": task}, toolset=Toolset(pool=tools, workspace_factory=None)
    )
    return envs["t"].for_trial(runtime_id)


def _call_in_threads(env, calls: list[tuple[str, dict]]) -> None:
    """Fire several `call_tool` s concurrently, one per thread, and join."""
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(env.call_tool, name, args) for name, args in calls]
        for future in futures:
            future.result(timeout=5)


def test_tool_defaults_to_serial():
    @tool
    def plain(x: int = Field(description="x")) -> str:
        """A plain tool."""
        return str(x)

    assert plain.concurrency == ToolConcurrency.SERIAL


def test_tool_accepts_concurrency_string_and_enum():
    @tool(concurrency="read_only")
    def reader(x: int = Field(description="x")) -> str:
        """A read-only tool."""
        return str(x)

    @tool(concurrency=ToolConcurrency.CONCURRENT, concurrency_key="gpu")
    def writer(x: int = Field(description="x")) -> str:
        """A concurrent tool."""
        return str(x)

    assert reader.concurrency == ToolConcurrency.READ_ONLY
    assert writer.concurrency == ToolConcurrency.CONCURRENT
    assert writer.concurrency_key == "gpu"


def test_tool_accepts_concurrent_read_split():
    @tool(concurrency="concurrent_read", concurrency_key="struct")
    def inspect(x: int = Field(description="x")) -> str:
        """A shared-read tool."""
        return str(x)

    assert inspect.concurrency == ToolConcurrency.CONCURRENT_READ
    assert inspect.concurrency_key == "struct"


def test_serial_tools_never_overlap():
    rec = _Recorder()
    env = _trial_runtime({"probe": _ProbeTool("probe", rec)})
    _call_in_threads(env, [("probe", {}), ("probe", {}), ("probe", {})])
    assert rec.max_active == 1
    assert len(env.state.tool_calls) == 3  # every call recorded, none lost


def test_read_only_tools_overlap():
    rec = _Recorder()
    env = _trial_runtime(
        {"probe": _ProbeTool("probe", rec, concurrency=ToolConcurrency.READ_ONLY)}
    )
    _call_in_threads(env, [("probe", {}), ("probe", {}), ("probe", {})])
    assert rec.max_active == 3
    assert len(env.state.tool_calls) == 3


def test_same_concurrency_key_serialises():
    rec = _Recorder()
    env = _trial_runtime(
        {
            "a": _ProbeTool(
                "a", rec, concurrency=ToolConcurrency.CONCURRENT, concurrency_key="k"
            ),
            "b": _ProbeTool(
                "b", rec, concurrency=ToolConcurrency.CONCURRENT, concurrency_key="k"
            ),
        }
    )
    _call_in_threads(env, [("a", {}), ("b", {})])
    assert rec.max_active == 1


def test_different_concurrency_keys_overlap():
    rec = _Recorder()
    env = _trial_runtime(
        {
            "a": _ProbeTool(
                "a", rec, concurrency=ToolConcurrency.CONCURRENT, concurrency_key="a"
            ),
            "b": _ProbeTool(
                "b", rec, concurrency=ToolConcurrency.CONCURRENT, concurrency_key="b"
            ),
        }
    )
    _call_in_threads(env, [("a", {}), ("b", {})])
    assert rec.max_active == 2


def test_same_key_readers_overlap():
    """Two `CONCURRENT_READ` calls on one key share the read side and overlap."""
    rec = _Recorder()
    env = _trial_runtime(
        {
            "r1": _ProbeTool(
                "r1",
                rec,
                concurrency=ToolConcurrency.CONCURRENT_READ,
                concurrency_key="struct",
            ),
            "r2": _ProbeTool(
                "r2",
                rec,
                concurrency=ToolConcurrency.CONCURRENT_READ,
                concurrency_key="struct",
            ),
        }
    )
    _call_in_threads(env, [("r1", {}), ("r2", {})])
    assert rec.max_active == 2


def test_reader_and_writer_same_key_serialise():
    """A `CONCURRENT_READ` reader excludes a `CONCURRENT` writer on one key."""
    rec = _Recorder()
    env = _trial_runtime(
        {
            "reader": _ProbeTool(
                "reader",
                rec,
                concurrency=ToolConcurrency.CONCURRENT_READ,
                concurrency_key="struct",
            ),
            "writer": _ProbeTool(
                "writer",
                rec,
                concurrency=ToolConcurrency.CONCURRENT,
                concurrency_key="struct",
            ),
        }
    )
    _call_in_threads(env, [("reader", {}), ("writer", {})])
    assert rec.max_active == 1


def test_readers_on_different_keys_overlap():
    """Readers on distinct keys never contend, regardless of the read side."""
    rec = _Recorder()
    env = _trial_runtime(
        {
            "a": _ProbeTool(
                "a",
                rec,
                concurrency=ToolConcurrency.CONCURRENT_READ,
                concurrency_key="a",
            ),
            "b": _ProbeTool(
                "b",
                rec,
                concurrency=ToolConcurrency.CONCURRENT_READ,
                concurrency_key="b",
            ),
        }
    )
    _call_in_threads(env, [("a", {}), ("b", {})])
    assert rec.max_active == 2


def test_reader_and_background_job_same_key_serialise():
    """A background job is a writer, so it excludes a same-key foreground reader."""
    rec = _Recorder()
    probe = _ProbeTool(
        "probe",
        rec,
        hold=0.2,
        concurrency=ToolConcurrency.CONCURRENT,
        concurrency_key="gpu",
        background_capable=True,
    )
    reader = _ProbeTool(
        "reader",
        rec,
        hold=0.2,
        concurrency=ToolConcurrency.CONCURRENT_READ,
        concurrency_key="gpu",
    )
    env = _trial_runtime({"probe": probe, "reader": reader})
    assert env.job_manager is not None

    handle = json.loads(env.call_tool("start_probe", {}).result)
    job_id = handle["job_id"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        foreground = pool.submit(env.call_tool, "reader", {})
        done = json.loads(
            env.call_tool(
                "wait_for_job", {"job_id": job_id, "timeout_seconds": 3}
            ).result
        )
        foreground.result(timeout=5)

    assert done["status"] == JobStatus.SUCCEEDED.value
    assert rec.max_active == 1
    env.shutdown_jobs()


def test_distinct_runtimes_do_not_serialise():
    rec = _Recorder()
    task = TaskDefinition(
        name="t",
        description="d",
        tools=["probe"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    template = build_environments(
        {"t": task},
        toolset=Toolset(
            pool={"probe": _ProbeTool("probe", rec)}, workspace_factory=None
        ),
    )["t"]
    # Two independent runtimes of the same task: each owns its own state lock,
    # so their SERIAL tools still overlap across runtimes.
    env_a = template.for_trial("tr_a")
    env_b = template.for_trial("tr_b")

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(env_a.call_tool, "probe", {})
        fb = pool.submit(env_b.call_tool, "probe", {})
        fa.result(timeout=5)
        fb.result(timeout=5)

    assert rec.max_active == 2


def test_foreground_and_background_share_key_lock():
    rec = _Recorder()
    probe = _ProbeTool(
        "probe",
        rec,
        hold=0.2,
        concurrency=ToolConcurrency.CONCURRENT,
        concurrency_key="gpu",
        background_capable=True,
    )
    env = _trial_runtime({"probe": probe})
    assert env.job_manager is not None

    # Launch the background job, then immediately race a foreground call of the
    # same tool. They share concurrency_key "gpu", so they must not overlap.
    handle = json.loads(env.call_tool("start_probe", {}).result)
    job_id = handle["job_id"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        foreground = pool.submit(env.call_tool, "probe", {})
        done = json.loads(
            env.call_tool(
                "wait_for_job", {"job_id": job_id, "timeout_seconds": 3}
            ).result
        )
        foreground.result(timeout=5)

    assert done["status"] == JobStatus.SUCCEEDED.value
    assert rec.max_active == 1
    env.shutdown_jobs()


def test_mcp_calls_offloaded_and_overlap_across_runtimes():
    rec = _Recorder()
    task = TaskDefinition(
        name="t",
        description="d",
        tools=["slow"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments(
        {"t": task},
        toolset=Toolset(
            pool={"slow": _ProbeTool("slow", rec, hold=0.3)}, workspace_factory=None
        ),
    )
    app = create_benchmark_server(envs)
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "slow", "arguments": {}},
    }

    async def _drive() -> None:
        # Enter the app's lifespan manually so the MCP session managers run;
        # ASGITransport does not emit lifespan events on its own.
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                rid_a = (
                    await client.post("/tasks/t/trials", json={"trial_index": 0})
                ).json()["trial_runtime_id"]
                rid_b = (
                    await client.post("/tasks/t/trials", json={"trial_index": 1})
                ).json()["trial_runtime_id"]

                async def _call(rid: str) -> None:
                    resp = await client.post(
                        f"/trials/{rid}/mcp", json=body, headers=_MCP_HEADERS
                    )
                    assert resp.status_code == 200

                async with anyio.create_task_group() as tg:
                    tg.start_soon(_call, rid_a)
                    tg.start_soon(_call, rid_b)

    anyio.run(_drive)
    # If the handler blocked the loop instead of offloading, the second request
    # could not start until the first finished, capping max_active at 1.
    assert rec.max_active == 2


def _empty_server(mcp_thread_pool_size=None):
    task = TaskDefinition(
        name="t",
        description="d",
        tools=[],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments({"t": task}, toolset=Toolset(workspace_factory=None))
    kwargs = {}
    if mcp_thread_pool_size is not None:
        kwargs["mcp_thread_pool_size"] = mcp_thread_pool_size
    return create_benchmark_server(envs, **kwargs)


def test_server_raises_thread_pool_at_startup():
    """Entering the app lifespan lifts anyio's default thread pool to the ceiling."""
    app = _empty_server()

    async def _drive() -> int:
        async with app.router.lifespan_context(app):
            return anyio.to_thread.current_default_thread_limiter().total_tokens

    tokens = anyio.run(_drive)
    assert tokens >= DEFAULT_MCP_THREAD_POOL_SIZE


def test_server_thread_pool_size_is_configurable():
    app = _empty_server(mcp_thread_pool_size=200)

    async def _drive() -> int:
        async with app.router.lifespan_context(app):
            return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(_drive) >= 200


def test_server_never_lowers_existing_thread_pool():
    """A smaller configured ceiling never shrinks an already-larger pool."""
    app = _empty_server(mcp_thread_pool_size=1)

    async def _drive() -> int:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 300  # something else sized it up first
        async with app.router.lifespan_context(app):
            return limiter.total_tokens

    assert anyio.run(_drive) == 300
