"""Process-worker backend for trial runtimes (Environment Concurrency Isolation, Phase 2).

Phase 1 hid every `/trials/{id}/*` endpoint behind the
:class:`~corral.backend.trial_runtime.TrialRuntime` protocol, with
:class:`~corral.backend.trial_runtime.InProcessTrialRuntime` the default backend
that runs a trial in the server process. That is correct for envs whose tools
are pure/reentrant, but it corrupts any env that keeps **process-global mutable
state** (wetlab's reaktoro module globals) when concurrent trials interleave.

This module adds the second backend: a runtime that executes a trial in its own
**OS process** — the only domain that isolates arbitrary Python + native global
state. It reuses the Phase-1 pieces wholesale: each worker builds real
`Environment` objects and drives them through an `InProcessTrialRuntime`, so
the behaviour inside the worker is byte-for-byte the in-process behaviour; only
the transport (a pickling pipe instead of a direct call) differs.

Three pieces:

* :func:`_worker_loop` — the child-process RPC loop. It holds one or more live
  `InProcessTrialRuntime` objects (keyed by `trial_runtime_id`) plus the
  per-*episode* dependency stores, and answers request/response messages.
* :class:`TrialWorkerPool` — a fixed set of worker processes. Leasing blocks
  when all are busy, so the pool size *is* the process-trial concurrency limit
  (natural back-pressure).
* :class:`WorkerTrialRuntime` — the proxy that satisfies `TrialRuntime` by
  forwarding each method to its leased worker over the pipe. The HTTP layer
  never learns which backend a `trial_runtime_id` uses.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from corral.backend.env import Environment
from corral.backend.trial_runtime import InProcessTrialRuntime

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from corral.backend.schema import ToolCall, TrialCompletionResponse
    from corral.router.verbosity import ToolVerbosity

# A builder that (re)constructs the full environments map inside a worker
# process. It must be picklable so `spawn`/`forkserver` can ship it to the child
# — a module-level function or a `functools.partial` of one (corral's task envs
# are built by exactly such functions, e.g. `create_qualysis_environments`).
BuildEnvs = Callable[[], Mapping[str, Environment]]

# Default number of worker processes. Only spawned when a build_envs is supplied
# *and* at least one env is `concurrency="process"`, so a pure run pays nothing.
# Leasing blocks past this many simultaneous process-trials (back-pressure), so
# an undersized pool serialises rather than corrupts. Overridable per server.
DEFAULT_TRIAL_WORKER_POOL_SIZE = 4


class WorkerError(RuntimeError):
    """An exception raised inside a worker, re-raised in the server process.

    Carries the worker-side message and formatted traceback so a failure that
    happened across the process boundary is still debuggable server-side (the
    REST/MCP layers already render tool errors to strings, so this surfaces the
    same way an in-process failure would).
    """

    def __init__(self, detail: str, worker_traceback: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.worker_traceback = worker_traceback


def _worker_loop(conn: Connection, build_envs: BuildEnvs) -> None:
    """RPC loop for one worker process.

    Builds the environments once (importing the task's heavy deps — reaktoro,
    its database — a single time per worker), then serves request/response
    messages until it receives the `None` shutdown sentinel. Every request is
    `(command, payload)`; every reply is `("ok", data)` or
    `("err", {"detail", "traceback"})`. Replies are ordinary picklable objects
    (`ToolCall`, `TrialCompletionResponse`, plain dicts), which the pipe
    serialises for us.

    The worker holds runtimes keyed by `trial_runtime_id` (not a single one),
    so several runtimes of the same *episode* — pinned here to share their
    dependency-output store — can coexist while a chain runs. `episode_stores`
    holds one shared store per `episode_id`; a runtime built for that episode
    reads its upstream siblings' outputs through it.
    """
    try:
        environments = build_envs()
    except Exception:
        # A worker that cannot build its envs is useless; log and exit so the
        # parent's `recv` raises (EOFError) and surfaces the failure on lease.
        logger.error(
            f"Trial worker failed to build environments:\n{traceback.format_exc()}"
        )
        return

    runtimes: dict[str, InProcessTrialRuntime] = {}
    episode_stores: dict[str, dict] = {}

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:  # shutdown sentinel
            break

        cmd, payload = msg
        try:
            data = _dispatch(cmd, payload, environments, runtimes, episode_stores)
            conn.send(("ok", data))
        except Exception as exc:  # every worker-side failure crosses as an err
            conn.send(
                ("err", {"detail": str(exc), "traceback": traceback.format_exc()})
            )

    # Drain any still-open runtimes so background-job threads don't outlive us.
    for runtime in runtimes.values():
        with contextlib.suppress(Exception):  # best-effort cleanup on shutdown
            runtime.close()
    conn.close()


def _dispatch(
    cmd: str,
    payload: dict,
    environments: Mapping[str, Environment],
    runtimes: dict[str, InProcessTrialRuntime],
    episode_stores: dict[str, dict],
):
    """Execute one worker command against the worker's live runtimes."""
    if cmd == "open":
        rid = payload["trial_runtime_id"]
        episode_id = payload["episode_id"]
        # Pin the episode's shared dependency store in this worker so a chain's
        # downstream task reads its upstream siblings' outputs (the store cannot
        # cross the process boundary, so all of an episode's runtimes live here).
        store = episode_stores.setdefault(episode_id, {}) if episode_id else None
        env = environments[payload["task_id"]].for_trial(
            rid,
            episode_task_runs=store,
            max_job_concurrency=payload["tool_jobs_per_trial"],
        )
        env.state.run_id = payload["benchmark_run_id"]
        env.state.episode_id = episode_id
        runtimes[rid] = InProcessTrialRuntime(env)
        return {"workspace": runtimes[rid].workspace}

    if cmd == "drop_episode":
        episode_stores.pop(payload["episode_id"], None)
        return {}

    rid = payload["trial_runtime_id"]
    runtime = runtimes.get(rid)
    if runtime is None:
        raise KeyError(f"trial runtime {rid!r} is not open in this worker")

    if cmd == "prompt":
        return {"prompt": runtime.get_task_prompt()}
    if cmd == "guide":
        return {"guide": runtime.guide(payload["verbosity"])}
    if cmd == "tools":
        return runtime.tools_payload(payload["verbosity"])
    if cmd == "mcp_tools":
        return runtime.mcp_tools_payload(payload["verbosity"])
    if cmd == "execute":
        return {"result": runtime.call_tool(payload["tool"], payload["arguments"])}
    if cmd == "snapshot":
        return runtime.snapshot()
    if cmd == "status":
        return runtime.status()
    if cmd == "configure":
        return runtime.configure()
    if cmd == "submit":
        return {"response": runtime.submit(payload["answer"])}
    if cmd == "surrender":
        return {"response": runtime.surrender()}
    if cmd == "close":
        runtime.close()
        runtimes.pop(rid, None)
        return {}

    raise ValueError(f"unknown worker command {cmd!r}")


def _default_start_method() -> str:
    """Pick a safe multiprocessing start method for spawning trial workers.

    A bare `fork` from the multithreaded FastAPI server can deadlock (a child
    inherits locks held by threads that don't exist in it), so never use it.
    `spawn` is portable and safe on every platform (and required on macOS);
    `forkserver` is the Phase-3 performance option (it amortises the task's
    imports via a preloaded template), but plain `spawn` is the low-risk
    default here. Callers can override per server.
    """
    available = multiprocessing.get_all_start_methods()
    for method in ("spawn", "forkserver"):
        if method in available:
            return method
    return available[0]


@dataclass
class WorkerHandle:
    """The server-side end of one worker process.

    `lock` serialises access to `conn`: a single worker answers one request
    at a time, and — because an episode's runtimes share one handle — sibling
    calls must not interleave send/recv on the pipe. Holding the lock across the
    blocking `recv` is intentional: the worker cannot do two things at once
    regardless, so this only mirrors that.
    """

    conn: Connection
    lock: threading.Lock
    process: BaseProcess


class TrialWorkerPool:
    """A fixed pool of worker processes leased one-per-active-process-trial.

    `lease` blocks when every worker is busy, so the pool size caps how many
    `"process"` trials execute at once — the pool *is* the limiter, giving
    natural back-pressure without a separate semaphore. Workers are reused
    across trials (imports stay amortised); an episode holds one worker for its
    whole duration (see the server's pinning), so size the pool for the peak
    number of simultaneous process-trials/episodes.
    """

    def __init__(
        self,
        build_envs: BuildEnvs,
        size: int,
        *,
        start_method: str | None = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1, got {size}")
        ctx = multiprocessing.get_context(start_method or _default_start_method())
        self._idle: queue.Queue[WorkerHandle] = queue.Queue()
        self._handles: list[WorkerHandle] = []
        self._closed = False
        for _ in range(size):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop, args=(child_conn, build_envs), daemon=True
            )
            process.start()
            # The parent keeps only its end of the pipe; the child owns the other.
            child_conn.close()
            handle = WorkerHandle(
                conn=parent_conn, lock=threading.Lock(), process=process
            )
            self._handles.append(handle)
            self._idle.put(handle)
        logger.info(
            f"Started trial worker pool: {size} process(es) "
            f"(start method: {ctx.get_start_method()})"
        )

    @property
    def size(self) -> int:
        """Number of worker processes in the pool (the process-trial limit)."""
        return len(self._handles)

    def lease(self) -> WorkerHandle:
        """Take an idle worker, blocking until one is free (back-pressure)."""
        return self._idle.get()

    def release(self, handle: WorkerHandle) -> None:
        """Return a worker to the idle set for the next trial to reuse."""
        self._idle.put(handle)

    def shutdown(self) -> None:
        """Stop every worker: send the shutdown sentinel, then join/terminate."""
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            try:
                with handle.lock:
                    handle.conn.send(None)
            except Exception:  # a dead worker's pipe may already be closed
                pass
        for handle in self._handles:
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.terminate()
            with contextlib.suppress(Exception):
                handle.conn.close()


class WorkerTrialRuntime:
    """A :class:`TrialRuntime` that runs its trial in a worker process.

    Every method is a synchronous request/response over the leased worker's
    pipe; the worker executes the real `InProcessTrialRuntime` and ships back
    a picklable result. The two properties (`task_id`, `workspace`) are cached
    locally so the hot REST/MCP metadata reads need no round-trip.

    `release_on_close` decides what :meth:`close` does with the worker: a
    standalone process-trial returns it to the pool, while an *episode*-pinned
    runtime leaves it parked (the server releases it when the whole episode
    closes, keeping the episode's dependency store alive across its tasks).
    """

    def __init__(
        self,
        pool: TrialWorkerPool,
        handle: WorkerHandle,
        task_id: str,
        *,
        release_on_close: bool,
    ) -> None:
        self._pool = pool
        self._handle = handle
        self._task_id = task_id
        self._release_on_close = release_on_close
        self._trial_runtime_id: str | None = None
        self._workspace: str | None = None

    def open(
        self,
        *,
        trial_runtime_id: str,
        episode_id: str | None,
        benchmark_run_id: str | None,
        tool_jobs_per_trial: int | None,
    ) -> str | None:
        """Build the trial's environment in the worker; return its workspace."""
        self._trial_runtime_id = trial_runtime_id
        data = self._rpc(
            "open",
            trial_runtime_id=trial_runtime_id,
            task_id=self._task_id,
            episode_id=episode_id,
            benchmark_run_id=benchmark_run_id,
            tool_jobs_per_trial=tool_jobs_per_trial,
        )
        self._workspace = data["workspace"]
        return self._workspace

    def _rpc(self, cmd: str, **payload):
        """Send one command to the worker and return its reply (or raise)."""
        payload.setdefault("trial_runtime_id", self._trial_runtime_id)
        with self._handle.lock:
            self._handle.conn.send((cmd, payload))
            tag, data = self._handle.conn.recv()
        if tag == "err":
            raise WorkerError(data["detail"], data.get("traceback"))
        return data

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def workspace(self) -> str | None:
        return self._workspace

    def get_task_prompt(self) -> str | list[dict]:
        return self._rpc("prompt")["prompt"]

    def guide(self, verbosity: ToolVerbosity) -> str:
        return self._rpc("guide", verbosity=verbosity)["guide"]

    def tools_payload(self, verbosity: ToolVerbosity) -> dict:
        return self._rpc("tools", verbosity=verbosity)

    def mcp_tools_payload(self, verbosity: ToolVerbosity) -> dict:
        return self._rpc("mcp_tools", verbosity=verbosity)

    def call_tool(self, name: str, arguments: dict) -> ToolCall:
        return self._rpc("execute", tool=name, arguments=arguments)["result"]

    def snapshot(self) -> dict:
        return self._rpc("snapshot")

    def status(self) -> dict:
        return self._rpc("status")

    def configure(self) -> dict:
        return self._rpc("configure")

    def submit(self, answer: str) -> TrialCompletionResponse:
        return self._rpc("submit", answer=answer)["response"]

    def surrender(self) -> TrialCompletionResponse:
        return self._rpc("surrender")["response"]

    def close(self) -> None:
        """Shut the trial down in the worker, then free the worker if standalone."""
        try:
            self._rpc("close")
        finally:
            if self._release_on_close:
                self._pool.release(self._handle)
