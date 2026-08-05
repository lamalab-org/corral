from __future__ import annotations

import contextlib
import threading
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from corral.backend.executors import (
    DEFAULT_EXECUTOR,
    DEFAULT_JOB_CONCURRENCY,
    JobCancelled,
    JobExecutor,
    JobWork,
    ThreadExecutor,
    build_executor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from corral.backend.tool import Tool

# Re-exported for backwards compatibility: callers historically imported
# `JobExecutor` / `ThreadExecutor` / `DEFAULT_JOB_CONCURRENCY` from this
# module. The implementations now live in :mod:`corral.backend.executors`.
__all__ = [
    "DEFAULT_JOB_CONCURRENCY",
    "DEFAULT_POLL_AFTER_SECONDS",
    "JobContext",
    "JobExecutor",
    "JobManager",
    "JobRecord",
    "JobStatus",
    "ThreadExecutor",
]

# Hint returned in a job handle telling the agent how long to wait before its
# first status poll. Advisory only — the agent may poll whenever it likes.
DEFAULT_POLL_AFTER_SECONDS = 5


class ExclusiveLock(Protocol):
    """The minimal lock interface a `concurrency_key` factory must return.

    A background job is a *writer* on its resource, so it acquires the exclusive
    side. A plain `threading.Lock` satisfies this, and so does the runtime's
    `_ReadWriteLock` via its write-side `acquire`/`release` aliases, which
    lets one key serialise foreground writers, readers, and jobs consistently.
    """

    def acquire(self) -> bool: ...

    def release(self) -> None: ...


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class JobStatus(str, Enum):
    """Lifecycle of a background job.

    A job starts `QUEUED`, moves to `RUNNING` when a worker picks it up, and
    ends in exactly one terminal state: `SUCCEEDED` (result available),
    `FAILED` (the tool raised), `CANCELLED` (the agent cancelled it), or
    `TIMED_OUT` (a job-level deadline elapsed). Subclassing `str` keeps the
    value JSON-serialisable straight out of :meth:`JobRecord.to_dict`.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Whether the job has finished and its status will no longer change."""
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }


@dataclass(frozen=True)
class JobContext:
    """Immutable provenance a job is permanently bound to at submit time.

    `arguments` are the *visible* arguments the agent supplied and are safe to
    surface in reports. `call_arguments` additionally carry any injected hidden
    arguments (scoring secrets, fixed paths) and are used only to execute the
    tool — they are **never** serialised. `hidden_arg_names` records *which*
    arguments were injected without leaking their values.
    """

    job_id: str
    tool_name: str
    arguments: dict[str, Any]
    call_arguments: dict[str, Any]
    hidden_arg_names: tuple[str, ...]
    workspace: str | None
    benchmark_run_id: str | None
    episode_id: str | None
    trial_runtime_id: str | None
    concurrency_key: str | None = None


@dataclass
class JobRecord:
    """Mutable state of one background job: its context plus lifecycle timing."""

    context: JobContext
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    result: str | None = None
    error: str | None = None

    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    def to_dict(self, include_result: bool = True) -> dict[str, Any]:
        """Serialise a job for a tool response or a report.

        The hidden argument *values* in `call_arguments` are deliberately
        excluded; only the visible arguments and the names of injected hidden
        arguments are reported, mirroring how :meth:`CorralState.snapshot`
        redacts `hidden_args`.
        """
        ctx = self.context
        data: dict[str, Any] = {
            "job_id": ctx.job_id,
            "tool_name": ctx.tool_name,
            "status": self.status.value,
            "arguments": ctx.arguments,
            "hidden_arg_names": list(ctx.hidden_arg_names),
            "workspace": ctx.workspace,
            "concurrency_key": ctx.concurrency_key,
            "benchmark_run_id": ctx.benchmark_run_id,
            "episode_id": ctx.episode_id,
            "trial_runtime_id": ctx.trial_runtime_id,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration_seconds(),
            "error": self.error,
        }
        if include_result:
            data["result"] = self.result
        return data


class JobManager:
    """Per-runtime registry that runs background jobs and tracks their state.

    One manager belongs to one trial runtime, so its jobs share that runtime's
    provenance and never mix with another trial's. Bookkeeping is guarded by a
    single lock; the actual tool execution happens off-lock in the executor.

    Cancellation is best-effort by nature: a queued job is dropped before it
    starts, but Python cannot force-stop a thread already inside a tool call, so
    a running job is *marked* cancelled and its result discarded while the
    underlying work runs to completion in the background. Executors backed by a
    subprocess can implement true cancellation by killing the process.
    """

    def __init__(
        self,
        *,
        executor: JobExecutor | None = None,
        executors: dict[str, JobExecutor] | None = None,
        max_concurrency: int = DEFAULT_JOB_CONCURRENCY,
        provenance_provider: Callable[[], dict[str, str | None]] | None = None,
        key_lock_factory: Callable[[str], ExclusiveLock] | None = None,
    ) -> None:
        # Executors are keyed by the name a tool requests via
        # `@tool(executor=...)` (`thread`/`process`/`subprocess`/`slurm`/
        # `modal`). Built-in backends are created lazily on first use; a
        # deployment injects preconfigured ones (e.g. a SlurmExecutor with cluster
        # flags, or a ModalExecutor bound to a deployed function) through
        # `executors`. A bare `executor=` overrides the default (`thread`)
        # backend, preserving the historical single-executor constructor.
        self._max_concurrency = max(1, max_concurrency)
        self._executors: dict[str, JobExecutor] = dict(executors or {})
        if executor is not None:
            self._executors.setdefault(DEFAULT_EXECUTOR, executor)
        self._executors_guard = threading.Lock()
        self._provenance_provider = provenance_provider or dict
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._cancelled: set[str] = set()
        # Per-job cancellation signals. Set by `cancel` so a cancellation-aware
        # executor (subprocess/Slurm/Modal) can abort work already running; the
        # thread/process executors cannot interrupt a running call and ignore it.
        self._cancel_events: dict[str, threading.Event] = {}
        # One lock per concurrency_key so two jobs that touch the same resource
        # (e.g. two solvers writing the same structure file) never run at once.
        # When the owning runtime supplies a `key_lock_factory` (Section 9) those
        # locks are shared with foreground CONCURRENT tool calls, so a key
        # serialises jobs *and* direct calls; otherwise the manager owns them.
        self._key_lock_factory = key_lock_factory
        self._key_locks: dict[str, ExclusiveLock] = {}

    def _resolve_executor(self, tool: Tool) -> JobExecutor:
        """Return the executor a tool's work should run on, building it lazily.

        The name comes from `tool.executor` (default `thread`). Instances are
        cached so every job of the same kind shares one pool; an injected or
        previously-built executor is reused.
        """
        name = getattr(tool, "executor", None) or DEFAULT_EXECUTOR
        with self._executors_guard:
            executor = self._executors.get(name)
            if executor is None:
                executor = build_executor(name, self._max_concurrency)
                self._executors[name] = executor
            return executor

    def submit(
        self,
        tool: Tool,
        *,
        visible_arguments: dict[str, Any],
        call_arguments: dict[str, Any],
        hidden_arg_names: tuple[str, ...] = (),
        workspace: str | None = None,
        concurrency_key: str | None = None,
    ) -> JobRecord:
        """Register a job and hand its work to the executor immediately.

        Returns the :class:`JobRecord` right away (status `QUEUED`); the tool
        runs on a worker so the caller — and therefore the agent — is never
        blocked. All provenance is snapshotted into an immutable
        :class:`JobContext` here, at submit time.
        """
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        prov = self._provenance_provider()
        context = JobContext(
            job_id=job_id,
            tool_name=tool.name,
            arguments=dict(visible_arguments),
            call_arguments=dict(call_arguments),
            hidden_arg_names=tuple(hidden_arg_names),
            workspace=workspace,
            benchmark_run_id=prov.get("benchmark_run_id"),
            episode_id=prov.get("episode_id"),
            trial_runtime_id=prov.get("trial_runtime_id"),
            concurrency_key=concurrency_key or getattr(tool, "concurrency_key", None),
        )
        record = JobRecord(context=context)
        executor = self._resolve_executor(tool)
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = record
            self._cancel_events[job_id] = cancel_event
        future = executor.submit(
            partial(self._run, tool, record, executor, cancel_event)
        )
        with self._lock:
            self._futures[job_id] = future
        logger.debug(
            f"Submitted background job {job_id} for tool {tool.name!r} on "
            f"{getattr(tool, 'executor', None) or DEFAULT_EXECUTOR!r} executor"
        )
        return record

    def _run(
        self,
        tool: Tool,
        record: JobRecord,
        executor: JobExecutor,
        cancel_event: threading.Event,
    ) -> None:
        """Orchestrate one job on a worker thread, recording the outcome.

        This runs the bookkeeping — cancellation checks, `concurrency_key`
        serialisation, status/timing transitions — in the manager's own process;
        only the tool call itself is handed to `executor.run_tool`, which may
        run it in a child process, on Slurm, or on Modal. Never raises: the
        executor's future always resolves cleanly so a waiter blocking on it
        wakes up, and every terminal state is captured on the record.

        A job stays `QUEUED` while its worker waits on the `concurrency_key`
        lock (it is genuinely queued behind a same-key job); `RUNNING` /
        `started_at` mark the moment the tool actually begins, so a job's
        recorded execution window never overlaps another job sharing its key.
        """
        job_id = record.context.job_id

        with self._lock:
            if job_id in self._cancelled:
                return  # cancelled before it started

        # Serialise on the concurrency key *before* flipping to RUNNING, so the
        # recorded start time reflects real execution rather than queue time.
        key_lock = self._key_lock(record.context.concurrency_key)
        if key_lock is not None:
            key_lock.acquire()
        try:
            with self._lock:
                # A cancel may have landed while we waited on the key lock.
                if job_id in self._cancelled:
                    return
                record.status = JobStatus.RUNNING
                record.started_at = _utcnow()
            work = JobWork(
                job_id=job_id,
                tool_name=tool.name,
                tool=tool,
                call_arguments=record.context.call_arguments,
                workspace=record.context.workspace,
            )
            rendered = executor.run_tool(work, cancel_event)
            with self._lock:
                if job_id in self._cancelled:
                    return
                record.result = rendered
                record.status = JobStatus.SUCCEEDED
                record.ended_at = _utcnow()
        except JobCancelled:
            # The executor stopped the work in response to a cancel; `cancel`
            # already marked the record CANCELLED, so leave it untouched.
            return
        except Exception as exc:  # captured onto the record, never propagated
            with self._lock:
                if job_id not in self._cancelled:
                    record.error = str(exc)
                    record.status = JobStatus.FAILED
                    record.ended_at = _utcnow()
            logger.warning(f"Background job {job_id} ({tool.name!r}) failed: {exc}")
        finally:
            if key_lock is not None:
                key_lock.release()

    def _key_lock(self, key: str | None) -> ExclusiveLock | None:
        if key is None:
            return None
        # Prefer the runtime's shared factory so foreground and background work
        # on the same key contend on one lock (Section 9).
        if self._key_lock_factory is not None:
            return self._key_lock_factory(key)
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _require(self, job_id: str) -> tuple[JobRecord, Future | None]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record, self._futures.get(job_id)

    def status(self, job_id: str) -> JobStatus:
        """Return a job's current status. Raises `KeyError` if unknown."""
        record, _ = self._require(job_id)
        return record.status

    def result(
        self, job_id: str, *, wait: bool = False, timeout: float | None = None
    ) -> dict[str, Any]:
        """Return a job's serialised state, optionally blocking until it ends.

        With `wait=False` this is a non-blocking poll — a still-running job
        comes back with no `result`/`error` yet. With `wait=True` the call
        blocks (up to `timeout` seconds) for the job to reach a terminal state;
        on timeout it returns the current (still-running) view rather than
        raising, so the agent can decide whether to keep waiting.
        """
        record, future = self._require(job_id)
        if wait and future is not None and not record.status.is_terminal:
            # `_run` never propagates, so this only blocks for completion; on
            # timeout we fall through and return the current (running) view.
            with contextlib.suppress(FutureTimeoutError):
                future.result(timeout=timeout)
        with self._lock:
            return record.to_dict()

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a job (best-effort). Raises `KeyError` if unknown.

        A queued job is prevented from starting; a running job is marked
        cancelled and its eventual result discarded, though the underlying work
        cannot be force-stopped in a thread and runs to completion in the
        background.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status.is_terminal:
                return record.to_dict()
            self._cancelled.add(job_id)
            future = self._futures.get(job_id)
            cancel_event = self._cancel_events.get(job_id)
            record.status = JobStatus.CANCELLED
            record.ended_at = _utcnow()
            snapshot = record.to_dict()
        # Signal a cancellation-aware executor (subprocess/Slurm/Modal) to stop
        # work already running; `future.cancel` only drops a job still queued.
        if cancel_event is not None:
            cancel_event.set()
        if future is not None:
            future.cancel()  # only succeeds if it has not started yet
        return snapshot

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return a compact view of every job (without result payloads)."""
        with self._lock:
            return [
                record.to_dict(include_result=False) for record in self._jobs.values()
            ]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Full `{job_id: record}` view for embedding in a trial report."""
        with self._lock:
            return {job_id: record.to_dict() for job_id, record in self._jobs.items()}

    def shutdown(self, wait: bool = False) -> None:
        """Cancel outstanding jobs and tear down every executor it created.

        Called when a trial runtime closes so no job thread (or child process /
        cluster job) outlives the trial that owns it. Each running job's cancel
        event is set so a cancellation-aware executor stops its work.
        """
        with self._lock:
            for job_id, record in self._jobs.items():
                if not record.status.is_terminal:
                    self._cancelled.add(job_id)
                    record.status = JobStatus.CANCELLED
                    record.ended_at = _utcnow()
                    event = self._cancel_events.get(job_id)
                    if event is not None:
                        event.set()
            executors = list(self._executors.values())
        for executor in executors:
            executor.shutdown(wait=wait)
