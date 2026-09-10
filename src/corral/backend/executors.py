from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import cloudpickle

from corral.runtime import permissions

if TYPE_CHECKING:
    from collections.abc import Callable

    from corral.core.tool import Tool

# Default per-execution background-job concurrency. Bounds how many jobs actually
# execute at once inside one execution runtime; jobs beyond it queue. Kept modest
# because background tools are typically expensive (simulations, solvers). This
# is the single source of truth; :mod:`corral.backend.jobs` re-exports it.
DEFAULT_JOB_CONCURRENCY = 4

# How often the polling executors (process / subprocess) wake to
# check whether their out-of-process work has finished or the job was cancelled.
_DEFAULT_POLL_INTERVAL = 0.2

# Grace period between SIGTERM and SIGKILL when cancelling a subprocess.
_TERMINATE_GRACE_SECONDS = 5.0


def render_result(result: Any) -> str:
    """Render a tool's return value as the string stored on the job record.

    A plain string passes through unchanged; anything else is JSON-encoded with
    a `str` fallback for non-serialisable objects. Kept here (rather than in
    `JobManager`) so an out-of-process worker renders identically to an
    in-process one — the executor boundary always carries a string back.
    """
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


class JobCancelled(Exception):
    """Raised by an executor's `run_tool` when a job is cancelled mid-flight.

    The manager treats this as "the record is already `CANCELLED`, leave it
    alone." Subprocess execution can stop running work; thread and process-pool
    execution can only prevent queued work or discard a running call's result.
    """


@dataclass(frozen=True)
class JobWork:
    """Everything an executor needs to run one job's tool, and nothing more.

    Assembled by the :class:`~corral.backend.jobs.JobManager` from the job's
    immutable :class:`~corral.backend.jobs.JobContext` at execution time. The
    `call_arguments` already include any injected hidden arguments; the
    `workspace` was resolved at submit time so an out-of-process job runs in
    the right directory even after the environment has moved on.
    """

    job_id: str
    tool_name: str
    tool: Tool
    call_arguments: dict[str, Any]
    workspace: str | None = None


@runtime_checkable
class JobExecutor(Protocol):
    """Pluggable backend deciding *where* a job's work runs.

    `submit` runs the manager's `_run` orchestrator on a worker thread and
    returns its :class:`~concurrent.futures.Future` (so a waiter can block on
    completion). `run_tool` runs one job's tool to completion and returns the
    rendered result string — this is the seam an out-of-process executor
    overrides. A threading event requests cancellation of synchronous work.
    `shutdown` releases the executor's resources.

    All provenance, status tracking, and `concurrency_key` serialisation live
    in the manager, so thread, process, and subprocess backends share the
    same bookkeeping.
    """

    def submit(self, fn: Callable[[], Any]) -> Future: ...

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str: ...

    def shutdown(self, wait: bool = False) -> None: ...


class _PooledExecutor:
    """Base giving every executor a lazy worker-thread pool for `submit`.

    The pool runs the :class:`~corral.backend.jobs.JobManager`'s `_run`
    orchestrator (which handles status transitions, `concurrency_key` locking,
    and recording). For :class:`ThreadExecutor` the tool itself also runs on that
    thread; for the out-of-process executors the thread simply blocks in
    `run_tool` while the real work happens in a child process.
    The pool is created on first :meth:`submit`, so merely *having* a
    background-capable tool (e.g. on a per-task template environment that never
    runs a job) costs nothing.
    """

    _POOL_PREFIX = "corral-job"

    def __init__(self, max_workers: int = DEFAULT_JOB_CONCURRENCY) -> None:
        self._max_workers = max(1, max_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def submit(self, fn: Callable[[], Any]) -> Future:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=self._POOL_PREFIX,
                )
            return self._pool.submit(fn)

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        raise NotImplementedError

    def shutdown(self, wait: bool = False) -> None:
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            # cancel_futures drops jobs that never started; a running thread
            # cannot be force-killed (see JobManager.cancel).
            pool.shutdown(wait=wait, cancel_futures=True)
        self._shutdown_backend(wait)

    def _shutdown_backend(self, wait: bool) -> None:
        """Tear down any extra backend resource (process pool, …). Default no-op."""


class ThreadExecutor(_PooledExecutor):
    """In-process :class:`JobExecutor` backed by a thread pool (the default).

    Suitable for I/O-bound tools and for tools that themselves shell out to an
    external process (which is where the real work, and any true parallelism,
    then happens). CPU-bound pure-Python tools are bounded by the GIL and want a
    :class:`ProcessExecutor` instead; scientific command-line programs want a
    :class:`SubprocessExecutor` for real isolation and cancellation.

    Cancellation is best-effort: Python cannot interrupt a running call in a
    thread, so a cancelled job runs to completion in the background and its
    result is discarded by the manager.
    """

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        # A running call in a thread cannot be interrupted, so cancellation stays
        # best-effort (the manager discards a cancelled job's result); we can only
        # honour a cancel that arrived before the call started.
        if cancel.is_set():
            raise JobCancelled(work.job_id)
        return render_result(work.tool.execute(**work.call_arguments))


class RestrictedExecutor(_PooledExecutor):
    """Apply the same private-input boundary to foreground and background tools."""

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        if cancel.is_set():
            raise JobCancelled(work.job_id)
        result = permissions.execute_job(
            work.tool, work.call_arguments, work.workspace, cancel=cancel
        )
        return render_result(result)


def _run_cloudpickled(blob: bytes) -> str:
    """Child-side entry point for :class:`ProcessExecutor`.

    Receives a cloudpickled `(tool, call_arguments)` pair (so `@tool`
    closures survive the process boundary that plain `pickle` would reject),
    runs the tool, and returns the rendered string. Defined at module scope so
    it is importable in a spawned child.
    """
    tool, call_arguments = cloudpickle.loads(blob)
    return render_result(tool.execute(**call_arguments))


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate a POSIX group created by `start_new_session=True`."""
    # The child's PID is also its original PGID, even after the leader exits.
    # Waiting only for the leader would leave SIGTERM-resistant descendants
    # running, including descendants that closed their inherited output pipes.
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while True:
            proc.poll()  # Reap the leader so its zombie cannot keep the group alive.
            # EPERM still means the group exists; macOS can also return it
            # while the group's last member is exiting. Keep waiting, but let
            # permission failures from actual TERM/KILL delivery propagate.
            with suppress(PermissionError):
                os.killpg(proc.pid, 0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(min(_DEFAULT_POLL_INTERVAL, remaining))
    proc.wait()


def _terminate_process(
    proc: subprocess.Popen, *, start_new_session: bool = False
) -> None:
    """Terminate and reap a child, including its own POSIX group when enabled."""
    if os.name == "posix" and start_new_session:
        _terminate_process_group(proc)
        return
    # Windows has no killpg; without a new POSIX session the child shares a
    # process group with its caller, which must never be signalled as a group.
    if proc.poll() is not None:
        return
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


class ProcessExecutor(_PooledExecutor):
    """Run a job's Python tool in a separate process, dodging the GIL.

    For CPU-bound pure-Python tools that would otherwise serialise behind the
    interpreter lock. Work is dispatched to a :class:`ProcessPoolExecutor`; the
    `(tool, arguments)` pair is cloudpickled so `@tool`-decorated closures
    survive the boundary (plain `pickle` cannot serialise the locally-defined
    tool class).

    Reused pool workers cannot be force-killed mid-task, so cancellation is
    best-effort here (a queued task is dropped; a running one runs to completion
    and its result is discarded). For true cancellation of a heavy external
    program, prefer :class:`SubprocessExecutor`.
    """

    def __init__(self, max_workers: int = DEFAULT_JOB_CONCURRENCY) -> None:
        super().__init__(max_workers)
        self._proc_pool: ProcessPoolExecutor | None = None
        self._proc_lock = threading.Lock()

    def _process_pool(self) -> ProcessPoolExecutor:
        with self._proc_lock:
            if self._proc_pool is None:
                self._proc_pool = ProcessPoolExecutor(max_workers=self._max_workers)
            return self._proc_pool

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        blob = cloudpickle.dumps((work.tool, work.call_arguments))
        future = self._process_pool().submit(_run_cloudpickled, blob)
        while True:
            try:
                return future.result(timeout=_DEFAULT_POLL_INTERVAL)
            except FutureTimeoutError:
                if cancel.is_set():
                    # Best-effort: only cancels if it has not started running.
                    future.cancel()
                    raise JobCancelled(work.job_id) from None

    def _shutdown_backend(self, wait: bool) -> None:
        with self._proc_lock:
            pool, self._proc_pool = self._proc_pool, None
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=True)


class SubprocessExecutor(_PooledExecutor):
    """Run a job's tool in a fresh child interpreter — the scientific default.

    Spawns one throwaway `python -m corral.backend._job_worker` process per
    job, handing it a cloudpickled `(tool, arguments)` payload and reading a
    cloudpickled result back through temp files. This is the executor
    `make_efficiency.md` §8 recommends for LAMMPS / Gaussian / xTB and other
    command-line programs, because a dedicated child process gives:

    * **process isolation** — a crash can't take the benchmark worker down;
    * **true cancellation** — the child is killed, along with its process group
      on POSIX when `start_new_session=True` (the default);
    * **stdout/stderr capture** — surfaced on failure for provenance;
    * **a clean working directory** — the job's resolved workspace is its cwd.

    `extra_env` is merged into the child's environment (e.g. `OMP_NUM_THREADS`
    or a scheduler's variables). On POSIX, `start_new_session=True` puts the
    child in its own process group so cancellation reaches grandchildren that
    remain in that group. SIGTERM is followed by SIGKILL after a grace period
    if the group still exists, even if the child has already exited. On Windows,
    or with `start_new_session=False`, cancellation stops only the direct child.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_JOB_CONCURRENCY,
        *,
        python_executable: str | None = None,
        extra_env: dict[str, str] | None = None,
        start_new_session: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        super().__init__(max_workers)
        self._python = python_executable or sys.executable
        self._extra_env = dict(extra_env or {})
        self._start_new_session = start_new_session
        self._poll_interval = poll_interval

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        tmpdir = Path(tempfile.mkdtemp(prefix="corral-job-"))
        in_path = tmpdir / "in.pkl"
        out_path = tmpdir / "out.pkl"
        try:
            with in_path.open("wb") as fh:
                cloudpickle.dump((work.tool, work.call_arguments), fh)

            cwd = (
                work.workspace
                if work.workspace and Path(work.workspace).is_dir()
                else None
            )
            env = {**os.environ, **self._extra_env} if self._extra_env else None
            proc = subprocess.Popen(
                [
                    self._python,
                    "-m",
                    "corral.backend._job_worker",
                    str(in_path),
                    str(out_path),
                ],
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=self._start_new_session,
            )
            stderr = self._wait_for_process(proc, cancel)
            return self._collect_result(work, proc, out_path, stderr)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _wait_for_process(
        self, proc: subprocess.Popen, cancel: threading.Event
    ) -> bytes:
        """Block until the child exits, killing it if the job is cancelled."""
        while True:
            try:
                _, stderr = proc.communicate(timeout=self._poll_interval)
                return stderr or b""
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    _terminate_process(proc, start_new_session=self._start_new_session)
                    raise JobCancelled(proc.pid) from None

    def _collect_result(
        self,
        work: JobWork,
        proc: subprocess.Popen,
        out_path: Path,
        stderr: bytes,
    ) -> str:
        if not out_path.exists():
            detail = stderr.decode(errors="replace").strip() or "no output produced"
            raise RuntimeError(
                f"subprocess for {work.tool_name!r} exited with code "
                f"{proc.returncode} before producing a result: {detail}"
            )
        with out_path.open("rb") as fh:
            payload = cloudpickle.load(fh)
        if payload.get("ok"):
            return payload["result"]
        raise RuntimeError(
            payload.get("error")
            or f"subprocess for {work.tool_name!r} failed without a message"
        )


# Built-in executor names a tool may request via `@tool(executor="...")`.
_BUILTIN_EXECUTORS: dict[str, Callable[[int], JobExecutor]] = {
    "thread": ThreadExecutor,
    "process": ProcessExecutor,
    "subprocess": SubprocessExecutor,
}

DEFAULT_EXECUTOR = "thread"


def build_executor(
    name: str, max_workers: int = DEFAULT_JOB_CONCURRENCY
) -> JobExecutor:
    """Construct a local executor by name (`thread`/`process`/`subprocess`)."""
    factory = _BUILTIN_EXECUTORS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown job executor {name!r}; expected one of {sorted(_BUILTIN_EXECUTORS)}."
        )
    return factory(max_workers)
