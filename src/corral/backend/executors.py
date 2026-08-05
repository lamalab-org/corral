from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import cloudpickle
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from corral.backend.tool import Tool

# Default per-trial background-job concurrency. Bounds how many jobs actually
# execute at once inside one trial runtime; jobs beyond it queue. Kept modest
# because background tools are typically expensive (simulations, solvers). This
# is the single source of truth; :mod:`corral.backend.jobs` re-exports it.
DEFAULT_JOB_CONCURRENCY = 4

# How often the polling executors (process / subprocess / slurm / modal) wake to
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

    Only the executors that can *actually* stop running work (subprocess, Slurm,
    Modal) raise this; the manager treats it as "the record is already
    `CANCELLED`, leave it alone." Thread/Process execution cannot interrupt a
    running call, so they never raise it (cancellation stays best-effort there).
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
    overrides. `shutdown` releases the executor's resources.

    All provenance, status tracking, and `concurrency_key` serialisation live
    in the manager, so this protocol stays tiny enough for a process, subprocess,
    Slurm, or Modal backend to implement without re-deriving any bookkeeping.
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
    `run_tool` while the real work happens in a child process / on a cluster /
    on Modal. The pool is created on first :meth:`submit`, so merely *having* a
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


def _run_cloudpickled(blob: bytes) -> str:
    """Child-side entry point for :class:`ProcessExecutor`.

    Receives a cloudpickled `(tool, call_arguments)` pair (so `@tool`
    closures survive the process boundary that plain `pickle` would reject),
    runs the tool, and returns the rendered string. Defined at module scope so
    it is importable in a spawned child.
    """
    tool, call_arguments = cloudpickle.loads(blob)
    return render_result(tool.execute(**call_arguments))


def _terminate_process(proc: subprocess.Popen) -> None:
    """Terminate a child process, escalating to kill after a short grace."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()


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

    * **process isolation** — a crash can't take the benchmark server down;
    * **true cancellation** — the child (and its whole process group) is killed;
    * **stdout/stderr capture** — surfaced on failure for provenance;
    * **a clean working directory** — the job's resolved workspace is its cwd.

    `extra_env` is merged into the child's environment (e.g. `OMP_NUM_THREADS`
    or a scheduler's variables); `start_new_session` puts the child in its own
    process group so cancellation reaches any grandchildren it spawns.
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
                    _terminate_process(proc)
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


# Slurm job states we treat as "still going" vs. terminal. Anything terminal and
# not COMPLETED is a failure (FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, …).
_SLURM_ACTIVE_STATES = frozenset(
    {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}
)
_SLURM_SUCCESS_STATE = "COMPLETED"


@runtime_checkable
class SlurmRunner(Protocol):
    """Seam over the `sbatch` / `sacct` / `scancel` commands.

    Injecting this keeps :class:`SlurmExecutor` testable without a real cluster
    and lets deployments swap in a bespoke submission wrapper (accounting flags,
    partitions, container images).
    """

    def submit(self, script_path: str, *, job_name: str, workspace: str | None) -> str:
        """Submit `script_path` and return the Slurm job id."""

    def poll(self, slurm_job_id: str) -> str:
        """Return the job's current Slurm state (e.g. `RUNNING`, `COMPLETED`)."""

    def cancel(self, slurm_job_id: str) -> None:
        """Cancel the Slurm job."""


class SubprocessSlurmRunner:
    """Default :class:`SlurmRunner` shelling out to the real Slurm CLIs."""

    def __init__(
        self, sbatch: str = "sbatch", sacct: str = "sacct", scancel: str = "scancel"
    ) -> None:
        self._sbatch = sbatch
        self._sacct = sacct
        self._scancel = scancel

    def submit(self, script_path: str, *, job_name: str, workspace: str | None) -> str:
        result = subprocess.run(
            [self._sbatch, "--parsable", "--job-name", job_name, script_path],
            cwd=workspace or None,
            capture_output=True,
            text=True,
            check=True,
        )
        # `--parsable` prints "<jobid>" or "<jobid>;<cluster>".
        return result.stdout.strip().split(";")[0]

    def poll(self, slurm_job_id: str) -> str:
        result = subprocess.run(
            [
                self._sacct,
                "-j",
                slurm_job_id,
                "--format=State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # sacct lists the job step(s); the first line is the primary job state.
        for line in result.stdout.splitlines():
            state = line.strip().split()[0] if line.strip() else ""
            if state:
                return state
        return "PENDING"  # not yet visible in the accounting DB

    def cancel(self, slurm_job_id: str) -> None:
        subprocess.run([self._scancel, slurm_job_id], check=False)


class SlurmExecutor(_PooledExecutor):
    """Run a job's tool as a Slurm batch job on an HPC scheduler.

    Writes the same cloudpickled payload as :class:`SubprocessExecutor`, wraps it
    in an `sbatch` script that runs :mod:`corral.backend._job_worker` on a
    compute node, submits it, and polls `sacct` until the job reaches a
    terminal state — then reads the result back from the shared filesystem.
    Cancellation issues `scancel`.

    The scheduler interaction goes through an injectable :class:`SlurmRunner`
    (default :class:`SubprocessSlurmRunner`), so the orchestration is unit-testable
    without a live cluster and deployments can customise submission. `sbatch_options`
    are extra `#SBATCH` directives (partition, time limit, gpus); the payload
    directory must live on a filesystem the compute nodes can read.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_JOB_CONCURRENCY,
        *,
        runner: SlurmRunner | None = None,
        payload_dir: str | None = None,
        sbatch_options: list[str] | None = None,
        python_executable: str | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        super().__init__(max_workers)
        self._runner = runner or SubprocessSlurmRunner()
        self._payload_dir = payload_dir
        self._sbatch_options = list(sbatch_options or [])
        self._python = python_executable or sys.executable
        self._poll_interval = poll_interval

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        tmpdir = Path(tempfile.mkdtemp(prefix="corral-slurm-", dir=self._payload_dir))
        in_path = tmpdir / "in.pkl"
        out_path = tmpdir / "out.pkl"
        script_path = tmpdir / "job.sbatch"
        try:
            with in_path.open("wb") as fh:
                cloudpickle.dump((work.tool, work.call_arguments), fh)
            self._write_script(script_path, in_path, out_path, work)

            slurm_id = self._runner.submit(
                str(script_path),
                job_name=f"corral-{work.job_id}",
                workspace=work.workspace,
            )
            logger.debug(
                f"Submitted Slurm job {slurm_id} for {work.tool_name!r} (job {work.job_id})"
            )
            self._await_completion(slurm_id, cancel)
            return self._collect_result(work, out_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _write_script(
        self, script_path: Path, in_path: Path, out_path: Path, work: JobWork
    ) -> None:
        directives = "\n".join(f"#SBATCH {opt}" for opt in self._sbatch_options)
        worker = (
            f'"{self._python}" -m corral.backend._job_worker "{in_path}" "{out_path}"'
        )
        cd = f'cd "{work.workspace}"\n' if work.workspace else ""
        script_path.write_text(f"#!/bin/bash\n{directives}\n{cd}{worker}\n")
        script_path.chmod(0o755)

    def _await_completion(self, slurm_id: str, cancel: threading.Event) -> None:
        while True:
            if cancel.is_set():
                self._runner.cancel(slurm_id)
                raise JobCancelled(slurm_id)
            state = self._runner.poll(slurm_id).upper()
            if state in _SLURM_ACTIVE_STATES:
                time.sleep(self._poll_interval)
                continue
            if state == _SLURM_SUCCESS_STATE:
                return
            if state.startswith("CANCELLED"):
                raise JobCancelled(slurm_id)
            raise RuntimeError(f"Slurm job {slurm_id} ended in state {state!r}")

    def _collect_result(self, work: JobWork, out_path: Path) -> str:
        if not out_path.exists():
            raise RuntimeError(
                f"Slurm job for {work.tool_name!r} completed but produced no result "
                f"at {str(out_path)!r} (payload dir not shared with the compute node?)"
            )
        with out_path.open("rb") as fh:
            payload = cloudpickle.load(fh)
        if payload.get("ok"):
            return payload["result"]
        raise RuntimeError(
            payload.get("error") or f"Slurm job for {work.tool_name!r} failed"
        )


@runtime_checkable
class ModalCaller(Protocol):
    """Seam over a Modal function invocation (spawn → poll → cancel).

    Injecting this keeps :class:`ModalExecutor` runnable and testable without a
    deployed Modal app, and lets a deployment point the executor at whatever
    remote function actually runs its tools.
    """

    def spawn(self, tool: Tool, call_arguments: dict[str, Any]) -> Any:
        """Start the remote call and return an opaque handle."""

    def poll(self, handle: Any) -> tuple[bool, str | None]:
        """Return `(done, rendered_result)`; raise if the remote call failed."""

    def cancel(self, handle: Any) -> None:
        """Cancel the remote call."""


class _ModalFunctionCaller:
    """Default :class:`ModalCaller` wrapping a deployed `modal.Function`.

    The wrapped function must accept a single `bytes` argument — a cloudpickled
    `(tool, call_arguments)` pair — and return the rendered result string
    (`render_result` applied on the remote side). We drive it asynchronously:
    `spawn` starts the call, `poll` does a non-blocking `get`.
    """

    def __init__(self, modal_function: Any) -> None:
        self._fn = modal_function

    def spawn(self, tool: Tool, call_arguments: dict[str, Any]) -> Any:
        blob = cloudpickle.dumps((tool, call_arguments))
        return self._fn.spawn(blob)

    def poll(self, handle: Any) -> tuple[bool, str | None]:
        try:
            result = handle.get(timeout=0)
        except TimeoutError:
            return False, None
        return True, render_result(result)

    def cancel(self, handle: Any) -> None:
        with contextlib.suppress(Exception):
            handle.cancel()


class ModalExecutor(_PooledExecutor):
    """Run a job's tool on Modal via a deployed function.

    Corral already runs some tools on Modal; this executor lets a
    `background_capable` tool declare `executor="modal"` and be dispatched
    there. Provide either a `caller` (any :class:`ModalCaller`) or a
    `modal_function` (a deployed `modal.Function` taking a cloudpickled
    `(tool, arguments)` blob and returning the rendered string). Without one,
    :meth:`run_tool` raises with guidance rather than guessing at an app — there
    is no universal remote entry point to fall back to.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_JOB_CONCURRENCY,
        *,
        caller: ModalCaller | None = None,
        modal_function: Any = None,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__(max_workers)
        if caller is None and modal_function is not None:
            caller = _ModalFunctionCaller(modal_function)
        self._caller = caller
        self._poll_interval = poll_interval

    def run_tool(self, work: JobWork, cancel: threading.Event) -> str:
        if self._caller is None:
            raise RuntimeError(
                "ModalExecutor needs a `modal_function=` (a deployed modal.Function "
                "taking a cloudpickled (tool, arguments) blob) or a custom `caller=`; "
                f"tool {work.tool_name!r} declared executor='modal' but none was configured."
            )
        handle = self._caller.spawn(work.tool, work.call_arguments)
        while True:
            if cancel.is_set():
                self._caller.cancel(handle)
                raise JobCancelled(work.job_id)
            done, result = self._caller.poll(handle)
            if done:
                return result if result is not None else ""
            time.sleep(self._poll_interval)


# Built-in executor names a tool may request via `@tool(executor="...")`.
_BUILTIN_EXECUTORS: dict[str, Callable[[int], JobExecutor]] = {
    "thread": ThreadExecutor,
    "process": ProcessExecutor,
    "subprocess": SubprocessExecutor,
    "slurm": SlurmExecutor,
    "modal": ModalExecutor,
}

DEFAULT_EXECUTOR = "thread"


def build_executor(
    name: str, max_workers: int = DEFAULT_JOB_CONCURRENCY
) -> JobExecutor:
    """Construct a built-in executor by name (`thread`/`process`/…).

    Slurm and Modal are built with their defaults (real `sbatch`; no Modal
    function): a tool asking for them expects that backend to be present, and a
    deployment that needs custom wiring injects a configured instance through
    :class:`~corral.backend.jobs.JobManager`'s `executors` map instead.
    """
    factory = _BUILTIN_EXECUTORS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown job executor {name!r}; expected one of {sorted(_BUILTIN_EXECUTORS)}."
        )
    return factory(max_workers)
