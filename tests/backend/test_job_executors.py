"""Tests for local job executors and routing to the declared backend.

* :class:`ProcessExecutor` runs a tool in a separate process (via cloudpickle,
  so `@tool` closures survive the boundary);
* :class:`SubprocessExecutor` runs it in a fresh child interpreter with real
  isolation, stdout/stderr-backed error reporting, and *true* cancellation;
* the :class:`JobManager` routes a job to the executor named by `tool.executor`
  and honours injected executor instances.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import Field

from corral.backend.executors import (
    JobCancelled,
    JobWork,
    ProcessExecutor,
    SubprocessExecutor,
    build_executor,
    render_result,
)
from corral.backend.jobs import JobManager, JobStatus
from corral.core.tool import tool


@tool
def cube(x: float = Field(description="value to cube")) -> str:
    """Cube a number (CPU-bound stand-in)."""
    return str(x * x * x)


@tool
def explode(x: float = Field(description="value")) -> str:
    """Always raise, to exercise the failure path."""
    raise ValueError(f"nope-{x}")


@tool
def nap(seconds: float = Field(description="seconds to sleep")) -> str:
    """Sleep, then report (stand-in for a long external job)."""
    time.sleep(seconds)
    return "awake"


def _work(tool_obj, **arguments):
    return JobWork(
        job_id="job_test",
        tool_name=tool_obj.name,
        tool=tool_obj,
        call_arguments=arguments,
    )


def test_render_result_passes_strings_and_json_encodes_others():
    assert render_result("hi") == "hi"
    assert render_result({"a": 1}) == '{"a": 1}'
    assert render_result([1, 2]) == "[1, 2]"


def test_process_executor_runs_tool_in_a_process():
    executor = ProcessExecutor(max_workers=2)
    try:
        assert executor.run_tool(_work(cube, x=3.0), threading.Event()) == "27.0"
    finally:
        executor.shutdown(wait=True)


def test_process_executor_propagates_tool_error():
    executor = ProcessExecutor(max_workers=2)
    try:
        with pytest.raises(ValueError, match="nope-2.0"):
            executor.run_tool(_work(explode, x=2.0), threading.Event())
    finally:
        executor.shutdown(wait=True)


def test_subprocess_executor_runs_tool_in_child_interpreter():
    executor = SubprocessExecutor(max_workers=2, poll_interval=0.05)
    try:
        assert executor.run_tool(_work(cube, x=4.0), threading.Event()) == "64.0"
    finally:
        executor.shutdown(wait=True)


def test_subprocess_executor_reports_tool_failure():
    executor = SubprocessExecutor(max_workers=2, poll_interval=0.05)
    try:
        with pytest.raises(RuntimeError, match="nope-5.0"):
            executor.run_tool(_work(explode, x=5.0), threading.Event())
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("start_new_session", [False, True])
def test_subprocess_executor_cancellation_kills_the_process(start_new_session):
    executor = SubprocessExecutor(
        max_workers=2, poll_interval=0.05, start_new_session=start_new_session
    )
    cancel = threading.Event()
    outcome: dict[str, str] = {}

    def _run():
        try:
            executor.run_tool(_work(nap, seconds=30.0), cancel)
            outcome["result"] = "completed"
        except JobCancelled:
            outcome["result"] = "cancelled"

    worker = threading.Thread(target=_run)
    worker.start()
    time.sleep(0.4)  # let the child actually start napping
    cancel.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert outcome["result"] == "cancelled"
    executor.shutdown(wait=True)


def test_build_executor_maps_names_to_backends():
    assert isinstance(build_executor("process"), ProcessExecutor)
    assert isinstance(build_executor("subprocess"), SubprocessExecutor)


@pytest.mark.parametrize("name", ["quantum", "slurm", "modal"])
def test_build_executor_rejects_unknown_name(name):
    with pytest.raises(ValueError, match="Unknown job executor"):
        build_executor(name)


class _RecordingExecutor:
    """Minimal executor that records the tools it was asked to run."""

    def __init__(self):
        self.ran: list[str] = []
        self._pool = None

    def submit(self, fn):
        # Run inline on a real worker thread so the manager's Future resolves.
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1)
        return self._pool.submit(fn)

    def run_tool(self, work, cancel):
        self.ran.append(work.tool_name)
        return render_result(work.tool.execute(**work.call_arguments))

    def shutdown(self, wait=False):
        if self._pool is not None:
            self._pool.shutdown(wait=wait)


@tool(background_capable=True, executor="subprocess")
def sub_tool(x: float = Field(description="v")) -> str:
    """A tool that asks to run on the subprocess executor."""
    return str(x + 1)


@tool(background_capable=True)  # no executor -> default "thread"
def thread_tool(x: float = Field(description="v")) -> str:
    """A tool that runs on the default thread executor."""
    return str(x + 2)


def test_manager_routes_job_to_the_tools_declared_executor():
    injected = _RecordingExecutor()
    manager = JobManager(executors={"subprocess": injected})

    rec = manager.submit(
        sub_tool, visible_arguments={"x": 1.0}, call_arguments={"x": 1.0}
    )
    view = manager.result(rec.context.job_id, wait=True, timeout=3.0)
    assert view["status"] == JobStatus.SUCCEEDED.value
    assert view["result"] == "2.0"
    # The injected subprocess executor — not the default thread one — ran it.
    assert injected.ran == ["sub_tool"]
    manager.shutdown()


def test_manager_uses_default_executor_when_tool_declares_none():
    injected = _RecordingExecutor()
    # Register the recorder under "subprocess"; the thread tool must NOT hit it.
    manager = JobManager(executors={"subprocess": injected})
    rec = manager.submit(
        thread_tool, visible_arguments={"x": 1.0}, call_arguments={"x": 1.0}
    )
    view = manager.result(rec.context.job_id, wait=True, timeout=3.0)
    assert view["result"] == "3.0"
    assert injected.ran == []
    manager.shutdown()


def test_manager_end_to_end_over_subprocess_executor():
    manager = JobManager()  # builds the real SubprocessExecutor lazily
    rec = manager.submit(
        sub_tool, visible_arguments={"x": 9.0}, call_arguments={"x": 9.0}
    )
    view = manager.result(rec.context.job_id, wait=True, timeout=15.0)
    assert view["status"] == JobStatus.SUCCEEDED.value
    assert view["result"] == "10.0"
    manager.shutdown()
