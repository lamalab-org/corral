"""Cancellation must stop descendants even when the group leader exits first."""

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from corral.backend import executors
from corral.backend.executors import SubprocessExecutor
from corral.backend.jobs import JobManager, JobStatus
from corral.core.tool import tool


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for subprocess state"
        time.sleep(0.02)


def _is_running(pid):
    # An orphan can remain a zombie until the OS reaps it, particularly in
    # containers. Zombies have terminated, even though kill(pid, 0) succeeds.
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode in (0, 1), result.stderr
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
@pytest.mark.parametrize(
    ("worker_mode", "ignore_term"),
    [
        pytest.param("running", False, id="graceful-group-termination"),
        pytest.param("ignore_term", True, id="whole-group-needs-sigkill"),
        pytest.param("running", True, id="grandchild-outlives-leader"),
        pytest.param("exited", True, id="leader-already-exited"),
    ],
)
def test_cancel_job_kills_worker_and_grandchild(
    tmp_path, monkeypatch, worker_mode, ignore_term
):
    monkeypatch.setattr(executors, "_TERMINATE_GRACE_SECONDS", 0.2)
    worker_pid_path = tmp_path / "worker.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"

    @tool(background_capable=True, executor="subprocess")
    def spawn_sleeper() -> str:
        """Spawn a long-running external program for cancellation testing."""
        if worker_mode == "ignore_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        worker_pid_path.write_text(str(os.getpid()))
        sleeper = """
import os, signal, sys, time
from pathlib import Path
handler = signal.SIG_IGN if sys.argv[2] == 'True' else signal.SIG_DFL
signal.signal(signal.SIGTERM, handler)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
"""
        # Most cases close the inherited pipes so pipe EOF cannot be mistaken
        # for the process group exiting. The exited-leader case keeps them open
        # to leave communicate() pending after the worker has already died.
        output = None if worker_mode == "exited" else subprocess.DEVNULL
        child = subprocess.Popen(
            [sys.executable, "-c", sleeper, str(grandchild_pid_path), str(ignore_term)],
            stdout=output,
            stderr=output,
        )
        if worker_mode == "exited":
            os._exit(0)
        child.wait()
        return "finished"

    executor = SubprocessExecutor(poll_interval=0.02)
    finished = threading.Event()
    run_tool = executor.run_tool

    def run_and_notify(work, cancel):
        try:
            return run_tool(work, cancel)
        finally:
            finished.set()

    monkeypatch.setattr(executor, "run_tool", run_and_notify)
    manager = JobManager(executors={"subprocess": executor})
    record = manager.submit(spawn_sleeper, visible_arguments={}, call_arguments={})
    worker_pid = grandchild_pid = None
    try:
        # The grandchild publishes its PID only after installing its handler.
        # File readiness, rather than a fixed sleep, gates cancellation.
        _wait_until(
            lambda: all(
                path.exists() and path.read_text()
                for path in (worker_pid_path, grandchild_pid_path)
            )
        )
        worker_pid = int(worker_pid_path.read_text())
        grandchild_pid = int(grandchild_pid_path.read_text())
        assert os.getpgid(grandchild_pid) == worker_pid
        assert _is_running(grandchild_pid)
        if worker_mode == "exited":
            _wait_until(lambda: not _is_running(worker_pid))
        else:
            assert _is_running(worker_pid)

        result = manager.cancel(record.context.job_id)
        assert result["status"] == JobStatus.CANCELLED.value
        # A cancelled record alone is not evidence that execution has stopped.
        assert finished.wait(timeout=5)
        _wait_until(lambda: not _is_running(worker_pid), timeout=3)
        _wait_until(lambda: not _is_running(grandchild_pid), timeout=3)
        with pytest.raises(ChildProcessError):
            os.waitpid(worker_pid, os.WNOHANG)  # the executor reaped its child
    finally:
        # Also clean up against the unfixed implementation or a failed assert.
        manager.cancel(record.context.job_id)
        if worker_pid is None and worker_pid_path.exists():
            contents = worker_pid_path.read_text()
            worker_pid = int(contents) if contents else None
        if worker_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(worker_pid, signal.SIGKILL)
        manager.shutdown(wait=True)


@pytest.mark.parametrize(
    ("platform", "start_new_session"),
    [("posix", False), ("nt", True), ("nt", False)],
)
@pytest.mark.parametrize("needs_kill", [False, True])
def test_cancellation_without_own_posix_group_only_targets_child(
    monkeypatch, platform, start_new_session, needs_kill
):
    # Patch this module's os reference, not os.name globally (which also
    # changes pathlib and pytest). The mock has no killpg on Windows.
    platform_os = SimpleNamespace(name=platform)
    if platform == "posix":
        platform_os.killpg = Mock(side_effect=AssertionError("Shared process group"))
    monkeypatch.setattr(executors, "os", platform_os)
    proc = Mock()
    proc.poll.return_value = None
    if needs_kill:
        proc.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.2), 0]

    executors._terminate_process(proc, start_new_session=start_new_session)

    proc.terminate.assert_called_once_with()
    assert proc.kill.call_count == int(needs_kill)
    # SIGKILL must also be followed by a wait, to avoid leaving a zombie.
    assert proc.wait.call_count == 1 + int(needs_kill)


@pytest.mark.parametrize("disappears_on_call", [1, 2, 3])
def test_group_disappearing_during_cancellation_is_harmless(
    monkeypatch, disappears_on_call
):
    killpg = Mock(
        side_effect=[None] * (disappears_on_call - 1) + [ProcessLookupError()]
    )
    monkeypatch.setattr(executors, "os", SimpleNamespace(name="posix", killpg=killpg))
    # The constants used by the POSIX branch need not exist on Windows.
    monkeypatch.setattr(executors, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    monkeypatch.setattr(executors, "_TERMINATE_GRACE_SECONDS", 0)
    proc = Mock()
    proc.poll.return_value = 0  # The leader exiting must not skip group cleanup.

    executors._terminate_process(proc, start_new_session=True)

    assert killpg.call_count == disappears_on_call
    proc.wait.assert_called_once_with()


def test_group_cancellation_does_not_hide_permission_errors(monkeypatch):
    killpg = Mock(side_effect=PermissionError())
    monkeypatch.setattr(executors, "os", SimpleNamespace(name="posix", killpg=killpg))
    proc = Mock()

    with pytest.raises(PermissionError):
        executors._terminate_process(proc, start_new_session=True)


def test_group_probe_permission_error_does_not_abort_cancellation(monkeypatch):
    killpg = Mock(side_effect=[None, PermissionError(), ProcessLookupError()])
    monkeypatch.setattr(executors, "os", SimpleNamespace(name="posix", killpg=killpg))
    monkeypatch.setattr(
        executors, "time", SimpleNamespace(monotonic=lambda: 0, sleep=Mock())
    )
    proc = Mock()

    executors._terminate_process(proc, start_new_session=True)

    assert killpg.call_count == 3
    proc.wait.assert_called_once_with()
