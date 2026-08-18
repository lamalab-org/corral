"""Tests for background jobs (PR 4 of `make_efficiency.md`).

These cover the second concurrency layer: one agent starting a long-running
tool, getting a job handle back immediately, and polling / waiting / cancelling
it later — instead of the tool call blocking the agent.

* :class:`JobManager` lifecycle: submit → running → succeeded, failure capture,
  non-blocking poll vs. blocking wait, best-effort cancellation;
* `concurrency_key` serialises same-key jobs but lets different keys overlap;
* provenance (workspace, ids, hidden-arg names) is bound at submit time and
  hidden argument *values* are never serialised;
* an :class:`Environment` grows `start_<tool>` + control tools for a
  background-capable task, resolves hidden args, and folds jobs into State.
"""

import json
import threading
import time

from pydantic import Field

from corral.backend.jobs import JobManager, JobStatus, ThreadExecutor
from corral.core.action import Action
from corral.core.environment import Toolset, build_environments
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.core.transition import execute_action, propose_action


class _FakeTool:
    """Minimal duck-typed tool for JobManager unit tests."""

    def __init__(self, name, fn, concurrency_key=None):
        self.name = name
        self._fn = fn
        self.concurrency_key = concurrency_key
        self.background_capable = True

    def execute(self, **kwargs):
        return self._fn(**kwargs)


def _wait_until(predicate, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _call(env, state, name, arguments):
    action = Action(name=name, arguments=arguments)
    child = execute_action(env, propose_action(state, action), action)
    return child, child.messages[-1]


def test_submit_returns_immediately_then_succeeds():
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def _work(**_):
        started.set()
        release.wait(2.0)
        return "done"

    record = manager.submit(
        _FakeTool("work", _work),
        visible_arguments={},
        call_arguments={},
    )
    # The handle comes back before the tool finishes.
    assert started.wait(2.0)
    assert manager.status(record.context.job_id) == JobStatus.RUNNING

    # A non-blocking poll shows no result yet.
    polled = manager.result(record.context.job_id)
    assert polled["result"] is None

    release.set()
    final = manager.result(record.context.job_id, wait=True, timeout=2.0)
    assert final["status"] == JobStatus.SUCCEEDED.value
    assert final["result"] == "done"
    assert final["duration"] is not None
    manager.shutdown()


def test_failed_job_records_error():
    manager = JobManager()

    def _boom(**_):
        raise RuntimeError("kaboom")

    record = manager.submit(
        _FakeTool("boom", _boom), visible_arguments={}, call_arguments={}
    )
    final = manager.result(record.context.job_id, wait=True, timeout=2.0)
    assert final["status"] == JobStatus.FAILED.value
    assert "kaboom" in final["error"]
    assert final["result"] is None
    manager.shutdown()


def test_wait_timeout_returns_running_view_without_raising():
    manager = JobManager()
    release = threading.Event()

    record = manager.submit(
        _FakeTool("slow", lambda **_: release.wait(2.0) or "ok"),
        visible_arguments={},
        call_arguments={},
    )
    view = manager.result(record.context.job_id, wait=True, timeout=0.05)
    assert view["status"] == JobStatus.RUNNING.value
    release.set()
    manager.shutdown()


def test_unknown_job_id_raises_keyerror():
    manager = JobManager()
    for call in (
        lambda: manager.status("nope"),
        lambda: manager.result("nope"),
        lambda: manager.cancel("nope"),
    ):
        try:
            call()
        except KeyError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected KeyError for unknown job id")
    manager.shutdown()


def test_cancel_queued_job_never_runs():
    # A single-slot manager: the second job is stuck QUEUED behind the first.
    manager = JobManager(max_concurrency=1)
    release = threading.Event()
    ran_second = threading.Event()

    first = manager.submit(
        _FakeTool("first", lambda **_: release.wait(2.0) or "1"),
        visible_arguments={},
        call_arguments={},
    )
    second = manager.submit(
        _FakeTool("second", lambda **_: ran_second.set() or "2"),
        visible_arguments={},
        call_arguments={},
    )
    # Second is waiting in the pool queue, not running.
    assert _wait_until(
        lambda: manager.status(first.context.job_id) == JobStatus.RUNNING
    )
    assert manager.status(second.context.job_id) == JobStatus.QUEUED

    cancelled = manager.cancel(second.context.job_id)
    assert cancelled["status"] == JobStatus.CANCELLED.value

    release.set()
    manager.result(first.context.job_id, wait=True, timeout=2.0)
    time.sleep(0.05)
    assert not ran_second.is_set()
    assert manager.status(second.context.job_id) == JobStatus.CANCELLED
    manager.shutdown()


def _intervals_overlap(a, b):
    return a["started_at"] < b["ended_at"] and b["started_at"] < a["ended_at"]


def test_same_concurrency_key_serialises():
    manager = JobManager(max_concurrency=4)

    def _work(**_):
        time.sleep(0.1)
        return "ok"

    r1 = manager.submit(
        _FakeTool("t1", _work, concurrency_key="gpu"),
        visible_arguments={},
        call_arguments={},
    )
    r2 = manager.submit(
        _FakeTool("t2", _work, concurrency_key="gpu"),
        visible_arguments={},
        call_arguments={},
    )
    v1 = manager.result(r1.context.job_id, wait=True, timeout=3.0)
    v2 = manager.result(r2.context.job_id, wait=True, timeout=3.0)
    assert not _intervals_overlap(v1, v2)
    manager.shutdown()


def test_different_concurrency_keys_overlap():
    manager = JobManager(max_concurrency=4)

    def _work(**_):
        time.sleep(0.1)
        return "ok"

    r1 = manager.submit(
        _FakeTool("t1", _work, concurrency_key="a"),
        visible_arguments={},
        call_arguments={},
    )
    r2 = manager.submit(
        _FakeTool("t2", _work, concurrency_key="b"),
        visible_arguments={},
        call_arguments={},
    )
    v1 = manager.result(r1.context.job_id, wait=True, timeout=3.0)
    v2 = manager.result(r2.context.job_id, wait=True, timeout=3.0)
    assert _intervals_overlap(v1, v2)
    manager.shutdown()


def test_job_context_redacts_hidden_values():
    manager = JobManager()
    record = manager.submit(
        _FakeTool("t", lambda **_: "ok"),
        visible_arguments={"x": 1},
        call_arguments={"x": 1, "secret": "s3cr3t"},
        hidden_arg_names=("secret",),
        workspace="/tmp/ws",
    )
    view = manager.result(record.context.job_id, wait=True, timeout=2.0)
    assert view["workspace"] == "/tmp/ws"
    assert view["arguments"] == {"x": 1}
    assert view["hidden_arg_names"] == ["secret"]
    # The secret value must never appear anywhere in the serialised job.
    assert "s3cr3t" not in json.dumps(view)
    manager.shutdown()


def test_thread_executor_is_pluggable():
    executor = ThreadExecutor(max_workers=2)
    manager = JobManager(executor=executor)
    record = manager.submit(
        _FakeTool("t", lambda **_: "ok"), visible_arguments={}, call_arguments={}
    )
    assert (
        manager.result(record.context.job_id, wait=True, timeout=2.0)["result"] == "ok"
    )
    manager.shutdown()


@tool(background_capable=True, concurrency_key="sim")
def slow_square(x: float = Field(description="value to square")) -> str:
    """Square a number after a short delay (stands in for a long simulation)."""
    time.sleep(0.05)
    return str(x * x)


@tool(background_capable=True, hidden_args=["secret_scale"])
def scaled(
    x: float = Field(description="value"),
    secret_scale: float = Field(default=10.0, description="hidden multiplier"),
) -> str:
    """Multiply a value by a hidden scale factor."""
    return str(x * secret_scale)


@tool
def plain_add(
    x: float = Field(description="a"), y: float = Field(description="b")
) -> str:
    """Add two numbers (not background-capable)."""
    return str(x + y)


def _bg_env(*, hidden=False):
    tools = {"slow_square": slow_square, "scaled": scaled, "plain_add": plain_add}
    names = ["scaled"] if hidden else ["slow_square", "plain_add"]
    task = TaskDefinition(
        name="t",
        description="d",
        tools=names,
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments(
        {"t": task}, toolset=Toolset(pool=tools, workspace_factory=None)
    )
    return envs["t"]


def test_env_attaches_background_and_control_tools():
    env = _bg_env().for_task("task_1")
    assert env.job_manager is not None
    # start_<tool> only for the background-capable tool, plus the control tools.
    assert "start_slow_square" in env.tools
    assert "start_plain_add" not in env.tools
    for control in (
        "get_job_status",
        "get_job_result",
        "wait_for_job",
        "cancel_job",
        "list_jobs",
    ):
        assert control in env.tools
    # The original (blocking) tool is still directly callable.
    assert "slow_square" in env.tools
    env.shutdown_jobs()


def test_env_without_background_tools_has_no_job_manager():
    task = TaskDefinition(
        name="t",
        description="d",
        tools=["plain_add"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    env = build_environments(
        {"t": task},
        toolset=Toolset(pool={"plain_add": plain_add}, workspace_factory=None),
    )["t"].for_task("task_1")
    assert env.job_manager is None
    assert not any(name.startswith("start_") for name in env.tools)


def test_for_task_sizes_job_pool_from_override():
    template = _bg_env()
    # A pinned per-execution limit sizes both the runtime and its JobManager.
    runtime = template.for_task("task_lim", max_job_concurrency=9)
    assert runtime.max_job_concurrency == 9
    # (whitebox) the value reaches the manager that bounds simultaneous jobs.
    assert runtime.job_manager._max_concurrency == 9
    runtime.shutdown_jobs()


def test_for_task_defaults_to_template_job_concurrency():
    template = _bg_env()
    # Omitting the override keeps the template's own default (unchanged path).
    runtime = template.for_task("task_default")
    assert runtime.max_job_concurrency == template.max_job_concurrency
    runtime.shutdown_jobs()


def test_env_start_tool_runs_job_end_to_end():
    env = _bg_env().for_task("task_1")
    state = env.initial_state()
    state, call = _call(env, state, "start_slow_square", {"x": 4})
    handle = json.loads(call["content"])
    assert handle["status"] in {"queued", "running"}
    job_id = handle["job_id"]

    state, call = _call(
        env, state, "wait_for_job", {"job_id": job_id, "timeout_seconds": 3}
    )
    done = json.loads(call["content"])
    assert done["status"] == "succeeded"
    assert done["result"] == "16"
    env.shutdown_jobs()


def test_env_submit_job_injects_hidden_args_and_redacts_them():
    env = _bg_env(hidden=True).for_task("task_1")
    state = env.initial_state()
    state = state.fork(
        environment={
            **dict(state.environment),
            "hidden_arguments": {"secret_scale": 3.0},
        }
    )

    state, call = _call(env, state, "start_scaled", {"x": 5})
    handle = json.loads(call["content"])
    state, call = _call(
        env,
        state,
        "wait_for_job",
        {"job_id": handle["job_id"], "timeout_seconds": 3},
    )
    done = json.loads(call["content"])
    assert done["result"] == "15.0"  # 5 * hidden 3.0
    assert done["hidden_arg_names"] == ["secret_scale"]
    assert "3.0" not in json.dumps(done["arguments"])
    env.shutdown_jobs()


def test_env_submit_job_errors_when_hidden_arg_missing():
    env = _bg_env(hidden=True).for_task("task_1")
    state = env.initial_state()
    _, call = _call(env, state, "start_scaled", {"x": 5})
    assert call["metadata"]["success"] is False
    assert "secret_scale" in call["content"]
    env.shutdown_jobs()


def test_environment_jobs_contain_no_runner_identity():
    env = _bg_env().for_task("task_1")
    state = env.initial_state()
    state, call = _call(env, state, "start_slow_square", {"x": 2})
    handle = json.loads(call["content"])
    state, _ = _call(
        env,
        state,
        "wait_for_job",
        {"job_id": handle["job_id"], "timeout_seconds": 3},
    )

    jobs = state.environment["jobs"]
    assert handle["job_id"] in jobs
    record = jobs[handle["job_id"]]
    assert record["tool_name"] == "slow_square"
    assert "execution_id" not in record
    assert "benchmark_run_id" not in record
    assert record["result"] == "4"
    env.shutdown_jobs()


def test_unknown_job_id_via_tool_returns_error():
    env = _bg_env().for_task("task_1")
    state = env.initial_state()
    _, call = _call(env, state, "get_job_status", {"job_id": "job_missing"})
    result = json.loads(call["content"])
    assert "error" in result
    env.shutdown_jobs()
