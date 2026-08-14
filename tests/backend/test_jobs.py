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
  background-capable task, resolves hidden args, and folds jobs into its report;
* the generated tools are served over both the REST and MCP trial surfaces.
"""

import json
import threading
import time

from fastapi.testclient import TestClient
from pydantic import Field

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.jobs import JobManager, JobStatus, ThreadExecutor
from corral.backend.server import create_benchmark_server
from corral.backend.task import TaskDefinition
from corral.backend.tool import tool

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


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


def test_provenance_bound_at_submit_and_hidden_values_redacted():
    manager = JobManager(
        provenance_provider=lambda: {
            "benchmark_run_id": "run_1",
            "episode_id": "ep_1",
            "trial_runtime_id": "tr_1",
        }
    )
    record = manager.submit(
        _FakeTool("t", lambda **_: "ok"),
        visible_arguments={"x": 1},
        call_arguments={"x": 1, "secret": "s3cr3t"},
        hidden_arg_names=("secret",),
        workspace="/tmp/ws",
    )
    view = manager.result(record.context.job_id, wait=True, timeout=2.0)
    assert view["benchmark_run_id"] == "run_1"
    assert view["episode_id"] == "ep_1"
    assert view["trial_runtime_id"] == "tr_1"
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
    env = _bg_env().for_trial("tr_1")
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
    )["t"].for_trial("tr_1")
    assert env.job_manager is None
    assert not any(name.startswith("start_") for name in env.tools)


def test_for_trial_sizes_job_pool_from_override():
    template = _bg_env()
    # A pinned per-trial limit sizes both the runtime and its JobManager.
    runtime = template.for_trial("tr_lim", max_job_concurrency=9)
    assert runtime.max_job_concurrency == 9
    # (whitebox) the value reaches the manager that bounds simultaneous jobs.
    assert runtime.job_manager._max_concurrency == 9
    runtime.shutdown_jobs()


def test_for_trial_defaults_to_template_job_concurrency():
    template = _bg_env()
    # Omitting the override keeps the template's own default (unchanged path).
    runtime = template.for_trial("tr_default")
    assert runtime.max_job_concurrency == template.max_job_concurrency
    runtime.shutdown_jobs()


def test_env_start_tool_runs_job_end_to_end():
    env = _bg_env().for_trial("tr_1")
    handle = json.loads(env.call_tool("start_slow_square", {"x": 4}).result)
    assert handle["status"] in {"queued", "running"}
    job_id = handle["job_id"]

    done = json.loads(
        env.call_tool("wait_for_job", {"job_id": job_id, "timeout_seconds": 3}).result
    )
    assert done["status"] == "succeeded"
    assert done["result"] == "16"
    env.shutdown_jobs()


def test_env_submit_job_injects_hidden_args_and_redacts_them():
    env = _bg_env(hidden=True).for_trial("tr_1")
    env.state.hidden_args = {"secret_scale": 3.0}

    handle = json.loads(env.call_tool("start_scaled", {"x": 5}).result)
    done = json.loads(
        env.call_tool(
            "wait_for_job", {"job_id": handle["job_id"], "timeout_seconds": 3}
        ).result
    )
    assert done["result"] == "15.0"  # 5 * hidden 3.0
    assert done["hidden_arg_names"] == ["secret_scale"]
    assert "3.0" not in json.dumps(done["arguments"])
    env.shutdown_jobs()


def test_env_submit_job_errors_when_hidden_arg_missing():
    env = _bg_env(hidden=True).for_trial("tr_1")
    # No hidden_args configured on the state.
    result = json.loads(env.call_tool("start_scaled", {"x": 5}).result)
    assert "error" in result
    assert "secret_scale" in result["error"]
    env.shutdown_jobs()


def test_completed_trial_data_includes_job_provenance():
    env = _bg_env().for_trial("tr_1")
    env.state.run_id = "run_1"
    handle = json.loads(env.call_tool("start_slow_square", {"x": 2}).result)
    env.call_tool("wait_for_job", {"job_id": handle["job_id"], "timeout_seconds": 3})

    data = env.get_completed_trial_data()
    jobs = data["state"]["jobs"]
    assert handle["job_id"] in jobs
    record = jobs[handle["job_id"]]
    assert record["tool_name"] == "slow_square"
    assert record["trial_runtime_id"] == "tr_1"
    assert record["benchmark_run_id"] == "run_1"
    assert record["result"] == "4"
    env.shutdown_jobs()


def test_unknown_job_id_via_tool_returns_error():
    env = _bg_env().for_trial("tr_1")
    result = json.loads(
        env.call_tool("get_job_status", {"job_id": "job_missing"}).result
    )
    assert "error" in result
    env.shutdown_jobs()


def _server_env():
    tools = {"slow_square": slow_square}
    task = TaskDefinition(
        name="task_a",
        description="d",
        tools=["slow_square"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    return build_environments(
        {"task_a": task}, toolset=Toolset(pool=tools, workspace_factory=None)
    )


def test_rest_trial_exposes_and_runs_background_tools():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        names = {
            t["function"]["name"]
            for t in client.get(f"/trials/{rid}/tools").json()["tools"]
        }
        assert {"start_slow_square", "get_job_result", "list_jobs"} <= names

        start = client.post(
            f"/trials/{rid}/tools/execute",
            json={"tool_name": "start_slow_square", "arguments": {"x": 6}},
        ).json()["result"]
        job_id = json.loads(start["result"])["job_id"]

        done = client.post(
            f"/trials/{rid}/tools/execute",
            json={
                "tool_name": "wait_for_job",
                "arguments": {"job_id": job_id, "timeout_seconds": 3},
            },
        ).json()["result"]
        assert json.loads(done["result"])["result"] == "36"

        # The live state snapshot reflects the job, then close cleans it up.
        state = client.get(f"/trials/{rid}/state").json()
        assert job_id in state["jobs"]
        assert client.delete(f"/trials/{rid}").status_code == 200


def test_create_trial_over_http_sizes_job_pool():
    """`tool_jobs_per_trial` in the create_trial body reaches `for_trial`.

    Section 3 wiring: the per-trial background-job limit travels over HTTP so the
    server sizes each runtime's JobManager, instead of a fixed server default.
    """
    captured: list[int | None] = []

    class _RecordingEnv(Environment):
        def for_trial(
            self, trial_runtime_id, episode_task_runs=None, *, max_job_concurrency=None
        ):
            captured.append(max_job_concurrency)
            return super().for_trial(
                trial_runtime_id,
                episode_task_runs,
                max_job_concurrency=max_job_concurrency,
            )

    task = TaskDefinition(
        name="task_a",
        description="d",
        tools=["slow_square"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    env = _RecordingEnv(
        "task_a",
        task,
        "",
        toolset=Toolset(pool={"slow_square": slow_square}, workspace_factory=None),
    )
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        # Pinned value is forwarded verbatim.
        rid = client.post(
            "/tasks/task_a/trials",
            json={"trial_index": 0, "tool_jobs_per_trial": 9},
        ).json()["trial_runtime_id"]
        # Omitted → None, so the server keeps the environment's own default.
        client.post("/tasks/task_a/trials", json={"trial_index": 1})
        assert client.delete(f"/trials/{rid}").status_code == 200

    assert captured == [9, None]


def test_mcp_trial_lists_background_tools():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp = client.post(f"/trials/{rid}/mcp", json=body, headers=_MCP_HEADERS)
        names = {t["name"] for t in resp.json()["result"]["tools"]}
        assert "start_slow_square" in names
        assert "get_job_status" in names
