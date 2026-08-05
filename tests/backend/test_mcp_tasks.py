"""Tests for the MCP Tasks extension surface (Section 7 of `make_efficiency.md`).

PR 4 exposed background jobs through explicit `start_<tool>` / `get_job_*`
fallback tools. This suite covers the *native* MCP representation layered on top
of the same :class:`~corral.backend.jobs.JobManager`:

* a background-capable tool advertises `execution.taskSupport = "optional"` so
  a task-aware client may task-augment its call;
* the low-level MCP server advertises the `tasks` capability;
* a task-augmented `tools/call` returns a durable `CreateTaskResult` handle
  instead of blocking, and the job is a real JobManager job (same provenance,
  same trial-report entry);
* `tasks/get` / `tasks/result` / `tasks/list` / `tasks/cancel` dispatch
  to that runtime's JobManager, with spec-compliant errors for unknown/terminal
  tasks;
* JobStatus maps onto the extension's task-status vocabulary.
"""

import threading
import time

import anyio
from fastapi.testclient import TestClient
from mcp.server.experimental.request_context import Experimental
from mcp.types import (
    TASK_STATUS_CANCELLED,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_WORKING,
    CreateTaskResult,
    TaskMetadata,
)
from pydantic import Field

from corral.backend import mcp_tasks
from corral.backend.env import Toolset, build_environments
from corral.backend.jobs import JobStatus
from corral.backend.mcp_server import build_trial_mcp_server
from corral.backend.server import create_benchmark_server
from corral.backend.task import TaskDefinition
from corral.backend.tool import tool
from corral.router.verbosity import ToolVerbosity

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

INVALID_PARAMS = -32602


@tool(background_capable=True, concurrency_key="sim")
def slow_square(x: float = Field(description="value to square")) -> str:
    """Square a number after a short delay (stands in for a long simulation)."""
    time.sleep(0.05)
    return str(x * x)


@tool
def plain_add(
    x: float = Field(description="a"), y: float = Field(description="b")
) -> str:
    """Add two numbers (not background-capable)."""
    return str(x + y)


def _bg_env(trial_id="tr_1"):
    tools = {"slow_square": slow_square, "plain_add": plain_add}
    task = TaskDefinition(
        name="t",
        description="d",
        tools=["slow_square", "plain_add"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments(
        {"t": task}, toolset=Toolset(pool=tools, workspace_factory=None)
    )
    return envs["t"].for_trial(trial_id)


def _server_env():
    tools = {"slow_square": slow_square, "plain_add": plain_add}
    task = TaskDefinition(
        name="task_a",
        description="d",
        tools=["slow_square", "plain_add"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    return build_environments(
        {"task_a": task}, toolset=Toolset(pool=tools, workspace_factory=None)
    )


def _call(client, rid, method, params, req_id=1):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    return client.post(f"/trials/{rid}/mcp", headers=_MCP_HEADERS, json=body).json()


def test_background_tool_advertises_task_support():
    # Only a background-capable tool offers task-augmented invocation.
    assert slow_square.to_mcp()["execution"] == {"taskSupport": "optional"}
    assert "execution" not in plain_add.to_mcp()


def test_job_status_maps_onto_task_status():
    assert mcp_tasks.task_status_for(JobStatus.QUEUED.value) == TASK_STATUS_WORKING
    assert mcp_tasks.task_status_for(JobStatus.RUNNING.value) == TASK_STATUS_WORKING
    assert mcp_tasks.task_status_for(JobStatus.SUCCEEDED.value) == TASK_STATUS_COMPLETED
    assert mcp_tasks.task_status_for(JobStatus.FAILED.value) == TASK_STATUS_FAILED
    assert mcp_tasks.task_status_for(JobStatus.CANCELLED.value) == TASK_STATUS_CANCELLED
    # No task "timed_out": a deadline-elapsed job reports as failed.
    assert mcp_tasks.task_status_for(JobStatus.TIMED_OUT.value) == TASK_STATUS_FAILED


def test_trial_server_advertises_tasks_capability():
    server = build_trial_mcp_server({}, ToolVerbosity.FULL)
    caps = server.create_initialization_options().capabilities
    assert caps.tasks is not None


def test_maybe_start_task_submits_background_job():
    env = _bg_env("tr_start")
    exp = Experimental(task_metadata=TaskMetadata(ttl=60000))
    result = anyio.run(mcp_tasks.maybe_start_task, env, "slow_square", {"x": 4}, exp)
    assert isinstance(result, CreateTaskResult)
    assert result.task.status == TASK_STATUS_WORKING
    assert result.task.ttl == 60000
    # The handle is a real JobManager job in this runtime.
    assert env.job_manager.status(result.task.taskId) is not None
    env.shutdown_jobs()


def test_maybe_start_task_ignores_non_augmented_call():
    env = _bg_env("tr_plain")
    exp = Experimental(task_metadata=None)  # not task-augmented
    assert (
        anyio.run(mcp_tasks.maybe_start_task, env, "slow_square", {"x": 4}, exp) is None
    )
    env.shutdown_jobs()


def test_maybe_start_task_ignores_non_background_tool():
    env = _bg_env("tr_add")
    exp = Experimental(task_metadata=TaskMetadata(ttl=1000))
    # plain_add never advertised task support, so a task-augmented call falls
    # through to synchronous execution.
    assert (
        anyio.run(mcp_tasks.maybe_start_task, env, "plain_add", {"x": 1, "y": 2}, exp)
        is None
    )
    env.shutdown_jobs()


def test_mcp_tools_list_advertises_task_support():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        tools = {
            t["name"]: t
            for t in _call(client, rid, "tools/list", {})["result"]["tools"]
        }
        assert tools["slow_square"]["execution"]["taskSupport"] == "optional"
        # The generated fallback variant is an ordinary tool, not task-augmentable.
        assert (tools["start_slow_square"].get("execution") or {}).get(
            "taskSupport"
        ) is None
        assert client.delete(f"/trials/{rid}").status_code == 200


def test_task_augmented_call_returns_handle_and_result():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        # A task-augmented tools/call returns a durable handle, not a blocking result.
        start = _call(
            client,
            rid,
            "tools/call",
            {"name": "slow_square", "arguments": {"x": 4}, "task": {"ttl": 60000}},
        )["result"]
        assert start["task"]["status"] == TASK_STATUS_WORKING
        assert start["task"]["pollInterval"] == mcp_tasks.POLL_INTERVAL_MS
        task_id = start["task"]["taskId"]

        # tasks/get reports a live status.
        got = _call(client, rid, "tasks/get", {"taskId": task_id}, req_id=2)["result"]
        assert got["taskId"] == task_id
        assert got["status"] in {TASK_STATUS_WORKING, TASK_STATUS_COMPLETED}

        # tasks/result blocks until the job finishes and returns the tool payload,
        # carrying the spec-mandated related-task metadata.
        payload = _call(client, rid, "tasks/result", {"taskId": task_id}, req_id=3)[
            "result"
        ]
        assert payload["content"][0]["text"] == "16"
        assert mcp_tasks.RELATED_TASK_METADATA_KEY in payload["_meta"]

        # tasks/list surfaces the task, and it is folded into the trial state
        # exactly like a fallback job (same JobManager, same provenance).
        listed = _call(client, rid, "tasks/list", {}, req_id=4)["result"]
        assert task_id in {t["taskId"] for t in listed["tasks"]}
        assert task_id in client.get(f"/trials/{rid}/state").json()["jobs"]
        assert client.delete(f"/trials/{rid}").status_code == 200


def test_non_augmented_call_stays_synchronous():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        # No `task` field → the original blocking behaviour: the result comes
        # back inline, not as a task handle.
        result = _call(
            client, rid, "tools/call", {"name": "slow_square", "arguments": {"x": 3}}
        )["result"]
        assert "task" not in result
        assert result["content"][0]["text"] == "9"
        assert client.delete(f"/trials/{rid}").status_code == 200


def test_tasks_get_unknown_task_is_invalid_params():
    with TestClient(create_benchmark_server(_server_env())) as client:
        rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
            "trial_runtime_id"
        ]
        resp = _call(client, rid, "tasks/get", {"taskId": "job_missing"})
        assert resp["error"]["code"] == INVALID_PARAMS
        assert client.delete(f"/trials/{rid}").status_code == 200


def test_tasks_cancel_running_then_terminal_errors():
    gate = threading.Event()

    @tool(background_capable=True)
    def gated(x: float = Field(description="value")) -> str:
        """Block on a test-controlled gate (stands in for a long-running job)."""
        gate.wait(timeout=5)
        return str(x)

    task = TaskDefinition(
        name="task_a",
        description="d",
        tools=["gated"],
        scoring_fn=lambda a: 1.0,
        submission_format={},
    )
    envs = build_environments(
        {"task_a": task}, toolset=Toolset(pool={"gated": gated}, workspace_factory=None)
    )
    try:
        with TestClient(create_benchmark_server(envs)) as client:
            rid = client.post("/tasks/task_a/trials", json={"trial_index": 0}).json()[
                "trial_runtime_id"
            ]
            start = _call(
                client,
                rid,
                "tools/call",
                {"name": "gated", "arguments": {"x": 1}, "task": {"ttl": 60000}},
            )["result"]
            task_id = start["task"]["taskId"]

            # Cancelling a live task moves it to the cancelled terminal state.
            cancelled = _call(client, rid, "tasks/cancel", {"taskId": task_id}, 2)[
                "result"
            ]
            assert cancelled["status"] == TASK_STATUS_CANCELLED

            # Cancelling an already-terminal task is an INVALID_PARAMS error.
            again = _call(client, rid, "tasks/cancel", {"taskId": task_id}, 3)
            assert again["error"]["code"] == INVALID_PARAMS
            assert client.delete(f"/trials/{rid}").status_code == 200
    finally:
        gate.set()  # release the worker thread


def test_task_scoped_server_serves_tasks_too():
    """The sequential /tasks/{id}/mcp surface also serves the Tasks extension."""
    with TestClient(create_benchmark_server(_server_env())) as client:
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "slow_square",
                "arguments": {"x": 5},
                "task": {"ttl": 60000},
            },
        }
        start = client.post(
            "/tasks/task_a/mcp", headers=_MCP_HEADERS, json=body
        ).json()["result"]
        task_id = start["task"]["taskId"]
        res = client.post(
            "/tasks/task_a/mcp",
            headers=_MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tasks/result",
                "params": {"taskId": task_id},
            },
        ).json()["result"]
        assert res["content"][0]["text"] == "25"
