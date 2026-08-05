"""Tests for the process-worker trial backend (Environment Concurrency Isolation, Phase 2).

Two layers:

* The worker machinery in isolation — :class:`TrialWorkerPool` (leasing bounds
  concurrency, workers are reused) and :class:`WorkerTrialRuntime` (a trial runs
  end-to-end over the pipe; worker-side failures cross the boundary as a
  :class:`WorkerError`).
* The backend behind the server — `POST /tasks/{id}/trials` routes a
  `concurrency="process"` env to a worker, and every `/trials/{id}/*` REST
  and MCP handler behaves identically to the in-process backend, including
  episode-pinned dependency chains.

Workers are spawned with the `spawn` start method so the tests behave the same
on Linux (default `fork`) and macOS.
"""

import threading

import pytest
import worker_fixtures
from fastapi.testclient import TestClient

from corral.backend.schema import ToolCallStatus
from corral.backend.server import create_benchmark_server
from corral.backend.trial_worker import (
    TrialWorkerPool,
    WorkerError,
    WorkerTrialRuntime,
)

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _make_client(build_envs, *, pool_size: int = 1) -> TestClient:
    """A server whose process envs are backed by a `spawn` worker pool."""
    app = create_benchmark_server(
        build_envs(),
        build_envs=build_envs,
        trial_worker_pool_size=pool_size,
        trial_worker_start_method="spawn",
    )
    return TestClient(app)


def _create_trial(client: TestClient, task_id: str, **body) -> dict:
    resp = client.post(f"/tasks/{task_id}/trials", json={"trial_index": 0, **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_pool_reuses_workers():
    """A released worker is handed back out — imports stay amortised."""
    pool = TrialWorkerPool(
        worker_fixtures.build_process_calc_envs, 1, start_method="spawn"
    )
    try:
        handle = pool.lease()
        pool.release(handle)
        assert pool.lease() is handle  # same process, not a fresh one
    finally:
        pool.shutdown()


def test_pool_lease_blocks_until_release():
    """Leasing past the pool size blocks (the pool *is* the limiter)."""
    pool = TrialWorkerPool(
        worker_fixtures.build_process_calc_envs, 1, start_method="spawn"
    )
    try:
        first = pool.lease()

        leased = threading.Event()
        second: list = []

        def _lease_second():
            second.append(pool.lease())
            leased.set()

        waiter = threading.Thread(target=_lease_second, daemon=True)
        waiter.start()

        # The only worker is out, so the second lease cannot complete yet.
        assert not leased.wait(timeout=0.5)

        pool.release(first)
        assert leased.wait(timeout=5)
        assert second[0] is first
    finally:
        pool.shutdown()


def test_worker_runtime_round_trip():
    pool = TrialWorkerPool(
        worker_fixtures.build_process_calc_envs, 1, start_method="spawn"
    )
    try:
        runtime = WorkerTrialRuntime(
            pool, pool.lease(), "task_a", release_on_close=True
        )
        runtime.open(
            trial_runtime_id="tr_rt",
            episode_id=None,
            benchmark_run_id="run-1",
            tool_jobs_per_trial=None,
        )

        # configure/prompt/snapshot all cross the pipe like the in-process path.
        assert runtime.configure()["task_id"] == "task_a"
        assert "a task" in runtime.get_task_prompt()

        tool_call = runtime.call_tool("calc", {"operation": "add", "x": 2, "y": 5})
        assert tool_call.result == "7"
        assert tool_call.status == ToolCallStatus.SUCCESS

        assert runtime.status()["tool_statistics"]["total_calls"] == 1
        assert runtime.snapshot()["tool_statistics"]["total_calls"] == 1

        completion = runtime.submit("7")
        assert completion.score == 1.0
        assert completion.surrendered is False

        runtime.close()
        # The worker was returned to the pool and is reusable.
        assert pool.lease() is runtime._handle
    finally:
        pool.shutdown()


def test_worker_error_crosses_boundary():
    """A raise inside the worker surfaces server-side as a WorkerError + traceback."""
    pool = TrialWorkerPool(
        worker_fixtures.build_process_calc_envs, 1, start_method="spawn"
    )
    try:
        runtime = WorkerTrialRuntime(
            pool, pool.lease(), "task_a", release_on_close=True
        )
        # Address a runtime id the worker never opened -> the worker raises,
        # which is re-raised in this process with the message + traceback.
        runtime._trial_runtime_id = "tr_never_opened"
        with pytest.raises(WorkerError) as excinfo:
            runtime.status()
        assert "tr_never_opened" in str(excinfo.value)
        assert excinfo.value.worker_traceback is not None
        assert "KeyError" in excinfo.value.worker_traceback
    finally:
        pool.shutdown()


def test_worker_backend_execute_and_submit():
    with _make_client(worker_fixtures.build_process_calc_envs) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        exec_resp = client.post(
            f"/trials/{rid}/tools/execute",
            json={
                "tool_name": "calc",
                "arguments": {"operation": "mul", "x": 3, "y": 4},
            },
        )
        assert exec_resp.status_code == 200, exec_resp.text
        assert exec_resp.json()["result"]["result"] == "12"

        submit_resp = client.post(f"/trials/{rid}/submit", json={"answer": "12"})
        assert submit_resp.status_code == 200, submit_resp.text
        assert submit_resp.json()["score"] == 1.0


def test_worker_backend_tool_error_surfaces():
    """A tool that raises comes back as an EXECUTION_ERROR record, message intact."""
    with _make_client(worker_fixtures.build_process_calc_envs) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]
        resp = client.post(
            f"/trials/{rid}/tools/execute",
            json={"tool_name": "boom", "arguments": {"message": "kaboom"}},
        )
        assert resp.status_code == 200, resp.text
        result = resp.json()["result"]
        assert result["status"] == ToolCallStatus.EXECUTION_ERROR.value
        assert result["error_message"] == "kaboom"


def test_worker_backend_two_trials_isolated():
    with _make_client(worker_fixtures.build_process_calc_envs, pool_size=2) as client:
        rid_a = _create_trial(client, "task_a")["trial_runtime_id"]
        rid_b = _create_trial(client, "task_a")["trial_runtime_id"]
        assert rid_a != rid_b

        client.post(
            f"/trials/{rid_a}/tools/execute",
            json={"tool_name": "calc", "arguments": {"operation": "add", "x": 1}},
        )

        status_a = client.get(f"/trials/{rid_a}/status").json()
        status_b = client.get(f"/trials/{rid_b}/status").json()

    # Separate processes, separate state: the call in A never leaks into B.
    assert status_a["tool_statistics"]["total_calls"] == 1
    assert status_b["tool_statistics"]["total_calls"] == 0


def test_worker_backend_prompt_and_tools_endpoints():
    with _make_client(worker_fixtures.build_process_calc_envs) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        prompt = client.get(f"/trials/{rid}/prompt").json()["prompt"]
        assert "a task" in prompt

        tools = client.get(f"/trials/{rid}/tools", params={"verbosity": "brief"}).json()
        assert sorted(t["function"]["name"] for t in tools["tools"]) == ["boom", "calc"]


def _mcp_rpc(client: TestClient, rid: str, method: str, params: dict) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = client.post(f"/trials/{rid}/mcp", json=body, headers=_MCP_HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_worker_backend_mcp_tools_list_and_call():
    with _make_client(worker_fixtures.build_process_calc_envs) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        listed = _mcp_rpc(client, rid, "tools/list", {})
        assert sorted(t["name"] for t in listed["result"]["tools"]) == ["boom", "calc"]

        called = _mcp_rpc(
            client,
            rid,
            "tools/call",
            {"name": "calc", "arguments": {"operation": "mul", "x": 3, "y": 4}},
        )
        assert called["result"]["isError"] is False
        assert called["result"]["content"][0]["text"] == "12"


def test_worker_backend_episode_pinning_shares_outputs():
    with _make_client(worker_fixtures.build_process_chain_envs) as client:
        rid_up = _create_trial(client, "upstream", episode_id="E1")["trial_runtime_id"]
        assert (
            client.post(
                f"/trials/{rid_up}/submit", json={"answer": "hello"}
            ).status_code
            == 200
        )
        client.delete(f"/trials/{rid_up}")

        # Downstream, same episode -> same pinned worker -> sees upstream's output.
        rid_down = _create_trial(client, "downstream", episode_id="E1")[
            "trial_runtime_id"
        ]
        prompt = client.get(f"/trials/{rid_down}/prompt").json()["prompt"]
        assert "hello" in prompt

        client.delete(f"/trials/{rid_down}")
        # Closing the episode frees the pinned worker for reuse.
        assert client.delete("/episodes/E1").status_code == 200


def test_worker_backend_episodes_are_isolated():
    with _make_client(worker_fixtures.build_process_chain_envs, pool_size=2) as client:
        rid_e1 = _create_trial(client, "upstream", episode_id="E1")["trial_runtime_id"]
        client.post(f"/trials/{rid_e1}/submit", json={"answer": "hello"})
        rid_e2 = _create_trial(client, "upstream", episode_id="E2")["trial_runtime_id"]
        client.post(f"/trials/{rid_e2}/submit", json={"answer": "world"})

        down_e1 = _create_trial(client, "downstream", episode_id="E1")[
            "trial_runtime_id"
        ]
        prompt_e1 = client.get(f"/trials/{down_e1}/prompt").json()["prompt"]

    assert "hello" in prompt_e1
    assert "world" not in prompt_e1


def test_process_env_without_build_envs_falls_back_in_process():
    """No build_envs -> no pool -> the process env still runs (in-process)."""
    app = create_benchmark_server(worker_fixtures.build_process_calc_envs())
    with TestClient(app) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]
        exec_resp = client.post(
            f"/trials/{rid}/tools/execute",
            json={
                "tool_name": "calc",
                "arguments": {"operation": "add", "x": 2, "y": 5},
            },
        )
        assert exec_resp.status_code == 200, exec_resp.text
        assert exec_resp.json()["result"]["result"] == "7"


# /trial_worker status endpoint (Phase 3): the scheduler reads this to decide
# whether "process" trials are worker-isolated (may overlap) or must serialise.
def test_trial_worker_endpoint_reports_active_pool():
    """With a builder + a process env, the endpoint advertises a live pool."""
    with _make_client(worker_fixtures.build_process_calc_envs, pool_size=2) as client:
        body = client.get("/trial_worker").json()
    assert body["active"] is True
    assert body["size"] == 2


def test_trial_worker_endpoint_inactive_without_pool():
    """No builder -> no pool -> the endpoint advertises the in-process backend."""
    app = create_benchmark_server(worker_fixtures.build_process_calc_envs())
    with TestClient(app) as client:
        body = client.get("/trial_worker").json()
    assert body["active"] is False
    assert body["size"] is None
