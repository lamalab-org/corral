"""Tests for per-trial *runtimes* (PR 2 of `make_efficiency.md`).

These cover the second stage: an isolated runtime per trial so repeated trials
of the *same* task no longer share a single mutable server-side environment.

* `Environment.for_trial()` builds a fresh, workspace-namespaced runtime;
* `POST /tasks/{id}/trials` mints one and returns its MCP url;
* the runtime-scoped REST endpoints operate on that runtime only;
* two runtimes of one task are fully isolated (tool calls don't cross over);
* the shared `/trials` MCP mount dispatches by `trial_runtime_id`;
* `DELETE` frees the runtime.
"""

from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import Field

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import create_benchmark_server
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import tool

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def calc(
    operation: Literal["add", "mul"] = Field(description="operation to perform"),
    x: float = Field(description="first operand"),
    y: float = Field(default=1.0, description="second operand"),
) -> str:
    """Perform a basic math operation."""
    return str(x + y if operation == "add" else x * y)


def _make_task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        name=task_id,
        description="a task",
        tools=["calc"],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
    )


def _build_environments(*task_ids: str, base_work_dir: str = "") -> dict:
    """Plain (non-subclassed) environments so `for_trial` reconstructs cleanly."""
    tasks = {task_id: _make_task(task_id) for task_id in task_ids}
    toolset = Toolset(
        pool={"calc": tool(calc)},
        # No workspace tools unless a base dir is given.
        workspace_factory=None if not base_work_dir else Toolset().workspace_factory,
    )
    return build_environments(tasks, base_work_dir=base_work_dir, toolset=toolset)


def _create_trial(client: TestClient, task_id: str) -> dict:
    resp = client.post(f"/tasks/{task_id}/trials", json={"trial_index": 0})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_for_trial_is_isolated_and_namespaced(tmp_path):
    template = _build_environments("task_a", base_work_dir=str(tmp_path))["task_a"]

    r1 = template.for_trial("tr_1")
    r2 = template.for_trial("tr_2")

    # Distinct state objects and distinct, id-namespaced workspaces.
    assert r1.state is not r2.state
    assert r1.state is not template.state
    assert r1.current_work_dir != r2.current_work_dir
    assert "tr_1" in r1.current_work_dir
    assert "tr_2" in r2.current_work_dir

    # A tool call on one runtime does not leak into the other.
    r1.call_tool("calc", {"operation": "add", "x": 1, "y": 2})
    assert r1.state.tool_statistics()["total_calls"] == 1
    assert r2.state.tool_statistics()["total_calls"] == 0


def test_template_workspace_naming_unchanged(tmp_path):
    """Templates (no runtime id) keep the historical `{task}_trial_{n}` name."""
    template = _build_environments("task_a", base_work_dir=str(tmp_path))["task_a"]
    assert template.current_work_dir.endswith("task_a_trial_0")


def test_create_trial_returns_runtime_descriptor():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        data = _create_trial(client, "task_a")

    assert data["task_id"] == "task_a"
    assert data["trial_runtime_id"].startswith("tr_")
    assert data["mcp_url"] == f"/trials/{data['trial_runtime_id']}/mcp"


def test_promote_trial_artifacts_overlays_the_scored_workspace(tmp_path):
    envs = _build_environments("task_a", base_work_dir=str(tmp_path))
    with TestClient(create_benchmark_server(envs)) as client:
        canonical = _create_trial(client, "task_a")
        branch = _create_trial(client, "task_a")
        canonical_workspace = Path(canonical["workspace"])
        branch_workspace = Path(branch["workspace"])
        (canonical_workspace / "keep.txt").write_text("keep", encoding="utf-8")
        artifact = branch_workspace / "results" / "final_results.json"
        artifact.parent.mkdir()
        artifact.write_text('{"value": 42}', encoding="utf-8")

        response = client.post(
            f"/trials/{branch['trial_runtime_id']}/artifacts/promote",
            json={"destination_trial_runtime_id": canonical["trial_runtime_id"]},
        )

        assert response.status_code == 200, response.text
        assert response.json()["files"] == ["results/final_results.json"]
        promoted = canonical_workspace / "results" / "final_results.json"
        assert promoted.read_text(encoding="utf-8") == '{"value": 42}'
        assert (canonical_workspace / "keep.txt").read_text() == "keep"


def test_create_trial_unknown_task_404():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        resp = client.post("/tasks/nope/trials", json={})
    assert resp.status_code == 404


def test_trial_execute_and_submit():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
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

        submit_resp = client.post(f"/trials/{rid}/submit", json={"answer": "7"})
        assert submit_resp.status_code == 200, submit_resp.text
        completion = submit_resp.json()
        assert completion["score"] == 1.0
        assert completion["surrendered"] is False
        assert "tool_statistics" in completion["state"]


def test_two_runtimes_of_same_task_are_isolated():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        rid_a = _create_trial(client, "task_a")["trial_runtime_id"]
        rid_b = _create_trial(client, "task_a")["trial_runtime_id"]
        assert rid_a != rid_b

        # Record a tool call in runtime A only.
        client.post(
            f"/trials/{rid_a}/tools/execute",
            json={"tool_name": "calc", "arguments": {"operation": "add", "x": 1}},
        )

        status_a = client.get(f"/trials/{rid_a}/status").json()
        status_b = client.get(f"/trials/{rid_b}/status").json()

    assert status_a["tool_statistics"]["total_calls"] == 1
    assert status_b["tool_statistics"]["total_calls"] == 0


def test_trial_prompt_and_tools_endpoints():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        prompt = client.get(f"/trials/{rid}/prompt").json()["prompt"]
        assert "a task" in prompt

        tools = client.get(f"/trials/{rid}/tools", params={"verbosity": "brief"}).json()
        assert [t["function"]["name"] for t in tools["tools"]] == ["calc"]


def test_close_trial_frees_runtime():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        assert client.delete(f"/trials/{rid}").status_code == 200
        # A second close and any further access now 404.
        assert client.delete(f"/trials/{rid}").status_code == 404
        assert client.get(f"/trials/{rid}/status").status_code == 404


def _mcp_rpc(client: TestClient, rid: str, method: str, params: dict) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = client.post(f"/trials/{rid}/mcp", json=body, headers=_MCP_HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_trial_mcp_tools_list_and_call():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        rid = _create_trial(client, "task_a")["trial_runtime_id"]

        listed = _mcp_rpc(client, rid, "tools/list", {})
        assert [t["name"] for t in listed["result"]["tools"]] == ["calc"]

        called = _mcp_rpc(
            client,
            rid,
            "tools/call",
            {"name": "calc", "arguments": {"operation": "mul", "x": 3, "y": 4}},
        )
        assert called["result"]["isError"] is False
        assert called["result"]["content"][0]["text"] == "12"


def test_trial_mcp_unknown_runtime_lists_no_tools():
    envs = _build_environments("task_a")
    with TestClient(create_benchmark_server(envs)) as client:
        listed = _mcp_rpc(client, "tr_does_not_exist", "tools/list", {})
    assert listed["result"]["tools"] == []


# Episode-scoped dependency stores (PR 3)
#
# A chained runtime bound to an `episode_id` reads its upstream siblings'
# outputs through the episode's shared store; different episodes never share.
def _build_chain() -> dict:
    """A two-task chain: `downstream` consumes `upstream`'s answer."""
    upstream = TaskDefinition(
        name="upstream",
        description="produce a value",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        # Store the answer verbatim so the downstream prompt shows it unchanged.
        resolve_answer=False,
    )
    downstream = TaskDefinition(
        name="downstream",
        description="consume the value",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        input_map={"from_upstream": InputRef("upstream", "answer")},
    )
    return build_environments(
        {"upstream": upstream, "downstream": downstream},
        toolset=Toolset(workspace_factory=None),
    )


def _create_episode_trial(client: TestClient, task_id: str, episode_id: str) -> str:
    resp = client.post(
        f"/tasks/{task_id}/trials",
        json={"episode_id": episode_id, "trial_index": 0},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["trial_runtime_id"]


def test_episode_store_shares_outputs_across_a_chain():
    with TestClient(create_benchmark_server(_build_chain())) as client:
        # Upstream submits inside episode E1.
        rid_up = _create_episode_trial(client, "upstream", "E1")
        assert (
            client.post(
                f"/trials/{rid_up}/submit", json={"answer": "hello"}
            ).status_code
            == 200
        )

        # Downstream, in the *same* episode, resolves the shared output.
        rid_down = _create_episode_trial(client, "downstream", "E1")
        prompt = client.get(f"/trials/{rid_down}/prompt").json()["prompt"]

    assert "hello" in prompt


def test_episodes_are_isolated_from_each_other():
    with TestClient(create_benchmark_server(_build_chain())) as client:
        # Same task, two episodes, two different upstream answers.
        rid_e1 = _create_episode_trial(client, "upstream", "E1")
        client.post(f"/trials/{rid_e1}/submit", json={"answer": "hello"})
        rid_e2 = _create_episode_trial(client, "upstream", "E2")
        client.post(f"/trials/{rid_e2}/submit", json={"answer": "world"})

        prompt_e1 = client.get(
            f"/trials/{_create_episode_trial(client, 'downstream', 'E1')}/prompt"
        ).json()["prompt"]
        prompt_e2 = client.get(
            f"/trials/{_create_episode_trial(client, 'downstream', 'E2')}/prompt"
        ).json()["prompt"]

    assert "hello" in prompt_e1
    assert "world" not in prompt_e1
    assert "world" in prompt_e2
    assert "hello" not in prompt_e2


def test_trial_without_episode_has_isolated_store():
    """A runtime created with no `episode_id` never shares dependency outputs."""
    app = create_benchmark_server(_build_chain())
    # Return the 500 response instead of re-raising, so we can assert on it.
    with TestClient(app, raise_server_exceptions=False) as client:
        rid_up = _create_episode_trial(client, "upstream", "E1")
        client.post(f"/trials/{rid_up}/submit", json={"answer": "hello"})

        # No episode id -> its own empty store -> upstream is unresolved.
        resp = client.post("/tasks/downstream/trials", json={"trial_index": 0})
        rid_down = resp.json()["trial_runtime_id"]
        prompt_resp = client.get(f"/trials/{rid_down}/prompt")

    # Resolving an unsatisfied dependency errors, not a silent cross-episode read.
    assert prompt_resp.status_code == 500


def test_close_episode_frees_the_store():
    app = create_benchmark_server(_build_chain())
    with TestClient(app, raise_server_exceptions=False) as client:
        rid_up = _create_episode_trial(client, "upstream", "E1")
        client.post(f"/trials/{rid_up}/submit", json={"answer": "hello"})

        assert client.delete("/episodes/E1").status_code == 200
        # A second delete 404s: the store is gone.
        assert client.delete("/episodes/E1").status_code == 404

        # A downstream runtime re-opening E1 now starts from an empty store.
        rid_down = _create_episode_trial(client, "downstream", "E1")
        assert client.get(f"/trials/{rid_down}/prompt").status_code == 500


class _ProcessEnv(Environment):
    """A stateful env declaring it must be process-isolated / serialised."""

    DEFAULT_CONCURRENCY = "process"


def test_environment_defaults_to_thread_concurrency():
    env = _build_environments("task_a")["task_a"]
    assert env.concurrency == "thread"
    # The mode is carried onto every per-trial runtime.
    assert env.for_trial("tr_1").concurrency == "thread"


def test_environment_subclass_declares_concurrency():
    tasks = {"task_a": _make_task("task_a")}
    toolset = Toolset(pool={"calc": tool(calc)}, workspace_factory=None)
    env = build_environments(tasks, toolset=toolset, env_cls=_ProcessEnv)["task_a"]
    assert env.concurrency == "process"
    # A subclass-declared mode survives the definition-only for_trial rebuild.
    assert env.for_trial("tr_1").concurrency == "process"


def test_environment_concurrency_kwarg_overrides_default():
    tasks = {"task_a": _make_task("task_a")}
    toolset = Toolset(pool={"calc": tool(calc)}, workspace_factory=None)
    env = build_environments(tasks, toolset=toolset, concurrency="serial")["task_a"]
    assert env.concurrency == "serial"


def test_environment_rejects_invalid_concurrency():
    tasks = {"task_a": _make_task("task_a")}
    toolset = Toolset(pool={"calc": tool(calc)}, workspace_factory=None)
    with pytest.raises(ValueError, match="concurrency"):
        build_environments(tasks, toolset=toolset, concurrency="nonsense")


def test_concurrency_endpoint_reports_modes():
    tasks = {"safe": _make_task("safe"), "stateful": _make_task("stateful")}
    toolset = Toolset(pool={"calc": tool(calc)}, workspace_factory=None)
    envs = {
        **build_environments({"safe": tasks["safe"]}, toolset=toolset),
        **build_environments(
            {"stateful": tasks["stateful"]}, toolset=toolset, env_cls=_ProcessEnv
        ),
    }
    with TestClient(create_benchmark_server(envs)) as client:
        resp = client.get("/concurrency")

    assert resp.status_code == 200
    assert resp.json() == {"safe": "thread", "stateful": "process"}
