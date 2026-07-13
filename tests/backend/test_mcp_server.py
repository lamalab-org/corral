"""Tests for the task-scoped MCP transport mounted on the benchmark server."""

from typing import Literal

from fastapi.testclient import TestClient
from pydantic import Field

from corral.backend.env import Environment
from corral.backend.mcp_server import execute_task_tool, task_mcp_tools
from corral.backend.server import create_benchmark_server
from corral.backend.task import TaskDefinition
from corral.backend.tool import tool
from corral.router.verbosity import ToolVerbosity

_DUMMY_TASK = TaskDefinition(
    name="dummy",
    description="dummy task",
    tools=[],
    scoring_fn=lambda answer: 1.0,
    submission_format={},
)

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


class DummyEnv(Environment):
    def __init__(self, task_id: str, base_work_dir: str = "", fs_manager=None):
        super().__init__(
            task_id, _DUMMY_TASK, base_work_dir, fs_manager=fs_manager
        )

    def get_task_prompt(self) -> str | list[dict]:
        return "dummy prompt"

    def score(self) -> float:
        return 1.0

    def configure_additional_apps(self):
        return "none"


def _env_with_calc(task_id: str) -> DummyEnv:
    env = DummyEnv(task_id=task_id, base_work_dir="", fs_manager=None)
    env.add_tool(tool(calc))
    return env


def documented(x: float = Field(description="operand")) -> str:
    """[BRIEF]Short summary.[/BRIEF]

    [DETAILED]A longer description only shown at higher verbosity.[/DETAILED]
    """
    return str(x)


def _env_with_documented(task_id: str) -> DummyEnv:
    env = DummyEnv(task_id=task_id, base_work_dir="", fs_manager=None)
    env.add_tool(tool(documented))
    return env


def _rpc(
    client: TestClient,
    task_id: str,
    method: str,
    params: dict,
    req_id: int = 1,
    query: str = "",
):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    url = f"/tasks/{task_id}/mcp{query}"
    response = client.post(url, json=body, headers=_MCP_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


# --- Pure helpers ---------------------------------------------------------


def test_task_mcp_tools_uses_to_mcp_schema():
    env = _env_with_calc("t")
    tools = task_mcp_tools(env, ToolVerbosity.FULL)

    assert len(tools) == 1
    calc = tools[0]
    assert calc.name == "calc"
    # `to_mcp` drops defaulted parameters from `required`.
    assert set(calc.inputSchema["required"]) == {"operation", "x"}
    assert "y" in calc.inputSchema["properties"]


def test_execute_task_tool_success():
    env = _env_with_calc("t")
    result = execute_task_tool(env, "t", "calc", {"operation": "add", "x": 2, "y": 5})
    assert result[0].text == "7"


def test_execute_task_tool_rejects_unknown_tool():
    env = _env_with_calc("t")
    result = execute_task_tool(env, "t", "does_not_exist", {})
    assert result.isError is True
    assert "not available for task 't'" in result.content[0].text


# --- End-to-end over the mounted Streamable-HTTP endpoint ------------------


def test_tools_list_over_mcp_endpoint():
    env = _env_with_calc("task_a")
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        data = _rpc(client, "task_a", "tools/list", {})

    names = [t["name"] for t in data["result"]["tools"]]
    assert names == ["calc"]


def test_tools_call_over_mcp_endpoint():
    env = _env_with_calc("task_a")
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        data = _rpc(
            client,
            "task_a",
            "tools/call",
            {"name": "calc", "arguments": {"operation": "mul", "x": 3, "y": 4}},
        )

    result = data["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"] == "12"


def test_tools_list_honors_request_verbosity():
    """The `?verbosity=` query param filters tool descriptions per request."""
    env = _env_with_documented("task_a")
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        brief = _rpc(client, "task_a", "tools/list", {}, query="?verbosity=brief")
        detailed = _rpc(client, "task_a", "tools/list", {}, query="?verbosity=detailed")

    brief_desc = brief["result"]["tools"][0]["description"]
    detailed_desc = detailed["result"]["tools"][0]["description"]
    assert brief_desc == "Short summary."
    # The detailed condition additionally includes the [DETAILED] section.
    assert "longer description" in detailed_desc
    assert detailed_desc != brief_desc


def test_tools_list_falls_back_to_mount_verbosity():
    """Without a query param, the server's mount-time verbosity is used."""
    env = _env_with_documented("task_a")
    # create_benchmark_server mounts MCP at the FULL default, which returns the
    # raw docstring (tags intact) rather than a filtered description.
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        data = _rpc(client, "task_a", "tools/list", {})

    desc = data["result"]["tools"][0]["description"]
    assert "[BRIEF]" in desc


def test_mcp_schema_endpoint_returns_tools_and_digest():
    env = _env_with_calc("task_a")
    with TestClient(create_benchmark_server({"task_a": env})) as client:
        resp = client.get("/tasks/task_a/tools/mcp", params={"verbosity": "brief"})
        assert resp.status_code == 200
        data = resp.json()

    assert [t["name"] for t in data["tools"]] == ["calc"]
    # Digest is stable and hex-encoded sha256.
    assert len(data["mcp_schema_sha256"]) == 64
    int(data["mcp_schema_sha256"], 16)


def test_mcp_endpoint_is_task_scoped():
    """A task's endpoint exposes only its own tools and rejects foreign ones."""
    envs = {"task_a": _env_with_calc("task_a"), "task_b": _env_with_calc("task_b")}
    # task_b has no tools registered.
    envs["task_b"] = DummyEnv(task_id="task_b", base_work_dir="", fs_manager=None)

    with TestClient(create_benchmark_server(envs)) as client:
        listed = _rpc(client, "task_b", "tools/list", {})
        assert listed["result"]["tools"] == []

        called = _rpc(
            client,
            "task_b",
            "tools/call",
            {"name": "calc", "arguments": {"operation": "add", "x": 1}},
        )
        assert called["result"]["isError"] is True
        assert (
            "not available for task 'task_b'" in called["result"]["content"][0]["text"]
        )
