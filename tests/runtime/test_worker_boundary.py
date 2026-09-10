"""Portable checks for the data allowed to cross the worker boundary."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import cloudpickle
import pytest
from tests.agents.commit_session import start_session

from corral.agents.llm_planner import LLMPlanner
from corral.core.environment import Environment, Toolset
from corral.core.state import EnvironmentState, ExecutionState
from corral.core.task import EnvironmentSetup, TaskDefinition
from corral.core.tool import tool
from corral.core.transition import ToolExecutionResult
from corral.runtime import permissions
from corral.runtime.agent_worker import RemoteSession, _snapshot
from corral.workspace import WorkspaceFilesystem, build_terminal_tool


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio()
async def test_planner_delegates_without_reopening_packaged_prompts(monkeypatch):
    planner = LLMPlanner(model="offline-test")
    payload = permissions.serialize(planner)
    prompt_store = Mock(side_effect=PermissionError("packaged prompts are private"))
    monkeypatch.setattr("corral.agents.base_agent.PromptStore", prompt_store)
    planner = cloudpickle.loads(payload)
    model_call = AsyncMock(
        side_effect=[
            SimpleNamespace(
                content="Use the executor to submit the answer.",
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            ),
            SimpleNamespace(
                content=(
                    "<action>submit_answer</action>"
                    '<action_input>{"answer":"42"}</action_input>'
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2},
            ),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    task = TaskDefinition(
        name="planner-task",
        description="Compute the answer",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    session = await start_session(
        Environment(
            "planner-task", task, toolset=Toolset(pool={}, workspace_factory=None)
        ),
        max_iterations=2,
    )
    try:
        result = await planner.run_session(session)
        assert result.status == "completed"
        assert result.answer == session.submission == "42"
        assert result.metadata["executor"] == "ReActAgent"
        assert result.usage.llm_calls == model_call.await_count == 2
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3
        prompt_store.assert_not_called()
    finally:
        await session._test_commit_store.aclose()


@pytest.mark.anyio()
async def test_agent_snapshot_contains_no_private_projection():
    secret = str(uuid4())
    task = TaskDefinition(
        name="public-task",
        description="Public instructions",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        scoring_inputs={"answer": secret},
        submission_format={},
        setup_fn=lambda env, state: EnvironmentSetup(
            hidden_arguments={"answer": secret}
        ),
    )
    session = await start_session(
        Environment(
            "public-task", task, toolset=Toolset(pool={}, workspace_factory=None)
        )
    )
    try:
        await session.record_message({"role": "user", "content": "public history"})
        session.previous_state = session.state
        session._last_score = {"score": 0.25, "trial_id": "attempt", "private": secret}
        snapshot = _snapshot(session)
        assert secret not in json.dumps(snapshot)
        assert (
            not {"state", "previous_state", "context", "runtime_actor"}
            & snapshot.keys()
        )
        assert snapshot["previous_evaluation"] == {"score": 0.25, "trial_id": "attempt"}
        assert snapshot["previous_messages"][-1]["content"] == "public history"
        remote = RemoteSession("localhost", 0, "token", "handle", snapshot)
        assert not hasattr(remote, "state")
        assert not hasattr(remote, "previous_state")
    finally:
        await session._test_commit_store.aclose()


@pytest.fixture()
def private_state():
    return ExecutionState(
        execution_id="private",
        branch_id="main",
        through_commit_hash="a" * 64,
        environment=EnvironmentState(
            values={"hidden_arguments": {"answer": str(uuid4())}}
        ),
    )


def test_serialization_rejects_private_objects_in_agent_closures(private_state):
    class CapturingAgent:
        async def run_session(self, session):
            return private_state.submission

    with pytest.raises(ValueError, match="private controller object"):
        permissions.serialize(CapturingAgent())
    with pytest.raises(ValueError, match="private controller object"):
        permissions.serialize({"nested": [private_state]})


def test_shell_payload_is_only_command_options(tmp_path, private_state, monkeypatch):
    terminal = build_terminal_tool(WorkspaceFilesystem(tmp_path))
    terminal.accidentally_captured_state = private_state
    environment = SimpleNamespace(workspace_path=str(tmp_path), private=private_state)
    requests = []

    def run(kind, payload, workspace, **kwargs):
        requests.append((kind, payload, workspace))
        return {"content": "public output"}

    monkeypatch.setattr(permissions, "run_worker", run)
    assert (
        permissions.execute_tool(
            environment, private_state, terminal, {"command": "pwd"}
        )
        == "public output"
    )
    assert requests == [("terminal", {"command": "pwd"}, str(tmp_path))]


@tool(hidden_args=["secret"])
def private_code_tool(code: str, secret: str) -> str:
    """An unsafe tool that must never receive hidden inputs in Docker."""
    return code + secret


@pytest.mark.parametrize("background", [False, True])
def test_untrusted_tools_with_private_inputs_fail_closed(
    tmp_path, monkeypatch, background
):
    monkeypatch.setattr(
        permissions,
        "run_worker",
        lambda *args, **kwargs: pytest.fail("worker must not start"),
    )
    arguments = {"code": "print('public')", "secret": str(uuid4())}
    with pytest.raises(PermissionError, match="cannot receive hidden arguments"):
        if background:
            permissions.execute_job(private_code_tool, arguments, str(tmp_path))
        else:
            permissions.execute_tool(
                SimpleNamespace(workspace_path=str(tmp_path)),
                None,
                private_code_tool,
                arguments,
            )


@tool(hidden_args=["work_dir"], workspace_args=("work_dir",))
def workspace_tool(work_dir: str) -> str:
    """Return the public workspace binding."""
    return work_dir


def test_workspace_binding_is_rebuilt_without_private_inputs(tmp_path, monkeypatch):
    secret = str(uuid4())

    def run(kind, payload, workspace, **kwargs):
        assert secret.encode() not in permissions.serialize(payload)
        assert payload[1] == {"work_dir": str(tmp_path)}
        return {"content": str(tmp_path)}

    monkeypatch.setattr(permissions, "run_worker", run)
    assert permissions.execute_job(
        workspace_tool, {"work_dir": secret}, str(tmp_path)
    ) == str(tmp_path)


def test_public_schema_rejects_injected_hidden_arguments():
    assert (
        permissions.visible_argument_error(private_code_tool, {"code": "public"})
        is None
    )
    assert "secret" in permissions.visible_argument_error(
        private_code_tool, {"code": "public", "secret": "override"}
    )


def test_background_result_cannot_publish_private_state(tmp_path):
    @tool(trusted=True)
    def private_result() -> ToolExecutionResult:
        """Return a private transition, which is valid only in the foreground."""
        return ToolExecutionResult(
            content="public", environment={"private": str(uuid4())}
        )

    with pytest.raises(ValueError, match="must execute in the foreground"):
        permissions.execute_job(private_result, {}, str(tmp_path))


def test_node_creation_and_copy_reject_symlink_ancestors(tmp_path):
    parent = tmp_path / "trial"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private data")
    nodes = parent / permissions.NODE_WORKSPACE_DIR
    nodes.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        permissions.create_node_workspace(str(parent), str(parent))
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(OSError):
        permissions.copy_workspace(nodes / "nested", destination)
    assert not list(destination.iterdir())
    assert (outside / "secret.txt").read_text() == "private data"
