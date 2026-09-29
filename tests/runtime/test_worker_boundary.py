"""Portable checks for the data allowed to cross the worker boundary."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import cloudpickle
import pytest
from tests.agents.commit_session import start_session

from corral.agents.llm_planner import LLMPlanner
from corral.core.environment import Environment, EnvironmentSetup, Toolset
from corral.core.state import EnvironmentState, ExecutionState
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.core.transition import ToolExecutionResult
from corral.runtime import agent_worker, permissions
from corral.runtime.agent_worker import RemoteSession, _snapshot
from corral.runtime.tool_execution import (
    PreparedToolCall,
    execute_prepared_background_call,
)
from corral.tools.python_repl import create_python_repl_tool
from corral.workspace import WorkspaceFilesystem, build_terminal_tool


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
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


@pytest.mark.anyio
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


@pytest.fixture
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


def test_run_worker_defaults_to_no_workspace_access(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    accesses = []

    def run(kind, payload, assigned_workspace, workspace_fd, **kwargs):
        del payload, workspace_fd
        assert assigned_workspace == str(workspace)
        accesses.append((kind, kwargs["workspace_access"]))
        return {"kind": kind}, {}

    monkeypatch.setattr(permissions, "_enabled", True)
    monkeypatch.setattr(permissions, "_root", tmp_path)
    monkeypatch.setattr(permissions, "_run_worker", run)

    assert permissions.run_worker("tool", object(), str(workspace)) == {"kind": "tool"}
    assert permissions.run_worker("agent", object(), str(workspace)) == {
        "kind": "agent"
    }
    assert accesses == [("tool", "none"), ("agent", "scratch")]


def test_shell_payload_is_only_command_options(tmp_path, private_state, monkeypatch):
    terminal = build_terminal_tool(WorkspaceFilesystem(tmp_path))
    terminal.accidentally_captured_state = private_state
    requests = []

    def run(kind, payload, workspace, **kwargs):
        requests.append((kind, payload, workspace))
        return {"content": "public output"}

    monkeypatch.setattr(permissions, "run_worker", run)
    assert (
        permissions.execute_restricted_tool(terminal, {"command": "pwd"}, str(tmp_path))
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
            execute_prepared_background_call(
                PreparedToolCall.capture(
                    private_code_tool,
                    arguments,
                    workspace=str(tmp_path),
                    execution_kind="restricted",
                )
            )
        else:
            permissions.execute_restricted_tool(
                private_code_tool, arguments, str(tmp_path)
            )


@tool(hidden_args=["work_dir"], workspace_args=("work_dir",))
def workspace_tool(work_dir: str) -> str:
    """Return the public workspace binding."""
    return work_dir


def test_workspace_binding_is_rebuilt_without_private_inputs(tmp_path, monkeypatch):
    secret = str(uuid4())

    def run(kind, payload, workspace, **kwargs):
        assert secret.encode() not in permissions.serialize(payload)
        assert payload[1] == {"work_dir": "/workspace"}
        return {"content": "/workspace"}

    monkeypatch.setattr(permissions, "run_worker", run)
    assert (
        execute_prepared_background_call(
            PreparedToolCall.capture(
                workspace_tool,
                {"work_dir": secret},
                workspace=str(tmp_path),
                execution_kind="restricted",
            )
        )
        == "/workspace"
    )


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
        execute_prepared_background_call(
            PreparedToolCall.capture(private_result, {}, workspace=str(tmp_path))
        )


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


def test_worker_response_ceiling_is_configurable(monkeypatch):
    """The control-reply ceiling is tunable in either direction."""
    monkeypatch.delenv("CORRAL_MAX_WORKER_RESPONSE_BYTES", raising=False)
    assert (
        permissions.max_worker_response_bytes()
        == permissions.DEFAULT_MAX_WORKER_RESPONSE_BYTES
    )

    monkeypatch.setenv("CORRAL_MAX_WORKER_RESPONSE_BYTES", str(4 * 1024**3))
    assert permissions.max_worker_response_bytes() == 4 * 1024**3
    # The session channel carries snapshots, so it follows the same setting.
    assert agent_worker._max_message() == 4 * 1024**3


def test_bulk_ceiling_is_separate_and_larger_than_the_reply_ceiling(monkeypatch):
    """Bulk payloads spool to disk, so they get their own, far higher ceiling.

    Keeping them apart is what lets the reply ceiling stay small enough to be a
    real memory guard while an environment's session grows as large as it needs.
    """
    monkeypatch.delenv("CORRAL_MAX_WORKER_BULK_BYTES", raising=False)
    monkeypatch.delenv("CORRAL_MAX_WORKER_RESPONSE_BYTES", raising=False)
    assert (
        permissions.max_worker_bulk_bytes() == permissions.DEFAULT_MAX_WORKER_BULK_BYTES
    )
    assert permissions.max_worker_bulk_bytes() > permissions.max_worker_response_bytes()

    monkeypatch.setenv("CORRAL_MAX_WORKER_BULK_BYTES", str(2 * 1024**3))
    assert permissions.max_worker_bulk_bytes() == 2 * 1024**3


@pytest.mark.parametrize("value", ["nonsense", "0", "-1"])
def test_bulk_ceiling_rejects_unusable_values(monkeypatch, value):
    monkeypatch.setenv("CORRAL_MAX_WORKER_BULK_BYTES", value)
    with pytest.raises(ValueError, match="CORRAL_MAX_WORKER_BULK_BYTES"):
        permissions.max_worker_bulk_bytes()


def test_control_reply_carries_a_length_prefix():
    """The prefix is what lets the controller refuse a reply before reading it."""
    framed = permissions.framed_reply({"ok": True, "result": {"content": "hello"}})
    (length,) = permissions._REPLY_PREFIX.unpack(
        framed[: permissions._REPLY_PREFIX.size]
    )
    body = framed[permissions._REPLY_PREFIX.size :]
    assert length == len(body)
    assert json.loads(body)["result"]["content"] == "hello"


def test_send_bulk_refuses_when_no_channel_is_installed():
    """Outside a worker there is no bulk descriptor, and that must not pass quietly."""
    assert permissions._bulk_fd is None
    with pytest.raises(RuntimeError, match="no bulk channel"):
        permissions.send_bulk("checkpoint", "payload")


def test_send_bulk_frames_name_and_payload(tmp_path):
    """A record names its payload and states its length before the bytes."""
    spool = tmp_path / "bulk"
    descriptor = os.open(spool, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        permissions.set_bulk_channel(descriptor)
        permissions.send_bulk("checkpoint", "opaque-base64")
    finally:
        permissions.set_bulk_channel(None)
        os.close(descriptor)
    written = spool.read_bytes()
    header = permissions._BULK_RECORD.size
    name_length, length = permissions._BULK_RECORD.unpack(written[:header])
    assert written[header : header + name_length] == b"checkpoint"
    assert written[header + name_length :] == b"opaque-base64"
    assert length == len(b"opaque-base64")


@pytest.mark.parametrize("value", ["nonsense", "0", "-1"])
def test_worker_response_ceiling_rejects_unusable_values(monkeypatch, value):
    monkeypatch.setenv("CORRAL_MAX_WORKER_RESPONSE_BYTES", value)
    with pytest.raises(ValueError, match="CORRAL_MAX_WORKER_RESPONSE_BYTES"):
        permissions.max_worker_response_bytes()


def test_repl_tool_carries_its_own_response_ceiling():
    """An environment can raise the ceiling without changing it globally."""
    repl = create_python_repl_tool(name="PythonREPL", max_response_bytes=1024**3)
    assert repl.max_response_bytes == 1024**3
    assert create_python_repl_tool(name="PythonREPL").max_response_bytes is None
