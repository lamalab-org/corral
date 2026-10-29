"""Regression tests for workspace restoration in async agent sessions."""

from pathlib import Path
from unittest.mock import Mock

import anyio
import pytest
from anyio.from_thread import BlockingPortal

from corral.agents.ai_scientist.agent import _BranchSessionRegistry
from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSession, run_agent_session
from corral.core import Action, ActorRef, AgentStarted, CommitRequest
from corral.core.action import submit_answer_action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool_catalog import state_tool_catalog
from corral.persistence import SQLiteCommitStore
from corral.report.projection import _token_usage, _tool_statistics
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults
from corral.workspace import AbsoluteWorkspaceFilesystem, build_terminal_tool


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def workspace_session(tmp_path):
    environment = Environment(
        "workspace-task",
        TaskDefinition(
            name="workspace-task",
            description="Read the restored workspace.",
            tools=[],
            scoring_fn=lambda _answer: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
        base_work_dir=str(tmp_path / "workspaces"),
        task_execution_id="workspace-execution",
    )
    (Path(environment.workspace_path) / "notes.txt").write_text(
        "restored contents", encoding="utf-8"
    )
    runtime_actor = ActorRef(kind="runtime", actor_id="corral", run_id="runtime")
    actor = ActorRef(kind="agent", actor_id="test-agent", run_id="agent")
    event = await anyio.to_thread.run_sync(
        lambda: environment.initial_event(
            execution_id="workspace-execution", actor_id=actor.actor_id
        )
    )
    async with SQLiteCommitStore(
        tmp_path / "commits.sqlite3", "workspace-execution"
    ) as store:
        root = await store.append(
            CommitRequest(
                request_id="execution:started",
                branch_id="main",
                based_on_hash=None,
                author=runtime_actor,
                event=event,
            )
        )
        await store.append(
            CommitRequest(
                request_id="agent:started",
                branch_id="main",
                based_on_hash=root.hash,
                author=runtime_actor,
                event=AgentStarted(agent_run_id=actor.run_id, agent_id=actor.actor_id),
            )
        )
        state = await store.materialize("main")
        assert "notes.txt" in state.workspace.files
        yield AgentSession(
            environment,
            state,
            actor=actor,
            runtime_actor=runtime_actor,
            state_store=store,
            max_iterations=10,
        )


@pytest.mark.anyio
@pytest.mark.parametrize("after_submission", [False, True])
async def test_sdk_tools_persist_and_appear_in_verbose_reports(
    workspace_session, after_submission
):
    session = workspace_session
    if after_submission:
        assert (await session.execute(submit_answer_action("done"))).success

    native = Action(id="native-read", name="Read", arguments={"path": "notes.txt"})
    failed = Action(id="native-shell", name="Bash", arguments={"command": "false"})
    # SDK call ids differ from the ids assigned by the Corral MCP server.
    echoed = Action(id="sdk-submit", name="submit_answer", arguments={"answer": "done"})
    prefixed = Action(id="claude-submit", name="mcp__corral__submit_answer")
    await session.record_messages(
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call.to_tool_call() for call in (native, echoed, prefixed, failed)
                ],
            },
            {"role": "tool", "tool_call_id": echoed.id, "content": "submitted"},
            {"role": "tool", "tool_call_id": prefixed.id, "content": "submitted"},
        ),
        usage={"input_tokens": 10, "output_tokens": 2},
    )
    # An in-flight SDK tool must not block Corral actions or become recovery work.
    assert native.id not in session.state.actions
    if not after_submission:
        assert (await session.execute(submit_answer_action("done"))).success
    await session.record_messages(
        (
            {
                "role": "tool",
                "tool_call_id": native.id,
                "name": "tool_result",
                "content": "contents",
            },
            {
                "role": "tool",
                "tool_call_id": failed.id,
                "content": "exit 1",
                "is_error": True,
                "duration_ms": 25,
            },
        )
    )

    state = await session.state_store.materialize(session.branch_id)
    assert state.usage.tool_calls == 3
    assert state.usage.llm_calls == 1
    assert state.usage.input_tokens == 10
    assert state.tool_statistics == {"submit_answer": 1, "Read": 1, "Bash": 1}
    messages = state.model_dump(mode="json")["conversations"][session.actor.run_id]
    calls = [call for message in messages for call in message.get("tool_calls", ())]
    assert {call["function"]["name"] for call in calls} == {
        "submit_answer",
        "Read",
        "Bash",
    }
    assert len(calls) == 3

    trial = TaskTrialResult(
        task_id="task",
        trial_id="trial",
        score=1.0,
        state=state.model_dump(mode="json"),
        tool_statistics=_tool_statistics(state),
        messages=messages,
        token_usage=_token_usage(state),
    )
    report = BenchmarkResult(
        task_results={"task": TaskTrialResults(task_id="task", trials=[trial])},
        verbose=True,
    )
    data = report._prepare_report_data({})["task_results"]["task"]["trials"][0]
    by_name = {call["tool_name"]: call for call in data["tool_calls"]}
    assert by_name["Read"]["arguments"] == {"path": "notes.txt"}
    assert by_name["Read"]["result"] == "contents"
    assert by_name["Read"]["status"] == "completed"
    assert by_name["Bash"]["status"] == "failed"
    assert by_name["Bash"]["error_message"] == "exit 1"
    assert by_name["Bash"]["duration"] == 0.025
    assert data["token_usage"]["tool_calls"] == 3
    report.verbose = False
    quiet = report._prepare_report_data({})["task_results"]["task"]["trials"][0]
    assert "messages" not in quiet
    assert "tool_calls" not in quiet


@pytest.mark.anyio
async def test_run_agent_session_restores_nonempty_workspace(workspace_session):
    session = workspace_session
    restored_file = Path(session.workspace) / "notes.txt"
    restored_file.unlink()

    class Reader:
        async def run_session(self, session):
            assert restored_file.read_text(encoding="utf-8") == "restored contents"
            return AgentOutcome(status="iteration_limit", error="budget ended")

    result = await run_agent_session(
        Reader(),
        session.environment,
        session.state,
        actor=session.actor,
        state_store=session.state_store,
        max_iterations=10,
    )

    assert result.state.workspace == session.state.workspace


@pytest.mark.anyio
async def test_fork_branch_restores_nonempty_workspace(workspace_session):
    session = workspace_session

    fork = await session.fork_branch(branch_id="experiment")

    assert fork.workspace != session.workspace
    assert fork.state.workspace == session.state.workspace
    assert (
        Path(fork.workspace, "notes.txt").read_text(encoding="utf-8")
        == "restored contents"
    )


@pytest.mark.anyio
async def test_fork_branch_preserves_custom_terminal_catalog(tmp_path):
    def workspace_tools(workspace):
        terminal = build_terminal_tool(AbsoluteWorkspaceFilesystem(workspace))
        terminal.description += " Use the task's relative output paths."
        return {"terminal": terminal}

    environment = Environment(
        "custom-terminal",
        TaskDefinition(
            name="custom-terminal",
            description="Exercise a task-specific terminal description.",
            tools=[],
            scoring_fn=lambda _answer: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
        base_work_dir=str(tmp_path / "workspaces"),
        task_execution_id="custom-terminal-execution",
        toolset=Toolset(workspace_factory=workspace_tools),
    )
    runtime_actor = ActorRef(kind="runtime", actor_id="corral", run_id="runtime")
    actor = ActorRef(kind="agent", actor_id="test-agent", run_id="agent")
    async with SQLiteCommitStore(
        tmp_path / "commits.sqlite3", "custom-terminal-execution"
    ) as store:
        event = await anyio.to_thread.run_sync(
            lambda: environment.initial_event(
                execution_id="custom-terminal-execution", actor_id=actor.actor_id
            )
        )
        root = await store.append(
            CommitRequest(
                request_id="execution:started",
                branch_id="main",
                based_on_hash=None,
                author=runtime_actor,
                event=event,
            )
        )
        await store.append(
            CommitRequest(
                request_id="agent:started",
                branch_id="main",
                based_on_hash=root.hash,
                author=runtime_actor,
                event=AgentStarted(agent_run_id=actor.run_id, agent_id=actor.actor_id),
            )
        )
        state = await store.materialize("main")
        session = AgentSession(
            environment,
            state,
            actor=actor,
            runtime_actor=runtime_actor,
            state_store=store,
            max_iterations=10,
        )
        fork = await session.fork_branch(branch_id="experiment")

        assert (
            fork.environment.tool_catalog_snapshot().fingerprint
            == state_tool_catalog(state).fingerprint
        )
        fork.environment.validate_state_tool_catalog(fork.state)


@pytest.mark.anyio
async def test_scientist_nodes_have_independent_files_and_promote_the_winner(
    workspace_session, monkeypatch
):
    parent = workspace_session
    Path(parent.workspace, "live.txt").write_text("live contents")
    shutdown_jobs = Mock()
    monkeypatch.setattr(parent.environment, "shutdown_jobs", shutdown_jobs)

    async with BlockingPortal() as portal:
        sessions = _BranchSessionRegistry(parent, portal)
        first = await anyio.to_thread.run_sync(sessions.create_branch)
        second = await anyio.to_thread.run_sync(sessions.create_branch)

        async def write(branch, content):
            response = await anyio.to_thread.run_sync(
                lambda: branch.execute(
                    Action(
                        name="write_file",
                        arguments={
                            "path": "/workspace/model.txt",
                            "content": content,
                        },
                    )
                )
            )
            assert response.success, response

        await write(first, "winning model")
        await write(second, "losing model")
        clone = await anyio.to_thread.run_sync(
            sessions.clone_branch, first.execution_id
        )
        assert Path(clone.workspace, "model.txt").read_text() == "winning model"
        await write(clone, "changed clone")
        assert len({branch.workspace for branch in (first, second, clone)}) == 3
        closed = []
        for branch in (first, second, clone):
            assert Path(branch.workspace).parent == Path(
                parent.workspace, ".corral-nodes"
            )
            branch_session = sessions.session(branch.execution_id)
            assert branch_session.environment is not parent.environment
            assert Path(branch.workspace, "live.txt").read_text() == "live contents"
            assert not Path(branch.workspace, ".corral-nodes").exists()
            mocked = Mock()
            monkeypatch.setattr(branch_session.environment, "shutdown_jobs", mocked)
            closed.append(mocked)
        assert Path(first.workspace, "model.txt").read_text() == "winning model"
        assert Path(second.workspace, "model.txt").read_text() == "losing model"
        assert not Path(parent.workspace, "model.txt").exists()
        assert parent.state.tool_statistics == {}

        response = await anyio.to_thread.run_sync(
            lambda: first.execute(
                Action(
                    name="read_file",
                    arguments={
                        "path": "/workspace/../"
                        + Path(second.workspace).name
                        + "/model.txt"
                    },
                )
            )
        )
        assert not response.success
        promoted = await anyio.to_thread.run_sync(
            sessions.promote_branch_artifacts, first.execution_id, parent.execution_id
        )
        assert set(promoted["files"]) == {"notes.txt", "live.txt", "model.txt"}
        assert Path(parent.workspace, "model.txt").read_text() == "winning model"
        snapshot = await parent.environment.workspace_manager.snapshot(parent.workspace)
        assert set(snapshot.files) == {"notes.txt", "live.txt", "model.txt"}
        for branch in (first, second, clone):
            await anyio.to_thread.run_sync(sessions.close_branch, branch.execution_id)
        for mocked in closed:
            mocked.assert_called_once_with()
        shutdown_jobs.assert_not_called()
