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
from corral.core.environment import Environment
from corral.core.task import TaskDefinition
from corral.persistence import SQLiteCommitStore


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
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


@pytest.mark.anyio()
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


@pytest.mark.anyio()
async def test_fork_branch_restores_nonempty_workspace(workspace_session):
    session = workspace_session

    fork = await session.fork_branch(branch_id="experiment")

    assert fork.workspace != session.workspace
    assert fork.state.workspace == session.state.workspace
    assert (
        Path(fork.workspace, "notes.txt").read_text(encoding="utf-8")
        == "restored contents"
    )


@pytest.mark.anyio()
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
                        arguments={"path": "model.txt", "content": content},
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
                        "path": "../" + Path(second.workspace).name + "/model.txt"
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
