"""Regression tests for workspace restoration in async agent sessions."""

from pathlib import Path

import anyio
import pytest

from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSession, run_agent_session
from corral.core import ActorRef, AgentStarted, CommitRequest
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
