"""AI Scientist checkpoints retain physical branch state across trial import."""

import json

import anyio
import pytest
from anyio.from_thread import BlockingPortal

from corral.agents import AIScientistAgent
from corral.agents.ai_scientist.agent import _BranchSessionRegistry
from corral.agents.ai_scientist.search.nodes import ResearchStage
from corral.agents.ai_scientist.state import ScientistState, TaskFormulation
from corral.agents.schema import AgentOutcome
from corral.core.action import Action
from corral.core.environment import Environment, EnvironmentSetup, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.observability import NoOpObserver
from corral.orchestration import ContinuationSource, RunTaskInput, RuntimeRegistry
from corral.persistence import ShardedCommitStore
from corral.run import execute_task
from corral.runtime import agent_worker
from corral.runtime.agent_worker import RemoteSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("use_remote_session", [False, True])
async def test_child_imports_scientist_branch_without_repeating_completed_tool(
    tmp_path, monkeypatch, use_remote_session
):
    calls = []

    if use_remote_session:

        def run_worker(kind, payload, _workspace, **_kwargs):
            assert kind == "agent"
            agent, (host, port, token, handle, snapshot) = payload
            remote = RemoteSession(host, port, token, handle, snapshot)
            return anyio.run(agent.run_session, remote).model_dump(mode="json")

        monkeypatch.setattr(agent_worker.permissions, "run_worker", run_worker)

    def measure(value: int) -> str:
        """Record one scalar measurement."""
        calls.append(value)
        return str(value)

    task = TaskDefinition(
        name="science",
        description="measure",
        tools=["measure"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
        setup_fn=lambda _env, _state: EnvironmentSetup(
            hidden_arguments={"secret": "private-checkpoint-value"},
            values={"nested_artifact": {"blob_ref": environment_blob.blob_ref}},
        ),
    )
    environment = Environment(
        "science",
        task,
        base_work_dir=str(tmp_path / "workspaces"),
        toolset=Toolset(pool={"measure": tool(measure)}),
    )
    store = ShardedCommitStore(tmp_path / "runs")
    environment_blob = await store.workspace_manager("source").artifact_store.put_bytes(
        b"private environment artifact"
    )

    class SourceAgent(AIScientistAgent):
        def __init__(self, *, resume=False, **kwargs):
            super().__init__(model="test-model", **kwargs)
            self.resume = resume

        async def run_session(self, session):
            if use_remote_session and not isinstance(session, RemoteSession):
                return await agent_worker.run_agent(self, session)
            if self.resume:
                return await super().run_session(session)
            async with BlockingPortal() as portal:
                branches = _BranchSessionRegistry(session, portal)
                handle = await anyio.to_thread.run_sync(branches.create_branch)
                written = await anyio.to_thread.run_sync(
                    lambda: handle.execute(
                        Action(
                            name="write_file",
                            arguments={
                                "path": "/workspace/measurement.txt",
                                "content": "saved",
                            },
                        )
                    )
                )
                assert written.success
                result = await anyio.to_thread.run_sync(
                    lambda: handle.execute(
                        Action(name="measure", arguments={"value": 7})
                    )
                )
                assert result.success
                branch = await anyio.to_thread.run_sync(
                    lambda: branches.checkpoint(handle.execution_id)
                )
                assert "environment" not in branch
                assert "private-checkpoint-value" not in json.dumps(branch)
                if use_remote_session:
                    branch_session = branches.session(handle.execution_id)
                    with pytest.raises(RuntimeError, match="issued checkpoint"):
                        await branch_session.restore_branch_state(
                            checkpoint={"blob_ref": "sha256:" + "0" * 64}
                        )
                    with pytest.raises(RuntimeError, match="issued checkpoint"):
                        await branch_session.restore_branch_state(
                            checkpoint={
                                "environment": {},
                                "workspace": branch["workspace"],
                            }
                        )
            state = ScientistState(
                task_prompt="measure",
                tools=[],
                formulation=TaskFormulation(
                    objective="measure", required_answer="number"
                ),
            )
            state.current_stage = ResearchStage.COMPLETE
            await session.set_agent_state(
                "ai_scientist",
                AIScientistAgent._state_payload(
                    state,
                    (),
                    status="partial",
                    branches=[
                        {"branch_id": "branch_0001", "head_node_id": None, **branch}
                    ],
                ),
            )
            submitted = await session.execute(
                Action(name="submit_answer", arguments={"answer": "source"})
            )
            assert submitted.success
            return AgentOutcome(status="completed", answer="source")

    source_agent = SourceAgent()
    source_registry = RuntimeRegistry(
        agents={"agent": source_agent},
        environments={"science": environment},
        workspace_manager_factory=store.workspace_manager,
    )
    source = RunTaskInput(
        execution_id="source",
        task_id="science",
        environment_id="science",
        agent_id="agent",
        max_iterations=10,
    )
    try:
        source_result = await execute_task(
            task=source,
            state_store=store,
            registry=source_registry,
            observer=NoOpObserver(),
        )
        assert source_result.status == "submitted", source_result.error
        assert calls == [7]
        checkpoint_hash = None
        async for commit in store.for_execution("source").iter_commits("main"):
            if (
                commit.event.type == "agent.state_updated"
                and commit.event.namespace == "ai_scientist"
            ):
                checkpoint_hash = commit.hash
        assert checkpoint_hash is not None
        await store.close_execution("source")

        class NoOldCalls:
            call_count = 0
            token_count = 0

            def generate(self, *_args, **_kwargs):
                raise AssertionError("old scientific model calls must not repeat")

        scientist = SourceAgent(resume=True, model_gateway=NoOldCalls())
        child_registry = RuntimeRegistry(
            agents={"agent": scientist},
            environments={"science": environment},
            workspace_manager_factory=store.workspace_manager,
        )
        monkeypatch.setattr(
            "corral.agents.ai_scientist.agent.FinalSynthesizer.generate",
            lambda *_args, **_kwargs: "42",
        )
        child = RunTaskInput(
            execution_id="child",
            task_id="science",
            environment_id="science",
            agent_id="agent",
            max_iterations=10,
            continuation=ContinuationSource(
                trial=str(store.execution_dir("source")),
                commit_hash=checkpoint_hash,
            ),
        )
        child_result = await execute_task(
            task=child,
            state_store=store,
            registry=child_registry,
            observer=NoOpObserver(),
        )
        assert child_result.status == "submitted", child_result.error
        assert (
            await store.workspace_manager("child").artifact_store.get_bytes(
                environment_blob.blob_ref
            )
            == b"private environment artifact"
        )
        assert calls == [7]
        restored_files = list((tmp_path / "workspaces").rglob("measurement.txt"))
        assert any(path.read_text() == "saved" for path in restored_files)
    finally:
        source_registry.close()
        await store.aclose()
