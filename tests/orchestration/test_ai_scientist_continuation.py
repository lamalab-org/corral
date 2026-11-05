"""AI Scientist checkpoints retain physical branch state across trial import."""

from pathlib import Path

import anyio
import pytest
from anyio.from_thread import BlockingPortal

from corral.agents import AIScientistAgent
from corral.agents.ai_scientist.agent import _BranchSessionRegistry
from corral.agents.ai_scientist.search.nodes import ResearchStage
from corral.agents.ai_scientist.state import ScientistState, TaskFormulation
from corral.agents.schema import AgentOutcome
from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.observability import NoOpObserver
from corral.orchestration import ContinuationSource, RunTaskInput, RuntimeRegistry
from corral.persistence import ShardedCommitStore
from corral.run import execute_task


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_child_imports_scientist_branch_without_repeating_completed_tool(
    tmp_path, monkeypatch
):
    calls = []

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
    )
    environment = Environment(
        "science",
        task,
        base_work_dir=str(tmp_path / "workspaces"),
        toolset=Toolset(pool={"measure": tool(measure)}),
    )
    store = ShardedCommitStore(tmp_path / "runs")

    class SourceAgent(AIScientistAgent):
        checkpoint_hash = None

        def __init__(self, *, resume=False, **kwargs):
            super().__init__(model="test-model", **kwargs)
            self.resume = resume

        async def run_session(self, session):
            if self.resume:
                return await super().run_session(session)
            async with BlockingPortal() as portal:
                branches = _BranchSessionRegistry(session, portal)
                handle = await anyio.to_thread.run_sync(branches.create_branch)
                Path(handle.workspace, "measurement.txt").write_text(
                    "saved", encoding="utf-8"
                )
                result = await anyio.to_thread.run_sync(
                    lambda: handle.execute(
                        Action(name="measure", arguments={"value": 7})
                    )
                )
                assert result.success
                branch = await anyio.to_thread.run_sync(
                    lambda: branches.checkpoint(handle.execution_id)
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
            self.checkpoint_hash = session.state.through_commit_hash
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
        assert source_result.status == "submitted"
        assert calls == [7]
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
                commit_hash=source_agent.checkpoint_hash,
            ),
        )
        child_result = await execute_task(
            task=child,
            state_store=store,
            registry=child_registry,
            observer=NoOpObserver(),
        )
        assert child_result.status == "submitted", child_result.error
        assert calls == [7]
        restored_files = list((tmp_path / "workspaces").rglob("measurement.txt"))
        assert any(path.read_text() == "saved" for path in restored_files)
    finally:
        source_registry.close()
        await store.aclose()
