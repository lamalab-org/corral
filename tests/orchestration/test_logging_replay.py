"""Temporal workflow logging remains quiet during deterministic replay."""

from __future__ import annotations

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer

from corral.agents.schema import AgentOutcome
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.orchestration import (
    CorralActivities,
    RuntimeRegistry,
    TaskWorkflowInput,
    TemporalTaskExecutor,
    create_worker,
    create_workflow_runner,
)
from corral.orchestration.executor import task_workflow_id
from corral.orchestration.workflows import TaskWorkflow
from corral.persistence import SQLiteCommitStore
from corral.report.logging import LoggingConfig, LogSinkConfig, configure_logging


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class _SubmitAgent:
    async def run_session(self, session):
        await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        return AgentOutcome(status="completed", answer="42")


def _environment() -> Environment:
    task = TaskDefinition(
        name="replay-logging",
        description="submit 42",
        tools=[],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    return Environment(
        "replay-logging",
        task,
        toolset=Toolset(workspace_factory=None),
    )


@pytest.mark.anyio()
async def test_temporal_replay_does_not_duplicate_workflow_logs(tmp_path):
    records = []
    configure_logging(
        LoggingConfig(
            console=False,
            sinks=(LogSinkConfig(records.append, level="DEBUG"),),
        )
    )
    store = SQLiteCommitStore(tmp_path / "commits.sqlite3")
    registry = RuntimeRegistry(
        agents={"agent": _SubmitAgent()},
        environments={"replay-logging": _environment()},
    )

    try:
        async with (
            await WorkflowEnvironment.start_time_skipping() as temporal,
            create_worker(
                temporal.client,
                task_queue="corral-replay-logging",
                activities=CorralActivities(store, registry),
            ),
        ):
            request = TaskWorkflowInput(
                execution_id="replay-logging-1",
                task_id="replay-logging",
                environment_id="replay-logging",
                agent_id="agent",
                max_iterations=1,
            )
            await TemporalTaskExecutor(
                temporal.client,
                store,
                "corral-replay-logging",
            ).execute(request)
            handle = temporal.client.get_workflow_handle(
                task_workflow_id(request.execution_id)
            )
            history = await handle.fetch_history()

        workflow_events_before = [
            item.record["extra"]["event"]
            for item in records
            if item.record["extra"]["event"].startswith("task_workflow.")
        ]
        assert workflow_events_before == [
            "task_workflow.started",
            "task_workflow.completed",
        ]

        await Replayer(
            workflows=[TaskWorkflow],
            workflow_runner=create_workflow_runner(),
        ).replay_workflow(history)

        workflow_events_after = [
            item.record["extra"]["event"]
            for item in records
            if item.record["extra"]["event"].startswith("task_workflow.")
        ]
        assert workflow_events_after == workflow_events_before
    finally:
        registry.close()
        store.close()
        configure_logging()
