"""End-to-end Temporal tests for standalone and benchmark execution."""

import asyncio
from datetime import datetime, timezone

import pytest
from temporalio.testing import WorkflowEnvironment

from corral.agents.schema import AgentOutcome
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.environment import Environment, Toolset
from corral.core.task import InputRef, TaskDefinition
from corral.orchestration import (
    BenchmarkWorkflowInput,
    CorralActivities,
    RuntimeRegistry,
    TaskWorkflowInput,
    TemporalBenchmarkExecutor,
    TemporalTaskExecutor,
    create_worker,
    execute_task,
)
from corral.orchestration.models import RunTaskInput
from corral.persistence import JSONLStateStore
from corral.run import BenchmarkTaskMetadata, CorralRunner
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class SubmitAgent:
    model = "test-model"

    async def run_session(self, session):
        result = await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        assert result.success is True
        return AgentOutcome(status="completed", answer="42")


class PreviousStateAgent(SubmitAgent):
    def __init__(self):
        self.seen_previous_hash = None

    async def run_session(self, session):
        assert session.previous_state is not None
        assert session.previous_state.submission == "42"
        self.seen_previous_hash = session.previous_state.state_hash
        return await super().run_session(session)


class ConcurrentAgent(SubmitAgent):
    """Probe that can finish only when one shared instance runs concurrently."""

    def __init__(self):
        self.active = 0
        self.peak_active = 0
        self.both_started = asyncio.Event()

    async def run_session(self, session):
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if self.active == 2:
            self.both_started.set()
        try:
            await self.both_started.wait()
            return await super().run_session(session)
        finally:
            self.active -= 1


class _RecordingSpan:
    def update(self, **_kwargs):
        return None

    def end(self, error=None):
        del error


class RecordingObserver:
    def __init__(self):
        self.observations = []

    def start(self, observation):
        self.observations.append(observation)
        return _RecordingSpan()

    def flush(self):
        return None


def _environment(task_id: str, dependency: str | None = None) -> Environment:
    task = TaskDefinition(
        name=task_id,
        description="submit 42",
        tools=[],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        input_map=(
            {"upstream": InputRef(dependency)} if dependency is not None else {}
        ),
        resolve_answer=False,
    )
    return Environment(
        task_id,
        task,
        toolset=Toolset(workspace_factory=None),
    )


@pytest.mark.anyio()
async def test_activity_restores_prior_state_for_reflective_agents(tmp_path):
    store = JSONLStateStore(tmp_path / "prior-states.jsonl")
    environment = _environment("reflective")
    prior = await TaskRuntime(store).run(
        SubmitAgent(),
        environment.for_task("prior"),
        execution_id="prior",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_iterations=1,
    )
    agent = PreviousStateAgent()
    registry = RuntimeRegistry(
        agents={"agent": agent},
        environments={"reflective": environment},
    )
    registry.record_evaluation(
        "reflective",
        "prior",
        {"state_hash": prior.state_hash, "score": 1.0},
    )

    try:
        result = await CorralActivities(store, registry).run_task(
            RunTaskInput(
                execution_id="current",
                task_id="reflective",
                environment_id="reflective",
                agent_id="agent",
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat(),
                max_iterations=1,
            )
        )
    finally:
        registry.close()
        store.close()

    assert result.submission == "42"
    assert agent.seen_previous_hash == prior.state_hash


@pytest.mark.anyio()
async def test_activities_do_not_serialize_requests_by_agent_id(tmp_path):
    store = JSONLStateStore(tmp_path / "concurrent-states.jsonl")
    agent = ConcurrentAgent()
    registry = RuntimeRegistry(
        agents={"shared": agent},
        environments={"concurrent": _environment("concurrent")},
    )
    activities = CorralActivities(store, registry)
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()

    def request(execution_id):
        return RunTaskInput(
            execution_id=execution_id,
            task_id="concurrent",
            environment_id="concurrent",
            agent_id="shared",
            started_at=started_at,
            max_iterations=1,
        )

    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(
                activities.run_task(request("concurrent-1")),
                activities.run_task(request("concurrent-2")),
            ),
            timeout=2,
        )
    finally:
        registry.close()
        store.close()

    assert first.submission == "42"
    assert second.submission == "42"
    assert agent.peak_active == 2


@pytest.mark.anyio()
async def test_temporal_owns_task_and_benchmark_lifecycle(tmp_path):
    store = JSONLStateStore(tmp_path / "states.jsonl")
    registry = RuntimeRegistry(
        agents={"agent": SubmitAgent()},
        environments={
            "upstream": _environment("upstream"),
            "downstream": _environment("downstream", "upstream"),
        },
    )
    observer = RecordingObserver()
    activities = CorralActivities(store, registry, observer)

    try:
        async with (
            await WorkflowEnvironment.start_time_skipping() as temporal,
            create_worker(
                temporal.client,
                task_queue="corral-test",
                activities=activities,
            ),
        ):
            task_executor = TemporalTaskExecutor(
                temporal.client,
                store,
                "corral-test",
            )
            request = TaskWorkflowInput(
                execution_id="standalone-1",
                task_id="upstream",
                environment_id="upstream",
                agent_id="agent",
                max_iterations=1,
            )
            state = await execute_task(executor=task_executor, task=request)
            retried = await execute_task(executor=task_executor, task=request)

            assert state.runtime.status == "submitted"
            assert state.submission == "42"
            assert retried.state_hash == state.state_hash

            benchmark = await TemporalBenchmarkExecutor(
                temporal.client,
                "corral-test",
            ).execute(
                BenchmarkWorkflowInput(
                    benchmark_run_id="benchmark-1",
                    task_ids=("upstream", "downstream"),
                    trials_per_task=2,
                    agent_by_task={
                        "upstream": "agent",
                        "downstream": "agent",
                    },
                    environment_by_task={
                        "upstream": "upstream",
                        "downstream": "downstream",
                    },
                    dependency_graph={
                        "upstream": (),
                        "downstream": ("upstream",),
                    },
                    max_iterations_by_task={
                        "upstream": 1,
                        "downstream": 1,
                    },
                    max_parallel=2,
                    max_parallel_per_task=2,
                    rounds_per_run=1,
                    evaluate=True,
                )
            )

            assert len(benchmark.trials) == 4
            assert all(result.output_ready for result in benchmark.trials)
            assert all(
                result.evaluation is not None and result.evaluation.score == 1.0
                for result in benchmark.trials
            )

            runner = CorralRunner(
                TemporalBenchmarkExecutor(temporal.client, "corral-test"),
                {
                    "upstream": BenchmarkTaskMetadata(
                        agent_id="agent",
                        environment_id="upstream",
                        max_iterations=1,
                        model="test-model",
                    ),
                    "downstream": BenchmarkTaskMetadata(
                        agent_id="agent",
                        environment_id="downstream",
                        dependencies=("upstream",),
                        max_iterations=1,
                        model="test-model",
                    ),
                },
                state_store=store,
            )
            report = await runner.run(
                "benchmark-through-runner",
                trials_per_task=1,
                max_parallel=2,
                max_parallel_per_task=1,
            )

            assert report.verbosity == "brief"
            assert report.task_results["upstream"].trials[0].score == 1.0
            assert report.task_results["upstream"].trials[0].state["metadata"][
                "model"
            ] == {"name": "test-model"}
            assert report.task_results["downstream"].trials[0].output == {
                "answer": "42"
            }
            benchmark_evaluations = [
                observation
                for observation in observer.observations
                if observation.name == "task.evaluate"
                and observation.context.benchmark_run_id == "benchmark-1"
            ]
            assert len(benchmark_evaluations) == 4
            assert {
                observation.context.task_id for observation in benchmark_evaluations
            } == {
                "upstream",
                "downstream",
            }
            assert all(
                observation.context.temporal_workflow_id
                for observation in benchmark_evaluations
            )
    finally:
        registry.close()
        store.close()
