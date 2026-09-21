"""End-to-end tests for standalone and benchmark execution."""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from corral import run
from corral.agents.schema import AgentOutcome
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.environment import Environment, Toolset
from corral.core.task import InputRef, TaskDefinition
from corral.observability import NoOpObserver
from corral.orchestration import (
    AgentRuntimeDefinition,
    DockerSandboxSpec,
    EnvironmentRuntimeDefinition,
    EvaluationRef,
    RetryPolicy,
    RuntimeRegistry,
    SandboxMode,
    SandboxProfile,
    StateRef,
    execute_task,
)
from corral.orchestration.evaluation import evaluate_task
from corral.orchestration.models import EvaluateTaskInput, RunTaskInput
from corral.persistence import ShardedCommitStore, SQLiteCommitStore
from corral.run import BenchmarkTaskMetadata, CorralRunner
from corral.runtime import TaskRuntime


@pytest.fixture
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
        self.seen_previous_hash = session.previous_state.through_commit_hash
        return await super().run_session(session)


class RecordingPreviousAttemptAgent(SubmitAgent):
    def __init__(self):
        self.previous_evaluations = []
        self.previous_states = []

    async def run_session(self, session):
        self.previous_evaluations.append(session.previous_evaluation)
        self.previous_states.append(session.previous_state)
        return await super().run_session(session)


class FileSubmitAgent:
    model = "test-model"

    async def run_session(self, session):
        written = await session.execute(
            Action(
                name="write_file",
                arguments={
                    "path": "/workspace/answer.txt",
                    "content": "durable",
                },
            )
        )
        assert written.success is True
        submitted = await session.execute(
            Action(
                name=SUBMIT_ANSWER_TOOL_NAME,
                arguments={"answer": "/workspace/answer.txt"},
            )
        )
        assert submitted.success is True
        return AgentOutcome(status="completed", answer="/workspace/answer.txt")


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

    def record_commit(self, commit, *, context=None):
        del commit, context

    def flush(self):
        return None


class RecordingDockerLauncher:
    def __init__(self):
        self.requests = []

    async def run(self, request, *, observation_context=None):
        self.requests.append((request, observation_context))
        return StateRef(
            commit_hash="d" * 64,
            execution_id=request.execution_id,
            branch_id="main",
            sequence=4,
            status="submitted",
            agent_steps=1,
            submission="42",
            output={"answer": "42"},
        )


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


@pytest.mark.anyio
async def test_execution_restores_prior_projection_for_reflective_agents(tmp_path):
    store = SQLiteCommitStore(tmp_path / "prior-commits.sqlite3")
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
        {"commit_hash": prior.through_commit_hash, "score": 1.0},
    )

    try:
        result = await execute_task(
            state_store=store,
            registry=registry,
            task=RunTaskInput(
                execution_id="current",
                task_id="reflective",
                environment_id="reflective",
                agent_id="agent",
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat(),
                max_iterations=1,
            ),
        )
    finally:
        registry.close()
        await store.aclose()

    assert result.submission == "42"
    assert agent.seen_previous_hash == prior.through_commit_hash


@pytest.mark.anyio
async def test_benchmark_evaluates_prior_attempt_before_starting_the_next(tmp_path):
    store = SQLiteCommitStore(tmp_path / "prior-attempt-commits.sqlite3")
    agent = RecordingPreviousAttemptAgent()
    registry = RuntimeRegistry(
        agents={"agent": agent},
        environments={"reflective": _environment("reflective")},
    )
    runner = CorralRunner(
        registry,
        {
            "reflective": BenchmarkTaskMetadata(
                agent_id="agent",
                environment_id="reflective",
                max_iterations=1,
            )
        },
        state_store=store,
        observer=NoOpObserver(),
    )

    try:
        report = await runner.run("reflective-benchmark", trials_per_task=2)
    finally:
        registry.close()
        await store.aclose()

    assert [trial.score for trial in report.all_results] == [1.0, 1.0]
    assert agent.previous_evaluations[0] is None
    assert agent.previous_states[0] is None
    assert agent.previous_evaluations[1]["score"] == 1.0
    assert agent.previous_evaluations[1]["trial_id"] == (
        "reflective-benchmark:reflective:0"
    )
    assert agent.previous_states[1].submission == "42"


@pytest.mark.anyio
async def test_executions_do_not_serialize_requests_by_agent_id(tmp_path):
    store = SQLiteCommitStore(tmp_path / "concurrent-commits.sqlite3")
    agent = ConcurrentAgent()
    registry = RuntimeRegistry(
        agents={"shared": agent},
        environments={"concurrent": _environment("concurrent")},
    )
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
                execute_task(
                    task=request("concurrent-1"), state_store=store, registry=registry
                ),
                execute_task(
                    task=request("concurrent-2"), state_store=store, registry=registry
                ),
            ),
            timeout=2,
        )
    finally:
        registry.close()
        await store.aclose()

    assert first.submission == "42"
    assert second.submission == "42"
    assert agent.peak_active == 2


@pytest.mark.anyio
async def test_evaluation_uses_durable_snapshot_after_live_workspace_is_deleted(
    tmp_path,
):
    task = TaskDefinition(
        name="file-task",
        description="write a result",
        tools=[],
        scoring_fn=lambda path: float(Path(path).read_text() == "durable"),
        submission_format={"answer": "path"},
        resolve_answer=True,
    )
    environment = Environment(
        "file-task", task, base_work_dir=str(tmp_path / "live-workspaces")
    )
    store = ShardedCommitStore(tmp_path / ".corral")
    registry = RuntimeRegistry(
        agents={"agent": FileSubmitAgent()},
        environments={"file-task": environment},
        workspace_manager_factory=store.workspace_manager,
    )
    execution_id = "durable-file-evaluation"

    try:
        state = await execute_task(
            state_store=store,
            registry=registry,
            task=RunTaskInput(
                execution_id=execution_id,
                task_id="file-task",
                environment_id="file-task",
                agent_id="agent",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            ),
        )
        live_workspace = Path(
            registry.environment("file-task", execution_id).workspace_path
        )
        shutil.rmtree(live_workspace)

        evaluation = await evaluate_task(
            EvaluateTaskInput(
                execution_id=execution_id,
                environment_id="file-task",
                commit_hash=state.commit_hash,
                task_id="file-task",
            ),
            state_store=store,
            registry=registry,
            observer=NoOpObserver(),
        )
    finally:
        registry.close()
        await store.aclose()

    assert not live_workspace.exists()
    assert evaluation.score == 1.0


@pytest.mark.anyio
async def test_direct_task_and_benchmark_lifecycle(tmp_path):
    store = SQLiteCommitStore(tmp_path / "commits.sqlite3")
    registry = RuntimeRegistry(
        agents={"agent": SubmitAgent()},
        environments={
            "upstream": _environment("upstream"),
            "downstream": _environment("downstream", "upstream"),
        },
    )
    observer = RecordingObserver()
    try:
        request = RunTaskInput(
            execution_id="standalone-1",
            task_id="upstream",
            environment_id="upstream",
            agent_id="agent",
            max_iterations=1,
        )
        state = await execute_task(task=request, state_store=store, registry=registry)
        retried = await execute_task(task=request, state_store=store, registry=registry)
        assert state.status == "submitted"
        assert state.submission == "42"
        assert retried.commit_hash == state.commit_hash

        runner = CorralRunner(
            registry,
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
            observer=observer,
        )
        report = await runner.run(
            "benchmark-1",
            trials_per_task=2,
            max_parallel=2,
            max_parallel_per_task=2,
        )
        assert len(report.all_results) == 4
        assert all(trial.score == 1.0 for trial in report.all_results)
        assert report.verbosity == "brief"
        assert report.task_results["upstream"].trials[0].state["task"]["model"] == {
            "name": "test-model"
        }
        assert report.task_results["downstream"].trials[0].output == {"answer": "42"}
        evaluations = [o for o in observer.observations if o.name == "task.evaluate"]
        assert len(evaluations) == 4
        assert {o.context.task_id for o in evaluations} == {"upstream", "downstream"}
        assert all(o.context.benchmark_run_id == "benchmark-1" for o in evaluations)
    finally:
        registry.close()
        await store.aclose()


@pytest.mark.anyio
async def test_execution_forwards_docker_runtime_definitions(tmp_path):
    store = SQLiteCommitStore(tmp_path / "docker-payload.sqlite3")
    registry = RuntimeRegistry(
        agents={"agent": SubmitAgent()},
        environments={"task": _environment("task")},
    )
    docker_launcher = RecordingDockerLauncher()
    try:
        result = await execute_task(
            state_store=store,
            registry=registry,
            docker_launcher=docker_launcher,
            task=RunTaskInput(
                execution_id="docker-payload",
                task_id="task",
                environment_id="task",
                agent_id="agent",
                sandbox=SandboxProfile(
                    mode=SandboxMode.DOCKER,
                    docker=DockerSandboxSpec(
                        image="corral:test", image_digest="sha256:" + "c" * 64
                    ),
                ),
                agent_runtime=AgentRuntimeDefinition(name="react", model="test-model"),
                environment_runtime=EnvironmentRuntimeDefinition(
                    name="samplemath", options={"level": 1}
                ),
            ),
        )
    finally:
        registry.close()
        await store.aclose()
    assert result.submission == "42"
    request, context = docker_launcher.requests[0]
    assert request.sandbox.mode == "docker"
    assert request.sandbox.docker.memory == "4g"
    assert request.environment_runtime.options == {"level": 1}
    assert context.execution_id == "docker-payload"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("limits", "expected_peak"),
    [
        ({"max_parallel": 2}, 2),
        ({"max_parallel_per_task": 1}, 2),
        ({"max_parallel_by_model": {"test-model": 1}}, 1),
        ({"max_parallel_by_environment": {"shared": 1}}, 1),
    ],
)
async def test_benchmark_enforces_concurrency_limits(
    monkeypatch, limits, expected_peak
):
    active = 0
    peak = 0

    async def launch(*, task, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return await RecordingDockerLauncher().run(task)
        finally:
            active -= 1

    monkeypatch.setattr(run, "execute_task", launch)
    runner = CorralRunner(
        None,
        {
            task_id: BenchmarkTaskMetadata("agent", "shared", model="test-model")
            for task_id in ("first", "second")
        },
        state_store=None,
        observer=NoOpObserver(),
    )
    options = {"max_parallel": 8, "max_parallel_per_task": 4, **limits}
    report = await runner.run(
        "concurrent", trials_per_task=4, evaluate=False, **options
    )
    assert len(report.all_results) == 8
    assert peak == expected_peak
    assert active == 0


@pytest.mark.anyio
async def test_evaluation_does_not_consume_execution_capacity(monkeypatch):
    evaluation_started = asyncio.Event()
    second_started = asyncio.Event()
    release_evaluations = asyncio.Event()

    async def launch(*, task, **kwargs):
        if task.task_id == "second":
            second_started.set()
            await evaluation_started.wait()
        return await RecordingDockerLauncher().run(task)

    async def evaluate(request, **kwargs):
        if request.task_id == "first":
            evaluation_started.set()
            await second_started.wait()
        await release_evaluations.wait()
        return EvaluationRef(request.commit_hash, 1.0, {}, "test")

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {
            task_id: BenchmarkTaskMetadata("agent", task_id)
            for task_id in ("first", "second")
        },
        state_store=None,
        observer=NoOpObserver(),
    )
    benchmark = asyncio.create_task(
        runner.run(
            "overlap",
            max_parallel=1,
            max_parallel_evaluations=1,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(evaluation_started.wait(), second_started.wait()),
            timeout=2,
        )
        assert not benchmark.done()
    finally:
        release_evaluations.set()

    report = await asyncio.wait_for(benchmark, timeout=2)
    assert [trial.score for trial in report.all_results] == [1.0, 1.0]


@pytest.mark.anyio
async def test_dependency_execution_does_not_wait_for_upstream_evaluation(
    monkeypatch,
):
    downstream_started = asyncio.Event()

    async def launch(*, task, **kwargs):
        if task.task_id == "downstream":
            assert task.dependency_outputs == {"upstream": {"answer": "42"}}
            downstream_started.set()
        return await RecordingDockerLauncher().run(task)

    async def evaluate(request, **kwargs):
        if request.task_id == "upstream":
            await downstream_started.wait()
        return EvaluationRef(request.commit_hash, 1.0, {}, "test")

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {
            "upstream": BenchmarkTaskMetadata("agent", "upstream"),
            "downstream": BenchmarkTaskMetadata("agent", "downstream", ("upstream",)),
        },
        state_store=None,
        observer=NoOpObserver(),
    )

    report = await asyncio.wait_for(
        runner.run(
            "dependencies",
            max_parallel=1,
            max_parallel_evaluations=1,
        ),
        timeout=2,
    )

    assert downstream_started.is_set()
    assert all(trial.score == 1.0 for trial in report.all_results)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_parallel_evaluations", "expected_peak"),
    [(None, 2), (1, 1)],
)
async def test_benchmark_enforces_evaluation_concurrency_limit(
    monkeypatch, max_parallel_evaluations, expected_peak
):
    active = 0
    peak = 0

    async def launch(*, task, **kwargs):
        return await RecordingDockerLauncher().run(task)

    async def evaluate(request, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return EvaluationRef(request.commit_hash, 1.0, {}, "test")
        finally:
            active -= 1

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {"task": BenchmarkTaskMetadata("agent", "environment")},
        state_store=None,
        observer=NoOpObserver(),
    )

    report = await runner.run(
        "evaluation-limit",
        trials_per_task=4,
        max_parallel=2,
        max_parallel_per_task=4,
        max_parallel_evaluations=max_parallel_evaluations,
    )

    assert len(report.all_results) == 4
    assert peak == expected_peak
    assert active == 0


@pytest.mark.anyio
async def test_benchmark_enforces_environment_evaluation_concurrency_limit(
    monkeypatch,
):
    active = 0
    peak = 0

    async def launch(*, task, **kwargs):
        return await RecordingDockerLauncher().run(task)

    async def evaluate(request, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return EvaluationRef(request.commit_hash, 1.0, {}, "test")
        finally:
            active -= 1

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {
            task_id: BenchmarkTaskMetadata(
                "agent",
                f"{task_id}-environment",
                environment_runtime=EnvironmentRuntimeDefinition("shared"),
            )
            for task_id in ("first", "second")
        },
        state_store=None,
        observer=NoOpObserver(),
    )

    report = await runner.run(
        "environment-evaluation-limit",
        trials_per_task=2,
        max_parallel=4,
        max_parallel_per_task=2,
        max_parallel_evaluations=4,
        max_parallel_evaluations_by_environment={"shared": 1},
    )

    assert len(report.all_results) == 4
    assert peak == 1
    assert active == 0


@pytest.mark.anyio
async def test_benchmark_enforces_total_concurrency_limit(monkeypatch):
    active = 0
    peak = 0
    evaluation_started = asyncio.Event()

    async def launch(*, task, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if task.task_id == "second":
                await evaluation_started.wait()
            return await RecordingDockerLauncher().run(task)
        finally:
            active -= 1

    async def evaluate(request, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        evaluation_started.set()
        try:
            await asyncio.sleep(0.01)
            return EvaluationRef(request.commit_hash, 1.0, {}, "test")
        finally:
            active -= 1

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {
            task_id: BenchmarkTaskMetadata("agent", task_id)
            for task_id in ("first", "second")
        },
        state_store=None,
        observer=NoOpObserver(),
    )

    report = await asyncio.wait_for(
        runner.run(
            "total-limit",
            max_parallel=2,
            max_parallel_evaluations=2,
            max_parallel_total=2,
        ),
        timeout=2,
    )

    assert len(report.all_results) == 2
    assert peak == 2
    assert active == 0


@pytest.mark.anyio
async def test_failed_trial_blocks_only_its_own_descendants(monkeypatch):
    requests = []

    async def launch(*, task, **kwargs):
        requests.append(task)
        if task.task_id == "upstream" and task.trial_index == 0:
            raise RuntimeError("launch failed")
        return await RecordingDockerLauncher().run(task)

    monkeypatch.setattr(run, "execute_task", launch)
    runner = CorralRunner(
        None,
        {
            "upstream": BenchmarkTaskMetadata("agent", "upstream"),
            "downstream": BenchmarkTaskMetadata("agent", "downstream", ("upstream",)),
        },
        state_store=None,
        observer=NoOpObserver(),
    )
    report = await runner.run(
        "failures", trials_per_task=2, max_parallel=2, evaluate=False
    )
    assert report.task_results["upstream"].trials[0].error_message == "launch failed"
    assert report.task_results["downstream"].trials[0].state["unreachable"] is True
    assert report.task_results["downstream"].trials[1].output == {"answer": "42"}
    downstream = [request for request in requests if request.task_id == "downstream"]
    assert len(downstream) == 1
    assert downstream[0].trial_index == 1
    assert downstream[0].dependency_outputs == {"upstream": {"answer": "42"}}


@pytest.mark.anyio
async def test_evaluation_failure_preserves_downstream_output(monkeypatch):
    async def launch(*, task, **kwargs):
        return await RecordingDockerLauncher().run(task)

    async def fail_evaluation(*args, **kwargs):
        raise RuntimeError("scorer unavailable")

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", fail_evaluation)
    runner = CorralRunner(
        None,
        {
            "upstream": BenchmarkTaskMetadata("agent", "upstream"),
            "downstream": BenchmarkTaskMetadata("agent", "downstream", ("upstream",)),
        },
        state_store=None,
        observer=NoOpObserver(),
    )
    report = await runner.run(
        "evaluation", retry_policy=RetryPolicy(maximum_attempts=1)
    )
    assert all(trial.output == {"answer": "42"} for trial in report.all_results)
    assert all(
        trial.evaluation_error == "scorer unavailable" for trial in report.all_results
    )
    assert all(trial.error_message is None for trial in report.all_results)


@pytest.mark.anyio
async def test_task_retries_failed_launches(monkeypatch):
    attempts = 0
    requests = []

    async def launch(self, request, **kwargs):
        nonlocal attempts
        requests.append(request)
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary failure")
        return await RecordingDockerLauncher().run(request)

    monkeypatch.setattr(run.LocalTaskLauncher, "run", launch)
    result = await execute_task(
        task=RunTaskInput("retry", "task", "env", "agent"),
        registry=None,
        state_store=None,
        observer=NoOpObserver(),
        retry_policy=RetryPolicy(initial_interval_seconds=0.001),
    )
    assert result.submission == "42"
    assert attempts == 3
    assert all(request is requests[0] for request in requests)


@pytest.mark.anyio
async def test_benchmark_cancellation_finishes_active_trials(monkeypatch):
    started = asyncio.Event()
    active = 0

    async def launch(*args, **kwargs):
        nonlocal active
        active += 1
        if active == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    monkeypatch.setattr(run.LocalTaskLauncher, "run", launch)
    runner = CorralRunner(
        None,
        {"task": BenchmarkTaskMetadata("agent", "env")},
        state_store=None,
        observer=NoOpObserver(),
    )
    task = asyncio.create_task(
        runner.run(
            "cancel",
            trials_per_task=3,
            max_parallel=2,
            max_parallel_per_task=2,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert active == 0


@pytest.mark.anyio
async def test_benchmark_cancellation_drains_active_evaluation(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    calls = 0

    async def launch(*, task, **kwargs):
        return await RecordingDockerLauncher().run(task)

    async def evaluate(request, **kwargs):
        nonlocal active, calls
        calls += 1
        active += 1
        started.set()
        try:
            await release.wait()
            return EvaluationRef(request.commit_hash, 1.0, {}, "test")
        finally:
            active -= 1

    monkeypatch.setattr(run, "execute_task", launch)
    monkeypatch.setattr(run, "evaluate_task", evaluate)
    runner = CorralRunner(
        None,
        {"task": BenchmarkTaskMetadata("agent", "environment")},
        state_store=None,
        observer=NoOpObserver(),
    )
    benchmark = asyncio.create_task(
        runner.run(
            "cancel-evaluation",
            trials_per_task=2,
            max_parallel=2,
            max_parallel_per_task=1,
            max_parallel_evaluations=1,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    benchmark.cancel()
    await asyncio.sleep(0)
    assert not benchmark.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(benchmark, timeout=2)
    assert calls == 1
    assert active == 0
