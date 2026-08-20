"""Tests for the thin Temporal benchmark metadata adapter."""

from datetime import datetime, timedelta, timezone

import pytest

from corral.core import submit_answer_action
from corral.core.environment import Environment, Toolset
from corral.core.state import (
    ActionState,
    ExecutionState,
    RuntimeState,
    ToolInvocationState,
    UsageState,
)
from corral.core.task import InputRef, TaskDefinition
from corral.orchestration import (
    BenchmarkWorkflowResult,
    EvaluationRef,
    StateRef,
    TaskWorkflowResult,
)
from corral.run import BenchmarkTaskMetadata, CorralRunner


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class RecordingExecutor:
    def __init__(self, result: BenchmarkWorkflowResult) -> None:
        self.result = result
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return self.result


class MemoryCommitStore:
    def __init__(self, state: ExecutionState) -> None:
        self.state = state
        self.loaded = []

    def for_execution(self, execution_id: str):
        assert execution_id == self.state.execution_id
        return self

    async def materialize(self, branch_id: str, commit_hash: str) -> ExecutionState:
        self.loaded.append(commit_hash)
        assert branch_id == self.state.branch_id
        assert commit_hash == self.state.through_commit_hash
        return self.state


def _submitted_state() -> ExecutionState:
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    action = submit_answer_action("42", action_id="submit-1")
    invocation_id = "submit-invocation"
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="benchmark-1:upstream:0",
        branch_id="main",
        conversations={
            "main": (
                action.to_message(),
                {
                    "role": "tool",
                    "tool_call_id": action.id,
                    "name": action.name,
                    "content": "answer accepted",
                    "metadata": {
                        "status": "success",
                        "success": True,
                        "duration_ms": 5.0,
                    },
                },
            )
        },
        actions={
            action.id: ActionState(
                action=action,
                requested_by_run_id="main",
                status="completed",
                invocation_ids=(invocation_id,),
            )
        },
        tool_invocations={
            invocation_id: ToolInvocationState(
                invocation_id=invocation_id,
                action_id=action.id,
                requested_by_run_id="main",
                tool_name=action.name,
                started_commit_hash="b" * 64,
                status="completed",
                observation="answer accepted",
                duration_ms=5.0,
            )
        },
        usage_by_run={
            "main": UsageState(
                input_tokens=12,
                output_tokens=3,
                reasoning_tokens=2,
                llm_calls=1,
                tool_calls=1,
                agent_steps=1,
            )
        },
        runtime=RuntimeState(
            status="submitted",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=2),
        ),
        submission="42",
    )


def _metadata() -> dict[str, BenchmarkTaskMetadata]:
    return {
        "upstream": BenchmarkTaskMetadata(
            agent_id="agent-a",
            environment_id="env-upstream",
            max_iterations=3,
            model="model-a",
        ),
        "downstream": BenchmarkTaskMetadata(
            agent_id="agent-b",
            environment_id="env-downstream",
            dependencies=("upstream",),
            max_iterations=7,
            model="model-b",
            task_queue="wetlab",
        ),
    }


def _environment(task_id: str, *dependencies: str) -> Environment:
    return Environment(
        task_id,
        TaskDefinition(
            name=task_id,
            description=f"Run {task_id}",
            tools=[],
            scoring_fn=lambda answer: float(bool(answer)),
            submission_format={"answer": "string"},
            input_map={dependency: InputRef(dependency) for dependency in dependencies},
            resolve_answer=False,
        ),
        toolset=Toolset(workspace_factory=None),
    )


def test_runner_infers_metadata_from_environments_by_default():
    result = BenchmarkWorkflowResult("benchmark", (), 1, ())
    runner = CorralRunner(
        RecordingExecutor(result),
        environments={
            "upstream": _environment("upstream"),
            "downstream": _environment("downstream", "upstream"),
        },
        agent_id="tool-calling",
        model="openai/gpt-4o",
        max_iterations=12,
    )

    assert runner.tasks == {
        "upstream": BenchmarkTaskMetadata(
            agent_id="tool-calling",
            environment_id="upstream",
            model="openai/gpt-4o",
            max_iterations=12,
        ),
        "downstream": BenchmarkTaskMetadata(
            agent_id="tool-calling",
            environment_id="downstream",
            dependencies=("upstream",),
            model="openai/gpt-4o",
            max_iterations=12,
        ),
    }


def test_runner_requires_one_metadata_source():
    result = BenchmarkWorkflowResult("benchmark", (), 1, ())
    executor = RecordingExecutor(result)

    with pytest.raises(ValueError, match="exactly one"):
        CorralRunner(executor)
    with pytest.raises(ValueError, match="exactly one"):
        CorralRunner(executor, _metadata(), environments={})


def test_runner_builds_dependency_closed_temporal_metadata():
    executor = RecordingExecutor(BenchmarkWorkflowResult("benchmark", (), 1, ()))
    runner = CorralRunner(executor, _metadata())

    request = runner.build_workflow_input(
        "benchmark",
        task_ids=("downstream", "upstream"),
        trials_per_task=4,
        max_parallel=8,
        max_parallel_per_task=2,
        max_parallel_by_model={"model-a": 1},
        max_parallel_by_environment={"env-downstream": 1},
    )

    assert request.task_ids == ("upstream", "downstream")
    assert request.agent_by_task == {
        "upstream": "agent-a",
        "downstream": "agent-b",
    }
    assert request.environment_by_task["downstream"] == "env-downstream"
    assert request.dependency_graph["downstream"] == ("upstream",)
    assert request.max_iterations_by_task == {"upstream": 3, "downstream": 7}
    assert request.model_by_task == {
        "upstream": "model-a",
        "downstream": "model-b",
    }
    assert request.task_queue_by_task == {"downstream": "wetlab"}
    assert request.max_parallel == 8
    assert request.max_parallel_per_task == 2


def test_runner_rejects_an_incomplete_task_selection():
    result = BenchmarkWorkflowResult("benchmark", (), 1, ())
    runner = CorralRunner(RecordingExecutor(result), _metadata())

    with pytest.raises(ValueError, match="dependency-closed"):
        runner.build_workflow_input(
            "benchmark",
            task_ids=("downstream",),
            include_dependencies=False,
        )


def test_runner_includes_dependencies_by_default():
    result = BenchmarkWorkflowResult("benchmark", (), 1, ())
    runner = CorralRunner(RecordingExecutor(result), _metadata())

    request = runner.build_workflow_input(
        "benchmark",
        task_ids=("downstream",),
    )

    assert request.task_ids == ("upstream", "downstream")


@pytest.mark.anyio()
async def test_invalid_reporting_metadata_does_not_start_workflow():
    result = BenchmarkWorkflowResult("benchmark", ("upstream",), 1, ())
    executor = RecordingExecutor(result)
    runner = CorralRunner(
        executor,
        {"upstream": _metadata()["upstream"]},
    )

    with pytest.raises(ValueError, match="greater than the number of trials"):
        await runner.run("benchmark", trials_per_task=1, k_values=[2])

    assert executor.requests == []


@pytest.mark.anyio()
async def test_runner_delegates_once_and_projects_state_for_reporting():
    state = _submitted_state()
    state_ref = StateRef(
        commit_hash=state.through_commit_hash,
        execution_id=state.execution_id,
        branch_id=state.branch_id,
        sequence=9,
        status="submitted",
        agent_steps=1,
        submission="42",
        output={"answer": "42"},
    )
    evaluation = EvaluationRef(
        commit_hash=state.through_commit_hash,
        score=1.0,
        metrics={"score": 1.0},
        scorer_version="test:v1",
    )
    workflow_result = BenchmarkWorkflowResult(
        benchmark_run_id="benchmark-1",
        task_ids=("upstream",),
        trials_per_task=1,
        trials=(
            TaskWorkflowResult(
                task_id="upstream",
                trial_index=0,
                execution_id="benchmark-1:upstream:0",
                state=state_ref,
                evaluation=evaluation,
            ),
        ),
    )
    executor = RecordingExecutor(workflow_result)
    store = MemoryCommitStore(state)
    runner = CorralRunner(
        executor,
        {"upstream": _metadata()["upstream"]},
        state_store=store,
    )

    report = await runner.run("benchmark-1")

    assert len(executor.requests) == 1
    assert executor.requests[0].benchmark_run_id == "benchmark-1"
    assert store.loaded == [state.through_commit_hash]
    trial = report.task_results["upstream"].trials[0]
    assert trial.trial_id == "benchmark-1:upstream:0"
    assert trial.output == {"answer": "42"}
    assert trial.score == 1.0
    assert trial.duration == 2.0
    assert trial.token_usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "reasoning_tokens": 2,
        "llm_calls": 1,
        "tool_calls": 1,
        "agent_steps": 1,
    }
    assert trial.tool_statistics["tool_calls"][0]["action_id"] == "submit-1"
    assert trial.state["runtime"]["status"] == "submitted"
    assert report.verbosity == "brief"


@pytest.mark.anyio()
async def test_unreachable_trials_project_without_loading_state():
    workflow_result = BenchmarkWorkflowResult(
        benchmark_run_id="benchmark-2",
        task_ids=("upstream", "downstream"),
        trials_per_task=1,
        trials=(
            TaskWorkflowResult(
                task_id="downstream",
                trial_index=0,
                execution_id="benchmark-2:downstream:0",
                state=None,
                unreachable_dependency="upstream",
            ),
        ),
    )
    runner = CorralRunner(RecordingExecutor(workflow_result), _metadata())

    report = await runner.run("benchmark-2")

    trial = report.task_results["downstream"].trials[0]
    assert trial.state == {
        "unreachable": True,
        "missing_dependency": "upstream",
    }
    assert trial.output_ready is False
    assert trial.error_message is None
