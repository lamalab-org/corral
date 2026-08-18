"""Temporal Workflows owning task loops and benchmark scheduling."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# Temporal requires either Start-to-Close or Schedule-to-Close on every
# Activity. Public ``None`` means no operational deadline, so use a 100-year
# Schedule-to-Close value as the protocol-level representation of "unbounded".
_UNBOUNDED_ACTIVITY_TIMEOUT = timedelta(days=365 * 100)

with workflow.unsafe.imports_passed_through():
    from corral.orchestration.models import (
        ActivityPolicy,
        BenchmarkProgress,
        BenchmarkWorkflowInput,
        BenchmarkWorkflowResult,
        EvaluateTaskInput,
        EvaluationRef,
        RunTaskInput,
        StateRef,
        TaskWorkflowInput,
        TaskWorkflowResult,
    )


def _activity_options(policy: ActivityPolicy) -> dict[str, Any]:
    options: dict[str, Any] = {
        "retry_policy": RetryPolicy(
            initial_interval=timedelta(seconds=policy.initial_interval_seconds),
            backoff_coefficient=policy.backoff_coefficient,
            maximum_interval=timedelta(seconds=policy.maximum_interval_seconds),
            maximum_attempts=policy.maximum_attempts,
            non_retryable_error_types=policy.non_retryable_error_types,
        ),
    }
    if policy.start_to_close_seconds is None:
        options["schedule_to_close_timeout"] = _UNBOUNDED_ACTIVITY_TIMEOUT
    else:
        options["start_to_close_timeout"] = timedelta(
            seconds=policy.start_to_close_seconds
        )
    if policy.heartbeat_timeout_seconds is not None:
        options["heartbeat_timeout"] = timedelta(
            seconds=policy.heartbeat_timeout_seconds
        )
    return options


async def _execute_activity(
    name: str,
    argument: Any,
    *,
    result_type: type[Any],
    policy: ActivityPolicy,
) -> Any:
    return await workflow.execute_activity(
        name,
        argument,
        result_type=result_type,
        **_activity_options(policy),
    )


@workflow.defn(name="CorralTaskWorkflow")
class TaskWorkflow:
    """Run and optionally evaluate one task through task-level Activities."""

    @workflow.run
    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        policy = request.activity_policy
        started_at = request.started_at or workflow.now().isoformat()
        current = await _execute_activity(
            "corral.run_task",
            RunTaskInput(
                execution_id=request.execution_id,
                task_id=request.task_id,
                environment_id=request.environment_id,
                agent_id=request.agent_id,
                started_at=started_at,
                max_iterations=request.max_iterations,
                model=request.model,
                dependency_outputs=request.dependency_outputs,
                enable_surrender=request.enable_surrender,
                benchmark_run_id=request.benchmark_run_id,
            ),
            result_type=StateRef,
            policy=policy,
        )

        evaluation: EvaluationRef | None = None
        evaluation_error: str | None = None
        if request.evaluate and current.output is not None:
            try:
                evaluation = await _execute_activity(
                    "corral.evaluate_task",
                    EvaluateTaskInput(
                        execution_id=request.execution_id,
                        environment_id=request.environment_id,
                        state_hash=current.state_hash,
                        task_id=request.task_id,
                        benchmark_run_id=request.benchmark_run_id,
                    ),
                    result_type=EvaluationRef,
                    policy=policy,
                )
            except Exception as exc:  # evaluation cannot invalidate runtime output
                evaluation_error = str(exc)

        return TaskWorkflowResult(
            task_id=request.task_id,
            trial_index=request.trial_index,
            execution_id=request.execution_id,
            state=current,
            evaluation=evaluation,
            evaluation_error=evaluation_error,
            error=current.error if current.status == "failed" else None,
        )


class _NoopAsyncContext(AbstractAsyncContextManager[None]):
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        del args


def _validate_graph(request: BenchmarkWorkflowInput) -> None:
    selected = set(request.task_ids)
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in permanent:
            return
        if task_id in temporary:
            raise ApplicationError(
                f"dependency cycle involving {task_id!r}",
                type="InvalidBenchmarkGraph",
                non_retryable=True,
            )
        temporary.add(task_id)
        for dependency in request.dependency_graph.get(task_id, ()):
            if dependency not in selected:
                raise ApplicationError(
                    f"selected task {task_id!r} requires missing dependency "
                    f"{dependency!r}",
                    type="InvalidBenchmarkGraph",
                    non_retryable=True,
                )
            visit(dependency)
        temporary.remove(task_id)
        permanent.add(task_id)

    for task_id in request.task_ids:
        visit(task_id)


@workflow.defn(name="CorralBenchmarkWorkflow")
class BenchmarkWorkflow:
    """Track the complete benchmark DAG, retries, and bounded concurrency."""

    def __init__(self) -> None:
        self._run_id = ""
        self._total = 0
        self._status: dict[str, str] = {}
        self._carried: tuple[TaskWorkflowResult, ...] = ()

    @workflow.query
    def progress(self) -> BenchmarkProgress:
        statuses = tuple(self._status.values())
        carried_failed = sum(result.error is not None for result in self._carried)
        carried_unreachable = sum(
            result.unreachable_dependency is not None for result in self._carried
        )
        carried_completed = len(self._carried) - carried_failed - carried_unreachable
        completed = statuses.count("completed") + carried_completed
        failed = statuses.count("failed") + carried_failed
        unreachable = statuses.count("unreachable") + carried_unreachable
        running = statuses.count("running")
        terminal = completed + failed + unreachable
        return BenchmarkProgress(
            benchmark_run_id=self._run_id,
            total=self._total,
            pending=max(0, self._total - terminal - running),
            running=running,
            completed=completed,
            failed=failed,
            unreachable=unreachable,
        )

    @workflow.run
    async def run(self, request: BenchmarkWorkflowInput) -> BenchmarkWorkflowResult:
        _validate_graph(request)
        self._run_id = request.benchmark_run_id
        self._total = len(request.task_ids) * request.trials_per_task
        self._carried = request.completed

        global_gate = asyncio.Semaphore(request.max_parallel)
        task_gates = {
            task_id: asyncio.Semaphore(request.max_parallel_per_task)
            for task_id in request.task_ids
        }
        model_gates = {
            model: asyncio.Semaphore(limit)
            for model, limit in request.max_parallel_by_model.items()
        }
        environment_gates = {
            environment: asyncio.Semaphore(limit)
            for environment, limit in request.max_parallel_by_environment.items()
        }

        batch_start = request.next_trial_index
        batch_end = request.trials_per_task
        if request.rounds_per_run:
            batch_end = min(batch_end, batch_start + request.rounds_per_run)

        handles: dict[tuple[int, str], asyncio.Task[TaskWorkflowResult]] = {}

        async def run_one(trial_index: int, task_id: str) -> TaskWorkflowResult:
            key = f"{trial_index}:{task_id}"
            dependencies: dict[str, dict[str, Any]] = {}
            for dependency in request.dependency_graph.get(task_id, ()):
                result = await handles[(trial_index, dependency)]
                if not result.output_ready:
                    self._status[key] = "unreachable"
                    return TaskWorkflowResult(
                        task_id=task_id,
                        trial_index=trial_index,
                        execution_id=(
                            f"{request.benchmark_run_id}:{task_id}:{trial_index}"
                        ),
                        state=None,
                        unreachable_dependency=dependency,
                    )
                assert result.state is not None
                assert result.state.output is not None
                dependencies[dependency] = result.state.output

            model = request.model_by_task.get(task_id)
            model_gate: AbstractAsyncContextManager[None]
            if model is not None and model in model_gates:
                model_gate = model_gates[model]
            else:
                model_gate = _NoopAsyncContext()
            environment_id = request.environment_by_task[task_id]
            environment_gate: AbstractAsyncContextManager[None]
            if environment_id in environment_gates:
                environment_gate = environment_gates[environment_id]
            else:
                environment_gate = _NoopAsyncContext()

            self._status[key] = "queued"
            try:
                async with (
                    model_gate,
                    environment_gate,
                    task_gates[task_id],
                    global_gate,
                ):
                    self._status[key] = "running"
                    execution_id = f"{request.benchmark_run_id}:{task_id}:{trial_index}"
                    child_options: dict[str, Any] = {
                        "id": (
                            f"corral/{request.benchmark_run_id}/{task_id}/{trial_index}"
                        )
                    }
                    task_queue = request.task_queue_by_task.get(task_id)
                    if task_queue is not None:
                        child_options["task_queue"] = task_queue
                    result = await workflow.execute_child_workflow(
                        TaskWorkflow.run,
                        TaskWorkflowInput(
                            execution_id=execution_id,
                            task_id=task_id,
                            environment_id=request.environment_by_task[task_id],
                            agent_id=request.agent_by_task[task_id],
                            benchmark_run_id=request.benchmark_run_id,
                            trial_index=trial_index,
                            dependency_outputs=dependencies,
                            max_iterations=request.max_iterations_by_task.get(
                                task_id, 10
                            ),
                            model=model,
                            enable_surrender=request.enable_surrender,
                            evaluate=request.evaluate,
                            activity_policy=request.activity_policy,
                        ),
                        **child_options,
                    )
                self._status[key] = "failed" if result.error else "completed"
                return result
            except Exception as exc:
                self._status[key] = "failed"
                return TaskWorkflowResult(
                    task_id=task_id,
                    trial_index=trial_index,
                    execution_id=f"{request.benchmark_run_id}:{task_id}:{trial_index}",
                    state=None,
                    error=str(exc),
                )

        # Create every task in deterministic order before awaiting any. A node
        # can therefore await its same-round dependency handle regardless of the
        # order in which Temporal schedules the coroutines.
        for trial_index in range(batch_start, batch_end):
            for task_id in request.task_ids:
                handles[(trial_index, task_id)] = asyncio.create_task(
                    run_one(trial_index, task_id)
                )

        batch_results = await asyncio.gather(
            *(handles[key] for key in handles),
        )
        completed = (*request.completed, *batch_results)

        if batch_end < request.trials_per_task:
            workflow.continue_as_new(
                replace(
                    request,
                    next_trial_index=batch_end,
                    completed=completed,
                )
            )

        task_order = {task_id: index for index, task_id in enumerate(request.task_ids)}
        ordered = tuple(
            sorted(
                completed,
                key=lambda result: (
                    result.trial_index,
                    task_order[result.task_id],
                ),
            )
        )
        return BenchmarkWorkflowResult(
            benchmark_run_id=request.benchmark_run_id,
            task_ids=request.task_ids,
            trials_per_task=request.trials_per_task,
            trials=ordered,
        )


__all__ = ["BenchmarkWorkflow", "TaskWorkflow"]
