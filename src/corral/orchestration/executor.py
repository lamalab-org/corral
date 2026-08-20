"""Thin client facades for starting Corral Temporal Workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from corral.orchestration.models import BenchmarkWorkflowResult, TaskWorkflowResult
from corral.orchestration.workflows import BenchmarkWorkflow, TaskWorkflow

if TYPE_CHECKING:
    from temporalio.client import Client

    from corral.core.state import ExecutionState
    from corral.orchestration.models import (
        BenchmarkWorkflowInput,
        TaskWorkflowInput,
    )
    from corral.persistence import CommitStore


class TaskExecutionError(RuntimeError):
    """A Temporal task completed without a loadable final projection."""


def task_workflow_id(execution_id: str) -> str:
    """Return the deterministic Workflow ID for one logical task execution."""
    return f"corral/task/{execution_id}"


def benchmark_workflow_id(benchmark_run_id: str) -> str:
    """Return the deterministic Workflow ID for one logical benchmark run."""
    return f"corral/benchmark/{benchmark_run_id}"


@dataclass(frozen=True)
class TemporalTaskExecutor:
    """Start task Workflows; all execution control remains inside Temporal."""

    client: Client
    state_store: CommitStore
    task_queue: str = "corral"

    async def execute_result(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        workflow_id = task_workflow_id(request.execution_id)
        try:
            handle = await self.client.start_workflow(
                TaskWorkflow.run,
                request,
                id=workflow_id,
                task_queue=self.task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            )
        except WorkflowAlreadyStartedError:
            handle = self.client.get_workflow_handle(
                workflow_id,
                result_type=TaskWorkflowResult,
            )
        return await handle.result()

    async def execute(self, request: TaskWorkflowInput) -> ExecutionState:
        result = await self.execute_result(request)
        if result.state is None:
            raise TaskExecutionError(
                result.error or "task produced no final projection"
            )
        store = self.state_store.for_execution(result.state.execution_id)
        return await store.materialize(result.state.branch_id, result.state.commit_hash)


@dataclass(frozen=True)
class TemporalBenchmarkExecutor:
    """Start the parent Workflow that owns benchmark scheduling and tracking."""

    client: Client
    task_queue: str = "corral"

    async def execute(
        self,
        request: BenchmarkWorkflowInput,
    ) -> BenchmarkWorkflowResult:
        workflow_id = benchmark_workflow_id(request.benchmark_run_id)
        try:
            handle = await self.client.start_workflow(
                BenchmarkWorkflow.run,
                request,
                id=workflow_id,
                task_queue=self.task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            )
        except WorkflowAlreadyStartedError:
            handle = self.client.get_workflow_handle(
                workflow_id,
                result_type=BenchmarkWorkflowResult,
            )
        return await handle.result()


async def execute_task(
    *,
    executor: TemporalTaskExecutor,
    task: TaskWorkflowInput,
) -> ExecutionState:
    """Execute one task by delegating its complete lifecycle to Temporal.

    This public function intentionally contains no loop, retry handling,
    benchmark identity, concurrency gate, or checkpoint logic.
    """
    return await executor.execute(task)


__all__ = [
    "TaskExecutionError",
    "TemporalBenchmarkExecutor",
    "TemporalTaskExecutor",
    "benchmark_workflow_id",
    "execute_task",
    "task_workflow_id",
]
