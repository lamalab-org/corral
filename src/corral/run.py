"""Thin benchmark metadata adapter for Temporal orchestration.

The runner does not execute agents, schedule trials, manage concurrency, or
checkpoint progress. Temporal owns those responsibilities. This module only
turns immutable per-task metadata into a ``BenchmarkWorkflowInput``, delegates
it once, and asks the reporting layer to project the durable result.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from corral.core.task import assert_dependencies_selected, order_selected
from corral.orchestration.models import (
    ActivityPolicy,
    BenchmarkWorkflowInput,
    BenchmarkWorkflowResult,
)
from corral.report.projection import normalise_k_values, project_benchmark_result

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from corral.persistence import StateStore
    from corral.report.metrics import Metric
    from corral.report.results import BenchmarkResult


class BenchmarkExecutor(Protocol):
    """The only execution capability needed by :class:`CorralRunner`."""

    async def execute(
        self,
        request: BenchmarkWorkflowInput,
    ) -> BenchmarkWorkflowResult: ...


@dataclass(frozen=True, slots=True)
class BenchmarkTaskMetadata:
    """Durable worker IDs and scheduling metadata for one benchmark task.

    The mapping key supplied to :class:`CorralRunner` is the task ID. Agents
    and environments remain registered on Temporal workers and are referenced
    here only by their durable IDs; live Python objects never enter Workflow
    history.
    """

    agent_id: str
    environment_id: str
    dependencies: tuple[str, ...] = ()
    max_iterations: int = 10
    model: str | None = None
    task_queue: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.environment_id:
            raise ValueError("environment_id cannot be empty")
        if isinstance(self.dependencies, str):
            raise TypeError("dependencies must be an iterable of task IDs")
        dependencies = tuple(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if any(not dependency for dependency in dependencies):
            raise ValueError("dependencies cannot contain an empty task ID")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies cannot contain duplicates")
        if self.model == "":
            raise ValueError("model cannot be empty")
        if self.task_queue == "":
            raise ValueError("task_queue cannot be empty")


def _normalise_task_metadata(
    tasks: Mapping[str, BenchmarkTaskMetadata],
) -> dict[str, BenchmarkTaskMetadata]:
    if not tasks:
        raise ValueError("CorralRunner requires at least one task")

    normalised: dict[str, BenchmarkTaskMetadata] = {}
    for task_id, metadata in tasks.items():
        if not task_id:
            raise ValueError("task IDs cannot be empty")
        if not isinstance(metadata, BenchmarkTaskMetadata):
            raise TypeError(
                f"metadata for task {task_id!r} must be BenchmarkTaskMetadata"
            )
        if task_id in metadata.dependencies:
            raise ValueError(f"task {task_id!r} cannot depend on itself")
        normalised[task_id] = metadata

    known = set(normalised)
    for task_id, metadata in normalised.items():
        missing = set(metadata.dependencies) - known
        if missing:
            raise ValueError(
                f"task {task_id!r} depends on unknown task(s): {sorted(missing)}"
            )

    graph = {
        task_id: list(metadata.dependencies) for task_id, metadata in normalised.items()
    }
    order_selected(list(normalised), graph)  # validates the complete graph
    return normalised


def _selected_task_ids(
    tasks: Mapping[str, BenchmarkTaskMetadata],
    task_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    selected = list(tasks) if task_ids is None else list(task_ids)
    if not selected:
        raise ValueError("a benchmark must select at least one task")
    if len(set(selected)) != len(selected):
        raise ValueError("task_ids cannot contain duplicates")

    unknown = set(selected) - tasks.keys()
    if unknown:
        raise ValueError(f"unknown benchmark task(s): {sorted(unknown)}")

    graph = {task_id: list(tasks[task_id].dependencies) for task_id in tasks}
    assert_dependencies_selected(selected, graph)
    return tuple(order_selected(selected, graph))


class CorralRunner:
    """Build one Temporal benchmark request and project its durable result.

    This class intentionally has no synchronous execution path. Callers await
    :meth:`run`; Temporal handles retries, concurrency, task-DAG
    readiness, evaluation, and progress tracking.
    """

    def __init__(
        self,
        executor: BenchmarkExecutor,
        tasks: Mapping[str, BenchmarkTaskMetadata],
        *,
        state_store: StateStore | None = None,
        metrics: Iterable[Metric] | None = None,
    ) -> None:
        self.executor = executor
        self.tasks = MappingProxyType(_normalise_task_metadata(tasks))
        self.state_store = state_store
        self.metrics = tuple(metrics) if metrics is not None else None

    def build_workflow_input(
        self,
        benchmark_run_id: str,
        *,
        task_ids: Iterable[str] | None = None,
        trials_per_task: int = 1,
        max_parallel: int = 1,
        max_parallel_per_task: int = 1,
        max_parallel_by_model: Mapping[str, int] | None = None,
        max_parallel_by_environment: Mapping[str, int] | None = None,
        enable_surrender: bool = False,
        evaluate: bool = True,
        activity_policy: ActivityPolicy | None = None,
        rounds_per_run: int = 0,
    ) -> BenchmarkWorkflowInput:
        """Build the complete serializable benchmark plan for Temporal."""
        selected = _selected_task_ids(self.tasks, task_ids)
        metadata = {task_id: self.tasks[task_id] for task_id in selected}
        return BenchmarkWorkflowInput(
            benchmark_run_id=benchmark_run_id,
            task_ids=selected,
            trials_per_task=trials_per_task,
            agent_by_task={
                task_id: task.agent_id for task_id, task in metadata.items()
            },
            environment_by_task={
                task_id: task.environment_id for task_id, task in metadata.items()
            },
            dependency_graph={
                task_id: task.dependencies for task_id, task in metadata.items()
            },
            max_iterations_by_task={
                task_id: task.max_iterations for task_id, task in metadata.items()
            },
            model_by_task={
                task_id: task.model
                for task_id, task in metadata.items()
                if task.model is not None
            },
            task_queue_by_task={
                task_id: task.task_queue
                for task_id, task in metadata.items()
                if task.task_queue is not None
            },
            max_parallel=max_parallel,
            max_parallel_per_task=max_parallel_per_task,
            max_parallel_by_model=dict(max_parallel_by_model or {}),
            max_parallel_by_environment=dict(max_parallel_by_environment or {}),
            enable_surrender=enable_surrender,
            evaluate=evaluate,
            activity_policy=activity_policy or ActivityPolicy(),
            rounds_per_run=rounds_per_run,
        )

    async def run(
        self,
        benchmark_run_id: str,
        *,
        task_ids: Iterable[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | Iterable[int] | None = None,
        max_parallel: int = 1,
        max_parallel_per_task: int = 1,
        max_parallel_by_model: Mapping[str, int] | None = None,
        max_parallel_by_environment: Mapping[str, int] | None = None,
        enable_surrender: bool = False,
        evaluate: bool = True,
        activity_policy: ActivityPolicy | None = None,
        rounds_per_run: int = 0,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Execute the Temporal benchmark and return its reporting projection."""
        # Invalid reporting metadata must fail before an expensive Workflow is
        # started, not after the benchmark has completed.
        normalise_k_values(k_values, trials_per_task)
        request = self.build_workflow_input(
            benchmark_run_id,
            task_ids=task_ids,
            trials_per_task=trials_per_task,
            max_parallel=max_parallel,
            max_parallel_per_task=max_parallel_per_task,
            max_parallel_by_model=max_parallel_by_model,
            max_parallel_by_environment=max_parallel_by_environment,
            enable_surrender=enable_surrender,
            evaluate=evaluate,
            activity_policy=activity_policy,
            rounds_per_run=rounds_per_run,
        )

        started = perf_counter()
        workflow_result = await self.executor.execute(request)
        duration = perf_counter() - started
        if workflow_result.benchmark_run_id != benchmark_run_id:
            raise RuntimeError(
                "Temporal returned a result for benchmark "
                f"{workflow_result.benchmark_run_id!r}, expected {benchmark_run_id!r}"
            )
        if workflow_result.task_ids != request.task_ids:
            raise RuntimeError(
                "Temporal returned different task metadata: "
                f"{workflow_result.task_ids!r} != {request.task_ids!r}"
            )
        if workflow_result.trials_per_task != request.trials_per_task:
            raise RuntimeError(
                "Temporal returned a different trials_per_task value: "
                f"{workflow_result.trials_per_task} != {request.trials_per_task}"
            )
        return await project_benchmark_result(
            workflow_result,
            state_store=self.state_store,
            k_values=k_values,
            total_duration=duration,
            verbose=verbose,
            metrics=self.metrics,
        )


__all__ = [
    "BenchmarkExecutor",
    "BenchmarkTaskMetadata",
    "CorralRunner",
    "project_benchmark_result",
]
