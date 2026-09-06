"""Task execution and benchmark scheduling with asyncio."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from corral.core.environment import Environment
from corral.core.task import assert_dependencies_selected, order_selected
from corral.observability import (
    CompositeObserver,
    LoggingObserver,
    ObservationContext,
    observer_from_env,
)
from corral.orchestration.evaluation import evaluate_task
from corral.orchestration.launchers import LocalTaskLauncher
from corral.orchestration.models import (
    AgentRuntimeDefinition,
    BenchmarkExecutionResult,
    BenchmarkInput,
    EnvironmentRuntimeDefinition,
    EvaluateTaskInput,
    RetryPolicy,
    RunTaskInput,
    SandboxProfile,
    StateRef,
    TaskExecutionResult,
)
from corral.report.logging import event, exception_fields, log_context
from corral.report.projection import normalise_k_values, project_benchmark_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from corral.observability import Observer
    from corral.orchestration.launchers import DockerTaskLauncher, TaskLauncher
    from corral.orchestration.registry import RuntimeRegistry
    from corral.persistence import CommitStore
    from corral.report.metrics import Metric
    from corral.report.results import BenchmarkResult


T = TypeVar("T")


async def _with_retries(
    operation: Callable[[], Awaitable[T]], policy: RetryPolicy
) -> T:
    delay = policy.initial_interval_seconds
    for attempt in range(1, policy.maximum_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if (
                attempt == policy.maximum_attempts
                or type(exc).__name__ in policy.non_retryable_error_types
            ):
                raise
            await asyncio.sleep(min(delay, policy.maximum_interval_seconds))
            delay *= policy.backoff_coefficient
    raise AssertionError("retry policy must allow at least one attempt")


def _task_observer(observer: Observer | None) -> Observer:
    if observer is None:
        return observer_from_env()
    if isinstance(observer, LoggingObserver):
        return observer
    if isinstance(observer, CompositeObserver) and any(
        isinstance(child, LoggingObserver) for child in observer.observers
    ):
        return observer
    return CompositeObserver(LoggingObserver(), observer)


async def execute_task(
    *,
    task: RunTaskInput,
    state_store: CommitStore,
    registry: RuntimeRegistry,
    observer: Observer | None = None,
    docker_launcher: DockerTaskLauncher | None = None,
    retry_policy: RetryPolicy | None = None,
) -> StateRef:
    """Run one task and return a reference to its persisted final state."""
    launcher: TaskLauncher
    if task.sandbox.mode == "docker":
        if docker_launcher is None:
            raise ValueError("Docker tasks require a DockerTaskLauncher")
        launcher = docker_launcher
    else:
        launcher = LocalTaskLauncher(state_store, registry, _task_observer(observer))
    context = ObservationContext(
        execution_id=task.execution_id,
        benchmark_run_id=task.benchmark_run_id,
        task_id=task.task_id,
    )
    with log_context(
        execution_id=task.execution_id,
        benchmark_run_id=task.benchmark_run_id,
        task_id=task.task_id,
    ):
        return await _with_retries(
            lambda: launcher.run(task, observation_context=context),
            retry_policy or RetryPolicy(),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTaskMetadata:
    """Resource IDs and scheduling metadata for one benchmark task."""

    agent_id: str
    environment_id: str
    dependencies: tuple[str, ...] = ()
    max_iterations: int = 10
    model: str | None = None
    sandbox: SandboxProfile = field(default_factory=SandboxProfile.local)
    agent_runtime: AgentRuntimeDefinition | None = None
    environment_runtime: EnvironmentRuntimeDefinition | None = None

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


def _metadata_from_environments(
    environments: Mapping[str, Environment],
    *,
    agent_id: str,
    model: str | None,
    max_iterations: int,
    sandbox: SandboxProfile,
    agent_runtime: AgentRuntimeDefinition | None,
    environment_runtime: EnvironmentRuntimeDefinition | None,
) -> dict[str, BenchmarkTaskMetadata]:
    """Infer routine benchmark metadata from task-keyed environments."""
    if not environments:
        raise ValueError("CorralRunner requires at least one environment")
    if not agent_id:
        raise ValueError("agent_id cannot be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if model == "":
        raise ValueError("model cannot be empty")

    metadata: dict[str, BenchmarkTaskMetadata] = {}
    for task_id, environment in environments.items():
        if not isinstance(environment, Environment):
            raise TypeError(f"environment for task {task_id!r} must be an Environment")
        if environment.task_id != task_id:
            raise ValueError(
                f"environment mapping key {task_id!r} does not match its "
                f"task_id {environment.task_id!r}"
            )
        metadata[task_id] = BenchmarkTaskMetadata(
            agent_id=agent_id,
            environment_id=task_id,
            dependencies=tuple(sorted(environment.current_task.dependencies())),
            max_iterations=max_iterations,
            model=model,
            sandbox=sandbox,
            agent_runtime=agent_runtime,
            environment_runtime=environment_runtime,
        )
    return metadata


def _selected_task_ids(
    tasks: Mapping[str, BenchmarkTaskMetadata],
    task_ids: Iterable[str] | None,
    *,
    include_dependencies: bool,
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
    if include_dependencies:
        selected_set = set(selected)
        pending = list(selected)
        while pending:
            task_id = pending.pop()
            for dependency in graph[task_id]:
                if dependency not in selected_set:
                    selected.append(dependency)
                    selected_set.add(dependency)
                    pending.append(dependency)
    else:
        assert_dependencies_selected(selected, graph)
    return tuple(order_selected(selected, graph))


_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "credential",
    "password",
    "secret",
    "token",
)


def _redact_secrets(value: Any) -> Any:
    """Keep benchmark configuration useful without serializing credentials."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(marker in str(key).casefold() for marker in _SECRET_KEY_MARKERS)
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_secrets(item) for item in value]
    return value


def _report_metadata(
    request: BenchmarkInput,
    *,
    k_values: list[int],
    include_dependencies: bool,
    verbose: bool,
) -> dict[str, Any]:
    agent_runtime = {
        task_id: _redact_secrets(asdict(runtime))
        for task_id, runtime in request.agent_runtime_by_task.items()
    }
    environment_runtime = {
        task_id: _redact_secrets(asdict(runtime))
        for task_id, runtime in request.environment_runtime_by_task.items()
    }
    return {
        "schema_version": 1,
        "benchmark_run_id": request.benchmark_run_id,
        "agent": {
            "by_task": {
                task_id: {
                    "id": request.agent_by_task[task_id],
                    "runtime": agent_runtime.get(task_id),
                }
                for task_id in request.task_ids
            }
        },
        "model": {
            "by_task": {
                task_id: request.model_by_task.get(task_id)
                for task_id in request.task_ids
            }
        },
        "environment": {
            "by_task": {
                task_id: {
                    "id": request.environment_by_task[task_id],
                    "runtime": environment_runtime.get(task_id),
                }
                for task_id in request.task_ids
            }
        },
        "benchmark": {
            "task_ids": list(request.task_ids),
            "trials_per_task": request.trials_per_task,
            "k_values": k_values,
            "dependency_graph": {
                task_id: list(dependencies)
                for task_id, dependencies in request.dependency_graph.items()
            },
            "include_dependencies": include_dependencies,
            "max_iterations_by_task": dict(request.max_iterations_by_task),
            "max_parallel": request.max_parallel,
            "max_parallel_per_task": request.max_parallel_per_task,
            "max_parallel_by_model": dict(request.max_parallel_by_model),
            "max_parallel_by_environment": dict(request.max_parallel_by_environment),
            "enable_surrender": request.enable_surrender,
            "evaluate": request.evaluate,
            "retry_policy": asdict(request.retry_policy),
            "sandbox": asdict(request.sandbox),
            "verbose": verbose,
        },
    }


class CorralRunner:
    """Run dependency-ordered trials with bounded concurrency and reporting."""

    def __init__(
        self,
        registry: RuntimeRegistry,
        tasks: Mapping[str, BenchmarkTaskMetadata] | None = None,
        *,
        environments: Mapping[str, Environment] | None = None,
        agent_id: str = "agent",
        model: str | None = None,
        max_iterations: int = 10,
        state_store: CommitStore,
        observer: Observer | None = None,
        docker_launcher: DockerTaskLauncher | None = None,
        metrics: Iterable[Metric] | None = None,
        sandbox: SandboxProfile | None = None,
        agent_runtime: AgentRuntimeDefinition | None = None,
        environment_runtime: EnvironmentRuntimeDefinition | None = None,
    ) -> None:
        """Create a runner from environments or explicit advanced metadata.

        Passing `environments` is the convenience path: task dependencies,
        environment IDs, model metadata, and iteration budgets are inferred.
        Passing `tasks` preserves complete per-task control for deployments
        that use custom agents, environments, or budgets.
        """
        if (tasks is None) == (environments is None):
            raise ValueError("pass exactly one of tasks or environments")
        if environments is not None:
            tasks = _metadata_from_environments(
                environments,
                agent_id=agent_id,
                model=model,
                max_iterations=max_iterations,
                sandbox=sandbox or SandboxProfile.local(),
                agent_runtime=agent_runtime,
                environment_runtime=environment_runtime,
            )
        assert tasks is not None
        self.registry = registry
        self.observer = _task_observer(observer)
        self.docker_launcher = docker_launcher
        self.tasks = MappingProxyType(_normalise_task_metadata(tasks))
        self.state_store = state_store
        self.metrics = tuple(metrics) if metrics is not None else None

    def build_input(
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
        retry_policy: RetryPolicy | None = None,
        include_dependencies: bool = True,
    ) -> BenchmarkInput:
        """Build and validate the complete benchmark plan."""
        selected = _selected_task_ids(
            self.tasks,
            task_ids,
            include_dependencies=include_dependencies,
        )
        metadata = {task_id: self.tasks[task_id] for task_id in selected}
        profiles = [task.sandbox for task in metadata.values()]
        sandbox = profiles[0]
        if any(profile != sandbox for profile in profiles[1:]):
            raise ValueError(
                "one benchmark cannot mix local and Docker sandbox profiles"
            )
        return BenchmarkInput(
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
            max_parallel=max_parallel,
            max_parallel_per_task=max_parallel_per_task,
            max_parallel_by_model=dict(max_parallel_by_model or {}),
            max_parallel_by_environment=dict(max_parallel_by_environment or {}),
            enable_surrender=enable_surrender,
            evaluate=evaluate,
            retry_policy=retry_policy or RetryPolicy(),
            sandbox=sandbox,
            agent_runtime_by_task={
                task_id: task.agent_runtime
                for task_id, task in metadata.items()
                if task.agent_runtime is not None
            },
            environment_runtime_by_task={
                task_id: task.environment_runtime
                for task_id, task in metadata.items()
                if task.environment_runtime is not None
            },
        )

    async def _execute_benchmark(
        self, request: BenchmarkInput
    ) -> BenchmarkExecutionResult:
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
        handles: dict[tuple[int, str], asyncio.Task[TaskExecutionResult]] = {}

        async def run_one(trial_index: int, task_id: str) -> TaskExecutionResult:
            execution_id = f"{request.benchmark_run_id}:{task_id}:{trial_index}"
            dependencies: dict[str, dict[str, Any]] = {}
            for dependency in request.dependency_graph.get(task_id, ()):
                result = await handles[(trial_index, dependency)]
                if not result.output_ready:
                    return TaskExecutionResult(
                        task_id=task_id,
                        trial_index=trial_index,
                        execution_id=execution_id,
                        state=None,
                        unreachable_dependency=dependency,
                    )
                assert result.state is not None
                assert result.state.output is not None
                dependencies[dependency] = result.state.output

            model = request.model_by_task.get(task_id)
            environment_id = request.environment_by_task[task_id]
            try:
                async with AsyncExitStack() as gates:
                    for gate in (
                        model_gates.get(model),
                        environment_gates.get(environment_id),
                        task_gates[task_id],
                        global_gate,
                    ):
                        if gate is not None:
                            await gates.enter_async_context(gate)
                    current = await execute_task(
                        task=RunTaskInput(
                            execution_id=execution_id,
                            task_id=task_id,
                            environment_id=environment_id,
                            agent_id=request.agent_by_task[task_id],
                            benchmark_run_id=request.benchmark_run_id,
                            trial_index=trial_index,
                            dependency_outputs=dependencies,
                            max_iterations=request.max_iterations_by_task.get(
                                task_id, 10
                            ),
                            model=model,
                            enable_surrender=request.enable_surrender,
                            sandbox=request.sandbox,
                            agent_runtime=request.agent_runtime_by_task.get(task_id),
                            environment_runtime=request.environment_runtime_by_task.get(
                                task_id
                            ),
                        ),
                        state_store=self.state_store,
                        registry=self.registry,
                        observer=self.observer,
                        docker_launcher=self.docker_launcher,
                        retry_policy=request.retry_policy,
                    )
                    evaluation = None
                    evaluation_error = None
                    if request.evaluate and current.output is not None:
                        try:
                            evaluation = await _with_retries(
                                lambda: evaluate_task(
                                    EvaluateTaskInput(
                                        execution_id=execution_id,
                                        environment_id=environment_id,
                                        commit_hash=current.commit_hash,
                                        branch_id=current.branch_id,
                                        task_id=task_id,
                                        benchmark_run_id=request.benchmark_run_id,
                                    ),
                                    state_store=self.state_store,
                                    registry=self.registry,
                                    observer=self.observer,
                                ),
                                request.retry_policy,
                            )
                        except Exception as exc:
                            evaluation_error = str(exc) or type(exc).__name__
                            event(
                                "WARNING",
                                "evaluation.degraded",
                                subsystem="runtime",
                                execution_id=execution_id,
                                task_id=task_id,
                                **exception_fields(exc),
                            )
                    return TaskExecutionResult(
                        task_id=task_id,
                        trial_index=trial_index,
                        execution_id=execution_id,
                        state=current,
                        evaluation=evaluation,
                        evaluation_error=evaluation_error,
                        error=current.error if current.status == "failed" else None,
                    )
            except Exception as exc:
                event(
                    "ERROR",
                    "benchmark.task_failed",
                    subsystem="runtime",
                    execution_id=execution_id,
                    task_id=task_id,
                    **exception_fields(exc),
                )
                return TaskExecutionResult(
                    task_id=task_id,
                    trial_index=trial_index,
                    execution_id=execution_id,
                    state=None,
                    error=str(exc) or type(exc).__name__,
                )

        # Register every handle before a task can await its same-trial dependencies.
        for trial_index in range(request.trials_per_task):
            for task_id in request.task_ids:
                handles[(trial_index, task_id)] = asyncio.create_task(
                    run_one(trial_index, task_id)
                )
        try:
            trials = await asyncio.gather(*handles.values())
        finally:
            # Finish cancellation before callers close stores and environments.
            for handle in handles.values():
                if not handle.done():
                    handle.cancel()
            await asyncio.gather(*handles.values(), return_exceptions=True)
        return BenchmarkExecutionResult(
            benchmark_run_id=request.benchmark_run_id,
            task_ids=request.task_ids,
            trials_per_task=request.trials_per_task,
            trials=tuple(trials),
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
        retry_policy: RetryPolicy | None = None,
        include_dependencies: bool = True,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Execute the benchmark and return its reporting projection."""
        # Invalid reporting metadata must fail before an expensive benchmark is
        # started, not after the benchmark has completed.
        normalised_k = normalise_k_values(k_values, trials_per_task)
        request = self.build_input(
            benchmark_run_id,
            task_ids=task_ids,
            trials_per_task=trials_per_task,
            max_parallel=max_parallel,
            max_parallel_per_task=max_parallel_per_task,
            max_parallel_by_model=max_parallel_by_model,
            max_parallel_by_environment=max_parallel_by_environment,
            enable_surrender=enable_surrender,
            evaluate=evaluate,
            retry_policy=retry_policy,
            include_dependencies=include_dependencies,
        )

        started = perf_counter()
        with log_context(benchmark_run_id=benchmark_run_id):
            event(
                "INFO",
                "benchmark.started",
                subsystem="runtime",
                task_count=len(request.task_ids),
                trials_per_task=request.trials_per_task,
            )
            try:
                result = await self._execute_benchmark(request)
                duration = perf_counter() - started
                report = await project_benchmark_result(
                    result,
                    state_store=self.state_store,
                    k_values=normalised_k,
                    total_duration=duration,
                    verbose=verbose,
                    metrics=self.metrics,
                    metadata=_report_metadata(
                        request,
                        k_values=normalised_k,
                        include_dependencies=include_dependencies,
                        verbose=verbose,
                    ),
                )
            except Exception as exc:
                event(
                    "ERROR",
                    "benchmark.failed",
                    subsystem="runtime",
                    status="failed",
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    **exception_fields(exc),
                )
                raise
            event(
                "INFO",
                "benchmark.completed",
                subsystem="runtime",
                status="completed",
                duration_ms=round(duration * 1000, 3),
                trial_count=len(result.trials),
            )
            return report


__all__ = [
    "BenchmarkTaskMetadata",
    "CorralRunner",
    "execute_task",
    "project_benchmark_result",
]
