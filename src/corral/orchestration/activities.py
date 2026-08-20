"""Temporal Activities for Corral's task-boundary runtime."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from temporalio import activity

from corral.core.state import ExecutionState, TaskOutput
from corral.evaluation import TaskScorer
from corral.logging import event, exception_fields, log_context
from corral.observability import (
    CompositeObserver,
    LoggingObserver,
    Observation,
    ObservationContext,
    Observer,
    observe_safely,
    observer_from_env,
    update_safely,
)
from corral.orchestration.models import (
    EvaluateTaskInput,
    EvaluationRef,
    RunTaskInput,
    StateRef,
)
from corral.persistence import CommitNotFoundError
from corral.runtime import TaskRuntime

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from corral.agents.session import Agent
    from corral.core.commit import Commit
    from corral.core.environment import Environment
    from corral.persistence import CommitStore

T = TypeVar("T")


def state_ref(state: ExecutionState, commit: Commit) -> StateRef:
    """Project a materialized head into the small value kept in Workflow history."""
    output = (
        {"answer": state.submission}
        if state.submission is not None and state.runtime.status != "surrendered"
        else None
    )
    raw_error = state.runtime.metadata.get("error")
    return StateRef(
        commit_hash=commit.hash,
        execution_id=state.execution_id,
        branch_id=state.branch_id,
        sequence=commit.sequence,
        status=state.runtime.status,
        agent_steps=state.usage.agent_steps,
        submission=state.submission,
        output=output,
        error=str(raw_error) if raw_error is not None else None,
    )


def _heartbeat(details: str) -> None:
    with contextlib.suppress(RuntimeError):
        activity.heartbeat(details)
    # Keeping Activities directly unit-testable is useful; a direct call has
    # no Temporal activity context and therefore cannot emit a heartbeat.


async def _with_heartbeats(
    operation: Awaitable[T],
    *,
    details: str,
    interval_seconds: float = 5.0,
) -> T:
    """Heartbeat while an LLM call, setup hook, or synchronous tool is running."""
    task = asyncio.ensure_future(operation)
    _heartbeat(details)
    while True:
        done, _pending = await asyncio.wait({task}, timeout=interval_seconds)
        if done:
            return task.result()
        _heartbeat(details)


class RuntimeRegistry:
    """Worker-side live resources addressed by durable Workflow IDs.

    Temporal payloads never contain agents, environments, clients, locks, or
    credentials. Deployments construct the same logical registry on every
    worker polling a Corral task queue.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, Agent],
        environments: Mapping[str, Environment],
    ) -> None:
        self._agents = dict(agents)
        self._environments = dict(environments)
        self._bound: dict[tuple[str, str], Environment] = {}
        self._last_evaluations: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"Temporal worker has no agent {agent_id!r}") from exc

    def environment(self, environment_id: str, execution_id: str) -> Environment:
        key = (environment_id, execution_id)
        with self._lock:
            bound = self._bound.get(key)
            if bound is not None:
                return bound
            try:
                template = self._environments[environment_id]
            except KeyError as exc:
                raise KeyError(
                    f"Temporal worker has no environment {environment_id!r}"
                ) from exc
            bound = template.for_task(execution_id)
            self._bound[key] = bound
            return bound

    def record_evaluation(
        self,
        task_id: str,
        execution_id: str,
        result: Mapping[str, Any],
    ) -> None:
        """Expose an evaluation and its commit hash to the next task attempt."""
        with self._lock:
            self._last_evaluations[task_id] = {
                "trial_id": execution_id,
                **dict(result),
            }

    def last_evaluation(self, task_id: str | None) -> dict[str, Any] | None:
        if task_id is None:
            return None
        with self._lock:
            result = self._last_evaluations.get(task_id)
            return dict(result) if result is not None else None

    def close(self) -> None:
        with self._lock:
            environments = tuple(self._bound.values())
            self._bound.clear()
            self._last_evaluations.clear()
        for environment in environments:
            environment.shutdown_jobs()


class CorralActivities:
    """Bound Activity implementations registered by a Corral Temporal worker."""

    def __init__(
        self,
        state_store: CommitStore,
        registry: RuntimeRegistry,
        observer: Observer | None = None,
    ) -> None:
        self.state_store = state_store
        self.registry = registry
        self.observer = (
            observer_from_env()
            if observer is None
            else observer
            if isinstance(observer, LoggingObserver)
            else CompositeObserver(LoggingObserver(), observer)
        )
        self.runtime = TaskRuntime(state_store, self.observer)

    @staticmethod
    def _context(
        *,
        execution_id: str,
        task_id: str | None,
        benchmark_run_id: str | None,
    ) -> ObservationContext:
        workflow_id: str | None = None
        workflow_run_id: str | None = None
        with contextlib.suppress(RuntimeError):
            info = activity.info()
            workflow_id = info.workflow_id
            workflow_run_id = info.workflow_run_id
        return ObservationContext(
            execution_id=execution_id,
            benchmark_run_id=benchmark_run_id,
            task_id=task_id,
            temporal_workflow_id=workflow_id,
            temporal_run_id=workflow_run_id,
        )

    @staticmethod
    def _activity_fields(name: str) -> dict[str, Any]:
        fields: dict[str, Any] = {"activity": name, "attempt": 1}
        with contextlib.suppress(RuntimeError):
            info = activity.info()
            fields.update(
                attempt=info.attempt,
                workflow_id=info.workflow_id,
                workflow_run_id=info.workflow_run_id,
            )
        return fields

    @activity.defn(name="corral.run_task")
    async def run_task(self, request: RunTaskInput) -> StateRef:
        started = perf_counter()
        activity_fields = self._activity_fields("corral.run_task")
        correlation = {
            "benchmark_run_id": request.benchmark_run_id,
            "execution_id": request.execution_id,
            "task_id": request.task_id,
            **activity_fields,
        }
        event(
            "INFO",
            "activity.started",
            subsystem="orchestration",
            status="running",
            **correlation,
        )

        async def invoke() -> ExecutionState:
            agent = self.registry.agent(request.agent_id)
            environment = self.registry.environment(
                request.environment_id, request.execution_id
            )
            dependencies = {
                task_id: TaskOutput(output=output)
                for task_id, output in request.dependency_outputs.items()
            }
            observation_context = self._context(
                execution_id=request.execution_id,
                task_id=request.task_id,
                benchmark_run_id=request.benchmark_run_id,
            )
            last_evaluation = self.registry.last_evaluation(request.task_id)
            previous_state = None
            if last_evaluation is not None:
                previous_hash = last_evaluation.get("commit_hash")
                if isinstance(previous_hash, str):
                    try:
                        execution_store = self.state_store.for_execution(
                            str(last_evaluation.get("trial_id"))
                        )
                        previous_state = await execution_store.materialize(
                            "main", previous_hash
                        )
                    except CommitNotFoundError:
                        # Evaluation context remains useful even if a deployment
                        # pruned the referenced commit between attempts.
                        previous_state = None

            configured_model = request.model
            if configured_model is None:
                agent_model = getattr(agent, "model", None)
                if isinstance(agent_model, str) and agent_model:
                    configured_model = agent_model
            return await self.runtime.run(
                agent,
                environment,
                execution_id=request.execution_id,
                started_at=datetime.fromisoformat(request.started_at),
                max_iterations=request.max_iterations,
                dependency_outputs=dependencies,
                model_metadata=(
                    {"name": configured_model} if configured_model is not None else {}
                ),
                scaffold_metadata={
                    "name": type(agent).__name__,
                    "enable_surrender": request.enable_surrender,
                },
                last_score=last_evaluation,
                previous_state=previous_state,
                observation_context=observation_context,
            )

        try:
            with log_context(**correlation):
                state = await _with_heartbeats(
                    invoke(),
                    details=f"run-task:{request.execution_id}",
                )
        except Exception as exc:
            event(
                "WARNING",
                "activity.attempt_failed",
                subsystem="orchestration",
                status="retryable_failure",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                **correlation,
                **exception_fields(exc),
            )
            raise
        event(
            "INFO",
            "activity.completed",
            subsystem="orchestration",
            status="completed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            commit_hash=state.through_commit_hash,
            **correlation,
        )
        execution_store = self.state_store.for_execution(request.execution_id)
        head = await execution_store.head("main")
        if head is None:
            raise RuntimeError("task runtime returned without a branch head")
        return state_ref(state, head)

    @activity.defn(name="corral.evaluate_task")
    async def evaluate_task(self, request: EvaluateTaskInput) -> EvaluationRef:
        started = perf_counter()
        activity_fields = self._activity_fields("corral.evaluate_task")
        correlation = {
            "benchmark_run_id": request.benchmark_run_id,
            "execution_id": request.execution_id,
            "task_id": request.task_id,
            **activity_fields,
        }
        event(
            "INFO",
            "activity.started",
            subsystem="orchestration",
            status="running",
            **correlation,
        )
        try:
            execution_store = self.state_store.for_execution(request.execution_id)
            state = await execution_store.materialize(
                request.branch_id, request.commit_hash
            )
            observation_context = self._context(
                execution_id=request.execution_id,
                task_id=request.task_id,
                benchmark_run_id=request.benchmark_run_id,
            )
            with observe_safely(
                self.observer,
                Observation(
                    name="restore",
                    context=observation_context,
                    based_on_hash=state.through_commit_hash,
                    input={"commit_hash": state.through_commit_hash},
                ),
            ) as restore:
                update_safely(restore)
            environment = self.registry.environment(
                request.environment_id, request.execution_id
            )
            with observe_safely(
                self.observer,
                Observation(
                    name="task.evaluate",
                    context=observation_context,
                    as_type="evaluator",
                    based_on_hash=state.through_commit_hash,
                    metadata={"recoverable": True},
                ),
            ) as evaluation:
                result = await _with_heartbeats(
                    asyncio.to_thread(
                        TaskScorer(
                            environment.current_task,
                            workspace=environment.workspace_path,
                        ).evaluate,
                        state,
                    ),
                    details=f"evaluate:{request.execution_id}",
                )
                if request.task_id is not None:
                    self.registry.record_evaluation(
                        request.task_id,
                        request.execution_id,
                        result.model_dump(mode="json"),
                    )
                update_safely(
                    evaluation,
                    output=result.model_dump(mode="json"),
                )
        except Exception as exc:
            event(
                "WARNING",
                "activity.attempt_failed",
                subsystem="orchestration",
                status="retryable_failure",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                **correlation,
                **exception_fields(exc),
            )
            raise
        event(
            "INFO",
            "activity.completed",
            subsystem="orchestration",
            status="completed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            commit_hash=state.through_commit_hash,
            score=result.score,
            **correlation,
        )
        return EvaluationRef(
            commit_hash=result.commit_hash,
            score=result.score,
            metrics=dict(result.metrics),
            scorer_version=result.scorer_version,
            feedback=result.feedback,
            metadata=dict(result.metadata),
        )

    @property
    def definitions(self) -> list[Any]:
        """Activity callables passed to :class:`temporalio.worker.Worker`."""
        return [
            self.run_task,
            self.evaluate_task,
        ]


__all__ = ["CorralActivities", "RuntimeRegistry", "state_ref"]
