"""JSON-serializable inputs and results carried by Temporal histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActivityPolicy:
    """Timeout and retry policy applied to Corral Activities."""

    start_to_close_seconds: float = 300.0
    heartbeat_timeout_seconds: float | None = 30.0
    maximum_attempts: int = 3
    initial_interval_seconds: float = 1.0
    maximum_interval_seconds: float = 30.0
    backoff_coefficient: float = 2.0
    non_retryable_error_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_to_close_seconds <= 0:
            raise ValueError("start_to_close_seconds must be greater than 0")
        if (
            self.heartbeat_timeout_seconds is not None
            and self.heartbeat_timeout_seconds <= 0
        ):
            raise ValueError("heartbeat_timeout_seconds must be greater than 0")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least 1")
        if self.initial_interval_seconds <= 0 or self.maximum_interval_seconds <= 0:
            raise ValueError("retry intervals must be greater than 0")
        if self.backoff_coefficient < 1:
            raise ValueError("backoff_coefficient must be at least 1")


@dataclass(frozen=True)
class StateRef:
    """Small Workflow-safe projection of a State stored outside Temporal."""

    state_hash: str
    state_id: str
    revision: int
    status: str
    agent_steps: int
    submission: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"submitted", "surrendered", "terminal", "failed"}


@dataclass(frozen=True)
class RunTaskInput:
    execution_id: str
    task_id: str
    environment_id: str
    agent_id: str
    started_at: str
    max_iterations: int = 10
    model: str | None = None
    dependency_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    enable_surrender: bool = False
    benchmark_run_id: str | None = None


@dataclass(frozen=True)
class EvaluateTaskInput:
    execution_id: str
    environment_id: str
    state_hash: str
    task_id: str | None = None
    benchmark_run_id: str | None = None


@dataclass(frozen=True)
class TaskWorkflowInput:
    """Durable specification for one task attempt.

    Agents and Environment definitions are deliberately addressed by IDs and
    resolved by workers. Live Python objects never enter Workflow history.
    """

    execution_id: str
    task_id: str
    environment_id: str
    agent_id: str
    trial_index: int = 0
    dependency_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_iterations: int = 10
    model: str | None = None
    enable_surrender: bool = False
    evaluate: bool = False
    activity_policy: ActivityPolicy = field(default_factory=ActivityPolicy)
    started_at: str | None = None
    benchmark_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ValueError("execution_id cannot be empty")
        if self.trial_index < 0:
            raise ValueError("trial_index cannot be negative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.model == "":
            raise ValueError("model cannot be empty")


@dataclass(frozen=True)
class EvaluationRef:
    state_hash: str
    score: float
    metrics: dict[str, float]
    scorer_version: str
    feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskWorkflowResult:
    task_id: str
    trial_index: int
    execution_id: str
    state: StateRef | None
    evaluation: EvaluationRef | None = None
    evaluation_error: str | None = None
    error: str | None = None
    unreachable_dependency: str | None = None

    @property
    def output_ready(self) -> bool:
        return self.state is not None and self.state.output is not None


@dataclass(frozen=True)
class BenchmarkWorkflowInput:
    """Everything Temporal needs to schedule and track a benchmark run."""

    benchmark_run_id: str
    task_ids: tuple[str, ...]
    trials_per_task: int
    agent_by_task: dict[str, str]
    environment_by_task: dict[str, str]
    dependency_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_iterations_by_task: dict[str, int] = field(default_factory=dict)
    model_by_task: dict[str, str] = field(default_factory=dict)
    task_queue_by_task: dict[str, str] = field(default_factory=dict)
    max_parallel: int = 1
    max_parallel_per_task: int = 1
    max_parallel_by_model: dict[str, int] = field(default_factory=dict)
    max_parallel_by_environment: dict[str, int] = field(default_factory=dict)
    enable_surrender: bool = False
    evaluate: bool = True
    activity_policy: ActivityPolicy = field(default_factory=ActivityPolicy)
    rounds_per_run: int = 0
    next_trial_index: int = 0
    completed: tuple[TaskWorkflowResult, ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_run_id:
            raise ValueError("benchmark_run_id cannot be empty")
        if not self.task_ids:
            raise ValueError("task_ids cannot be empty")
        if self.trials_per_task < 1:
            raise ValueError("trials_per_task must be at least 1")
        if self.max_parallel < 1 or self.max_parallel_per_task < 1:
            raise ValueError("parallelism limits must be at least 1")
        if self.rounds_per_run < 0:
            raise ValueError("rounds_per_run cannot be negative")
        missing_agents = set(self.task_ids) - self.agent_by_task.keys()
        missing_environments = set(self.task_ids) - self.environment_by_task.keys()
        if missing_agents or missing_environments:
            raise ValueError(
                "every task needs an agent and environment ID; "
                f"missing agents={sorted(missing_agents)}, "
                f"missing environments={sorted(missing_environments)}"
            )
        for model, limit in self.max_parallel_by_model.items():
            if limit < 1:
                raise ValueError(f"max_parallel_by_model[{model!r}] must be at least 1")
        for environment, limit in self.max_parallel_by_environment.items():
            if limit < 1:
                raise ValueError(
                    f"max_parallel_by_environment[{environment!r}] must be at least 1"
                )


@dataclass(frozen=True)
class BenchmarkProgress:
    benchmark_run_id: str
    total: int
    pending: int
    running: int
    completed: int
    failed: int
    unreachable: int


@dataclass(frozen=True)
class BenchmarkWorkflowResult:
    benchmark_run_id: str
    task_ids: tuple[str, ...]
    trials_per_task: int
    trials: tuple[TaskWorkflowResult, ...]


__all__ = [
    "ActivityPolicy",
    "BenchmarkProgress",
    "BenchmarkWorkflowInput",
    "BenchmarkWorkflowResult",
    "EvaluateTaskInput",
    "EvaluationRef",
    "RunTaskInput",
    "StateRef",
    "TaskWorkflowInput",
    "TaskWorkflowResult",
]
