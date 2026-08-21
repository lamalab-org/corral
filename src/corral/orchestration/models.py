"""JSON-serializable inputs and results carried by Temporal histories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

RUNTIME_PROTOCOL_VERSION = "1"


class SandboxMode(str, Enum):
    """Execution boundary used for one task attempt."""

    LOCAL = "local"
    DOCKER = "docker"


class SandboxRetention(str, Enum):
    """When a Docker container and workspace volume are retained."""

    NEVER = "never"
    ON_FAILURE = "on-failure"
    ALWAYS = "always"


@dataclass(frozen=True)
class DockerSandboxSpec:
    """Serializable, security-oriented Docker limits for a task container."""

    image: str = "corral-benchmark:latest"
    image_digest: str | None = None
    cpus: float = 2.0
    memory: str = "4g"
    pids_limit: int = 256
    network: str = "bridge"
    environment_allowlist: tuple[str, ...] = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )
    retention: str = SandboxRetention.NEVER.value
    registry_module: str | None = None
    runtime_protocol_version: str = RUNTIME_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "retention", SandboxRetention(self.retention).value)
        if not self.image.strip():
            raise ValueError("Docker sandbox image cannot be empty")
        if (
            self.image_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest) is None
        ):
            raise ValueError("image_digest must be an immutable sha256 reference")
        if self.cpus <= 0:
            raise ValueError("Docker sandbox cpus must be greater than zero")
        if not re.fullmatch(r"[1-9][0-9]*(?:[bkmgBKMG])?", self.memory):
            raise ValueError("Docker sandbox memory must look like '4096m' or '4g'")
        if self.pids_limit < 1:
            raise ValueError("Docker sandbox pids_limit must be at least one")
        if not self.network.strip():
            raise ValueError("Docker sandbox network cannot be empty")
        if not self.runtime_protocol_version.strip():
            raise ValueError("runtime_protocol_version cannot be empty")
        if self.registry_module == "":
            raise ValueError("registry_module cannot be empty")
        for name in self.environment_allowlist:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"invalid environment variable name {name!r}")
        if len(set(self.environment_allowlist)) != len(self.environment_allowlist):
            raise ValueError("environment_allowlist cannot contain duplicates")

    @property
    def immutable_image(self) -> str:
        """Return the digest selected during preflight when available."""
        return self.image_digest or self.image


@dataclass(frozen=True)
class SandboxProfile:
    """Sandbox selection carried through replayable Workflow history."""

    mode: str = SandboxMode.LOCAL.value
    docker: DockerSandboxSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SandboxMode(self.mode).value)
        if self.mode == SandboxMode.DOCKER.value and self.docker is None:
            raise ValueError("Docker sandbox mode requires a DockerSandboxSpec")
        if self.mode == SandboxMode.LOCAL.value and self.docker is not None:
            raise ValueError("local sandbox mode cannot carry Docker configuration")

    @classmethod
    def local(cls) -> SandboxProfile:
        return cls()


@dataclass(frozen=True)
class AgentRuntimeDefinition:
    """Values needed to reconstruct a built-in agent inside an image."""

    name: str
    model: str | None = None
    api_endpoint: str | None = None
    temperature: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("agent runtime name cannot be empty")


@dataclass(frozen=True)
class EnvironmentRuntimeDefinition:
    """Values needed to reconstruct a registered environment inside an image."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)
    repository_root: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("environment runtime name cannot be empty")
        if self.repository_root == "":
            raise ValueError("environment repository_root cannot be empty")


@dataclass(frozen=True)
class ActivityPolicy:
    """Timeout and retry policy applied to Corral Activities."""

    start_to_close_seconds: float | None = None
    heartbeat_timeout_seconds: float | None = None
    maximum_attempts: int = 3
    initial_interval_seconds: float = 1.0
    maximum_interval_seconds: float = 30.0
    backoff_coefficient: float = 2.0
    non_retryable_error_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_to_close_seconds is not None and self.start_to_close_seconds <= 0:
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
    """Small Workflow-safe reference to a materialized commit projection."""

    commit_hash: str
    execution_id: str
    branch_id: str
    sequence: int
    status: str
    agent_steps: int
    submission: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
    trial_index: int = 0
    sandbox: SandboxProfile = field(default_factory=SandboxProfile.local)
    agent_runtime: AgentRuntimeDefinition | None = None
    environment_runtime: EnvironmentRuntimeDefinition | None = None


@dataclass(frozen=True)
class EvaluateTaskInput:
    execution_id: str
    environment_id: str
    commit_hash: str
    branch_id: str = "main"
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
    sandbox: SandboxProfile = field(default_factory=SandboxProfile.local)
    agent_runtime: AgentRuntimeDefinition | None = None
    environment_runtime: EnvironmentRuntimeDefinition | None = None

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ValueError("execution_id cannot be empty")
        if self.trial_index < 0:
            raise ValueError("trial_index cannot be negative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.model == "":
            raise ValueError("model cannot be empty")
        if self.sandbox.mode == SandboxMode.DOCKER.value:
            docker = self.sandbox.docker
            assert docker is not None
            if docker.registry_module is None and (
                self.agent_runtime is None or self.environment_runtime is None
            ):
                raise ValueError(
                    "Docker tasks require reconstructable agent and environment "
                    "definitions or a registry_module"
                )


@dataclass(frozen=True)
class EvaluationRef:
    commit_hash: str
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
    sandbox: SandboxProfile = field(default_factory=SandboxProfile.local)
    agent_runtime_by_task: dict[str, AgentRuntimeDefinition] = field(
        default_factory=dict
    )
    environment_runtime_by_task: dict[str, EnvironmentRuntimeDefinition] = field(
        default_factory=dict
    )

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
        if self.sandbox.mode == SandboxMode.DOCKER.value:
            docker = self.sandbox.docker
            assert docker is not None
            if docker.registry_module is None:
                missing_agent_runtime = (
                    set(self.task_ids) - self.agent_runtime_by_task.keys()
                )
                missing_environment_runtime = (
                    set(self.task_ids) - self.environment_runtime_by_task.keys()
                )
                if missing_agent_runtime or missing_environment_runtime:
                    raise ValueError(
                        "Docker benchmarks need reconstructable runtime definitions; "
                        f"missing agents={sorted(missing_agent_runtime)}, "
                        "missing environments="
                        f"{sorted(missing_environment_runtime)}"
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
    "RUNTIME_PROTOCOL_VERSION",
    "ActivityPolicy",
    "AgentRuntimeDefinition",
    "BenchmarkProgress",
    "BenchmarkWorkflowInput",
    "BenchmarkWorkflowResult",
    "DockerSandboxSpec",
    "EnvironmentRuntimeDefinition",
    "EvaluateTaskInput",
    "EvaluationRef",
    "RunTaskInput",
    "SandboxMode",
    "SandboxProfile",
    "SandboxRetention",
    "StateRef",
    "TaskWorkflowInput",
    "TaskWorkflowResult",
]
