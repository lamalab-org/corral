"""Task and benchmark execution resources, lazily exported."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RetryPolicy": ("corral.orchestration.models", "RetryPolicy"),
    "AgentRuntimeDefinition": (
        "corral.orchestration.models",
        "AgentRuntimeDefinition",
    ),
    "BenchmarkInput": (
        "corral.orchestration.models",
        "BenchmarkInput",
    ),
    "BenchmarkExecutionResult": (
        "corral.orchestration.models",
        "BenchmarkExecutionResult",
    ),
    "DockerInfrastructureError": (
        "corral.orchestration.launchers",
        "DockerInfrastructureError",
    ),
    "DockerSandboxSpec": ("corral.orchestration.models", "DockerSandboxSpec"),
    "DockerTaskLauncher": (
        "corral.orchestration.launchers",
        "DockerTaskLauncher",
    ),
    "EnvironmentRuntimeDefinition": (
        "corral.orchestration.models",
        "EnvironmentRuntimeDefinition",
    ),
    "EvaluationRef": ("corral.orchestration.models", "EvaluationRef"),
    "RuntimeRegistry": ("corral.orchestration.registry", "RuntimeRegistry"),
    "SandboxMode": ("corral.orchestration.models", "SandboxMode"),
    "SandboxProfile": ("corral.orchestration.models", "SandboxProfile"),
    "SandboxRetention": ("corral.orchestration.models", "SandboxRetention"),
    "StateRef": ("corral.orchestration.models", "StateRef"),
    "RunTaskInput": ("corral.orchestration.models", "RunTaskInput"),
    "TaskExecutionResult": ("corral.orchestration.models", "TaskExecutionResult"),
    "execute_task": ("corral.run", "execute_task"),
}

__all__ = [
    "AgentRuntimeDefinition",
    "BenchmarkExecutionResult",
    "BenchmarkInput",
    "DockerInfrastructureError",
    "DockerSandboxSpec",
    "DockerTaskLauncher",
    "EnvironmentRuntimeDefinition",
    "EvaluationRef",
    "RetryPolicy",
    "RunTaskInput",
    "RuntimeRegistry",
    "SandboxMode",
    "SandboxProfile",
    "SandboxRetention",
    "StateRef",
    "TaskExecutionResult",
    "execute_task",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
