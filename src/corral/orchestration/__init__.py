"""Temporal-backed task and benchmark orchestration, lazily exported."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ActivityPolicy": ("corral.orchestration.models", "ActivityPolicy"),
    "AgentRuntimeDefinition": (
        "corral.orchestration.models",
        "AgentRuntimeDefinition",
    ),
    "BenchmarkProgress": ("corral.orchestration.models", "BenchmarkProgress"),
    "BenchmarkWorkflow": ("corral.orchestration.workflows", "BenchmarkWorkflow"),
    "BenchmarkWorkflowInput": (
        "corral.orchestration.models",
        "BenchmarkWorkflowInput",
    ),
    "BenchmarkWorkflowResult": (
        "corral.orchestration.models",
        "BenchmarkWorkflowResult",
    ),
    "CorralActivities": ("corral.orchestration.activities", "CorralActivities"),
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
    "RuntimeRegistry": ("corral.orchestration.activities", "RuntimeRegistry"),
    "SandboxMode": ("corral.orchestration.models", "SandboxMode"),
    "SandboxProfile": ("corral.orchestration.models", "SandboxProfile"),
    "SandboxRetention": ("corral.orchestration.models", "SandboxRetention"),
    "StateRef": ("corral.orchestration.models", "StateRef"),
    "TaskExecutionError": (
        "corral.orchestration.executor",
        "TaskExecutionError",
    ),
    "TaskWorkflow": ("corral.orchestration.workflows", "TaskWorkflow"),
    "TaskWorkflowInput": ("corral.orchestration.models", "TaskWorkflowInput"),
    "TaskWorkflowResult": ("corral.orchestration.models", "TaskWorkflowResult"),
    "TemporalBenchmarkExecutor": (
        "corral.orchestration.executor",
        "TemporalBenchmarkExecutor",
    ),
    "TemporalTaskExecutor": (
        "corral.orchestration.executor",
        "TemporalTaskExecutor",
    ),
    "benchmark_workflow_id": (
        "corral.orchestration.executor",
        "benchmark_workflow_id",
    ),
    "create_worker": ("corral.orchestration.worker", "create_worker"),
    "create_workflow_runner": (
        "corral.orchestration.worker",
        "create_workflow_runner",
    ),
    "execute_task": ("corral.orchestration.executor", "execute_task"),
    "task_workflow_id": ("corral.orchestration.executor", "task_workflow_id"),
}

__all__ = [
    "ActivityPolicy",
    "AgentRuntimeDefinition",
    "BenchmarkProgress",
    "BenchmarkWorkflow",
    "BenchmarkWorkflowInput",
    "BenchmarkWorkflowResult",
    "CorralActivities",
    "DockerInfrastructureError",
    "DockerSandboxSpec",
    "DockerTaskLauncher",
    "EnvironmentRuntimeDefinition",
    "EvaluationRef",
    "RuntimeRegistry",
    "SandboxMode",
    "SandboxProfile",
    "SandboxRetention",
    "StateRef",
    "TaskExecutionError",
    "TaskWorkflow",
    "TaskWorkflowInput",
    "TaskWorkflowResult",
    "TemporalBenchmarkExecutor",
    "TemporalTaskExecutor",
    "benchmark_workflow_id",
    "create_worker",
    "create_workflow_runner",
    "execute_task",
    "task_workflow_id",
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
