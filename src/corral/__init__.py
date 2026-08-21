"""Public Corral API, loaded lazily for Temporal Workflow sandbox safety."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ActivityPolicy": ("corral.orchestration", "ActivityPolicy"),
    "AgentRuntimeDefinition": ("corral.orchestration", "AgentRuntimeDefinition"),
    "BenchmarkTaskMetadata": ("corral.run", "BenchmarkTaskMetadata"),
    "BenchmarkWorkflowInput": (
        "corral.orchestration",
        "BenchmarkWorkflowInput",
    ),
    "CorralActivities": ("corral.orchestration", "CorralActivities"),
    "CorralRunner": ("corral.run", "CorralRunner"),
    "DockerSandboxSpec": ("corral.orchestration", "DockerSandboxSpec"),
    "EvaluationResult": ("corral.evaluation", "EvaluationResult"),
    "ENVIRONMENT_NAMES": (
        "corral.runtime.environment_loader",
        "ENVIRONMENT_NAMES",
    ),
    "ENVIRONMENT_PRESETS": (
        "corral.runtime.environment_loader",
        "ENVIRONMENT_PRESETS",
    ),
    "EnvironmentPreset": (
        "corral.runtime.environment_loader",
        "EnvironmentPreset",
    ),
    "EnvironmentName": ("corral.runtime.environment_loader", "EnvironmentName"),
    "EnvironmentRuntimeDefinition": (
        "corral.orchestration",
        "EnvironmentRuntimeDefinition",
    ),
    "Metric": ("corral.report.metrics", "Metric"),
    "MetricMetadata": ("corral.report.metrics", "MetricMetadata"),
    "LangfuseObserver": ("corral.observability", "LangfuseObserver"),
    "NoOpObserver": ("corral.observability", "NoOpObserver"),
    "Observer": ("corral.observability", "Observer"),
    "RuntimeRegistry": ("corral.orchestration", "RuntimeRegistry"),
    "SandboxMode": ("corral.orchestration", "SandboxMode"),
    "SandboxProfile": ("corral.orchestration", "SandboxProfile"),
    "SandboxRetention": ("corral.orchestration", "SandboxRetention"),
    "Scorer": ("corral.evaluation", "Scorer"),
    "TaskMetric": ("corral.report.metrics", "TaskMetric"),
    "TaskScorer": ("corral.evaluation", "TaskScorer"),
    "TaskWorkflowInput": ("corral.orchestration", "TaskWorkflowInput"),
    "TemporalBenchmarkExecutor": (
        "corral.orchestration",
        "TemporalBenchmarkExecutor",
    ),
    "TemporalTaskExecutor": (
        "corral.orchestration",
        "TemporalTaskExecutor",
    ),
    "create_worker": ("corral.orchestration", "create_worker"),
    "execute_task": ("corral.orchestration", "execute_task"),
    "get_default_metrics": ("corral.report.metrics", "get_default_metrics"),
    "load_environment_group": (
        "corral.runtime.environment_loader",
        "load_environment_group",
    ),
    "normalise_environment_name": (
        "corral.runtime.environment_loader",
        "normalise_environment_name",
    ),
    "project_benchmark_result": ("corral.run", "project_benchmark_result"),
}

__all__ = [
    "ENVIRONMENT_NAMES",
    "ENVIRONMENT_PRESETS",
    "ActivityPolicy",
    "AgentRuntimeDefinition",
    "BenchmarkTaskMetadata",
    "BenchmarkWorkflowInput",
    "CorralActivities",
    "CorralRunner",
    "DockerSandboxSpec",
    "EnvironmentName",
    "EnvironmentPreset",
    "EnvironmentRuntimeDefinition",
    "EvaluationResult",
    "LangfuseObserver",
    "Metric",
    "MetricMetadata",
    "NoOpObserver",
    "Observer",
    "RuntimeRegistry",
    "SandboxMode",
    "SandboxProfile",
    "SandboxRetention",
    "Scorer",
    "TaskMetric",
    "TaskScorer",
    "TaskWorkflowInput",
    "TemporalBenchmarkExecutor",
    "TemporalTaskExecutor",
    "create_worker",
    "execute_task",
    "get_default_metrics",
    "load_environment_group",
    "normalise_environment_name",
    "project_benchmark_result",
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
