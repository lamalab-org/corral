"""Metrics system for benchmark results."""

from loguru import logger

from corral.report.metrics.base import Metric, MetricContext, MetricMetadata, TaskMetric

# Import core metrics for easy access
from corral.report.metrics.core import (
    AverageDurationMetric,
    AverageScoreMetric,
    PassAtKMetric,
    PassHatKMetric,
    SuccessRateMetric,
    TaskAverageDurationMetric,
    TaskAverageScoreMetric,
    TaskPassAtKMetric,
    TaskPassHatKMetric,
    TaskSuccessRateMetric,
    TaskTotalTokenUsageMetric,
    TotalDurationMetric,
    TotalSurrenderedTrialsMetric,
    TotalTasksMetric,
    TotalTokenUsageMetric,
    TotalToolCallsMetric,
    TotalToolExecutionDurationMetric,
)
from corral.report.metrics.registry import MetricRegistry, get_metrics_registry


# Explicit Metric Lists (for direct use without global registry)
def _create_overall_metrics() -> list[Metric]:
    """Create instances of all overall (non-task) metrics."""
    return [
        AverageScoreMetric(),
        SuccessRateMetric(),
        TotalTasksMetric(),
        TotalSurrenderedTrialsMetric(),
        TotalDurationMetric(),
        AverageDurationMetric(),
        TotalToolExecutionDurationMetric(),
        TotalTokenUsageMetric(),
        TotalToolCallsMetric(),
    ]


def _create_task_metrics() -> list[Metric]:
    """Create instances of all task-level metrics (excluding pass@k)."""
    return [
        TaskSuccessRateMetric(),
        TaskAverageScoreMetric(),
        TaskAverageDurationMetric(),
        TaskTotalTokenUsageMetric(),
    ]


def get_pass_metrics(k_values: list[int]) -> list[Metric]:
    """Get pass@k and pass^k metrics for specified k values.

    This function creates new metric instances without registering them
    to any registry. Use this when you want explicit control over metrics.

    Args:
        k_values: List of k values for pass@k and pass^k metrics.

    Returns:
        List of Metric instances for the specified k values.

    Example:
        >>> metrics = get_pass_metrics([1, 3, 5])
        >>> # Returns PassAtKMetric(1), PassHatKMetric(1), TaskPassAtKMetric(1), ...
    """
    metrics: list[Metric] = []
    for k in k_values:
        metrics.extend(
            [
                PassAtKMetric(k),
                PassHatKMetric(k),
                TaskPassAtKMetric(k),
                TaskPassHatKMetric(k),
            ]
        )
    return metrics


def get_default_metrics(k_values: list[int] | None = None) -> list[Metric]:
    """Get all default metrics without registering them to any registry.

    This function creates new metric instances for explicit use. Use this
    when you want full visibility and control over which metrics are used.

    Args:
        k_values: List of k values for pass@k and pass^k metrics.
                  If None, uses [1] as default.

    Returns:
        List of all default Metric instances.

    Example:
        >>> # Get all defaults with k=[1, 3]
        >>> metrics = get_default_metrics([1, 3])
        >>>
        >>> # Use explicitly with BenchmarkResult
        >>> result = BenchmarkResult(task_results=..., metrics=metrics)
        >>>
        >>> # Or combine with custom metrics
        >>> from my_metrics import CustomMetric
        >>> result = BenchmarkResult(
        ...     task_results=...,
        ...     metrics=get_default_metrics([1]) + [CustomMetric()]
        ... )
    """
    if k_values is None:
        k_values = [1]

    return (
        _create_overall_metrics() + _create_task_metrics() + get_pass_metrics(k_values)
    )


__all__ = [
    "AverageDurationMetric",
    "AverageScoreMetric",
    "Metric",
    "MetricContext",
    "MetricMetadata",
    "MetricRegistry",
    "PassAtKMetric",
    "PassHatKMetric",
    "SuccessRateMetric",
    "TaskAverageDurationMetric",
    "TaskAverageScoreMetric",
    "TaskMetric",
    "TaskPassAtKMetric",
    "TaskPassHatKMetric",
    "TaskSuccessRateMetric",
    "TaskTotalTokenUsageMetric",
    "TotalDurationMetric",
    "TotalSurrenderedTrialsMetric",
    "TotalTasksMetric",
    "TotalTokenUsageMetric",
    "TotalToolCallsMetric",
    "TotalToolExecutionDurationMetric",
    "get_default_metrics",
    "get_metrics_registry",
    "get_pass_metrics",
]
