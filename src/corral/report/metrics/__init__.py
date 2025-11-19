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
from corral.report.metrics.registry import MetricRegistry, get_registry


def register_default_metrics(k_values: list[int] | None = None) -> None:
    """Register all built-in metrics to global registry.

    This function is called automatically when the module is imported,
    registering all core metrics so they're immediately available for use.

    This function is idempotent - calling it multiple times will not duplicate metrics.

    Args:
        k_values: List of k values for pass@k and pass^k metrics.
                  If None, uses [1] as default.
    """
    registry = get_registry()

    if k_values is None:
        k_values = [1]

    # List of all default metrics to register
    default_metrics = [
        # Overall metrics
        AverageScoreMetric(),
        SuccessRateMetric(),
        TotalTasksMetric(),
        TotalSurrenderedTrialsMetric(),
        TotalDurationMetric(),
        AverageDurationMetric(),
        TotalToolExecutionDurationMetric(),
        TotalTokenUsageMetric(),
        TotalToolCallsMetric(),
        # Task-level metrics
        TaskSuccessRateMetric(),
        TaskAverageScoreMetric(),
        TaskAverageDurationMetric(),
        TaskTotalTokenUsageMetric(),
    ]

    # Add pass@k and pass^k metrics for each k value
    for k in k_values:
        default_metrics.extend(
            [
                PassAtKMetric(k),
                PassHatKMetric(k),
                TaskPassAtKMetric(k),
                TaskPassHatKMetric(k),
            ]
        )

    # Register each metric, skipping if already registered
    for metric in default_metrics:
        try:
            registry.register(metric)
        except ValueError:
            # Metric already registered, skip it
            logger.debug(
                f"Metric '{metric.metadata.name}' already registered, skipping"
            )

    logger.debug(f"Registered all default metrics for k values: {k_values}")


__all__ = [
    "AverageDurationMetric",
    # Core metrics
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
    "get_registry",
    "register_default_metrics",
]
