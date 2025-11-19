from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from loguru import logger

from .base import Metric

if TYPE_CHECKING:
    from corral.report.results import BenchmarkResult


class MetricRegistry:
    """Central registry for all metrics.

    The registry maintains a collection of metrics and provides methods for:
    - Registering and unregistering metrics
    - Querying metrics by name
    - Batch calculation of multiple metrics

    The registry is thread-safe for read operations but not for concurrent
    registration/unregistration. In typical usage, metrics are registered
    once at startup.

    Attributes:
        _metrics: Dictionary mapping metric names to Metric instances
    """

    def __init__(self):
        """Initialize an empty metric registry."""
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        """Register a metric in the registry.

        If a metric with the same name already exists, a ValueError is raised.

        Args:
            metric: The metric instance to register

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register(AverageScoreMetric())
        """
        name = metric.metadata.name

        if name in self._metrics:
            raise ValueError(f"Metric '{name}' already registered")

        self._metrics[name] = metric

        logger.debug(f"Registered metric: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a metric from the registry.

        Args:
            name: The name of the metric to unregister

        Raises:
            KeyError: If the metric is not found in the registry

        Example:
            >>> registry.unregister("average_score")
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found in registry")

        del self._metrics[name]

        logger.debug(f"Unregistered metric: {name}")

    def get(self, name: str) -> Metric:
        """Get a metric by name.

        Args:
            name: The name of the metric to retrieve

        Returns:
            The metric instance

        Raises:
            KeyError: If the metric is not found in the registry

        Example:
            >>> metric = registry.get("average_score")
            >>> value = metric.calculate(benchmark_result)
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self._metrics[name]

    def list_all(self) -> list[Metric]:
        """List all registered metrics.

        Returns:
            List of Metric objects for all registered metrics

        Example:
            >>> all_metrics = registry.list_all()
            >>> for metric in all_metrics:
            ...     print(f"{metric.metadata.name}: {metric.metadata.description}")
        """
        return list(self._metrics.values())

    def calculate_all(
        self,
        benchmark_result: "BenchmarkResult",
        enabled_only: list[str] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Calculate all metrics (or only enabled ones).

        This method implements batch calculation with error handling.
        If a metric fails validation or calculation, it logs an error
        and continues with other metrics.

        Args:
            benchmark_result: The benchmark result to calculate metrics from
            enabled_only: Optional list of metric names to calculate.
                         If None, all metrics are calculated.
            parallel: If True, calculate metrics in parallel using ThreadPoolExecutor.
                     Default is False for backward compatibility.
            max_workers: Maximum number of threads for parallel execution.
                        If None, defaults to min(32, (cpu_count or 1) + 4).
                        Only used when parallel=True.

        Returns:
            Dictionary mapping metric names to their calculated values.
            Metrics that failed calculation will have None as their value.
        """
        metrics_to_calc = (
            [self._metrics[name] for name in enabled_only if name in self._metrics]
            if enabled_only is not None
            else list(self._metrics.values())
        )

        if parallel:
            return self._calculate_parallel(
                benchmark_result, metrics_to_calc, max_workers
            )
        else:
            return self._calculate_sequential(benchmark_result, metrics_to_calc)

    def _calculate_sequential(
        self, benchmark_result: "BenchmarkResult", metrics: list[Metric]
    ) -> dict[str, Any]:
        """Calculate metrics sequentially.

        Args:
            benchmark_result: The benchmark result to calculate metrics from
            metrics: List of metric instances to calculate

        Returns:
            Dictionary mapping metric names to their calculated values
        """
        results = {}

        for metric in metrics:
            name = metric.metadata.name
            try:
                results[name] = metric.calculate(benchmark_result)
            except Exception as e:
                logger.error(f"Error calculating metric '{name}': {e}")
                results[name] = None

        return results

    def _calculate_parallel(
        self,
        benchmark_result: "BenchmarkResult",
        metrics: list[Metric],
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Calculate metrics in parallel using ThreadPoolExecutor.

        This method is useful when calculating many independent metrics,
        as it can significantly reduce total computation time.

        Args:
            benchmark_result: The benchmark result to calculate metrics from
            metrics: List of metric instances to calculate
            max_workers: Maximum number of threads. If None, uses default.

        Returns:
            Dictionary mapping metric names to their calculated values
        """
        results = {}

        def calculate_metric(metric: Metric) -> tuple[str, Any]:
            """Helper function to calculate a single metric."""
            name = metric.metadata.name
            try:
                value = metric.calculate(benchmark_result)
                return (name, value)
            except Exception as e:
                logger.error(f"Error calculating metric '{name}': {e}")
                return (name, None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all metric calculations
            future_to_metric = {
                executor.submit(calculate_metric, metric): metric for metric in metrics
            }

            # Collect results as they complete
            for future in as_completed(future_to_metric):
                name, value = future.result()
                results[name] = value

        return results


# Global registry instance
_global_registry = MetricRegistry()


def get_registry() -> MetricRegistry:
    """Get the global metric registry.

    Returns:
        The global MetricRegistry instance

    Example:
        >>> from corral.report.metrics.registry import get_registry
        >>> registry = get_registry()
        >>> registry.register(MyCustomMetric())
    """
    return _global_registry
