import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from corral.report.logging import logger

from .base import Metric

if TYPE_CHECKING:
    from corral.report.results import BenchmarkResult

# Combined estimated cost (in seconds, summed over the metrics to calculate)
# below which the process pool never pays off. Spinning one up costs a fresh
# interpreter per worker — a full re-import of the package under the `spawn`
# start method — plus pickling the entire benchmark result to each worker, so
# for the cheap in-memory reductions that make up the default metric set the
# overhead dwarfs the work. Only once the metrics collectively claim more work
# than this do we parallelise. Tune via the `cost_threshold` argument to
# `MetricRegistry.calculate_all`.
PARALLEL_COST_THRESHOLD_SECONDS = 2.0


def _calculate_single_metric(args: tuple) -> tuple[str, Any]:
    """Helper function to calculate a single metric (module-level for pickling).

    Args:
        args: Tuple of (metric, benchmark_result)

    Returns:
        Tuple of (metric_name, calculated_value or None if error)
    """
    metric, benchmark_result = args
    name = metric.metadata.name
    try:
        value = metric.calculate(benchmark_result)
        return (name, value)
    except Exception as e:
        logger.warning(f"Metric '{name}' could not be calculated: {e}")
        return (name, None)


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
        cost_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Calculate all metrics (or only enabled ones).

        This method implements batch calculation with error handling.
        If a metric fails validation or calculation, it logs an error
        and continues with other metrics.

        Args:
            benchmark_result: The benchmark result to calculate metrics from
            enabled_only: Optional list of metric names to calculate.
                         If None, all metrics are calculated.
            parallel: If True, calculate metrics with a process pool *only when*
                     their combined :attr:`~corral.report.metrics.base.Metric.estimated_cost_seconds`
                     clears `cost_threshold`; otherwise (and always when
                     False) they run sequentially. Cheap metrics never pay the
                     pool's spawn + pickling overhead. Default is False.
            max_workers: Maximum number of processes for parallel execution.
                        If None, uses one worker per metric, capped at the CPU
                        count. Only used when a parallel run is chosen.
            cost_threshold: Combined estimated cost (seconds) at or above which a
                        `parallel=True` request actually uses the pool. Defaults
                        to :data:`PARALLEL_COST_THRESHOLD_SECONDS`.

        Returns:
            Dictionary mapping metric names to their calculated values.
            Metrics that failed calculation will have None as their value.
        """
        metrics_to_calc = (
            [self._metrics[name] for name in enabled_only if name in self._metrics]
            if enabled_only is not None
            else list(self._metrics.values())
        )

        if parallel and self._should_parallelize(metrics_to_calc, cost_threshold):
            return self._calculate_parallel(
                benchmark_result, metrics_to_calc, max_workers
            )
        else:
            return self._calculate_sequential(benchmark_result, metrics_to_calc)

    @staticmethod
    def _estimated_cost(metrics: list[Metric]) -> float:
        """Combined estimated calculation cost (seconds) of `metrics`.

        A missing or non-numeric hint is treated as zero, and negative hints
        are clamped, so a stray value can never force (or block) a parallel run.
        """
        total = 0.0
        for metric in metrics:
            cost = getattr(metric, "estimated_cost_seconds", 0.0)
            try:
                total += max(0.0, float(cost))
            except (TypeError, ValueError):
                continue
        return total

    @classmethod
    def _should_parallelize(
        cls, metrics: list[Metric], cost_threshold: float | None
    ) -> bool:
        """Whether a batch is expensive enough to justify the process pool.

        Parallel calculation only pays off once the metrics' combined estimated
        cost beats the fixed overhead of spawning worker processes (a fresh
        interpreter re-importing the package under `spawn`) and pickling the
        whole benchmark result to each one. Cheap in-memory reductions (the
        default) stay below the threshold and run sequentially.
        """
        threshold = (
            PARALLEL_COST_THRESHOLD_SECONDS
            if cost_threshold is None
            else cost_threshold
        )
        return cls._estimated_cost(metrics) >= threshold

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
                logger.warning(f"Metric '{name}' could not be calculated: {e}")
                results[name] = None

        return results

    def _calculate_parallel(
        self,
        benchmark_result: "BenchmarkResult",
        metrics: list[Metric],
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Calculate metrics in parallel using ProcessPoolExecutor.

        This method is useful when calculating many independent metrics,
        as it can significantly reduce total computation time by utilizing
        multiple CPU cores.

        Args:
            benchmark_result: The benchmark result to calculate metrics from
            metrics: List of metric instances to calculate
            max_workers: Maximum number of processes. If None, uses one worker
                        per metric, capped at the CPU count.

        Returns:
            Dictionary mapping metric names to their calculated values
        """
        results = {}

        # Prepare arguments for the module-level helper function
        args_list = [(metric, benchmark_result) for metric in metrics]

        # One worker per metric is the most that can ever run at once; spawning
        # more just pays extra interpreter-startup cost for idle processes.
        if max_workers is None:
            max_workers = max(1, min(len(metrics), os.cpu_count() or 1))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all metric calculations
            future_to_metric = {
                executor.submit(_calculate_single_metric, args): args[0]
                for args in args_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_metric):
                name, value = future.result()
                results[name] = value

        return results


# Global registry instance
_global_registry = MetricRegistry()


def get_metrics_registry() -> MetricRegistry:
    """Get the global metric registry.

    Returns:
        The global MetricRegistry instance

    Example:
        >>> from corral.report.metrics.registry import get_metrics_registry
        >>> registry = get_metrics_registry()
        >>> registry.register(MyCustomMetric())
    """
    return _global_registry
