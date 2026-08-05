from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, validator

# Type alias for metric values
MetricValue = float | int | dict[str, Any]


@runtime_checkable
class MetricContext(Protocol):
    """Protocol defining what any metric context must provide.

    Any object that has these attributes/methods can be used with metrics,
    WITHOUT needing to inherit from anything.

    This makes the metrics system portable - you can use it with:
    - Corral's BenchmarkResult
    - Other benchmarking systems
    - Custom test harnesses
    - Mock objects for testing

    Example:
        >>> @dataclass
        >>> class MyBenchmark:
        ...     tasks: dict[str, TaskData]
        ...
        ...     @property
        ...     def all_task_ids(self) -> set[str]:
        ...         return set(self.tasks.keys())
        ...
        ...     def get_task_trials(self, task_id: str) -> TaskData | None:
        ...         return self.tasks.get(task_id)
        >>>
        >>> # MyBenchmark automatically satisfies MetricContext!
        >>> benchmark = MyBenchmark(tasks={...})
        >>> metric = SomeTaskMetric()
        >>> result = metric.calculate(benchmark)  # ✅ Works!
    """

    @property
    def all_task_ids(self) -> set[str]:
        """All task identifiers in this benchmark run.

        Returns:
            Set of unique task IDs
        """
        ...

    def get_task_trials(self, task_id: str) -> Any:
        """Get trial results for a specific task.

        Args:
            task_id: The task identifier

        Returns:
            Task trial results object, or None if task not found
        """
        ...


class MetricMetadata(BaseModel):
    """Metadata about a metric.

    Attributes:
        name: Unique metric identifier (must be valid Python identifier)
        display_name: Human-readable name for reports
        description: Detailed description of what the metric measures

    Example:
        >>> metadata = MetricMetadata(
        ...     name="avg_execution_time",
        ...     display_name="Average Execution Time",
        ...     description="Mean time to complete tasks"
        ... )
    """

    name: str = Field(..., description="Unique metric identifier")
    display_name: str
    description: str

    @validator("name")
    def validate_name(cls, v):
        """Ensure name is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError("name must be valid Python identifier")
        return v

    class Config:
        frozen = True  # Make immutable
        extra = "forbid"  # Reject unknown fields


class Metric(ABC):
    """Base class for all metrics.

    Works with any object that satisfies the MetricContext protocol.

    Metrics encapsulate the logic for calculating specific performance indicators
    from benchmark results. Each metric must define its metadata and calculation logic.

    Subclasses must implement:
    - metadata property: Return MetricMetadata instance
    - calculate method: Compute the metric value from context

    Example:
        >>> class SuccessRateMetric(Metric):
        ...     @property
        ...     def metadata(self) -> MetricMetadata:
        ...         return MetricMetadata(
        ...             name="success_rate",
        ...             display_name="Success Rate",
        ...             description="Percentage of tasks completed successfully"
        ...         )
        ...
        ...     def calculate(self, context):
        ...         total = len(context.all_task_ids)
        ...         successes = sum(1 for task_id in context.all_task_ids
        ...                        if context.get_task_trials(task_id).is_correct)
        ...         return (successes / total * 100) if total > 0 else 0.0
    """

    #: Rough estimate, in seconds, of how long :meth:`calculate` runs on a
    #: typical benchmark. It is used *only* to decide whether parallel
    #: (process-pool) calculation is worth its fixed overhead — spawning worker
    #: interpreters (a full re-import of the package under the `spawn` start
    #: method) and pickling the whole benchmark result to each one. The default
    #: of `0.0` marks a cheap in-memory reduction that should always run
    #: sequentially; a metric that does real work (heavy computation, I/O, or
    #: model calls) should raise this so a batch of such metrics can amortise
    #: the pool overhead. See :meth:`MetricRegistry.calculate_all`.
    estimated_cost_seconds: float = 0.0

    @property
    @abstractmethod
    def metadata(self) -> MetricMetadata:
        """Return metric metadata.

        Returns:
            MetricMetadata instance describing this metric
        """
        ...

    @abstractmethod
    def calculate(self, context: MetricContext) -> MetricValue:
        """Calculate the metric value.

        Args:
            context: Any object satisfying MetricContext protocol

        Returns:
            Calculated metric value (can be float, dict for multi-value metrics, or int)

        Raises:
            ValueError: If context is invalid or missing required data
        """
        ...


class TaskMetric(Metric):
    """Base class for metrics calculated per task.

    Automatically iterates over all tasks in the context and calculates
    a metric for each one. Returns a dictionary mapping task_id to value.

    TaskMetric extends Metric to handle per-task calculations. It automatically
    aggregates results across all tasks while allowing custom per-task logic.

    Subclasses should implement:
    - metadata property: Return MetricMetadata instance
    - calculate_for_task method: Compute metric for a single task

    The base calculate method automatically calls calculate_for_task for each
    task and returns a dictionary mapping task_id to metric value.

    Example:
        >>> class TaskExecutionTimeMetric(TaskMetric):
        ...     @property
        ...     def metadata(self) -> MetricMetadata:
        ...         return MetricMetadata(
        ...             name="task_execution_time",
        ...             display_name="Task Execution Time",
        ...             description="Time taken to complete each task"
        ...         )
        ...
        ...     def calculate_for_task(self, context, task_id):
        ...         trials = context.get_task_trials(task_id)
        ...         if not trials:
        ...             return 0.0
        ...         return sum(r.execution_time for r in trials.trials) / len(trials.trials)
    """

    @abstractmethod
    def calculate_for_task(self, context: MetricContext, task_id: str) -> MetricValue:
        """Calculate metric for a specific task.

        Args:
            context: Any object satisfying MetricContext protocol
            task_id: ID of the task to calculate metric for

        Returns:
            Calculated metric value for the specified task

        Raises:
            ValueError: If task_id is not found or data is invalid
        """
        ...

    def calculate(self, context: MetricContext) -> dict[str, MetricValue]:
        """Calculate metric for all tasks.

        This method aggregates per-task calculations into a dictionary.

        This method is already implemented - it automatically calls
        calculate_for_task() for each task in the context.

        Args:
            context: Any object satisfying MetricContext protocol

        Returns:
            Dictionary mapping task_id to metric value

        Example:
            >>> metric = TaskExecutionTimeMetric()
            >>> result = metric.calculate(context)
            >>> # result = {"task1": 2.5, "task2": 3.1, "task3": 1.8}
        """
        return {
            task_id: self.calculate_for_task(context, task_id)
            for task_id in context.all_task_ids
        }
