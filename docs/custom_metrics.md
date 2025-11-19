# Custom Metrics Guide

This guide explains how to create, register, and manage custom metrics in Corral.

## Table of Contents

- [Custom Metrics Guide](#custom-metrics-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Default Metrics](#default-metrics)
    - [The MetricContext Protocol](#the-metriccontext-protocol)
    - [Parallel Metric Calculation](#parallel-metric-calculation)
  - [Creating a Custom Metric](#creating-a-custom-metric)
    - [Example: Creating an Overall Metric](#example-creating-an-overall-metric)
    - [Example: Creating a Task Metric](#example-creating-a-task-metric)
  - [Registering Metrics](#registering-metrics)
    - [Programmatic Registration](#programmatic-registration)
    - [During Benchmark Initialization](#during-benchmark-initialization)
  - [Unregistering Metrics](#unregistering-metrics)
  - [Listing Available Metrics](#listing-available-metrics)
    - [List All Metrics](#list-all-metrics)
    - [Filter by Type](#filter-by-type)
    - [Check if a Metric Exists](#check-if-a-metric-exists)
  - [Enabling/Disabling Metrics](#enablingdisabling-metrics)
    - [Disable Specific Metrics](#disable-specific-metrics)
    - [Keep Only Specific Metrics](#keep-only-specific-metrics)
    - [Clear All and Register Only Custom Metrics](#clear-all-and-register-only-custom-metrics)
  - [Advanced Examples](#advanced-examples)
    - [Metric with Parameters](#metric-with-parameters)
    - [Metric with Dependencies](#metric-with-dependencies)

## Introduction

Corral provides a flexible metrics system that allows you to:

- Define custom metrics without modifying core code
- Register/unregister metrics dynamically
- Enable/disable specific metrics as needed

### Default Metrics

Corral automatically registers the following metrics when the module is imported:

**Overall Metrics:**

- `average_score`: Mean score across all task trials
- `overall_success_rate`: Percentage of successful trials across all tasks
- `total_tasks`: Number of unique tasks in the benchmark
- `total_surrendered_trials`: Number of trials that were surrendered
- `overall_total_duration`: Total duration across all trials (seconds)
- `overall_average_duration`: Average trial duration across all tasks (seconds)
- `total_tool_execution_duration`: Total time spent executing tools across all trials (seconds)
- `total_token_usage`: Total token usage across all trials (dict with token types)
- `total_tool_calls`: Total successful and failed tool calls (dict)
- `pass_at_k`: Probability that at least 1 of k trials succeeds (for each k value)
- `pass_hat_k`: Probability that all k trials succeed (for each k value)

**Task-Level Metrics:**

- `task_success_rate`: Success rate for each individual task
- `task_average_score`: Average score for each individual task
- `task_average_duration`: Average trial duration for each task (seconds)
- `task_total_token_usage`: Total token usage for each task (dict)
- `task_pass_at_k`: Pass@k for each individual task (for each k value)
- `task_pass_hat_k`: Pass^k for each individual task (for each k value)

The `k` values for pass@k and pass^k metrics default to `[1]` but can be configured when creating a `BenchmarkResult` or by calling `register_default_metrics(k_values=[1, 3, 5])`.

### The MetricContext Protocol

Metrics work with any object that satisfies the `MetricContext` protocol. This makes the metrics system portable and testable without tight coupling to `BenchmarkResult`.

A `MetricContext` must provide:

- `all_task_ids` property: Returns a `set[str]` of all task identifiers
- `get_task_trials(task_id: str)` method: Returns trial results for a specific task or `None` if not found

This protocol-based design allows you to:

- Use metrics with different benchmarking systems
- Create mock objects for testing
- Integrate with custom test harnesses

### Parallel Metric Calculation

The registry supports parallel calculation of metrics for improved performance:

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Calculate all metrics in parallel
results = registry.calculate_all(
    benchmark_result, parallel=True, max_workers=8  # Optional: control thread pool size
)

# Calculate only specific metrics in parallel
results = registry.calculate_all(
    benchmark_result, enabled_only=["average_score", "pass_at_1"], parallel=True
)
```

Parallel calculation is beneficial when you have many independent metrics to compute.

## Creating a Custom Metric

There are two types of metrics you can create:

1. **Overall Metrics** (`Metric`): Calculate aggregate statistics across all benchmark results
2. **Task Metrics** (`TaskMetric`): Calculate statistics for individual tasks

### Example: Creating an Overall Metric

```python
from corral.report.metrics import Metric, MetricMetadata


class MyCustomMetric(Metric):
    """A custom metric that counts total trials."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="total_trials",
            display_name="Total Trials",
            description="Total number of trials across all tasks",
        )

    def calculate(self, context):
        """Calculate the metric value.

        Args:
            context: Object satisfying MetricContext protocol (e.g., BenchmarkResult)

        Returns:
            The calculated metric value (int, float, dict, etc.)
        """
        total = 0
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                total += len(task_trials.trials)
        return total
```

### Example: Creating a Task Metric

```python
from corral.report.metrics import TaskMetric, MetricMetadata
from corral.types import TaskNotFoundError, NoResultsError


class TaskMaxScoreMetric(TaskMetric):
    """A custom metric that finds the maximum score for a task."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="task_max_score",
            display_name="Task Maximum Score",
            description="Maximum score achieved across all trials for a task",
        )

    def calculate_for_task(self, context, task_id):
        """Calculate the metric value for a specific task.

        Args:
            context: Object satisfying MetricContext protocol
            task_id: ID of the task to calculate metric for

        Returns:
            The calculated metric value
        """
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = task_trials.trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        return max(trial.score for trial in trials)
```

## Registering Metrics

### Programmatic Registration

You can register custom metrics at runtime:

```python
from corral.report.metrics import get_registry

# Create your metric
custom_metric = MyCustomMetric()

# Get the global registry
registry = get_registry()

# Register the metric
registry.register(custom_metric)

# Verify registration
registered_names = [m.metadata.name for m in registry.list_all()]
print(f"Registered metrics: {registered_names}")
```

### During Benchmark Initialization

You can register metrics before running a benchmark:

```python
from corral import CorralRunner
from corral.report.metrics import get_registry
from my_metrics import MyCustomMetric

# Register custom metric
registry = get_registry()
registry.register(MyCustomMetric())

# Run benchmark (custom metrics will be included)
runner = CorralRunner(...)
results = runner.run()
```

## Unregistering Metrics

You can unregister metrics by name:

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Unregister a default metric
registry.unregister("average_score")

# Unregister your custom metric
registry.unregister("total_trials")

# Verify
remaining_names = [m.metadata.name for m in registry.list_all()]
print(f"Remaining metrics: {remaining_names}")
```

**Note:** `unregister()` raises `KeyError` if the metric is not found in the registry.

## Listing Available Metrics

### List All Metrics

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Get all metrics
all_metrics = registry.list_all()

for metric in all_metrics:
    metadata = metric.metadata
    print(f"{metadata.name}: {metadata.display_name}")
    print(f"  Description: {metadata.description}")
```

### Filter by Type

```python
from corral.report.metrics import get_registry, Metric, TaskMetric

registry = get_registry()

# Get all metrics
all_metrics = registry.list_all()

# Filter by type
overall_metrics = [
    m for m in all_metrics if isinstance(m, Metric) and not isinstance(m, TaskMetric)
]
task_metrics = [m for m in all_metrics if isinstance(m, TaskMetric)]

print(f"Overall metrics: {[m.metadata.name for m in overall_metrics]}")
print(f"Task metrics: {[m.metadata.name for m in task_metrics]}")
```

### Check if a Metric Exists

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Check if metric exists
metric_names = [m.metadata.name for m in registry.list_all()]
if "average_score" in metric_names:
    print("Average score metric is available")

# Or use get() with error handling
try:
    metric = registry.get("my_custom_metric")
    print(f"Found metric: {metric.metadata.display_name}")
except KeyError:
    print("Custom metric not found")
```

## Enabling/Disabling Metrics

### Disable Specific Metrics

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Disable metrics you don't want
metrics_to_disable = ["pass_at_1", "pass_hat_1", "total_tool_calls"]

for metric_name in metrics_to_disable:
    try:
        registry.unregister(metric_name)
        print(f"Disabled: {metric_name}")
    except KeyError:
        print(f"Metric not found: {metric_name}")
```

### Keep Only Specific Metrics

```python
from corral.report.metrics import get_registry

registry = get_registry()

# Define metrics to keep
metrics_to_keep = ["average_score", "overall_success_rate", "total_tasks"]

# Get all current metrics
all_metrics = [m.metadata.name for m in registry.list_all()]

# Unregister everything except what we want to keep
for metric_name in all_metrics:
    if metric_name not in metrics_to_keep:
        registry.unregister(metric_name)
```

### Clear All and Register Only Custom Metrics

```python
from corral.report.metrics import get_registry
from my_metrics import MetricA, MetricB, MetricC

registry = get_registry()

# Clear all registered metrics
all_metric_names = [m.metadata.name for m in registry.list_all()]
for metric_name in all_metric_names:
    registry.unregister(metric_name)

# Register only your custom metrics
registry.register(MetricA())
registry.register(MetricB())
registry.register(MetricC())
```

## Advanced Examples

### Metric with Parameters

```python
from corral.report.metrics import Metric, MetricMetadata, get_registry


class TopNScoresMetric(Metric):
    """Get the top N scores across all results."""

    def __init__(self, n=5):
        self.n = n

    @property
    def metadata(self):
        return MetricMetadata(
            name=f"top_{self.n}_scores",
            display_name=f"Top {self.n} Scores",
            description=f"The {self.n} highest scores across all tasks",
        )

    def calculate(self, context):
        all_scores = []
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                all_scores.extend([trial.score for trial in task_trials.trials])

        all_scores.sort(reverse=True)
        return all_scores[: self.n]


# Register with different parameters
registry = get_registry()

registry.register(TopNScoresMetric(n=3))
registry.register(TopNScoresMetric(n=10))
```

### Metric with Dependencies

```python
from corral.report.metrics import Metric, MetricMetadata, get_registry


class SuccessRatioMetric(Metric):
    """Calculate ratio of success to total tasks."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="success_ratio",
            display_name="Success Ratio",
            description="Ratio of successful tasks to total tasks",
        )

    def calculate(self, context):
        # Use other metrics if needed
        registry = get_registry()

        success_rate = registry.get("overall_success_rate").calculate(context)
        total_tasks = registry.get("total_tasks").calculate(context)

        if total_tasks == 0:
            return 0.0

        return success_rate / total_tasks
```
