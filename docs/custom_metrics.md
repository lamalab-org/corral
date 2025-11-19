# Custom Metrics Guide

This guide explains how to create, register, and manage custom metrics in Corral.

## Table of Contents

- [Introduction](#introduction)
- [Creating a Custom Metric](#creating-a-custom-metric)
- [Registering Metrics](#registering-metrics)
- [Unregistering Metrics](#unregistering-metrics)
- [Listing Available Metrics](#listing-available-metrics)
- [Enabling/Disabling Metrics](#enablingdisabling-metrics)
- [Creating a Metrics Plugin](#creating-a-metrics-plugin)

## Introduction

Corral provides a flexible metrics system that allows you to:

- Define custom metrics without modifying core code
- Register/unregister metrics dynamically
- Distribute metrics as installable packages
- Enable/disable specific metrics as needed

## Creating a Custom Metric

There are two types of metrics you can create:

1. **Overall Metrics** (`Metric`): Calculate aggregate statistics across all benchmark results
2. **Task Metrics** (`TaskMetric`): Calculate statistics for individual tasks

### Example: Creating an Overall Metric

```python
from corral import Metric, MetricMetadata


class MyCustomMetric(Metric):
    """A custom metric that counts total trials."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="total_trials",
            display_name="Total Trials",
            description="Total number of trials across all tasks",
            category="overall",
        )

    def calculate(self, benchmark_result):
        """Calculate the metric value.

        Args:
            benchmark_result: BenchmarkResult instance containing all results

        Returns:
            The calculated metric value (int, float, dict, etc.)
        """
        total = 0
        for task_results in benchmark_result.results.values():
            total += len(task_results)
        return total
```

### Example: Creating a Task Metric

```python
from corral import TaskMetric, MetricMetadata


class TaskMaxScoreMetric(TaskMetric):
    """A custom metric that finds the maximum score for a task."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="task_max_score",
            display_name="Task Maximum Score",
            description="Maximum score achieved across all trials for a task",
            category="task",
        )

    def calculate(self, task_name, task_results):
        """Calculate the metric value for a specific task.

        Args:
            task_name: Name of the task
            task_results: List of BenchmarkTaskResult for this task

        Returns:
            The calculated metric value
        """
        if not task_results:
            return 0.0

        return max(result.score for result in task_results)
```

## Registering Metrics

### Programmatic Registration

You can register custom metrics at runtime:

```python
from corral import get_registry

# Create your metric
custom_metric = MyCustomMetric()

# Get the global registry
registry = get_registry()

# Register the metric
registry.register(custom_metric)

# Verify registration
print(f"Registered metrics: {list(registry.list_metrics().keys())}")
```

### During Benchmark Initialization

You can register metrics before running a benchmark:

```python
from corral import CorralRunner, get_registry
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
from corral import get_registry

registry = get_registry()

# Unregister a default metric
registry.unregister("average_score")

# Unregister your custom metric
registry.unregister("total_trials")

# Verify
remaining_metrics = registry.list_metrics()
print(f"Remaining metrics: {list(remaining_metrics.keys())}")
```

## Listing Available Metrics

### List All Metrics

```python
from corral import get_registry

registry = get_registry()

# Get all metrics
all_metrics = registry.list_metrics()

for name, metric in all_metrics.items():
    metadata = metric.metadata
    print(f"{name}: {metadata.display_name}")
    print(f"  Description: {metadata.description}")
    print(f"  Category: {metadata.category}")
```

### Filter by Category

```python
from corral import get_registry

registry = get_registry()

# Get only overall metrics
overall_metrics = registry.list_metrics(category="overall")

# Get only task metrics
task_metrics = registry.list_metrics(category="task")

print(f"Overall metrics: {list(overall_metrics.keys())}")
print(f"Task metrics: {list(task_metrics.keys())}")
```

### Check if a Metric Exists

```python
from corral import get_registry

registry = get_registry()

if "average_score" in registry.list_metrics():
    print("Average score metric is available")

# Or use get() with default
metric = registry.get("my_custom_metric", default=None)
if metric is None:
    print("Custom metric not found")
```

## Enabling/Disabling Metrics

### Disable Specific Metrics

```python
from corral import get_registry

registry = get_registry()

# Disable metrics you don't want
metrics_to_disable = ["pass_at_1", "pass_hat_1", "total_tool_calls"]

for metric_name in metrics_to_disable:
    try:
        registry.unregister(metric_name)
        print(f"Disabled: {metric_name}")
    except ValueError:
        print(f"Metric not found: {metric_name}")
```

### Keep Only Specific Metrics

```python
from corral import get_registry

registry = get_registry()

# Define metrics to keep
metrics_to_keep = ["average_score", "success_rate", "total_tasks"]

# Get all current metrics
all_metrics = list(registry.list_metrics().keys())

# Unregister everything except what we want to keep
for metric_name in all_metrics:
    if metric_name not in metrics_to_keep:
        registry.unregister(metric_name)
```

### Clear All and Register Only Custom Metrics

```python
from corral import get_registry
from my_metrics import MetricA, MetricB, MetricC

registry = get_registry()

# Clear all registered metrics
all_metrics = list(registry.list_metrics().keys())
for metric_name in all_metrics:
    registry.unregister(metric_name)

# Register only your custom metrics
registry.register(MetricA())
registry.register(MetricB())
registry.register(MetricC())
```

## Creating a Metrics Plugin

You can distribute your custom metrics as an installable package using Python's entry points system.

### Package Structure

```
my_metrics_package/
├── pyproject.toml
├── README.md
├── src/
│   └── my_metrics/
│       ├── __init__.py
│       └── metrics.py
└── tests/
    └── test_metrics.py
```

### Define Your Metrics

`src/my_metrics/metrics.py`:

```python
from corral import Metric, MetricMetadata


class CustomMetricA(Metric):
    @property
    def metadata(self):
        return MetricMetadata(
            name="custom_metric_a",
            display_name="Custom Metric A",
            description="My first custom metric",
            category="overall",
        )

    def calculate(self, benchmark_result):
        # Your calculation logic
        return 42.0


class CustomMetricB(Metric):
    @property
    def metadata(self):
        return MetricMetadata(
            name="custom_metric_b",
            display_name="Custom Metric B",
            description="My second custom metric",
            category="overall",
        )

    def calculate(self, benchmark_result):
        # Your calculation logic
        return 100.0
```

`src/my_metrics/__init__.py`:

```python
from .metrics import CustomMetricA, CustomMetricB

__all__ = ["CustomMetricA", "CustomMetricB"]
```

### Configure Entry Points

#### Using `pyproject.toml` (Recommended)

```toml
[project]
name = "my-metrics-package"
version = "0.1.0"
description = "Custom metrics for Corral"
dependencies = [
    "corral>=0.2.0",
]

[project.entry-points."corral.metrics"]
custom_metric_a = "my_metrics.metrics:CustomMetricA"
custom_metric_b = "my_metrics.metrics:CustomMetricB"
```

#### Using `setup.py` (Legacy)

```python
from setuptools import setup, find_packages

setup(
    name="my-metrics-package",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "corral>=0.2.0",
    ],
    entry_points={
        "corral.metrics": [
            "custom_metric_a = my_metrics.metrics:CustomMetricA",
            "custom_metric_b = my_metrics.metrics:CustomMetricB",
        ],
    },
)
```

### Install Your Plugin

```bash
# Install in development mode
pip install -e .

# Or install from PyPI (after publishing)
pip install my-metrics-package
```

### Automatic Discovery

Once installed, your metrics will be automatically discovered and registered when Corral imports its metrics module:

```python
from corral import get_registry

# Your plugin metrics are automatically registered
registry = get_registry()
metrics = registry.list_metrics()

# Your custom metrics should appear
assert "custom_metric_a" in metrics
assert "custom_metric_b" in metrics
```

## Advanced Examples

### Metric with Parameters

```python
from corral import Metric, MetricMetadata


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
            category="overall",
        )

    def calculate(self, benchmark_result):
        all_scores = []
        for task_results in benchmark_result.results.values():
            all_scores.extend([r.score for r in task_results])

        all_scores.sort(reverse=True)
        return all_scores[: self.n]


# Register with different parameters
from corral import get_registry

registry = get_registry()

registry.register(TopNScoresMetric(n=3))
registry.register(TopNScoresMetric(n=10))
```

### Metric with Dependencies

```python
from corral import Metric, MetricMetadata


class SuccessRatioMetric(Metric):
    """Calculate ratio of success to total tasks."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="success_ratio",
            display_name="Success Ratio",
            description="Ratio of successful tasks to total tasks",
            category="overall",
        )

    def calculate(self, benchmark_result):
        # Use other metrics if needed
        registry = get_registry()

        success_rate = registry.get("success_rate").calculate(benchmark_result)
        total_tasks = registry.get("total_tasks").calculate(benchmark_result)

        if total_tasks == 0:
            return 0.0

        return success_rate / total_tasks
```

## Best Practices

1. **Clear Naming**: Use descriptive, unique names for your metrics
2. **Documentation**: Provide clear descriptions in `MetricMetadata`
3. **Error Handling**: Handle edge cases (empty results, division by zero, etc.)
4. **Performance**: Keep calculations efficient, especially for large result sets
5. **Testing**: Write unit tests for your custom metrics
6. **Versioning**: Version your metric plugins appropriately
7. **Dependencies**: Clearly specify dependencies in your package metadata

## Troubleshooting

### Plugin Not Discovered

If your plugin metrics aren't being discovered:

1. Verify the entry point configuration in `pyproject.toml` or `setup.py`
2. Reinstall the package: `pip install -e .`
3. Check for import errors: `python -c "from my_metrics.metrics import CustomMetricA"`
4. Enable debug logging to see discovery messages

### Metric Registration Errors

If you get registration errors:

- **"Metric already registered"**: The metric name is already in use
- **"Invalid metric type"**: Ensure your class inherits from `Metric` or `TaskMetric`
- **"Missing metadata"**: Implement the `metadata` property correctly

### Calculation Errors

If metrics fail during calculation:

- Add try/except blocks in your `calculate()` method
- Return sensible defaults for edge cases
- Log warnings for unexpected inputs

## See Also

- [API Documentation](API_specs.md)
- [Core Metrics Reference](../src/corral/report/metrics/core.py)
- [Metrics Registry API](../src/corral/report/metrics/registry.py)
