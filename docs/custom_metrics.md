# Custom Metrics Guide

This guide explains how to create, register, and manage custom metrics in Corral.

## Introduction

Corral provides a flexible metrics system that allows you to:

- Define custom metrics without modifying core code
- Explicitly configure which metrics are used for each benchmark
- Easily see what metrics are active via introspection methods

### Default Metrics

Corral provides the following default metrics:

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

The `k` values for pass@k and pass^k metrics default to `[1]` but can be configured when creating a `BenchmarkResult`.

## Understanding BenchmarkResult

`BenchmarkResult` is the core data structure that holds all your benchmark results and computed metrics. It's automatically created when you run a benchmark using `CorralRunner.bench()` or can be created manually from task results.

**What is it?**

A `BenchmarkResult` instance contains:

- All task trial results (successes, failures, scores, etc.)
- Configuration for metrics (like k values for pass@k)
- Timing information (total duration, per-trial durations)
- Tool usage statistics (token counts, tool calls, etc.)

### Viewing Active Metrics

Each `BenchmarkResult` instance has its own metric registry. You can easily see what metrics are configured:

```python
from corral.report.results import BenchmarkResult

# Create a benchmark result
result = BenchmarkResult(task_results=my_results, k=[1, 3])

# Print a formatted table of all active metrics
result.print_metrics()
```

**Output:**

```plaintext
======================================================================
Registered Metrics (21 total)
======================================================================
  [overall] average_score: Mean score across all task trials
  [overall] overall_average_duration: Average trial duration across all tasks
  [overall] overall_success_rate: Percentage of successful trials
  [overall] overall_total_duration: Total duration across all trials
  [overall] pass_at_1: Probability at least 1 of 1 trials succeeds
  [overall] pass_at_3: Probability at least 1 of 3 trials succeeds
  [overall] pass_hat_1: Probability all 1 trials succeed
  [overall] pass_hat_3: Probability all 3 trials succeed
  [overall] total_surrendered_trials: Number of surrendered trials
  [overall] total_tasks: Number of unique tasks
  [overall] total_token_usage: Total token usage across all trials
  [overall] total_tool_calls: Total tool calls across all trials
  [overall] total_tool_execution_duration: Total tool execution time
  [task   ] task_average_duration: Average trial duration per task
  [task   ] task_average_score: Average score per task
  [task   ] task_pass_at_1: Pass@1 per task
  [task   ] task_pass_at_3: Pass@3 per task
  [task   ] task_pass_hat_1: Pass^1 per task
  [task   ] task_pass_hat_3: Pass^3 per task
  [task   ] task_success_rate: Success rate per task
  [task   ] task_total_token_usage: Total token usage per task
======================================================================
```

You can also get metrics as a list of dictionaries for programmatic access:

```python
# Get metrics as structured data
metrics_info = result.list_metrics()

for m in metrics_info:
    print(f"{m['name']} ({m['type']}): {m['description']}")
```

### The MetricContext Protocol

Metrics work with any object that satisfies the `MetricContext` protocol.

A `MetricContext` must provide:

- `all_task_ids` property: Returns a `set[str]` of all task identifiers
- `get_task_trials(task_id: str)` method: Returns trial results for a specific task or `None` if not found

This protocol-based design allows you to:

- Use metrics with different benchmarking systems
- Create mock objects for testing
- Integrate with custom test harnesses

**Example: Custom Object Satisfying the Protocol**

Any object with these properties automatically satisfies the protocol:

```python
from dataclasses import dataclass


@dataclass
class MockTaskData:
    """Simple container for task trial data."""

    task_id: str
    trials: list


@dataclass
class CustomBenchmark:
    """Custom benchmark system that satisfies MetricContext protocol."""

    task_data: dict[str, MockTaskData]

    @property
    def all_task_ids(self) -> set[str]:
        """Required by MetricContext protocol."""
        return set(self.task_data.keys())

    def get_task_trials(self, task_id: str) -> MockTaskData | None:
        """Required by MetricContext protocol."""
        return self.task_data.get(task_id)


# Now you can use CustomBenchmark with any metric
from corral.report.metrics import AverageScoreMetric

benchmark = CustomBenchmark(
    task_data={
        "task1": MockTaskData("task1", [trial1, trial2]),
        "task2": MockTaskData("task2", [trial3]),
    }
)

metric = AverageScoreMetric()
result = metric.calculate(benchmark)  # ✅ Works!
```

### Parallel Metric Calculation

The registry supports parallel calculation of metrics using **thread-based parallelization** (`ThreadPoolExecutor`).

```python
from corral.report.metrics import get_metrics_registry

registry = get_metrics_registry()

# Calculate all metrics in parallel using threads
results = registry.calculate_all(
    benchmark_result, parallel=True, max_workers=8  # Optional: control thread pool size
)

# Calculate only specific metrics in parallel
results = registry.calculate_all(
    benchmark_result, enabled_only=["average_score", "pass_at_1"], parallel=True
)
```

!!! note "When to use parallel calculation"
    Parallel calculation is most beneficial when you have many independent metrics to compute. For a small number of metrics (< 5), sequential calculation may be faster due to threading overhead.

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

## Using Metrics with BenchmarkResult

Each `BenchmarkResult` instance has its own isolated metric registry. This design ensures:

- **Visibility**: You always know what metrics are active for a specific benchmark
- **Isolation**: Metrics in one benchmark don't affect others
- **Explicitness**: You can configure exactly which metrics to use

### Default Behavior

By default, `BenchmarkResult` uses all default metrics:

```python
from corral.report.results import BenchmarkResult

# Uses all default metrics with k=[1]
result = BenchmarkResult(task_results=my_results)

# Uses all default metrics with k=[1, 3, 5]
result = BenchmarkResult(task_results=my_results, k=[1, 3, 5])

# See what metrics are active
result.print_metrics()
```

### Configuring k Values for Pass@k and Pass^k Metrics

The `k` parameter controls which pass@k and pass^k metrics are calculated. By default, `k=[5]` is used, which means pass@5 and pass^5 metrics will be computed.

#### With BenchmarkResult

When creating a `BenchmarkResult` directly, pass the `k` parameter:

```python
from corral.report.results import BenchmarkResult

# Calculate pass@1, pass@3, and pass@5 metrics
result = BenchmarkResult(task_results=my_results, k=[1, 3, 5])

# Calculate only pass@1
result = BenchmarkResult(task_results=my_results, k=[1])

# Calculate pass@1 through pass@10
result = BenchmarkResult(task_results=my_results, k=list(range(1, 11)))
```

#### With CorralRunner

When using `CorralRunner`, pass the `k_values` parameter to the `bench()` method:

```python
from corral import CorralRunner

runner = CorralRunner(interface, agent)

# Run benchmark with pass@1, pass@3, and pass@5 metrics
result = runner.bench(trials_per_task=5, k_values=[1, 3, 5])

# Run benchmark with only pass@1
result = runner.bench(trials_per_task=3, k_values=[1])
```

!!! note "k values and trials"
    The maximum `k` value should not exceed the number of `trials_per_task`. For example, if you run 3 trials per task, you can calculate pass@1, pass@2, and pass@3, but not pass@5.

#### Previewing Metrics with Specific k Values

You can preview what metrics will be calculated before running:

```python
runner = CorralRunner(interface, agent)

# Preview metrics with k=[1, 3, 5]
runner.print_metrics(k_values=[1, 3, 5])
```

### Explicit Metrics Configuration

For full control, pass an explicit list of metrics:

```python
from corral.report.results import BenchmarkResult
from corral.report.metrics import (
    AverageScoreMetric,
    SuccessRateMetric,
    PassAtKMetric,
    TotalTasksMetric,
)

# Use only specific metrics
result = BenchmarkResult(
    task_results=my_results,
    metrics=[
        AverageScoreMetric(),
        SuccessRateMetric(),
        TotalTasksMetric(),
        PassAtKMetric(k=1),
        PassAtKMetric(k=3),
    ],
)

# Verify exactly what's configured
result.print_metrics()
```

### Combining Default and Custom Metrics

Use `get_default_metrics()` to get all defaults, then add your custom metrics:

```python
from corral.report.results import BenchmarkResult
from corral.report.metrics import get_default_metrics
from my_metrics import MyCustomMetric, AnotherCustomMetric

# Get all default metrics for k=[1, 3]
default_metrics = get_default_metrics(k_values=[1, 3])

# Combine with custom metrics
all_metrics = default_metrics + [MyCustomMetric(), AnotherCustomMetric()]

# Create benchmark with combined metrics
result = BenchmarkResult(task_results=my_results, metrics=all_metrics)

# See everything that's configured
result.print_metrics()
```

You can also use `get_pass_metrics()` to get just the pass@k metrics:

```python
from corral.report.metrics import (
    get_pass_metrics,
    AverageScoreMetric,
    SuccessRateMetric,
)

# Build a minimal set with custom k values
metrics = [
    AverageScoreMetric(),
    SuccessRateMetric(),
] + get_pass_metrics(k_values=[1, 5, 10])

result = BenchmarkResult(task_results=my_results, metrics=metrics)
```

## Listing Available Metrics

### List All Metrics

Use `list_metrics()` to get structured information about all metrics:

```python
from corral.report.results import BenchmarkResult

result = BenchmarkResult(task_results=my_results, k=[1, 3])

# Get all metrics as a list of dictionaries
metrics_info = result.list_metrics()

for m in metrics_info:
    print(f"[{m['type']}] {m['name']}")
    print(f"    Display: {m['display_name']}")
    print(f"    Description: {m['description']}")
```

Each dictionary contains:

- `name`: The metric's unique identifier
- `display_name`: Human-readable name for reports
- `description`: What the metric measures
- `type`: Either `"overall"` or `"task"`

### Print Metrics Table

For quick inspection, use `print_metrics()`:

```python
result = BenchmarkResult(task_results=my_results, k=[1, 3])
result.print_metrics()
```

This prints a formatted table showing all metrics sorted by type.

### Filter by Type

```python
from corral.report.results import BenchmarkResult

result = BenchmarkResult(task_results=my_results, k=[1])
metrics_info = result.list_metrics()

# Get only overall metrics
overall_metrics = [m for m in metrics_info if m["type"] == "overall"]
print(f"Overall metrics: {[m['name'] for m in overall_metrics]}")

# Get only task metrics
task_metrics = [m for m in metrics_info if m["type"] == "task"]
print(f"Task metrics: {[m['name'] for m in task_metrics]}")
```

## Using Metrics with CorralRunner

`CorralRunner` uses a `MetricRegistry` internally to manage metrics. This provides:

- Consistent metrics management throughout the codebase
- Parallel metric calculation support via `registry.calculate_all(parallel=True)`
- Thread-safe metric operations
- Direct access to the registry for advanced use cases

### Default Behavior (Backward Compatible)

By default, `CorralRunner` uses all default metrics. Existing code continues to work without changes:

```python
from corral import CorralRunner

# Uses all default metrics - same as before
runner = CorralRunner(interface, agent)
result = runner.bench(trials_per_task=3, k_values=[1, 3])
```

### Inspect Metrics Before Running

You can inspect what metrics will be used before running the benchmark:

```python
from corral import CorralRunner

runner = CorralRunner(interface, agent)

# See what metrics will be used with k=[1, 3]
runner.print_metrics(k_values=[1, 3])

# Or get metrics as structured data
metrics_info = runner.list_metrics(k_values=[1, 3])
for m in metrics_info:
    print(f"[{m['type']}] {m['name']}: {m['description']}")

# Then run the benchmark
result = runner.bench(trials_per_task=3, k_values=[1, 3])
```

**Output from `print_metrics()`:**

```plaintext
======================================================================
Configured Metrics (21 total)
======================================================================
  [overall] average_score: Mean score across all task trials
  [overall] overall_average_duration: Average trial duration across all tasks
  [overall] overall_success_rate: Percentage of successful trials
  ...
  [task   ] task_average_score: Average score per task
  [task   ] task_pass_at_1: Pass@1 per task
  ...
======================================================================
```

### Explicit Metrics at Initialization

For full control, pass an explicit list of metrics at initialization:

```python
from corral import CorralRunner
from corral.report.metrics import (
    AverageScoreMetric,
    SuccessRateMetric,
    PassAtKMetric,
    get_default_metrics,
)

# Option 1: Minimal metrics
runner = CorralRunner(
    interface,
    agent,
    metrics=[
        AverageScoreMetric(),
        SuccessRateMetric(),
        PassAtKMetric(k=1),
    ],
)

# Option 2: Defaults + custom
from my_metrics import MyCustomMetric

runner = CorralRunner(
    interface, agent, metrics=get_default_metrics([1, 3]) + [MyCustomMetric()]
)

# Verify configuration
runner.print_metrics()

# Run benchmark
result = runner.bench(trials_per_task=3)
```

### Register/Unregister Metrics Dynamically

You can modify metrics after initialization using `register_metric()` and `unregister_metric()`:

```python
from corral import CorralRunner
from corral.report.metrics import PassAtKMetric

runner = CorralRunner(interface, agent)

# Add a custom metric after initialization
from my_metrics import MyCustomMetric

runner.register_metric(MyCustomMetric())

# Check current metrics
runner.print_metrics()

# Remove a default metric you don't need
removed = runner.unregister_metric("total_tool_execution_duration")
if removed:
    print(f"Removed: {removed.metadata.display_name}")

# Replace an existing metric with a different configuration
runner.unregister_metric("pass_at_1")
runner.register_metric(PassAtKMetric(k=5))

# Run benchmark with modified metrics
result = runner.bench(trials_per_task=3)
```

### Reset or Clear Metrics

Use `clear_metrics()` to start fresh, or `reset_metrics()` to restore defaults:

```python
from corral import CorralRunner
from corral.report.metrics import SuccessRateMetric, AverageScoreMetric

runner = CorralRunner(interface, agent)

# Start fresh with no metrics
runner.clear_metrics()

# Add only what you need
runner.register_metric(SuccessRateMetric())
runner.register_metric(AverageScoreMetric())

# Check configuration
runner.print_metrics()  # Shows only 2 metrics

# Or reset to defaults with specific k values
runner.reset_metrics(k_values=[1, 3, 5])

# Run benchmark
result = runner.bench(trials_per_task=5)
```

### Accessing the Registry Directly

For advanced use cases, you can access the internal `MetricRegistry` directly:

```python
from corral import CorralRunner
from corral.report.metrics import AverageScoreMetric, SuccessRateMetric

runner = CorralRunner(
    interface, agent, metrics=[AverageScoreMetric(), SuccessRateMetric()]
)

# Access the internal registry
registry = runner.metric_registry

# Get a specific metric by name
avg_score = registry.get("average_score")

# List all registered metrics
all_metrics = registry.list_all()

# Use registry's parallel calculation (after benchmark completes)
result = runner.bench(trials_per_task=3)
metric_values = registry.calculate_all(result, parallel=True, max_workers=4)
```

### Benefits of CorralRunner Metrics Integration

| Aspect | Before | After |
|--------|--------|-------|
| **Visibility** | Metrics hidden inside `bench()` | Visible at runner creation |
| **Inspection** | Only after benchmark completes | Before and after with `print_metrics()` |
| **Configuration** | No way to customize | Explicit `metrics=` parameter |
| **Discoverability** | Must read source code | `list_metrics()` shows available metrics |
| **Dynamic Changes** | Not possible | `register_metric()` / `unregister_metric()` |
| **Flexibility** | Fixed at initialization | Modify anytime before `bench()` |
| **Registry Access** | N/A | Direct access via `metric_registry` property |

## Advanced Examples

### Metric with Parameters

```python
from corral.report.metrics import Metric, MetricMetadata
from corral.report.results import BenchmarkResult


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


# Use with explicit metrics configuration
from corral.report.metrics import get_default_metrics

metrics = get_default_metrics([1]) + [
    TopNScoresMetric(n=3),
    TopNScoresMetric(n=10),
]

result = BenchmarkResult(task_results=my_results, metrics=metrics)
```

### Sharing Metrics Between Benchmarks

If you need multiple `BenchmarkResult` instances to share the same metrics configuration, you can share a registry:

```python
from corral.report.results import BenchmarkResult
from corral.report.metrics import get_default_metrics
from corral.report.metrics.registry import MetricRegistry

# Create a shared registry
shared_registry = MetricRegistry()

# Register metrics to the shared registry
for metric in get_default_metrics([1, 3]):
    shared_registry.register(metric)

# Create multiple benchmarks sharing the same registry
result1 = BenchmarkResult(task_results=results_run_1, metric_registry=shared_registry)

result2 = BenchmarkResult(task_results=results_run_2, metric_registry=shared_registry)

# Both use the exact same metrics
result1.print_metrics()
result2.print_metrics()
```

## Using Custom Metrics with CLI and Docker

Custom metrics can be used with the Corral CLI and Docker runner by providing a metrics file.

### Creating a Metrics File

Create a Python file that exports a `METRICS` list containing your metric instances:

```python
# my_metrics.py
from corral.report.metrics import (
    Metric,
    MetricMetadata,
    TaskMetric,
    get_default_metrics,
)


class MyCustomMetric(Metric):
    """A custom metric example."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="my_custom_metric",
            display_name="My Custom Metric",
            description="A custom metric that counts trials",
        )

    def calculate(self, context):
        total = 0
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                total += len(task_trials.trials)
        return total


# METRICS list is required - this is what gets loaded
METRICS = get_default_metrics(k_values=[1, 3]) + [MyCustomMetric()]
```

### Using with the CLI

Use the `--metrics-file` option to load custom metrics:

```bash
# Run benchmark with custom metrics
corral bench run \
    --image ghcr.io/lamalab-org/corral-materials:latest \
    --metrics-file ./my_metrics.py \
    --trials 5

# Or use a config file
corral bench run --config benchmark_config.yaml
```

In your config file, add:

```yaml
# benchmark_config.yaml
image: ghcr.io/lamalab-org/corral-materials:latest
agent: ReActAgent
model: claude-sonnet-4-5-20250929
trials: 5
metrics_file: ./my_metrics.py  # Path to custom metrics
```

### How It Works with Docker

When you specify `--metrics-file`, Corral:

1. **Validates** the metrics file exists and is a valid Python file
2. **Mounts** the file into the Docker container at `/opt/corral-workspace/custom_metrics.py`
3. **Sets** the `METRICS_FILE` environment variable in the container
4. **Loads** the metrics dynamically using `load_metrics_from_file()`

This means:

- Your custom metrics file must be self-contained or only import from `corral`
- The file is mounted read-only for security
- If the file is invalid, the benchmark falls back to default metrics

### Loading Metrics Programmatically

You can also use the metrics loader directly in your code:

```python
from corral.report.metrics import load_metrics_from_file, validate_metrics_file

# Validate a metrics file before using it
result = validate_metrics_file("./my_metrics.py")
if result["valid"]:
    print(f"Found {result['metrics_count']} metrics: {result['metric_names']}")
else:
    print(f"Errors: {result['errors']}")

# Load metrics from file
metrics = load_metrics_from_file("./my_metrics.py")

# Use with CorralRunner
from corral import CorralRunner

runner = CorralRunner(interface, agent, metrics=metrics)
result = runner.bench(trials_per_task=5)
```

### Example: Minimal Custom Metrics File

```python
# minimal_metrics.py
from corral.report.metrics import Metric, MetricMetadata


class TrialCountMetric(Metric):
    @property
    def metadata(self):
        return MetricMetadata(
            name="trial_count",
            display_name="Trial Count",
            description="Total number of trials",
        )

    def calculate(self, context):
        return sum(
            len(context.get_task_trials(tid).trials or [])
            for tid in context.all_task_ids
            if context.get_task_trials(tid)
        )


# Export the metrics list
METRICS = [TrialCountMetric()]
```

### Example: Combining Defaults with Custom

```python
# combined_metrics.py
from corral.report.metrics import (
    get_default_metrics,
    Metric,
    MetricMetadata,
)


class PerfectScoreRate(Metric):
    @property
    def metadata(self):
        return MetricMetadata(
            name="perfect_score_rate",
            display_name="Perfect Score Rate",
            description="Percentage of trials with score >= 1.0",
        )

    def calculate(self, context):
        total, perfect = 0, 0
        for task_id in context.all_task_ids:
            trials = context.get_task_trials(task_id)
            if trials:
                for t in trials.trials:
                    total += 1
                    if t.score >= 1.0:
                        perfect += 1
        return (perfect / total * 100) if total > 0 else 0.0


# Get all defaults for k=1,3,5 and add custom metric
METRICS = get_default_metrics(k_values=[1, 3, 5]) + [PerfectScoreRate()]
```

See [`examples/custom_metrics.py`](../examples/custom_metrics.py) for a complete example with multiple metric types.
