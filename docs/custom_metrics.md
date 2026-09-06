# Custom metrics

Metrics operate on the reporting projection produced after a benchmark
completes. They do not participate in task execution or scheduling.

Implement `Metric` for one benchmark-wide value:

```python
from corral import Metric, MetricMetadata


class CompletedTrials(Metric):
    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="completed_trials",
            display_name="Completed trials",
            description="Number of trials with a submitted output",
        )

    def calculate(self, context) -> int:
        return sum(
            trial.output_ready
            for task_id in context.all_task_ids
            for trial in context.get_task_trials(task_id).trials
        )
```

Use `TaskMetric` when the result should contain one value per task:

```python
from corral import MetricMetadata, TaskMetric


class SubmittedPerTask(TaskMetric):
    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="submitted_per_task",
            display_name="Submitted trials per task",
            description="Number of trials that produced an output for each task",
        )

    def calculate_for_task(self, context, task_id: str) -> int:
        trials = context.get_task_trials(task_id)
        return sum(trial.output_ready for trial in trials.trials)
```

Pass metric instances when constructing the runner:

```python
runner = CorralRunner(
    registry,
    task_metadata,
    state_store=state_store,
    metrics=[CompletedTrials(), SubmittedPerTask()],
)

result = await runner.run("benchmark-run-1", trials_per_task=3)
```

`Metric.calculate()` receives a `MetricContext`. `BenchmarkResult` implements
that protocol with `all_task_ids` and `get_task_trials(task_id)`, so metrics can
also be tested directly against small compatible fixtures.
