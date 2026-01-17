"""Demonstration that MetricContext protocol works with non-BenchmarkResult objects.

This test shows that the metrics system is now portable and can work with
any object that satisfies the MetricContext protocol, not just BenchmarkResult.
"""

from dataclasses import dataclass

import pytest

from corral.report.metrics.base import Metric, MetricContext, MetricMetadata, TaskMetric


@dataclass
class MockTaskData:
    """Mock task trials data."""

    task_id: str
    trials: list


@dataclass
class MockBenchmark:
    """Custom benchmark system that satisfies MetricContext protocol.

    This demonstrates that any object with all_task_ids and get_task_trials
    can work with the metrics system - no need to inherit from BenchmarkResult!
    """

    task_data: dict[str, MockTaskData]

    @property
    def all_task_ids(self) -> set[str]:
        """Satisfies MetricContext protocol."""
        return set(self.task_data.keys())

    def get_task_trials(self, task_id: str) -> MockTaskData | None:
        """Satisfies MetricContext protocol."""
        return self.task_data.get(task_id)


class SimpleCountMetric(Metric):
    """Simple metric that counts tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_count",
            display_name="Task Count",
            description="Number of tasks",
        )

    def calculate(self, context: MetricContext) -> int:
        """Calculate using protocol methods only."""
        return len(context.all_task_ids)


class SimpleTaskMetric(TaskMetric):
    """Simple task metric that counts trials per task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="trial_count",
            display_name="Trial Count",
            description="Number of trials per task",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> int:
        """Calculate using protocol methods only."""
        trials = context.get_task_trials(task_id)
        if not trials:
            return 0
        return len(trials.trials)


def test_protocol_with_custom_benchmark():
    """Test that metrics work with custom objects satisfying MetricContext."""
    # Create a custom benchmark that's NOT a BenchmarkResult
    mock_benchmark = MockBenchmark(
        task_data={
            "task1": MockTaskData("task1", [1, 2, 3]),
            "task2": MockTaskData("task2", [1, 2]),
        }
    )

    # Verify it satisfies the protocol
    assert isinstance(mock_benchmark, MetricContext)

    # Use metrics with the custom benchmark
    count_metric = SimpleCountMetric()
    result = count_metric.calculate(mock_benchmark)
    assert result == 2

    task_metric = SimpleTaskMetric()
    results = task_metric.calculate(mock_benchmark)
    assert results == {"task1": 3, "task2": 2}


def test_protocol_runtime_check():
    """Test that @runtime_checkable allows isinstance checks."""

    # Object that satisfies the protocol
    @dataclass
    class GoodObject:
        @property
        def all_task_ids(self) -> set[str]:
            return {"task1"}

        def get_task_trials(self, task_id: str):
            return None

    # Object that doesn't satisfy the protocol
    @dataclass
    class BadObject:
        pass

    good = GoodObject()
    bad = BadObject()

    assert isinstance(good, MetricContext)
    assert not isinstance(bad, MetricContext)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
