"""Tests for the metrics base classes.

This module tests:
- MetricMetadata validation and configuration
- Metric abstract base class
- TaskMetric abstract base class
"""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from corral.report.metrics.base import Metric, MetricContext, MetricMetadata, TaskMetric
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults


class TestMetricMetadata:
    """Test MetricMetadata validation and behavior."""

    def test_valid_metadata(self):
        """Test creating valid MetricMetadata."""
        metadata = MetricMetadata(
            name="test_metric",
            display_name="Test Metric",
            description="A test metric",
        )

        assert metadata.name == "test_metric"
        assert metadata.display_name == "Test Metric"
        assert metadata.description == "A test metric"

    def test_invalid_name_with_spaces(self):
        """Test that names with spaces are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetricMetadata(
                name="invalid name",
                display_name="Invalid Name",
                description="Test",
            )
        assert "name must be valid Python identifier" in str(exc_info.value)

    def test_invalid_name_with_hyphens(self):
        """Test that names with hyphens are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetricMetadata(
                name="invalid-name",
                display_name="Invalid Name",
                description="Test",
            )
        assert "name must be valid Python identifier" in str(exc_info.value)

    def test_immutability(self):
        """Test that MetricMetadata is immutable."""
        metadata = MetricMetadata(
            name="test_metric",
            display_name="Test",
            description="Test",
        )

        with pytest.raises(ValidationError):
            metadata.name = "new_name"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            MetricMetadata(
                name="test_metric",
                display_name="Test",
                description="Test",
                extra_field="not_allowed",
            )


class TestMetric:
    """Test Metric abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Metric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Metric()


class TestTaskMetric:
    """Test TaskMetric abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that TaskMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TaskMetric()

    def test_concrete_task_metric_implementation(self):
        """Test a concrete implementation of TaskMetric."""

        class TestTaskMetric(TaskMetric):
            @property
            def metadata(self) -> MetricMetadata:
                return MetricMetadata(
                    name="test_task_metric",
                    display_name="Test Task Metric",
                    description="A test task metric",
                )

            def calculate_for_task(
                self, benchmark_result: BenchmarkResult, task_id: str
            ) -> float:
                task_results = benchmark_result.task_results.get(task_id)
                if task_results is None:
                    return 0.0
                return len(task_results.trials)

        metric = TestTaskMetric()
        assert metric.metadata.name == "test_task_metric"

        # Create a benchmark result with multiple tasks
        benchmark_result = BenchmarkResult(
            task_results={
                "task1": TaskTrialResults(
                    task_id="task1",
                    trials=[
                        TaskTrialResult(
                            task_id="task1",
                            trial_id="trial1",
                            score=1.0,
                            state={},
                            tool_statistics={},
                        ),
                        TaskTrialResult(
                            task_id="task1",
                            trial_id="trial2",
                            score=0.5,
                            state={},
                            tool_statistics={},
                        ),
                    ],
                ),
                "task2": TaskTrialResults(
                    task_id="task2",
                    trials=[
                        TaskTrialResult(
                            task_id="task2",
                            trial_id="trial1",
                            score=1.0,
                            state={},
                            tool_statistics={},
                        )
                    ],
                ),
            }
        )

        # Test calculate_for_task
        assert metric.calculate_for_task(benchmark_result, "task1") == 2
        assert metric.calculate_for_task(benchmark_result, "task2") == 1

        # Test calculate (should aggregate all tasks)
        results = metric.calculate(benchmark_result)
        assert isinstance(results, dict)
        assert results == {"task1": 2, "task2": 1}

    def test_task_metric_with_no_tasks(self):
        """Test TaskMetric with empty benchmark result."""

        class TestTaskMetric(TaskMetric):
            @property
            def metadata(self) -> MetricMetadata:
                return MetricMetadata(
                    name="test_task_metric",
                    display_name="Test",
                    description="Test",
                )

            def calculate_for_task(
                self, benchmark_result: BenchmarkResult, task_id: str
            ) -> float:
                return 0.0

        metric = TestTaskMetric()
        benchmark_result = BenchmarkResult(task_results={})

        results = metric.calculate(benchmark_result)
        assert results == {}

    def test_task_metric_returns_dict_types(self):
        """Test that TaskMetric can return different value types."""

        class TestTaskMetric(TaskMetric):
            @property
            def metadata(self) -> MetricMetadata:
                return MetricMetadata(
                    name="test_task_metric",
                    display_name="Test",
                    description="Test",
                )

            def calculate_for_task(
                self, benchmark_result: BenchmarkResult, task_id: str
            ) -> dict:
                return {"sub_metric_1": 1.0, "sub_metric_2": 2.0}

        metric = TestTaskMetric()
        benchmark_result = BenchmarkResult(
            task_results={
                "task1": TaskTrialResults(
                    task_id="task1",
                    trials=[
                        TaskTrialResult(
                            task_id="task1",
                            trial_id="trial1",
                            score=1.0,
                            state={},
                            tool_statistics={},
                        )
                    ],
                )
            }
        )

        results = metric.calculate(benchmark_result)
        assert results == {"task1": {"sub_metric_1": 1.0, "sub_metric_2": 2.0}}


def test_metric_context_protocol_is_runtime_checkable():
    """Any object exposing the context surface works, without subclassing."""

    @dataclass
    class GoodObject:
        @property
        def all_task_ids(self) -> set[str]:
            return {"task1"}

        def get_task_trials(self, task_id: str):
            return None

    @dataclass
    class BadObject:
        pass

    assert isinstance(GoodObject(), MetricContext)
    assert not isinstance(BadObject(), MetricContext)


def test_metrics_calculate_over_a_non_benchmark_result_context():
    """Metrics only use the MetricContext surface, not BenchmarkResult."""

    @dataclass
    class Trials:
        trials: list

    @dataclass
    class CustomContext:
        task_data: dict[str, Trials]

        @property
        def all_task_ids(self) -> set[str]:
            return set(self.task_data)

        def get_task_trials(self, task_id: str) -> Trials | None:
            return self.task_data.get(task_id)

    metadata = MetricMetadata(name="count", display_name="Count", description="Count")

    class TaskCount(Metric):
        @property
        def metadata(self) -> MetricMetadata:
            return metadata

        def calculate(self, context: MetricContext) -> int:
            return len(context.all_task_ids)

    class TrialCount(TaskMetric):
        @property
        def metadata(self) -> MetricMetadata:
            return metadata

        def calculate_for_task(self, context: MetricContext, task_id: str) -> int:
            return len(context.get_task_trials(task_id).trials)

    context = CustomContext({"task1": Trials([1, 2, 3]), "task2": Trials([1, 2])})

    assert TaskCount().calculate(context) == 2
    assert TrialCount().calculate(context) == {"task1": 3, "task2": 2}
