"""Tests for the metrics base classes.

This module tests:
- MetricMetadata validation and configuration
- Metric abstract base class
- TaskMetric abstract base class
"""

import pytest
from pydantic import ValidationError

from corral.report.metrics.base import Metric, MetricMetadata, TaskMetric
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

    def test_concrete_metric_implementation(self):
        """Test a concrete implementation of Metric."""

        class TestMetric(Metric):
            @property
            def metadata(self) -> MetricMetadata:
                return MetricMetadata(
                    name="test_metric",
                    display_name="Test Metric",
                    description="A test metric",
                )

            def calculate(self, benchmark_result: BenchmarkResult) -> float:
                return 42.0

        metric = TestMetric()
        assert metric.metadata.name == "test_metric"

        # Create a simple benchmark result
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

        result = metric.calculate(benchmark_result)
        assert result == 42.0


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
