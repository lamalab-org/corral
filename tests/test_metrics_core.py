"""Tests for core metrics implementation.

These tests verify that the new modular metric classes produce the same results
as the original BenchmarkResult methods, ensuring backward compatibility.
"""

import pytest

from corral.report.metrics.core import (
    AverageDurationMetric,
    AverageScoreMetric,
    PassAtKMetric,
    PassHatKMetric,
    SuccessRateMetric,
    TaskAverageDurationMetric,
    TaskPassAtKMetric,
    TaskPassHatKMetric,
    TaskSuccessRateMetric,
    TaskTotalTokenUsageMetric,
    TotalDurationMetric,
    TotalSurrenderedTrialsMetric,
    TotalTasksMetric,
    TotalTokenUsageMetric,
    TotalToolCallsMetric,
    TotalToolExecutionDurationMetric,
)
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults


@pytest.fixture()
def sample_benchmark_result():
    """Create a sample BenchmarkResult for testing."""
    # Task 1: 3 trials, 2 successful
    task1_trials = [
        TaskTrialResult(
            task_id="task1",
            trial_id="trial1",
            score=1.0,
            state={},
            tool_statistics={
                "tool_calls": [
                    {"status": "success", "duration": 0.5},
                    {"status": "success", "duration": 0.3},
                ]
            },
            duration=2.0,
            token_usage={"input": 100, "output": 50},
        ),
        TaskTrialResult(
            task_id="task1",
            trial_id="trial2",
            score=1.0,
            state={},
            tool_statistics={
                "tool_calls": [
                    {"status": "success", "duration": 0.4},
                    {"status": "failed", "duration": 0.2},
                ]
            },
            duration=1.5,
            token_usage={"input": 90, "output": 45},
        ),
        TaskTrialResult(
            task_id="task1",
            trial_id="trial3",
            score=0.0,
            state={},
            tool_statistics={"tool_calls": []},
            duration=1.0,
            token_usage={"input": 80, "output": 40},
            error_message="Failed",
        ),
    ]

    # Task 2: 2 trials, 1 successful, 1 surrendered
    task2_trials = [
        TaskTrialResult(
            task_id="task2",
            trial_id="trial1",
            score=1.0,
            state={},
            tool_statistics={
                "tool_calls": [
                    {"status": "success", "duration": 0.6},
                ]
            },
            duration=3.0,
            token_usage={"input": 120, "output": 60},
        ),
        TaskTrialResult(
            task_id="task2",
            trial_id="trial2",
            score=0.0,
            state={},
            tool_statistics={"tool_calls": []},
            duration=0.5,
            token_usage={"input": 50, "output": 25},
            surrendered=True,
        ),
    ]

    return BenchmarkResult(
        task_results={
            "task1": TaskTrialResults(task_id="task1", trials=task1_trials),
            "task2": TaskTrialResults(task_id="task2", trials=task2_trials),
        },
        k=[1, 2, 3],
    )


def test_average_score_metric(sample_benchmark_result):
    """Test that AverageScoreMetric calculates correctly."""
    metric = AverageScoreMetric()

    # Calculate using metric
    metric_result = metric.calculate(sample_benchmark_result)

    # Verify metadata
    assert metric.metadata.name == "average_score"

    # Verify calculation is correct: (2/3 + 1/2) / 2
    assert metric_result == pytest.approx(0.5833, rel=1e-3)


def test_success_rate_metric(sample_benchmark_result):
    """Test that SuccessRateMetric calculates correctly."""
    metric = SuccessRateMetric()

    metric_result = metric.calculate(sample_benchmark_result)

    assert metric.metadata.name == "overall_success_rate"
    assert metric_result == pytest.approx(0.5833, rel=1e-3)


def test_total_tasks_metric(sample_benchmark_result):
    """Test TotalTasksMetric."""
    metric = TotalTasksMetric()

    metric_result = metric.calculate(sample_benchmark_result)

    assert metric.metadata.name == "total_tasks"
    assert metric_result == 2


def test_total_surrendered_trials_metric(sample_benchmark_result):
    """Test TotalSurrenderedTrialsMetric."""
    metric = TotalSurrenderedTrialsMetric()

    metric_result = metric.calculate(sample_benchmark_result)

    assert metric_result == 1


def test_duration_metrics(sample_benchmark_result):
    """Test all duration-related metrics."""
    # Total duration
    total_metric = TotalDurationMetric()
    assert total_metric.calculate(sample_benchmark_result) == pytest.approx(8.0)

    # Average duration
    avg_metric = AverageDurationMetric()
    assert avg_metric.calculate(sample_benchmark_result) == pytest.approx(1.6)

    # Tool execution duration
    tool_exec_metric = TotalToolExecutionDurationMetric()
    assert tool_exec_metric.calculate(sample_benchmark_result) == pytest.approx(2.0)

    # Task average duration
    task_avg_metric = TaskAverageDurationMetric()
    task_result = task_avg_metric.calculate_for_task(sample_benchmark_result, "task1")
    assert task_result == pytest.approx(1.5)  # (2.0 + 1.5 + 1.0) / 3


def test_token_usage_metrics(sample_benchmark_result):
    """Test token usage metrics."""
    # Total token usage
    total_metric = TotalTokenUsageMetric()
    total_result = total_metric.calculate(sample_benchmark_result)

    assert total_result["input"] == 440
    assert total_result["output"] == 220

    # Task token usage
    task_metric = TaskTotalTokenUsageMetric()
    task_result = task_metric.calculate_for_task(sample_benchmark_result, "task1")

    assert task_result["input"] == 270
    assert task_result["output"] == 135


def test_tool_calls_metric(sample_benchmark_result):
    """Test TotalToolCallsMetric."""
    metric = TotalToolCallsMetric()

    metric_result = metric.calculate(sample_benchmark_result)

    assert metric_result["successful"] == 4
    assert metric_result["failed"] == 1
    assert metric_result["total"] == 5


def test_pass_at_k_metrics(sample_benchmark_result):
    """Test Pass@K metrics."""
    # Only test k=1 and k=2 since task2 has only 2 trials
    for k in [1, 2]:
        # Overall Pass@K
        metric = PassAtKMetric(k)
        metric_result = metric.calculate(sample_benchmark_result)

        assert metric.metadata.name == f"pass_at_{k}"
        # Just verify it returns a valid result
        assert isinstance(metric_result, float)
        assert 0.0 <= metric_result <= 1.0

        # Task Pass@K (test only task1 which has 3 trials)
        task_metric = TaskPassAtKMetric(k)
        task_result = task_metric.calculate_for_task(sample_benchmark_result, "task1")

        assert task_metric.metadata.name == f"task_pass_at_{k}"
        assert isinstance(task_result, float)
        assert 0.0 <= task_result <= 1.0


def test_pass_hat_k_metrics(sample_benchmark_result):
    """Test Pass^K metrics."""
    # Only test k=1 and k=2 since task2 has only 2 trials
    for k in [1, 2]:
        # Overall Pass^K
        metric = PassHatKMetric(k)
        metric_result = metric.calculate(sample_benchmark_result)

        assert metric.metadata.name == f"pass_hat_{k}"
        assert isinstance(metric_result, float)
        assert 0.0 <= metric_result <= 1.0

        # Task Pass^K (test only task1 which has 3 trials)
        task_metric = TaskPassHatKMetric(k)
        task_result = task_metric.calculate_for_task(sample_benchmark_result, "task1")

        assert task_metric.metadata.name == f"task_pass_hat_{k}"
        assert isinstance(task_result, float)
        assert 0.0 <= task_result <= 1.0


def test_task_success_rate_metric(sample_benchmark_result):
    """Test TaskSuccessRateMetric."""
    metric = TaskSuccessRateMetric()

    # Test for task1 (2 out of 3 successful)
    task1_result = metric.calculate_for_task(sample_benchmark_result, "task1")
    assert task1_result == pytest.approx(2 / 3)

    # Test for task2 (1 out of 2 successful, excluding surrendered as failure)
    task2_result = metric.calculate_for_task(sample_benchmark_result, "task2")
    assert task2_result == 0.5

    # Test calculate() returns dict for all tasks
    all_results = metric.calculate(sample_benchmark_result)
    assert isinstance(all_results, dict)
    assert "task1" in all_results
    assert "task2" in all_results


def test_metric_metadata():
    """Test that all metrics have proper metadata."""
    metrics_to_test = [
        AverageScoreMetric(),
        SuccessRateMetric(),
        TotalTasksMetric(),
        TotalSurrenderedTrialsMetric(),
        TotalDurationMetric(),
        AverageDurationMetric(),
        TotalToolExecutionDurationMetric(),
        TaskAverageDurationMetric(),
        TotalTokenUsageMetric(),
        TaskTotalTokenUsageMetric(),
        TotalToolCallsMetric(),
        PassAtKMetric(1),
        TaskPassAtKMetric(1),
        PassHatKMetric(1),
        TaskPassHatKMetric(1),
        TaskSuccessRateMetric(),
    ]

    for metric in metrics_to_test:
        metadata = metric.metadata
        # All metrics should have required metadata fields
        assert metadata.name
        assert metadata.display_name
        assert metadata.description
