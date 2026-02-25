"""Tests for benchmark reporting and metrics system."""

import json

import pytest

from corral.report.metrics import (
    AverageScoreMetric,
    Metric,
    MetricMetadata,
    get_default_metrics,
    get_metrics_registry,
)
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults


def create_dummy_results():
    return {
        "task_1": TaskTrialResults(
            task_id="task_1",
            trials=[
                TaskTrialResult(
                    task_id="task_1",
                    trial_id="1",
                    score=1.0,
                    state={},
                    tool_statistics={},
                ),
                TaskTrialResult(
                    task_id="task_1",
                    trial_id="2",
                    score=0.0,
                    state={},
                    tool_statistics={},
                ),
                TaskTrialResult(
                    task_id="task_1",
                    trial_id="3",
                    score=1.0,
                    state={},
                    tool_statistics={},
                ),
            ],
        ),
        "task_2": TaskTrialResults(
            task_id="task_2",
            trials=[
                TaskTrialResult(
                    task_id="task_2",
                    trial_id="1",
                    score=0.0,
                    state={},
                    tool_statistics={},
                ),
                TaskTrialResult(
                    task_id="task_2",
                    trial_id="2",
                    score=0.0,
                    state={},
                    tool_statistics={},
                ),
            ],
        ),
        "task_3": TaskTrialResults(
            task_id="task_3",
            trials=[
                TaskTrialResult(
                    task_id="task_3",
                    trial_id="1",
                    score=1.0,
                    state={},
                    tool_statistics={},
                ),
                TaskTrialResult(
                    task_id="task_3",
                    trial_id="2",
                    score=1.0,
                    state={},
                    tool_statistics={},
                ),
                TaskTrialResult(
                    task_id="task_3",
                    trial_id="3",
                    score=1.0,
                    state={},
                    tool_statistics={},
                ),
            ],
        ),
    }


def test_generate_report_json_output(tmp_path):
    """Test that generate_report produces consistent JSON output after refactoring."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1, 2])

    # Generate report with JSON output
    report_file = tmp_path / "test_report.json"
    benchmark.generate_report(report_path=str(report_file))

    # Verify file was created
    assert report_file.exists()

    # Load and verify JSON structure
    with open(report_file) as f:
        report_data = json.load(f)

    # Verify top-level structure
    assert "metrics" in report_data
    assert "task_results" in report_data

    # Verify metrics exist
    metrics = report_data["metrics"]
    assert "Average Score" in metrics
    assert "Overall Success Rate" in metrics
    assert "Pass@1" in metrics
    assert "Pass@2" in metrics
    assert "Pass^1" in metrics
    assert "Pass^2" in metrics
    assert "Total Tasks" in metrics
    assert metrics["Total Tasks"] == 3

    # Verify metrics values match calculated values from registry
    calculated_metrics = benchmark.calculate_metrics()
    assert metrics["Average Score"] == pytest.approx(
        calculated_metrics["average_score"]
    )
    assert metrics["Overall Success Rate"] == pytest.approx(
        calculated_metrics["overall_success_rate"]
    )
    assert metrics["Pass@1"] == pytest.approx(calculated_metrics["pass_at_1"])
    assert metrics["Pass@2"] == pytest.approx(calculated_metrics["pass_at_2"])
    assert metrics["Pass^1"] == pytest.approx(calculated_metrics["pass_hat_1"])
    assert metrics["Pass^2"] == pytest.approx(calculated_metrics["pass_hat_2"])

    # Verify task results
    assert len(report_data["task_results"]) == 3
    assert "task_1" in report_data["task_results"]
    assert "task_2" in report_data["task_results"]
    assert "task_3" in report_data["task_results"]

    # Verify task_1 structure
    task_1 = report_data["task_results"]["task_1"]
    assert "trials" in task_1
    assert "Task Success Rate" in task_1
    assert "Task Average Score" in task_1
    assert len(task_1["trials"]) == 3

    # Verify task_1 metrics values
    task_1_trials = benchmark.task_results["task_1"].trials
    expected_success_rate = sum(
        1 if trial.success else 0 for trial in task_1_trials
    ) / len(task_1_trials)
    expected_avg_score = sum(trial.score for trial in task_1_trials) / len(
        task_1_trials
    )
    assert task_1["Task Success Rate"] == pytest.approx(expected_success_rate)
    assert task_1["Task Average Score"] == pytest.approx(expected_avg_score)


def test_generate_report_console_output():
    """Test that generate_report console output works without crashing."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    # Test console output (without JSON file) - should not raise
    try:
        benchmark.generate_report(report_path=None)
    except Exception as e:
        pytest.fail(f"generate_report raised unexpected exception: {e}")


def test_prepare_report_data_structure():
    """Test that _prepare_report_data returns correct structure."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1, 2])

    # Calculate metrics first
    calculated_metrics = benchmark.calculate_metrics()
    report_data = benchmark._prepare_report_data(calculated_metrics)

    # Verify structure
    assert isinstance(report_data, dict)
    assert "metrics" in report_data
    assert "task_results" in report_data

    # Verify all tasks are included
    assert len(report_data["task_results"]) == 3
    for task_id in ["task_1", "task_2", "task_3"]:
        assert task_id in report_data["task_results"]
        task_data = report_data["task_results"][task_id]
        assert "trials" in task_data
        assert "Task Success Rate" in task_data
        assert "Task Average Score" in task_data


def test_display_console_report_no_crash():
    """Test that _display_console_report executes without crashing."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    # Should not raise any exceptions
    try:
        # Calculate metrics first
        calculated_metrics = benchmark.calculate_metrics()
        benchmark._display_console_report(calculated_metrics)
    except Exception as e:
        pytest.fail(f"_display_console_report raised unexpected exception: {e}")


# Auto-registration tests


def test_default_metrics_registered_on_import():
    """Test that default metrics are available via get_default_metrics().

    Note: With the new design, metrics are NOT auto-registered to the global
    registry on import. Instead, use get_default_metrics() to get metric instances,
    or pass explicit metrics to BenchmarkResult.
    """

    # get_default_metrics returns all default metrics without registering them
    default_metrics = get_default_metrics(k_values=[1])

    # Check that core metrics are available
    expected_metrics = [
        "average_score",
        "overall_success_rate",
        "total_tasks",
        "total_surrendered_trials",
        "overall_total_duration",
        "overall_average_duration",
        "total_tool_execution_duration",
        "total_token_usage",
        "total_tool_calls",
        "task_success_rate",
        "task_average_duration",
        "task_total_token_usage",
    ]

    metric_names = [m.metadata.name for m in default_metrics]

    for expected in expected_metrics:
        assert expected in metric_names, f"Metric '{expected}' not in default metrics"


def test_registry_has_metrics():
    """Test that BenchmarkResult instance registry contains metrics.

    Note: With the new design, each BenchmarkResult has its own registry.
    The global registry is no longer auto-populated on import.
    """
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    # Instance registry should have metrics
    metrics = benchmark.metric_registry.list_all()
    assert len(metrics) > 0, "Instance registry should contain metrics"
    assert all(isinstance(m, Metric) for m in metrics)


def test_can_get_registered_metrics():
    """Test that we can retrieve registered metrics from instance registry."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    # Should be able to get a known metric from instance registry
    metric = benchmark.metric_registry.get("average_score")
    assert isinstance(metric, AverageScoreMetric)
    assert metric.metadata.name == "average_score"


def test_custom_metric_registration():
    """Test that custom metrics can be registered alongside default ones."""

    class CustomMetric(Metric):
        @property
        def metadata(self) -> MetricMetadata:
            return MetricMetadata(
                name="custom_test_metric",
                display_name="Custom Test Metric",
                description="A test metric",
            )

        def calculate(self, benchmark_result) -> float:
            return 42.0

    registry = get_metrics_registry()

    # Register custom metric
    custom = CustomMetric()
    registry.register(custom)

    # Should be retrievable
    retrieved = registry.get("custom_test_metric")
    assert retrieved.metadata.name == "custom_test_metric"

    # Cleanup
    registry.unregister("custom_test_metric")
