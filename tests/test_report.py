"""Tests for benchmark reporting and metrics system."""

import json

import pytest

from corral.report.metrics import (
    AverageScoreMetric,
    Metric,
    MetricMetadata,
    discover_plugin_metrics,
    get_registry,
)
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults
from corral.types import (
    InsufficientTrialsError,
)


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


def test_pass_metrics_for_various_k_values():
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1, 2, 3, 4])

    # Use registry to get pass@k metrics
    pass_at_1_metric = benchmark.metric_registry.get("task_pass_at_1")
    pass_at_2_metric = benchmark.metric_registry.get("task_pass_at_2")

    assert pass_at_1_metric.calculate_for_task(benchmark, "task_1") > 0
    assert pass_at_2_metric.calculate_for_task(benchmark, "task_1") > 0

    # Test that k=3 raises error for task_2 (which has fewer trials)
    pass_at_3_metric = benchmark.metric_registry.get("task_pass_at_3")
    with pytest.raises(InsufficientTrialsError):
        pass_at_3_metric.calculate_for_task(benchmark, "task_2")  # More k than trials

    # Use registry for pass^k metrics
    pass_hat_1_metric = benchmark.metric_registry.get("task_pass_hat_1")
    pass_hat_2_metric = benchmark.metric_registry.get("task_pass_hat_2")
    pass_hat_3_metric = benchmark.metric_registry.get("task_pass_hat_3")

    assert pass_hat_1_metric.calculate_for_task(benchmark, "task_3") == 1.0
    assert pass_hat_2_metric.calculate_for_task(benchmark, "task_3") == 1.0
    assert pass_hat_3_metric.calculate_for_task(benchmark, "task_3") == 1.0


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

    # Verify metrics (using display names as used in JSON)
    metrics = report_data["metrics"]
    assert "Average Score" in metrics
    assert "Overall Success Rate" in metrics
    assert "Pass@1" in metrics
    assert "Pass@2" in metrics
    assert "Pass^1" in metrics
    assert "Pass^2" in metrics
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
    assert "success_rate" in task_1
    assert "average_score" in task_1
    assert len(task_1["trials"]) == 3

    # Verify task_1 metrics values
    task_1_trials = benchmark.task_results["task_1"].trials
    expected_success_rate = sum(
        1 if trial.success else 0 for trial in task_1_trials
    ) / len(task_1_trials)
    assert task_1["success_rate"] == pytest.approx(expected_success_rate)
    assert task_1["average_score"] == pytest.approx(
        sum(trial.score for trial in task_1_trials) / len(task_1_trials)
    )


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
        assert "success_rate" in task_data
        assert "average_score" in task_data


def test_display_console_report_no_crash():
    """Test that console report display works without crashing."""
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
    """Test that default metrics are automatically registered when module is imported."""
    registry = get_registry()

    # Check that core metrics are registered
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

    # registry.list_all() returns list of Metric objects
    registered_names = [m.metadata.name for m in registry.list_all()]

    for expected in expected_metrics:
        assert expected in registered_names, f"Metric '{expected}' not registered"


def test_registry_has_metrics():
    """Test that the registry contains metrics after import."""
    registry = get_registry()
    metrics = registry.list_all()

    assert len(metrics) > 0, "Registry should contain metrics after import"
    assert all(isinstance(m, Metric) for m in metrics)


def test_can_get_registered_metrics():
    """Test that we can retrieve registered metrics."""
    registry = get_registry()

    # Should be able to get a known metric
    metric = registry.get("average_score")
    assert isinstance(metric, AverageScoreMetric)
    assert metric.metadata.name == "average_score"


def test_can_list_all_metrics():
    """Test that we can list all metrics."""
    registry = get_registry()

    # Get all metrics
    all_metrics = registry.list_all()
    assert len(all_metrics) > 0

    # All should be Metric instances
    assert all(isinstance(m, Metric) for m in all_metrics)


def test_can_unregister_and_reregister_metrics():
    """Test that metrics can be unregistered and re-registered."""
    registry = get_registry()

    # Get a metric name
    metric_name = "average_score"

    # Get initial count
    initial_count = len(registry.list_all())
    assert initial_count > 0

    # Unregister a metric
    registry.unregister(metric_name)

    # Count should decrease
    after_unregister = len(registry.list_all())
    assert after_unregister == initial_count - 1

    # Re-register it
    registry.register(AverageScoreMetric())

    # Count should be back to original
    final_count = len(registry.list_all())
    assert final_count == initial_count


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

    registry = get_registry()

    # Register custom metric
    custom = CustomMetric()
    registry.register(custom)

    # Should be retrievable
    retrieved = registry.get("custom_test_metric")
    assert retrieved.metadata.name == "custom_test_metric"

    # Cleanup
    registry.unregister("custom_test_metric")


def test_plugin_discovery_handles_missing_entry_points():
    """Test that plugin discovery doesn't crash if no plugins are found."""
    # Should run without errors even if no plugins exist
    discover_plugin_metrics()


def test_registry_prevents_duplicate_registration():
    """Test that registry prevents duplicate metric registration."""
    registry = get_registry()

    # Try to register a metric that's already registered
    duplicate_metric = AverageScoreMetric()

    with pytest.raises(ValueError, match="already registered"):
        registry.register(duplicate_metric)
