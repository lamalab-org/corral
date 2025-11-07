import pytest

from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults
from corral.types import (
    InsufficientTrialsError,
    TaskNotFoundError,
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

    assert benchmark.task_pass_at_k("task_1", 1) > 0
    assert benchmark.task_pass_at_k("task_1", 2) > 0
    with pytest.raises(InsufficientTrialsError):
        benchmark.task_pass_at_k("task_2", 3)  # More k than trials

    assert benchmark.task_pass_hat_k("task_3", 1) == 1.0
    assert benchmark.task_pass_hat_k("task_3", 2) == 1.0
    assert benchmark.task_pass_hat_k("task_3", 3) == 1.0


def test_task_success_rates():
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    assert benchmark.task_success_rate("task_1") == pytest.approx(2 / 3)
    assert benchmark.task_success_rate("task_2") == 0.0
    assert benchmark.task_success_rate("task_3") == 1.0


def test_overall_metrics():
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    assert benchmark.average_score() == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert benchmark.overall_success_rate() == pytest.approx(((2 / 3) + 0 + 1) / 3)


def test_task_not_found():
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    with pytest.raises(TaskNotFoundError):
        benchmark.task_pass_at_k("task_x", 1)


def test_no_results_error():
    benchmark = BenchmarkResult(task_results={}, k=[1])

    with pytest.raises(ValueError):
        benchmark.average_score()
    with pytest.raises(ValueError):
        benchmark.overall_success_rate()


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
    import json

    with open(report_file) as f:
        report_data = json.load(f)

    # Verify top-level structure
    assert "metrics" in report_data
    assert "task_results" in report_data

    # Verify metrics
    metrics = report_data["metrics"]
    assert "average_score" in metrics
    assert "overall_success_rate" in metrics
    assert "pass@1" in metrics
    assert "pass@2" in metrics
    assert "pass^1" in metrics
    assert "pass^2" in metrics
    assert metrics["total_tasks"] == 3

    # Verify metrics values match calculated values
    assert metrics["average_score"] == pytest.approx(benchmark.average_score())
    assert metrics["overall_success_rate"] == pytest.approx(
        benchmark.overall_success_rate()
    )
    assert metrics["pass@1"] == pytest.approx(benchmark.overall_pass_at_k(1))
    assert metrics["pass@2"] == pytest.approx(benchmark.overall_pass_at_k(2))
    assert metrics["pass^1"] == pytest.approx(benchmark.overall_pass_hat_k(1))
    assert metrics["pass^2"] == pytest.approx(benchmark.overall_pass_hat_k(2))

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
    assert task_1["success_rate"] == pytest.approx(
        benchmark.task_success_rate("task_1")
    )
    assert task_1["average_score"] == pytest.approx(
        sum(trial.score for trial in benchmark.task_results["task_1"].trials)
        / len(benchmark.task_results["task_1"].trials)
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

    pass_at_k_results = {1: benchmark.overall_pass_at_k(1), 2: benchmark.overall_pass_at_k(2)}
    pass_hat_k_results = {1: benchmark.overall_pass_hat_k(1), 2: benchmark.overall_pass_hat_k(2)}

    report_data = benchmark._prepare_report_data(pass_at_k_results, pass_hat_k_results)

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
    """Test that _display_console_report executes without crashing."""
    task_results = create_dummy_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    pass_at_k_results = {1: benchmark.overall_pass_at_k(1)}
    pass_hat_k_results = {1: benchmark.overall_pass_hat_k(1)}

    # Should not raise any exceptions
    try:
        benchmark._display_console_report(pass_at_k_results, pass_hat_k_results)
    except Exception as e:
        pytest.fail(f"_display_console_report raised unexpected exception: {e}")
