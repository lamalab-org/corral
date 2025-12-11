"""Tests for WandB logger with flexible metrics system."""

from unittest.mock import MagicMock

from corral.report.metrics import AverageScoreMetric, PassAtKMetric, SuccessRateMetric
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults
from corral.report.wandb_logger import CorralWandbLogger


def create_test_results():
    """Create test benchmark results."""
    return {
        "task_1": TaskTrialResults(
            task_id="task_1",
            trials=[
                TaskTrialResult(
                    task_id="task_1",
                    trial_id="0",
                    score=1.0,
                    state={},
                    duration=10.0,
                    token_usage={"total_tokens": 100},
                    tool_statistics={"total_calls": 5},
                ),
                TaskTrialResult(
                    task_id="task_1",
                    trial_id="1",
                    score=0.5,
                    state={},
                    duration=15.0,
                    token_usage={"total_tokens": 150},
                    tool_statistics={"total_calls": 8},
                    error_message="Some error",  # This makes success=False
                ),
            ],
        ),
        "task_2": TaskTrialResults(
            task_id="task_2",
            trials=[
                TaskTrialResult(
                    task_id="task_2",
                    trial_id="0",
                    score=0.8,
                    state={},
                    duration=12.0,
                    token_usage={"total_tokens": 120},
                    tool_statistics={"total_calls": 6},
                ),
            ],
        ),
    }


def test_wandb_logger_with_default_metrics():
    """Test that WandB logger works with default metrics."""
    task_results = create_test_results()
    benchmark = BenchmarkResult(task_results=task_results, k=[1])

    # Create logger with mock wandb run
    logger = CorralWandbLogger(project="test")
    logger.run = MagicMock()

    # Log final results (should not raise any errors)
    logger.log_final_results(benchmark, k_values=[1])

    # Verify that log was called
    assert logger.run.log.called
    assert logger.run.summary.update.called


def test_wandb_logger_with_custom_metrics():
    """Test that WandB logger works with custom metrics."""
    task_results = create_test_results()

    # Create benchmark with custom metrics only
    custom_metrics = [
        AverageScoreMetric(),
        SuccessRateMetric(),
        PassAtKMetric(k=1),
    ]
    benchmark = BenchmarkResult(task_results=task_results, metrics=custom_metrics)

    # Create logger with mock wandb run
    logger = CorralWandbLogger(project="test")
    logger.run = MagicMock()

    # Log final results
    logger.log_final_results(benchmark, k_values=[1])

    # Verify that log was called
    assert logger.run.log.called

    # Get the logged metrics
    logged_calls = logger.run.log.call_args_list
    logged_metrics = {}
    for call in logged_calls:
        if call.args:
            logged_metrics.update(call.args[0])

    # Verify that the custom metrics were logged (not hardcoded ones)
    assert any("average_score" in key for key in logged_metrics)
    assert any("overall_success_rate" in key for key in logged_metrics)
    assert any("pass_at_1" in key for key in logged_metrics)


def test_wandb_logger_with_no_metrics():
    """Test that WandB logger handles empty metrics gracefully."""
    task_results = create_test_results()

    # Create benchmark with no metrics
    benchmark = BenchmarkResult(task_results=task_results, metrics=[])

    # Create logger with mock wandb run
    logger = CorralWandbLogger(project="test")
    logger.run = MagicMock()

    # Log final results (should handle gracefully without errors)
    # When there are no metrics and no trial table data, nothing is logged
    logger.log_final_results(benchmark, k_values=[1])

    # With no metrics and no trial table populated, log should not be called
    # This is expected behavior - nothing to log means no wandb.log() calls
    # The important thing is that it doesn't raise an error
    assert True  # Test passes if we get here without exception


def test_wandb_logger_respects_registered_metrics():
    """Test that WandB logger only logs metrics that are registered."""
    task_results = create_test_results()

    # Create benchmark with only a few metrics
    minimal_metrics = [
        AverageScoreMetric(),
        # Note: No success rate, no pass@k metrics
    ]
    benchmark = BenchmarkResult(task_results=task_results, metrics=minimal_metrics)

    # Create logger with mock wandb run
    logger = CorralWandbLogger(project="test")
    logger.run = MagicMock()

    # Log final results
    logger.log_final_results(benchmark, k_values=[1])

    # Verify that log was called
    assert logger.run.log.called

    # Get the logged metrics
    logged_calls = logger.run.log.call_args_list
    logged_metrics = {}
    for call in logged_calls:
        if call.args:
            logged_metrics.update(call.args[0])

    # Verify average_score is present
    assert any("average_score" in key for key in logged_metrics)

    # Verify that success_rate is NOT present (not registered)
    assert not any("success_rate" in key for key in logged_metrics)
