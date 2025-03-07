import pytest

from corral.report import (
    BenchmarkResult,
    InsufficientTrialsError,
    TaskNotFoundError,
    TaskTrailResult,
    TaskTrialResults,
)


def create_dummy_results():
    return {
        "task_1": TaskTrialResults(
            task_id="task_1",
            trials=[
                TaskTrailResult(
                    task_id="task_1", score=1.0, state={}, tool_statistics={}
                ),
                TaskTrailResult(
                    task_id="task_1", score=0.0, state={}, tool_statistics={}
                ),
                TaskTrailResult(
                    task_id="task_1", score=1.0, state={}, tool_statistics={}
                ),
            ],
        ),
        "task_2": TaskTrialResults(
            task_id="task_2",
            trials=[
                TaskTrailResult(
                    task_id="task_2", score=0.0, state={}, tool_statistics={}
                ),
                TaskTrailResult(
                    task_id="task_2", score=0.0, state={}, tool_statistics={}
                ),
            ],
        ),
        "task_3": TaskTrialResults(
            task_id="task_3",
            trials=[
                TaskTrailResult(
                    task_id="task_3", score=1.0, state={}, tool_statistics={}
                ),
                TaskTrailResult(
                    task_id="task_3", score=1.0, state={}, tool_statistics={}
                ),
                TaskTrailResult(
                    task_id="task_3", score=1.0, state={}, tool_statistics={}
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
