"""Example custom metrics file for Corral benchmarks.

This file demonstrates how to define custom metrics that can be used with
the CLI or Docker runner via the --metrics-file option.

Usage:
    # With CLI
    corral bench run --image my-env:latest --metrics-file ./custom_metrics.py

    # With config file
    # Add to your benchmark_config.yaml:
    # metrics_file: ./custom_metrics.py

The file must export a METRICS list containing instantiated Metric objects.
"""

from corral.report.metrics import (
    Metric,
    MetricMetadata,
    TaskMetric,
    # Include default metrics if you want them alongside custom ones
    get_default_metrics,
)

# =============================================================================
# Custom Overall Metric Example
# =============================================================================


class TotalTrialsMetric(Metric):
    """Count the total number of trials across all tasks."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="total_trials",
            display_name="Total Trials",
            description="Total number of trials executed across all tasks",
        )

    def calculate(self, context):
        """Calculate total trials.

        Args:
            context: Object satisfying MetricContext protocol (e.g., BenchmarkResult)

        Returns:
            Total number of trials
        """
        total = 0
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                total += len(task_trials.trials)
        return total


class PerfectScoreRateMetric(Metric):
    """Calculate the percentage of trials with perfect scores (1.0)."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="perfect_score_rate",
            display_name="Perfect Score Rate",
            description="Percentage of trials achieving a perfect score (1.0)",
        )

    def calculate(self, context):
        total_trials = 0
        perfect_trials = 0

        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                for trial in task_trials.trials:
                    total_trials += 1
                    if trial.score >= 1.0:
                        perfect_trials += 1

        if total_trials == 0:
            return 0.0
        return (perfect_trials / total_trials) * 100


# =============================================================================
# Custom Task Metric Example
# =============================================================================


class TaskMaxScoreMetric(TaskMetric):
    """Find the maximum score achieved for each task."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="task_max_score",
            display_name="Task Maximum Score",
            description="Maximum score achieved across all trials for each task",
        )

    def calculate_for_task(self, context, task_id):
        """Calculate the maximum score for a specific task.

        Args:
            context: Object satisfying MetricContext protocol
            task_id: ID of the task to calculate metric for

        Returns:
            Maximum score for this task, or 0.0 if no trials
        """
        task_trials = context.get_task_trials(task_id)
        if not task_trials or not task_trials.trials:
            return 0.0

        return max(trial.score for trial in task_trials.trials)


class TaskTrialCountMetric(TaskMetric):
    """Count trials per task."""

    @property
    def metadata(self):
        return MetricMetadata(
            name="task_trial_count",
            display_name="Task Trial Count",
            description="Number of trials executed for each task",
        )

    def calculate_for_task(self, context, task_id):
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            return 0
        return len(task_trials.trials)


# =============================================================================
# Parameterized Metric Example
# =============================================================================


class TopNScoresMetric(Metric):
    """Get the top N scores across all results."""

    def __init__(self, n: int = 5):
        self.n = n

    @property
    def metadata(self):
        return MetricMetadata(
            name=f"top_{self.n}_scores",
            display_name=f"Top {self.n} Scores",
            description=f"The {self.n} highest scores achieved across all tasks",
        )

    def calculate(self, context):
        all_scores = []
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                all_scores.extend([trial.score for trial in task_trials.trials])

        all_scores.sort(reverse=True)
        return all_scores[: self.n]


# =============================================================================
# METRICS Export - This is what gets loaded!
# =============================================================================

# Option 1: Custom metrics only (replaces all defaults)
# METRICS = [
#     TotalTrialsMetric(),
#     PerfectScoreRateMetric(),
#     TaskMaxScoreMetric(),
#     TaskTrialCountMetric(),
#     TopNScoresMetric(n=3),
# ]

# Option 2: Default metrics + custom metrics (recommended)
METRICS = [
    *get_default_metrics(k_values=[1, 3, 5]),  # Include defaults for k=1,3,5
    # Add custom metrics
    TotalTrialsMetric(),
    PerfectScoreRateMetric(),
    TaskMaxScoreMetric(),
    TaskTrialCountMetric(),
    TopNScoresMetric(n=3),
]

# Option 3: Minimal set with specific defaults + custom
# METRICS = [
#     # Core metrics
#     AverageScoreMetric(),
#     SuccessRateMetric(),
#     PassAtKMetric(k=1),
#     PassAtKMetric(k=3),
#     # Custom metrics
#     TotalTrialsMetric(),
#     PerfectScoreRateMetric(),
# ]
