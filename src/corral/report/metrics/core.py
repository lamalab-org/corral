from statistics import mean

from corral.report.metrics.base import Metric, MetricContext, MetricMetadata, TaskMetric
from corral.types import InsufficientTrialsError, NoResultsError, TaskNotFoundError


class AverageScoreMetric(Metric):
    """Overall average score across all tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="average_score",
            display_name="Average Score",
            description="Mean score across all task trials",
        )

    def calculate(self, context: MetricContext) -> float:
        """Calculate average score across all tasks.

        For each task, computes the mean score of all trials, then averages
        across all tasks.
        """
        if not context.all_task_ids:
            raise ValueError("No task results available.")

        task_averages = []
        for task_id in context.all_task_ids:
            trials = context.get_task_trials(task_id)
            if trials:
                task_averages.append(mean(trial.score for trial in trials.trials))

        return mean(task_averages)


class SuccessRateMetric(Metric):
    """Overall success rate across all tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="overall_success_rate",
            display_name="Overall Success Rate",
            description="Percentage of successful trials across all tasks",
        )

    def calculate(self, context: MetricContext) -> float:
        """Calculate overall success rate across all tasks.

        For each task, computes the success rate of all trials, then averages
        across all tasks.
        """
        if not context.all_task_ids:
            return 0.0

        task_success_rates = []
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials and task_trials.trials:
                success_rate = mean(
                    1 if trial.success else 0 for trial in task_trials.trials
                )
                task_success_rates.append(success_rate)

        return mean(task_success_rates) if task_success_rates else 0.0


class TotalTasksMetric(Metric):
    """Total number of unique tasks in the benchmark."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="total_tasks",
            display_name="Total Tasks",
            description="Number of unique tasks in the benchmark",
        )

    def calculate(self, context: MetricContext) -> int:
        """Get total number of tasks."""
        return len(context.all_task_ids)


class TotalSurrenderedTrialsMetric(Metric):
    """Total number of surrendered trials across all tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="total_surrendered_trials",
            display_name="Total Surrendered Trials",
            description="Number of trials that were surrendered",
        )

    def calculate(self, context: MetricContext) -> int:
        """Calculate total surrendered trials."""
        count = 0
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                count += sum(1 for trial in task_trials.trials if trial.surrendered)
        return count


class TotalDurationMetric(Metric):
    """Total duration across all trials."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="overall_total_duration",
            display_name="Overall Total Duration",
            description="Total duration across all trials (seconds)",
        )

    def calculate(self, context: MetricContext) -> float | None:
        """Calculate total duration."""
        total = 0.0
        has_duration = False
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                for trial in task_trials.trials:
                    if trial.duration is not None:
                        total += trial.duration
                        has_duration = True
        return total if has_duration else None


class AverageDurationMetric(Metric):
    """Overall average trial duration across all tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="overall_average_duration",
            display_name="Overall Average Duration",
            description="Average trial duration across all tasks (seconds)",
        )

    def calculate(self, context: MetricContext) -> float | None:
        """Calculate overall average duration."""
        durations = []
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                durations.extend(
                    trial.duration
                    for trial in task_trials.trials
                    if trial.duration is not None
                )
        return mean(durations) if durations else None


class TotalToolExecutionDurationMetric(Metric):
    """Total tool execution duration across all trials and tasks."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="total_tool_execution_duration",
            display_name="Total Tool Execution Duration",
            description="Total time spent executing tools across all trials (seconds)",
        )

    def calculate(self, context: MetricContext) -> float:
        """Calculate total tool execution duration."""
        total = 0.0
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                for trial in task_trials.trials:
                    total += trial.tool_execution_duration
        return total


class TaskAverageDurationMetric(TaskMetric):
    """Average trial duration per task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_average_duration",
            display_name="Task Average Duration",
            description="Average trial duration for each task (seconds)",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float | None:
        """Calculate average duration for a specific task."""
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        durations = [
            trial.duration for trial in task_trials.trials if trial.duration is not None
        ]

        return mean(durations) if durations else None


class TaskTotalDurationMetric(TaskMetric):
    """Total duration across all trials for a task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_total_duration",
            display_name="Task Total Duration",
            description="Total duration across all trials for each task (seconds)",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float | None:
        """Calculate total duration for a specific task."""
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        durations = [
            trial.duration for trial in task_trials.trials if trial.duration is not None
        ]

        return sum(durations) if durations else None


class TotalTokenUsageMetric(Metric):
    """Total token usage across all trials."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="total_token_usage",
            display_name="Total Token Usage",
            description="Total token usage across all trials",
        )

    def calculate(self, context: MetricContext) -> dict[str, int]:
        """Calculate total token usage."""
        total_tokens = {}
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                for trial in task_trials.trials:
                    if trial.token_usage:
                        for key, value in trial.token_usage.items():
                            total_tokens[key] = total_tokens.get(key, 0) + value
        return total_tokens


class TaskTotalTokenUsageMetric(TaskMetric):
    """Total token usage per task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_total_token_usage",
            display_name="Task Total Token Usage",
            description="Total token usage for each task",
        )

    def calculate_for_task(
        self, context: MetricContext, task_id: str
    ) -> dict[str, int]:
        """Calculate total token usage for a specific task."""
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        total_tokens = {}
        for trial in task_trials.trials:
            if trial.token_usage:
                for key, value in trial.token_usage.items():
                    total_tokens[key] = total_tokens.get(key, 0) + value

        return total_tokens


class TotalToolCallsMetric(Metric):
    """Total successful and failed tool calls across all trials."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="total_tool_calls",
            display_name="Total Tool Calls",
            description="Total successful and failed tool calls",
        )

    def calculate(self, context: MetricContext) -> dict[str, int]:
        """Calculate total tool calls."""
        total_calls = {"successful": 0, "failed": 0}
        for task_id in context.all_task_ids:
            task_trials = context.get_task_trials(task_id)
            if task_trials:
                for trial in task_trials.trials:
                    if "tool_calls" in trial.tool_statistics:
                        for tool_call in trial.tool_statistics["tool_calls"]:
                            # Check both 'status' field and 'error' field for failed calls
                            if tool_call.get("status") == "failed" or tool_call.get(
                                "error"
                            ):
                                total_calls["failed"] += 1
                            else:
                                total_calls["successful"] += 1
        total_calls["total"] = total_calls["successful"] + total_calls["failed"]
        return total_calls


class TaskPassAtKMetric(TaskMetric):
    """Pass@K metric calculated per task.

    pass@k is defined as: P(pass@k) = 1 - (1 - p)^k
    where p is the probability of a single trial succeeding.
    """

    def __init__(self, k: int):
        """Initialize with k value.

        Args:
            k: The number of trials to consider
        """
        self.k = k

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=f"task_pass_at_{self.k}",
            display_name=f"Task Pass@{self.k}",
            description=f"Pass@{self.k} for each individual task",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float:
        """Calculate pass@k for a specific task.

        Estimation formula:
        - If c (correct) == n (total), pass@k = 1.0
        - Otherwise, pass@k = 1 - (1 - c/n)^k

        Raises:
            TaskNotFoundError: If the task ID is not found
            InsufficientTrialsError: If there are fewer trials than k
        """
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = task_trials.trials

        if len(trials) < self.k:
            raise InsufficientTrialsError(
                f"Number of trials ({len(trials)}) is less than k ({self.k}) for task ID '{task_id}'."
            )

        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        return 1.0 if c == n else 1.0 - (1.0 - c / n) ** self.k


class PassAtKMetric(Metric):
    """Pass@K metric - probability at least one of K trials succeeds.

    Aggregates TaskPassAtKMetric results across all tasks.
    """

    def __init__(self, k: int):
        """Initialize with k value.

        Args:
            k: The number of trials to consider
        """
        self.k = k
        self._task_metric = TaskPassAtKMetric(k)

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=f"pass_at_{self.k}",
            display_name=f"Pass@{self.k}",
            description=f"Probability that at least 1 of {self.k} trials succeeds",
        )

    def calculate(self, context: MetricContext) -> float:
        """Calculate overall pass@k across all tasks.

        For each task, calculates pass@k, then averages across all tasks.
        """
        task_results = self._task_metric.calculate(context)
        return mean(task_results.values()) if task_results else 0.0


class TaskPassHatKMetric(TaskMetric):
    """Pass^K metric calculated per task.

    pass^k is defined as: P(pass^k) = p^k
    where p is the probability of a single trial succeeding.
    """

    def __init__(self, k: int):
        """Initialize with k value.

        Args:
            k: The number of trials to consider
        """
        self.k = k

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=f"task_pass_hat_{self.k}",
            display_name=f"Task Pass^{self.k}",
            description=f"Pass^{self.k} for each individual task",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float:
        """Calculate pass^k for a specific task.

        Estimation formula: pass^k = (c/n)^k
        where c is the number of correct/successful trials and n is total trials.

        Raises:
            TaskNotFoundError: If the task ID is not found
            NoResultsError: If there are no trials for the task
        """
        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = task_trials.trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        return (c / n) ** self.k


class PassHatKMetric(Metric):
    """Pass^K metric - probability all K trials succeed.

    Aggregates TaskPassHatKMetric results across all tasks.
    """

    def __init__(self, k: int):
        """Initialize with k value.

        Args:
            k: The number of trials to consider
        """
        self.k = k
        self._task_metric = TaskPassHatKMetric(k)

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=f"pass_hat_{self.k}",
            display_name=f"Pass^{self.k}",
            description=f"Probability that all {self.k} trials succeed",
        )

    def calculate(self, context: MetricContext) -> float:
        """Calculate overall pass^k across all tasks.

        For each task, calculates pass^k, then averages across all tasks.
        """
        task_results = self._task_metric.calculate(context)
        return mean(task_results.values()) if task_results else 0.0


class TaskSuccessRateMetric(TaskMetric):
    """Success rate per task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_success_rate",
            display_name="Task Success Rate",
            description="Success rate for each individual task",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float:
        """Calculate success rate for a specific task."""

        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = task_trials.trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        return mean(1 if trial.success else 0 for trial in trials)


class TaskAverageScoreMetric(TaskMetric):
    """Average score per task."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="task_average_score",
            display_name="Task Average Score",
            description="Average score for each individual task",
        )

    def calculate_for_task(self, context: MetricContext, task_id: str) -> float:
        """Calculate average score for a specific task."""

        task_trials = context.get_task_trials(task_id)
        if not task_trials:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = task_trials.trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        return mean(trial.score for trial in trials)
