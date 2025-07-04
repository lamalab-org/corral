import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from loguru import logger
from rich.table import Table


class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors"""


class TaskNotFoundError(BenchmarkError):
    """Raised when a task ID is not found in the benchmark results"""


class InsufficientTrialsError(BenchmarkError):
    """Raised when there aren't enough trials for a calculation"""


class NoResultsError(BenchmarkError):
    """Raised when trying to calculate metrics without results"""


@dataclass
class ToolResponse:
    """Response from a tool execution"""

    success: bool
    result: str | None
    error: str | None


@dataclass
class TaskTrailResult:
    """Result of a single trail in a task"""

    task_id: str
    trial_id: str
    score: float
    state: dict[str, Any]  # TODO replace Any with specific types
    tool_statistics: dict[str, Any]  # TODO replace Any with specific types
    duration: float | None = None
    token_usage: dict[str, int] | None = None

    @property
    def success(self) -> bool:
        """Whether the trial was successful"""
        return self.score > 0


@dataclass
class TaskTrialResults:
    """Collection of results of trials for a single task"""

    task_id: str
    trials: list[TaskTrailResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """
    This class provides methods for calculating various metrics about benchmark performance.

    Attributes:
        task_results: Dictionary mapping task IDs to their trial results
        k: The k value(s) to use for pass@k and pass^k calculations.
           Can be a single int or a list of ints.
    """

    task_results: dict[str, TaskTrialResults]
    k: list[int] = field(default_factory=lambda: [1])
    total_duration: float | None = None

    @property
    def all_task_ids(self) -> list[str]:
        """Get all task_ids in the benchmark"""
        return list(self.task_results.keys())

    @property
    def total_tasks(self) -> int:
        """Get number of unique tasks in the benchmark"""
        return len(self.task_results)

    @property
    def all_results(self) -> list[TaskTrailResult]:
        """Get a flat list of all individual trails across all tasks"""
        return [
            result
            for task_trials in self.task_results.values()
            for result in task_trials.trials
        ]

    def task_average_duration(self, task_id: str) -> float | None:
        """Calculate average trial duration for a task"""
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        durations = [t.duration for t in trials if t.duration is not None]

        return mean(durations) if durations else None

    def overall_average_duration(self) -> float | None:
        """Calculate overall average trial duration across all tasks"""
        all_durations = []
        for task_trials in self.task_results.values():
            durations = [
                t.duration for t in task_trials.trials if t.duration is not None
            ]
            all_durations.extend(durations)

        return mean(all_durations) if all_durations else None

    def overall_total_duration(self) -> float | None:
        """Calculate total duration across all trials"""
        all_durations = []
        for task_trials in self.task_results.values():
            durations = [
                t.duration for t in task_trials.trials if t.duration is not None
            ]
            all_durations.extend(durations)

        return sum(all_durations) if all_durations else None

    def _sum_token_usage(self, trials) -> dict[str, int]:
        """Helper method to sum token usage across trials"""
        total_tokens = {}

        for trial in trials:
            if trial.token_usage:
                for key, value in trial.token_usage.items():
                    total_tokens[key] = total_tokens.get(key, 0) + value

        return total_tokens

    def total_token_usage(self) -> dict[str, int]:
        """Calculate total token usage across all trials"""
        all_trials = []
        for task_trials in self.task_results.values():
            all_trials.extend(task_trials.trials)

        return self._sum_token_usage(all_trials)

    def task_total_token_usage(self, task_id: str) -> dict[str, int]:
        """Calculate total token usage for a specific task"""
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        return self._sum_token_usage(trials)

    def total_tool_calls(self) -> dict[str, int]:
        """Calculate total successful and failed tool calls across all trials"""
        successful_calls = 0
        failed_calls = 0

        for task_trials in self.task_results.values():
            for trial in task_trials.trials:
                if "tool_calls" in trial.tool_statistics:
                    for tool_call in trial.tool_statistics["tool_calls"]:
                        if tool_call.get("status") == "success":
                            successful_calls += 1
                        else:
                            failed_calls += 1

        return {
            "successful": successful_calls,
            "failed": failed_calls,
            "total": successful_calls + failed_calls,
        }

    def average_score(self) -> float:
        """Calculate average score across all results"""

        if not self.task_results:
            raise ValueError("No task results available.")

        task_averages = [
            sum(trial.score for trial in task.trials) / len(task.trials)
            for task in self.task_results.values()
        ]

        return sum(task_averages) / len(task_averages)

    def task_success_rate(self, task_id: str) -> float:
        """Calculate success rate for a specific task"""
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")
        return mean(1 if trial.success else 0 for trial in trials)

    def overall_success_rate(self) -> float:
        """Calculate overall success rate across all tasks"""
        if not self.task_results:
            raise ValueError("No tasks available to calculate overall success rate.")

        return mean(self.task_success_rate(task_id) for task_id in self.all_task_ids)

    def task_pass_at_k(self, task_id: str, k: int) -> float:
        """
        Calculate pass@k - probability that at least one out of k trials succeeds for a given k value.

        pass@k is defined as:
        .. math: P(pass@k) = 1 - (1 - p)^k

        Where p is the probability of a single trial succeeding.

        In practice, pass@k is estimated by:
        - Taking n samples. (the number of total trials for a given task). (where n ≥ k)
        - Calculating the number c, the number of correct or successful solutions
        - Estimating pass@k as
        .. math: 1 - (1 - c/n)^k when c < n, or 1 when c = n

        Args:
            task_id: The ID of the task to calculate pass@k for
            k: The k value to use

        Returns:
            The pass@k score

        Raises:
            TaskNotFoundError: If the task ID is not found
            InsufficientTrialsError: If there are fewer trials than k
        """
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials

        if len(trials) < k:
            raise InsufficientTrialsError(
                f"Number of trials ({len(trials)}) is less than k ({k}) for task ID '{task_id}'."
            )

        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        # Calculate pass@k
        return 1.0 if c == n else 1.0 - (1.0 - c / n) ** k

    def task_pass_hat_k(self, task_id: str, k: int) -> float:
        """
        Calculate pass^k - probability of all k trials succeeding for a given k value.

        pass^k is defined as:
        .. math: P(pass^k) = p^k

        Where p is the probability of a single trial succeeding.

        In practice, pass^k is estimated by:
        - Taking n samples. (the number of total trials for a given task). (where n ≥ k)
        - Calculating the number c, the number of correct or successful solutions
        - Estimating pass^k as
        .. math: (c/n)^k

        This measures the likelihood that ALL of k independent attempts will succeed.

        Args:
            task_id: The ID of the task to calculate pass^k for
            k: The k value to use

        Returns:
            The pass^k score

        Raises:
            TaskNotFoundError: If the task ID is not found
            NoResultsError: If there are no trials for the task
        """
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        # Calculate pass^k
        return (c / n) ** k

    def overall_pass_at_k(self, k: int) -> float:
        """
        Calculate average pass@k across all tasks for a given k value.

        Args:
            k: The k value to use

        Returns:
            The average pass@k score across all tasks
        """
        return mean(self.task_pass_at_k(task_id, k) for task_id in self.all_task_ids)

    def overall_pass_hat_k(self, k: int) -> float:
        """
        Calculate average pass^k across all tasks for a given k value.

        Args:
            k: The k value to use

        Returns:
            The average pass^k score across all tasks
        """
        return mean(self.task_pass_hat_k(task_id, k) for task_id in self.all_task_ids)

    def _build_summary_table(
        self, pass_at_k_results: dict, pass_hat_k_results: dict
    ) -> Table:
        summary_table = Table(
            title="Overall Metrics", show_header=False, header_style="bold magenta"
        )
        summary_table.add_column("Metric", justify="left", style="cyan")
        summary_table.add_column("Value", justify="right", style="white")
        summary_table.add_row("Total Tasks", str(self.total_tasks))
        summary_table.add_row("Average Score", f"{self.average_score():.3f}")
        summary_table.add_row(
            "Overall Success Rate", f"{self.overall_success_rate():.3f}"
        )

        # Add tool call statistics
        tool_call_stats = self.total_tool_calls()
        summary_table.add_row("Total Tool Calls", str(tool_call_stats["total"]))
        summary_table.add_row(
            "Successful Tool Calls", str(tool_call_stats["successful"])
        )
        summary_table.add_row("Failed Tool Calls", str(tool_call_stats["failed"]))

        # Add token usage statistics
        total_tokens = self.total_token_usage()
        if total_tokens:
            summary_table.add_row("--- Token Usage ---", "")
            for key, value in total_tokens.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        summary_table.add_row(f"{key}.{sub_key}", str(sub_value))
                else:
                    summary_table.add_row(key, str(value))

        # Add duration metrics
        if self.total_duration:
            summary_table.add_row("Total Benchmark Time", f"{self.total_duration:.2f}s")

        total_trial_duration = self.overall_total_duration()
        if total_trial_duration:
            summary_table.add_row(
                "Total Trial Duration", f"{total_trial_duration:.2f}s"
            )

        avg_duration = self.overall_average_duration()
        if avg_duration:
            summary_table.add_row("Avg Trial Duration", f"{avg_duration:.2f}s")

        for k_val in self.k:
            summary_table.add_row(f"Pass@{k_val}", f"{pass_at_k_results[k_val]:.3f}")
            summary_table.add_row(f"Pass^{k_val}", f"{pass_hat_k_results[k_val]:.3f}")
        return summary_table

    def _build_task_table(self, task_id: str) -> Table:
        """Build a table showing metrics for a specific task with trial IDs."""
        task_table = Table(
            title=f"Task: {task_id}",
            show_header=True,
            header_style="bold green",
            title_style="bold green",
        )
        task_table.add_column("Trial ID", style="yellow")
        task_table.add_column("Score", style="cyan")
        task_table.add_column("Success", style="white")
        task_table.add_column("Duration (s)", style="green")
        task_table.add_column("Tokens", style="orange")

        # Add columns for each k value
        for k_val in self.k:
            task_table.add_column(f"Pass@{k_val}", style="blue")
            task_table.add_column(f"Pass^{k_val}", style="magenta")

        # Add overall row with aggregated metrics
        task_success_rate = self.task_success_rate(task_id)
        pass_at_values = [f"{self.task_pass_at_k(task_id, k):.3f}" for k in self.k]
        pass_hat_values = [f"{self.task_pass_hat_k(task_id, k):.3f}" for k in self.k]
        avg_duration = self.task_average_duration(task_id)
        duration_str = f"{avg_duration:.2f}" if avg_duration else "-"

        task_tokens = self.task_total_token_usage(task_id)
        token_str = str(task_tokens.get("total", "-")) if task_tokens else "-"

        # Build the overall metrics row
        metrics_row = [
            "Overall",
            f"{self._calculate_task_average_score(task_id):.3f}",
            f"{task_success_rate:.3f}",
            duration_str,
            token_str,
        ]

        # Add pass@k and pass^k values
        for i in range(len(self.k)):
            metrics_row.append(pass_at_values[i])
            metrics_row.append(pass_hat_values[i])

        task_table.add_row(*metrics_row, style="bold")

        # Add individual trial rows
        for trial in self.task_results[task_id].trials:
            duration_str = f"{trial.duration:.2f}" if trial.duration else "-"
            token_str = (
                str(trial.token_usage.get("total", "-")) if trial.token_usage else "-"
            )
            trial_row = [
                trial.trial_id,
                f"{trial.score:.3f}",
                "✓" if trial.success else "✗",
                duration_str,
                token_str,
            ]

            # Add placeholder values for pass@k and pass^k (not applicable for individual trials)
            trial_row.extend(["-"] * (len(self.k) * 2))

            task_table.add_row(*trial_row)

        return task_table

    def _calculate_task_average_score(self, task_id: str) -> float:
        """Calculate the average score for a specific task."""
        trials = self.task_results[task_id].trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")
        return sum(trial.score for trial in trials) / len(trials)

    def _build_tool_usage_table(self, task_id: str) -> Table:
        """Builds a table showing tool usage statistics for a task."""
        tool_usage_table = Table(
            title="Tool Usage",
            show_header=True,
            header_style="bold blue",
        )
        tool_usage_table.add_column("Trial ID", style="yellow")
        tool_usage_table.add_column("Tool", style="cyan")
        tool_usage_table.add_column("Usage Stats", style="white")

        for trial in self.task_results[task_id].trials:
            if trial.tool_statistics:
                for tool, stats in trial.tool_statistics.items():
                    if isinstance(stats, int | float):
                        tool_usage_table.add_row(trial.trial_id, f"{tool}", str(stats))
                    elif isinstance(stats, list):
                        tool_usage_table.add_row(
                            trial.trial_id, f"{tool}", ", ".join(map(str, stats))
                        )
                    else:
                        tool_usage_table.add_row(trial.trial_id, f"{tool}", str(stats))
        return tool_usage_table

    def _build_tool_calls_table(self, task_id: str) -> Table:
        """Builds a table showing detailed tool calls for a task."""
        tool_calls_table = Table(
            title="Tool Calls Details",
            show_header=True,
            header_style="bold yellow",
        )
        tool_calls_table.add_column("Trial ID", style="yellow")
        tool_calls_table.add_column("Tool Name", style="cyan")
        tool_calls_table.add_column("Arguments", style="cyan")
        tool_calls_table.add_column("Result", style="green")
        tool_calls_table.add_column("Status", style="magenta")

        for trial in self.task_results[task_id].trials:
            if "tool_calls" in trial.tool_statistics:
                for tool_call in trial.tool_statistics["tool_calls"]:
                    tool_calls_table.add_row(
                        trial.trial_id,
                        tool_call["tool_name"],
                        str(tool_call["arguments"]),
                        str(tool_call["result"]),
                        tool_call["status"],
                    )
        return tool_calls_table

    def generate_report(self, report_path: str | None = None) -> None:
        """
        Display and optionally save a detailed report of the benchmark results.
        Shows overall metrics and detailed per-task tables including trial IDs.

        Args:
            report_path: Optional path to save the report. If provided, saves as JSON.
        """

        # Prepare pass metrics for the summary
        pass_at_k_results = {k_val: self.overall_pass_at_k(k_val) for k_val in self.k}
        pass_hat_k_results = {k_val: self.overall_pass_hat_k(k_val) for k_val in self.k}

        if report_path:
            try:
                # Create pass@k and pass^k dictionaries for the report
                pass_at_k_dict = {
                    f"pass@{k}": value for k, value in pass_at_k_results.items()
                }
                pass_hat_k_dict = {
                    f"pass^{k}": value for k, value in pass_hat_k_results.items()
                }

                # Get tool call statistics
                tool_call_stats = self.total_tool_calls()

                # Get token usage statistics
                total_tokens = self.total_token_usage()

                # Create report data with timing information
                report_data = {
                    "metrics": {
                        "average_score": self.average_score(),
                        "overall_success_rate": self.overall_success_rate(),
                        **pass_at_k_dict,
                        **pass_hat_k_dict,
                        "total_tasks": self.total_tasks,
                        "total_tool_calls": tool_call_stats["total"],
                        "successful_tool_calls": tool_call_stats["successful"],
                        "failed_tool_calls": tool_call_stats["failed"],
                        "total_token_usage": total_tokens,
                    }
                }

                # Add timing metrics to overall metrics
                if self.total_duration:
                    report_data["metrics"]["total_benchmark_duration"] = (
                        self.total_duration
                    )

                total_trial_duration = self.overall_total_duration()
                if total_trial_duration:
                    report_data["metrics"]["total_trial_duration"] = (
                        total_trial_duration
                    )

                avg_duration = self.overall_average_duration()
                if avg_duration:
                    report_data["metrics"]["average_trial_duration"] = avg_duration

                report_data["task_results"] = {}

                # Add task-specific results
                for task_id in self.all_task_ids:
                    # Calculate task-specific metrics for each k
                    task_pass_at_k = {}
                    task_pass_hat_k = {}

                    for k_val in self.k:
                        task_pass_at_k[k_val] = self.task_pass_at_k(task_id, k_val)
                        task_pass_hat_k[k_val] = self.task_pass_hat_k(task_id, k_val)

                    # Create task-specific pass@k and pass^k dictionaries
                    task_pass_at_k_dict = {
                        f"pass@{k}": value for k, value in task_pass_at_k.items()
                    }
                    task_pass_hat_k_dict = {
                        f"pass^{k}": value for k, value in task_pass_hat_k.items()
                    }

                    # Add trials information directly to the task results
                    trials_data = []
                    for trial in self.task_results[task_id].trials:
                        trial_data = {
                            "trial_id": trial.trial_id,
                            "score": trial.score,
                            "success": trial.success,
                        }

                        # Add duration if available
                        if trial.duration is not None:
                            trial_data["duration"] = trial.duration

                        # Add token usage if available
                        if trial.token_usage is not None:
                            trial_data["token_usage"] = trial.token_usage

                        # Add tool calls data if available
                        if "tool_calls" in trial.tool_statistics:
                            trial_data["tool_calls"] = [
                                {
                                    "tool_name": tool_call["tool_name"],
                                    "arguments": tool_call["arguments"],
                                    "result": tool_call["result"],
                                    "status": tool_call["status"],
                                    "error_message": tool_call.get("error_message"),
                                    "timestamp": tool_call.get("timestamp"),
                                }
                                for tool_call in trial.tool_statistics["tool_calls"]
                            ]

                        # Add other tool statistics
                        trial_data.update(
                            {
                                stat_key: stat_value
                                for stat_key, stat_value in trial.tool_statistics.items()
                                if stat_key != "tool_calls"
                            }
                        )

                        trials_data.append(trial_data)

                    task_result_data = {
                        "success_rate": self.task_success_rate(task_id),
                        "average_score": self._calculate_task_average_score(task_id),
                        **task_pass_at_k_dict,
                        **task_pass_hat_k_dict,
                        "trials": trials_data,
                        "total_token_usage": self.task_total_token_usage(task_id),
                    }

                    # Add task-level timing metrics
                    task_avg_duration = self.task_average_duration(task_id)
                    if task_avg_duration:
                        task_result_data["average_duration"] = task_avg_duration

                    report_data["task_results"][task_id] = task_result_data

                with Path(report_path).open("w") as f:
                    json.dump(report_data, f, indent=2)
                logger.info(f"Saved detailed report to: {report_path}")
            except Exception as e:
                logger.error(f"Error saving report file: {e}")
                raise

        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        # Print header panel
        console.print(
            Panel("BENCHMARK RESULTS REPORT", style="bold blue", expand=False)
        )

        summary_table = self._build_summary_table(pass_at_k_results, pass_hat_k_results)
        console.print(summary_table)
        console.print()  # Spacing

        # Create individual tables for each task
        for task_id, task_trials in self.task_results.items():
            task_table = self._build_task_table(task_id)
            console.print(task_table)

            # Add tool usage statistics if available
            trials = task_trials.trials
            if trials and trials[0].tool_statistics:
                tool_usage_table = self._build_tool_usage_table(task_id)
                console.print(tool_usage_table)

                # Add detailed tool calls if available
                if "tool_calls" in trials[0].tool_statistics:
                    tool_calls_table = self._build_tool_calls_table(task_id)
                    console.print(tool_calls_table)
                    console.print()  # Add spacing between tasks
                else:
                    console.print()  # Add spacing between tasks
            else:
                console.print()  # Add spacing between tasks
