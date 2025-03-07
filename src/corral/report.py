import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from loguru import logger


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
    score: float
    state: dict[str, Any]  # TODO replace Any with specific types
    tool_statistics: dict[str, Any]  # TODO replace Any with specific types

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
    """Results from running benchmark with multiple trials per task"""

    task_results: dict[str, TaskTrialResults]
    k: int = 1

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

    def average_score(self) -> float:
        """Calculate average score across all results"""
        all_results = self.all_results
        if not all_results:
            raise ValueError("No results available to calculate average score.")
        return mean(r.score for r in all_results)

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

    def task_pass_at_k(self, task_id: str) -> float:
        """Calculate pass@k for a specific task

        pass@k is defined as:
        .. math: P(pass@k) = 1 - (1 - p)^k

        In practice, pass@k is estimated by:
        - Taking n samples for each problem (where n ≥ k)
        - Calculating the number c of correct solutions
        - Estimating pass@k as
        .. math: 1 - (1 - c/n)^k when c < n, or 1 when c = n

        Args:
            task_id: The ID of the task to calculate pass@k for

        Returns:
            The pass@k score for the task

        Raises:
            TaskNotFoundError: If the task ID is not found
            InsufficientTrialsError: If there are fewer trials than k
        """
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        if len(trials) < self.k:
            raise InsufficientTrialsError(
                f"Number of trials ({len(trials)}) is less than k ({self.k}) for task ID '{task_id}'."
            )

        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        # Calculate pass@k
        if c == n:  # All trials succeeded
            return 1.0
        else:
            return 1.0 - (1.0 - c / n) ** self.k

    def task_pass_hat_k(self, task_id: str) -> float:
        """
        Calculate pass^k - probability of all k trials succeeding.

        pass^k is defined as:
        .. math: P(pass^k) = p^k

        Where p is the probability of a single trial succeeding.

        In practice, pass^k is estimated by:
        - Taking n samples for each problem (where n ≥ k)
        - Calculating the number c of correct solutions
        - Estimating pass^k as
        .. math: (c/n)^k

        This measures the likelihood that ALL of k independent attempts will succeed.

        Args:
            task_id: The ID of the task to calculate pass^k for

        Returns:
            The pass^k score for the task

        Raises:
            TaskNotFoundError: If the task ID is not found
            InsufficientTrialsError: If there are fewer trials than k
        """
        if task_id not in self.task_results:
            raise TaskNotFoundError(f"Task ID '{task_id}' not found.")

        trials = self.task_results[task_id].trials
        if not trials:
            raise NoResultsError(f"No trials available for task ID '{task_id}'.")

        # Calculate probability of success based on observed success rate
        c = sum(1 if trial.success else 0 for trial in trials)
        n = len(trials)

        # Calculate pass^k
        return (c / n) ** self.k

    def overall_pass_at_k(self) -> float:
        """Calculate average pass@k across all tasks"""
        return mean(self.task_pass_at_k(task_id) for task_id in self.all_task_ids)

    def overall_pass_hat_k(self) -> float:
        """Calculate average pass^k across all tasks"""
        return mean(self.task_pass_hat_k(task_id) for task_id in self.all_task_ids)

    def generate_report(self, report_path: str | None = None) -> None:
        """
        Display and optionally save a detailed report of the benchmark results.
        Shows overall metrics and detailed per-task tables.

        Args:
            k: Number of trials to consider for pass@k metrics
            report_path: Optional path to save the report. If provided, saves as JSON.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Print header panel
        console.print(
            Panel("BENCHMARK RESULTS REPORT", style="bold blue", expand=False)
        )

        # Summary Table
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
        summary_table.add_row(f"Pass@{self.k}", f"{self.overall_pass_at_k():.3f}")
        summary_table.add_row(f"Pass^{self.k}", f"{self.overall_pass_hat_k():.3f}")

        console.print(summary_table)
        console.print()  # Add spacing between tables

        # Create individual tables for each task
        for task_id, task_trials in self.task_results.items():
            task_table = Table(
                title=f"Task: {task_id}",
                show_header=False,
                header_style="bold green",
                title_style="bold green",
            )
            task_table.add_column("Metric", style="cyan")
            task_table.add_column("Value", style="white")

            # Add task metrics
            task_table.add_row("Success Rate", f"{self.task_success_rate(task_id):.3f}")
            task_table.add_row(f"Pass@{self.k}", f"{self.task_pass_at_k(task_id):.3f}")
            task_table.add_row(f"Pass^{self.k}", f"{self.task_pass_hat_k(task_id):.3f}")

            # Add tool usage statistics if available
            trials = task_trials.trials
            if trials and trials[0].tool_statistics:
                task_table.add_row("Tool Usage", "")
                for tool, stats in trials[0].tool_statistics.items():
                    if isinstance(stats, int | float):
                        task_table.add_row(f"  • {tool}", str(stats))
                    elif isinstance(stats, list):
                        task_table.add_row(f"  • {tool}", ", ".join(map(str, stats)))
                    else:
                        task_table.add_row(f"  • {tool}", str(stats))

                # Add detailed tool calls if available
                if "tool_calls" in trials[0].tool_statistics:
                    tool_calls_table = Table(
                        title="Tool Calls Details",
                        show_header=True,
                        header_style="bold yellow",
                    )
                    tool_calls_table.add_column("Tool Name", style="yellow")
                    tool_calls_table.add_column("Arguments", style="cyan")
                    tool_calls_table.add_column("Result", style="green")
                    tool_calls_table.add_column("Status", style="magenta")

                    for trial in trials:
                        if "tool_calls" in trial.tool_statistics:
                            for tool_call in trial.tool_statistics["tool_calls"]:
                                tool_calls_table.add_row(
                                    tool_call["tool_name"],
                                    str(tool_call["arguments"]),
                                    str(tool_call["result"]),
                                    tool_call["status"],
                                )

                    console.print(task_table)
                    console.print(tool_calls_table)
                    console.print()  # Add spacing between tasks
                else:
                    console.print(task_table)
                    console.print()  # Add spacing between tasks
            else:
                task_table.add_row("Tool Usage", "None")
                console.print(task_table)
                console.print()  # Add spacing between tasks

        # Save report if path provided
        if report_path:
            try:
                report_data = {
                    "metrics": {
                        "average_score": self.average_score(),
                        "overall_success_rate": self.overall_success_rate(),
                        f"pass@{self.k}": self.overall_pass_at_k(),
                        f"pass^{self.k}": self.overall_pass_hat_k(),
                        "total_tasks": self.total_tasks,
                    },
                    "task_results": {
                        task_id: {
                            "success_rate": self.task_success_rate(task_id),
                            f"pass@{self.k}": self.task_pass_at_k(task_id),
                            f"pass^{self.k}": self.task_pass_hat_k(task_id),
                            "tool_calls": [
                                {
                                    "tool_name": tool_call["tool_name"],
                                    "arguments": tool_call["arguments"],
                                    "result": tool_call["result"],
                                    "status": tool_call["status"],
                                    "error_message": tool_call.get("error_message"),
                                    "timestamp": tool_call.get("timestamp"),
                                }
                                for trial in self.task_results[task_id].trials
                                if "tool_calls" in trial.tool_statistics
                                for tool_call in trial.tool_statistics["tool_calls"]
                            ],
                        }
                        for task_id in self.all_task_ids
                    },
                }
                with Path(report_path).open("w") as f:
                    json.dump(report_data, f, indent=2)
                logger.info(f"Saved detailed report to: {report_path}")
            except Exception as e:
                logger.error(f"Error saving report file: {e}")
                raise
