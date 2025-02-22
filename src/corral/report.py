import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from loguru import logger


@dataclass
class ToolResponse:
    """Response from a tool execution"""

    success: bool
    result: str | None
    error: str | None


@dataclass
class TaskResult:
    """Result of a task submission with trial information"""

    score: float
    state: dict[str, Any]  # TODO replace Any with specific types
    tool_statistics: dict[str, Any]  # TODO replace Any with specific types

    @property
    def success(self) -> bool:
        """Whether the trial was successful"""
        return self.score > 0


@dataclass
class TaskTrials:
    """Collection of trials for a single task"""

    trials: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate across all trials"""
        return mean(1 if trial.success else 0 for trial in self.trials)

    def calculate_pass_at_k(self, k: int) -> float:
        """Calculate pass@k - probability of at least one success in k trials"""
        if len(self.trials) < k:
            return 0.0
        # For each group of k trials, check if at least one succeeded
        successes = sum(
            1 if any(t.success for t in self.trials[i : i + k]) else 0
            for i in range(0, len(self.trials), k)
        )
        return successes / (len(self.trials) // k)

    def calculate_pass_hat_k(self, k: int) -> float:
        """Calculate pass^k - probability of all k trials succeeding"""
        if len(self.trials) < k:
            return 0.0
        # For each group of k trials, check if all succeeded
        successes = sum(
            1 if all(t.success for t in self.trials[i : i + k]) else 0
            for i in range(0, len(self.trials), k)
        )
        return successes / (len(self.trials) // k)


@dataclass
class BenchmarkResult:
    """Results from running benchmark with multiple trials per task"""

    task_results: dict[str, TaskTrials]
    k: int  # Number of trials to consider for pass@k metrics
    total_tasks: int

    @property
    def average_score(self) -> float:
        """Calculate average score across all trials of all tasks"""
        all_scores = [
            trial.score
            for trials in self.task_results.values()
            for trial in trials.trials
        ]
        return mean(all_scores) if all_scores else 0.0

    @property
    def pass_at_k(self) -> float:
        """Calculate average pass@k across all tasks"""
        return mean(
            trials.calculate_pass_at_k(self.k) for trials in self.task_results.values()
        )

    @property
    def pass_hat_k(self) -> float:
        """Calculate average pass^k across all tasks"""
        return mean(
            trials.calculate_pass_hat_k(self.k) for trials in self.task_results.values()
        )

    def report(self, report_path: str | None = None) -> None:
        """
        Display and optionally save a detailed report of the benchmark results.
        Shows overall metrics and detailed per-task tables.

        Args:
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
        summary_table.add_row("Trials per Task (k)", str(self.k))
        summary_table.add_row("Average Score", f"{self.average_score:.3f}")
        summary_table.add_row(f"Pass@{self.k}", f"{self.pass_at_k:.3f}")
        summary_table.add_row(f"Pass^{self.k}", f"{self.pass_hat_k:.3f}")

        console.print(summary_table)
        console.print()  # Add spacing between tables

        # Create individual tables for each task
        for task_id, trials in self.task_results.items():
            task_table = Table(
                title=f"Task: {task_id}",
                show_header=False,
                header_style="bold green",
                title_style="bold green",
            )
            task_table.add_column("Metric", style="cyan")
            task_table.add_column("Value", style="white")

            # Add task metrics
            task_table.add_row("Success Rate", f"{trials.success_rate:.3f}")
            task_table.add_row(
                f"Pass@{self.k}", f"{trials.calculate_pass_at_k(self.k):.3f}"
            )
            task_table.add_row(
                f"Pass^{self.k}", f"{trials.calculate_pass_hat_k(self.k):.3f}"
            )

            # Add tool usage statistics if available
            if trials.trials and trials.trials[0].tool_statistics:
                task_table.add_row("Tool Usage", "")
                for tool, stats in trials.trials[0].tool_statistics.items():
                    if isinstance(stats, int | float):
                        task_table.add_row(f"  • {tool}", str(stats))
                    elif isinstance(stats, list):
                        task_table.add_row(f"  • {tool}", ", ".join(map(str, stats)))
                    else:
                        task_table.add_row(f"  • {tool}", str(stats))

                # Add detailed tool calls if available
                if "tool_calls" in trials.trials[0].tool_statistics:
                    tool_calls_table = Table(
                        title="Tool Calls Details",
                        show_header=True,
                        header_style="bold yellow",
                    )
                    tool_calls_table.add_column("Tool Name", style="yellow")
                    tool_calls_table.add_column("Arguments", style="cyan")
                    tool_calls_table.add_column("Result", style="green")
                    tool_calls_table.add_column("Status", style="magenta")

                    for trial in trials.trials:
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
                        "average_score": self.average_score,
                        f"pass@{self.k}": self.pass_at_k,
                        f"pass^{self.k}": self.pass_hat_k,
                        "total_tasks": self.total_tasks,
                        "k": self.k,
                    },
                    "task_results": {
                        task_id: {
                            "success_rate": trials.success_rate,
                            f"pass@{self.k}": trials.calculate_pass_at_k(self.k),
                            f"pass^{self.k}": trials.calculate_pass_hat_k(self.k),
                            "tool_calls": [
                                {
                                    "tool_name": tool_call["tool_name"],
                                    "arguments": tool_call["arguments"],
                                    "result": tool_call["result"],
                                    "status": tool_call["status"],
                                    "error_message": tool_call.get("error_message"),
                                    "timestamp": tool_call.get("timestamp"),
                                }
                                for trial in trials.trials
                                if "tool_calls" in trial.tool_statistics
                                for tool_call in trial.tool_statistics["tool_calls"]
                            ]
                            if trials.trials
                            else [],
                        }
                        for task_id, trials in self.task_results.items()
                    },
                }
                with Path(report_path).open("w") as f:
                    json.dump(report_data, f, indent=2)
                logger.info(f"Saved detailed report to: {report_path}")
            except Exception as e:
                logger.error(f"Error saving report file: {e}")
                raise
