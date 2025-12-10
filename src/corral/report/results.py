import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from corral.report.metrics.base import Metric, TaskMetric


@dataclass
class TaskTrialResult:
    """Result of a single trail in a task"""

    task_id: str
    trial_id: str
    score: float
    state: dict[str, Any]  # TODO replace Any with specific types
    tool_statistics: dict[str, Any]  # TODO replace Any with specific types
    messages: list[dict[str, Any]] | None = None  # Agent messages (verbose only)
    duration: float | None = None
    token_usage: dict[str, int] | None = None
    error_message: str | None = None
    surrendered: bool = False

    @property
    def success(self) -> bool:
        """Whether the trial was successful"""
        return self.score > 0 and self.error_message is None and not self.surrendered

    @property
    def tool_execution_duration(self) -> float:
        """Total tool execution duration for this trial"""
        duration = 0.0
        if "tool_calls" in self.tool_statistics:
            for tool_call in self.tool_statistics["tool_calls"]:
                if tool_call.get("duration") is not None:
                    duration += tool_call["duration"]
        return duration


@dataclass
class TaskTrialResults:
    """Collection of results of trials for a single task"""

    task_id: str
    trials: list[TaskTrialResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """
    This class provides methods for calculating various metrics about benchmark performance.

    Attributes:
        task_results: dictionary mapping task IDs to their trial results
        k: The k value(s) to use for pass@k and pass^k calculations.
           Can be a single int or a list of ints.
        total_duration: Total duration of the benchmark run
        verbosity: Verbosity level of tool description
        metrics: Explicit list of metrics to use. If None, uses default metrics.
                 Using explicit metrics is recommended for clarity and visibility.
        metric_registry: MetricRegistry instance. If None, creates an instance-level
                        registry (recommended). Pass a shared registry to share metrics
                        across multiple BenchmarkResult instances.
    """

    task_results: dict[str, TaskTrialResults]
    k: list[int] = field(default_factory=lambda: [1])
    total_duration: float | None = None
    verbosity: str | None = None
    metrics: list[Metric] | None = None  # Explicit metrics list
    metric_registry: Any = (
        None  # Instance-level by default, Any to avoid circular import
    )

    def __post_init__(self):
        """Initialize metric system with instance-level registry."""
        # Import here to avoid circular dependency
        from corral.report.metrics import get_default_metrics  # noqa:
        from corral.report.metrics.registry import MetricRegistry

        # Create instance-level registry by default (no global state)
        if self.metric_registry is None:
            self.metric_registry = MetricRegistry()

        # Determine which metrics to use
        if self.metrics is not None:
            # Explicit metrics provided - use them
            metrics_to_register = self.metrics
        else:
            # No explicit metrics - use defaults (backward compatible)
            metrics_to_register = get_default_metrics(k_values=self.k)

        # Register metrics to this instance's registry
        for metric in metrics_to_register:
            try:
                self.metric_registry.register(metric)
            except ValueError:
                # Metric already registered, skip it
                logger.debug(
                    f"Metric '{metric.metadata.name}' already registered, skipping"
                )

        # Log what metrics are active for visibility
        metric_names = sorted(
            [m.metadata.name for m in self.metric_registry.list_all()]
        )
        logger.debug(
            f"BenchmarkResult initialized with {len(metric_names)} metrics: "
            f"{', '.join(metric_names)}"
        )

    @property
    def all_task_ids(self) -> set[str]:
        """Get all task_ids in the benchmark.

        This property makes BenchmarkResult satisfy the MetricContext protocol.

        Returns:
            Set of unique task IDs
        """
        return set(self.task_results.keys())

    def get_task_trials(self, task_id: str) -> TaskTrialResults | None:
        """Get trial results for a specific task.

        This method makes BenchmarkResult satisfy the MetricContext protocol.

        Args:
            task_id: The task identifier

        Returns:
            TaskTrialResults for the task, or None if not found
        """
        return self.task_results.get(task_id)

    @property
    def total_tasks(self) -> int:
        """Get number of unique tasks in the benchmark"""
        return len(self.task_results)

    @property
    def all_results(self) -> list[TaskTrialResult]:
        """Get a flat list of all individual trails across all tasks"""
        return [
            result
            for task_trials in self.task_results.values()
            for result in task_trials.trials
        ]

    # Metric Calculation Methods
    def calculate_metrics(
        self, metrics_to_calculate: list[str] | None = None
    ) -> dict[str, Any]:
        """Calculate metrics using the registry.

        Args:
            metrics_to_calculate: Optional list of metric names to calculate.
                                 If None, calculates all registered metrics.

        Returns:
            Dictionary mapping metric names to their calculated values.
        """
        return self.metric_registry.calculate_all(
            benchmark_result=self, enabled_only=metrics_to_calculate, parallel=True
        )

    def _prepare_report_data(self, calculated_metrics: dict[str, Any]) -> dict:
        """
        Prepare report data for JSON export.
        Uses metric registry for calculating metrics.

        Args:
            calculated_metrics: Pre-calculated metrics dictionary

        Returns:
            Dictionary containing all report data ready for JSON export
        """
        # Map registry metric results to report format
        report_data = {"metrics": {}}

        # Add all metrics to the report
        for metric_name, metric_value in calculated_metrics.items():
            # Get display name from registry if available
            try:
                metric_obj = self.metric_registry.get(metric_name)
                display_name = metric_obj.metadata.display_name
            except KeyError:
                display_name = metric_name

            # Use display name as the key
            report_data["metrics"][display_name] = metric_value

        # Add non-metric fields
        report_data["metrics"]["tool_verbosity"] = self.verbosity
        if self.total_duration:
            report_data["metrics"]["total_benchmark_duration"] = self.total_duration

        report_data["task_results"] = {}

        # Add task-specific results
        for task_id in self.all_task_ids:
            # Add trials information directly to the task results
            trials_data = []
            for trial in self.task_results[task_id].trials:
                trial_data = {
                    "trial_id": trial.trial_id,
                    "score": trial.score,
                    "submitted_answer": trial.state.get("submitted_answer")
                    if trial.state and isinstance(trial.state, dict)
                    else None,
                    "success": trial.success,
                    "surrendered": trial.surrendered,
                    "tool_execution_duration": trial.tool_execution_duration,
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
                            "duration": tool_call.get("duration"),
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

            # Build task result data using registry metrics
            task_result_data = {
                "trials": trials_data,
            }

            # Add all task-level metrics from registry
            all_metrics = self.metric_registry.list_all()
            for metric in all_metrics:
                # Check if it's a TaskMetric
                if isinstance(metric, TaskMetric):
                    metric_name = metric.metadata.name
                    display_name = metric.metadata.display_name
                    try:
                        value = metric.calculate_for_task(self, task_id)
                        # Use display name as key in task_result_data
                        task_result_data[display_name] = value
                    except Exception as e:
                        logger.warning(
                            f"Error calculating {metric_name} for {task_id}: {e}"
                        )

            report_data["task_results"][task_id] = task_result_data

        return report_data

    def _build_summary_string(self, calculated_metrics: dict[str, Any]) -> str:
        """Build summary string using metrics from registry.

        Args:
            calculated_metrics: Pre-calculated metrics dictionary
        """
        lines = []

        # Add header
        lines.append("BENCHMARK RESULTS")
        lines.append("")

        # Add non-metric header info
        if "total_tasks" in calculated_metrics:
            lines.append(f"Total Tasks: {calculated_metrics['total_tasks']}")
        if self.verbosity:
            lines.append(f"Tool Verbosity: {self.verbosity}")

        # Add all metrics dynamically
        for metric_name, metric_value in calculated_metrics.items():
            if metric_name == "total_tasks":  # Already shown
                continue

            # Get display name from registry metadata
            try:
                metric_obj = self.metric_registry.get(metric_name)
                display_name = metric_obj.metadata.display_name
            except KeyError:
                # Fallback to metric name if not in registry
                display_name = metric_name

            # Handle dict values specially
            if isinstance(metric_value, dict):
                # Add rows for each key-value pair in the dict
                for key, val in metric_value.items():
                    if isinstance(val, dict):
                        # Nested dict - add sub-rows
                        for sub_key, sub_val in val.items():
                            lines.append(f"{display_name}.{key}.{sub_key}: {sub_val}")
                    else:
                        lines.append(f"{display_name}.{key}: {val}")
            else:
                # Simple value - format and add
                lines.append(
                    f"{display_name}: {self._format_metric_value(metric_value)}"
                )

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _format_metric_value(self, value: Any) -> str:
        """Format a metric value for display (non-dict values only)."""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _display_console_report(self, calculated_metrics: dict[str, Any]) -> None:
        """
        Display benchmark results to console.

        Args:
            calculated_metrics: Pre-calculated metrics dictionary
        """
        summary_string = self._build_summary_string(calculated_metrics)
        logger.info(f"\n{summary_string}")

    def generate_report(self, report_path: str | None = None) -> None:
        """
        Display and optionally save a detailed report of the benchmark results.
        Shows overall metrics and detailed per-task tables including trial IDs.

        Args:
            report_path: Optional path to save the report. If provided, saves as JSON.
        """
        # Calculate all metrics once
        calculated_metrics = self.calculate_metrics()

        # Save JSON report if path is provided
        if report_path:
            try:
                report_data = self._prepare_report_data(calculated_metrics)
                with Path(report_path).open("w") as f:
                    json.dump(report_data, f, indent=2)
                logger.info(f"Saved detailed report to: {report_path}")
            except Exception as e:
                logger.error(f"Error saving report file: {e}")
                raise

        # Display report to console
        self._display_console_report(calculated_metrics)
