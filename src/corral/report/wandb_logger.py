from pathlib import Path
from typing import Any

import wandb
from loguru import logger

from corral.report.results import (
    BenchmarkResult,
    TaskTrialResult,
)
from corral.report.utils import calculate_trial_tool_duration


class CorralWandbLogger:
    """Handles all wandb logging logic"""

    RUN_ID_FILE = Path(".wandb_run_id")

    def __init__(
        self,
        project: str = "corral",
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        run_id_file: str | Path | None = None,
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.name = name
        self.run = None
        self.trial_table = None
        # Always store as Path
        self.run_id_file = (
            Path(run_id_file) if run_id_file is not None else self.RUN_ID_FILE
        )
        self._run_id = None

    def _load_run_id(self, run_name: str) -> str | None:
        """
        Loads the run ID and run name from the local file if it exists and matches the current run_name.
        This enables resuming a previous wandb run with the same name, ensuring continuity in logging and avoiding duplicate runs.
        """
        try:
            if self.run_id_file.exists():
                with self.run_id_file.open() as f:
                    data = f.read().strip()
                    if data and "|" in data:
                        run_id, saved_run_name = data.split("|", 1)
                        if saved_run_name == run_name:
                            logger.info(
                                f"Loaded previous wandb run ID: {run_id} for run_name: {run_name}"
                            )
                            return run_id
        except Exception as e:
            logger.warning(f"Could not load wandb run ID: {e}")
        return None

    def _save_run_id(self, run_id: str, run_name: str) -> None:
        """
        Saves the run ID and run name to a local file for resuming.
        This allows future sessions to pick up the same wandb run, supporting robust experiment tracking.
        """
        try:
            # Save to file in format: run_id|run_name
            with self.run_id_file.open("w") as f:
                f.write(f"{run_id}|{run_name}")
            logger.info(f"Saved wandb run ID: {run_id} for run_name: {run_name}")
        except Exception as e:
            logger.warning(f"Could not save wandb run ID: {e}")

    def _remove_run_id(self) -> None:
        """
        Removes the run ID file after a successful finish.
        This cleanup step prevents accidental resumption of completed runs and keeps the workspace tidy.
        """
        try:
            if self.run_id_file.exists():
                self.run_id_file.unlink()
                logger.info(f"Removed wandb run ID file: {self.run_id_file!s}")
        except Exception as e:
            logger.warning(f"Could not remove wandb run ID file: {e}")

    def start_logging(self, config: dict[str, Any]) -> None:
        """
        Initializes a wandb run, resuming if interrupted and the run_name matches.
        This ensures experiment continuity and consistent logging, even if the process is restarted or interrupted.
        """
        run_name = self.name or f"{config['agent_type']}-{config['session_id']}"

        # Try to resume from previous run ID if available and run_name matches
        run_id = self._load_run_id(run_name)
        self._run_id = run_id
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=run_name,
                config=config,
                id=run_id,
                resume="allow" if run_id else None,
            )
            # Save the run ID and run_name for future resuming
            self._save_run_id(self.run.id, run_name)
        except Exception as e:
            logger.error(f"Failed to initialize wandb run: {e}")
            raise

        self.trial_table = wandb.Table(
            columns=[
                "task_id",
                "trial_id",
                "score",
                "success",
                "is_failure",
                "duration_s",
                "token_usage_prompt",
                "token_usage_completion",
                "token_usage_total",
                "tool_calls_total",
                "tool_calls_successful",
                "tool_calls_failed",
                "tool_execution_duration_s",
            ]
        )

        logger.info(f"Wandb run initialized: {self.run.url}")

    def log_trial(self, trial: TaskTrialResult) -> None:
        """
        Logs a single trial result to the wandb trial table.
        This enables detailed tracking of each trial's metrics and supports later aggregation and analysis.
        """
        if not self.trial_table:
            return

        tool_stats = (
            trial.tool_statistics if isinstance(trial.tool_statistics, dict) else {}
        )
        tool_duration = calculate_trial_tool_duration(trial)

        self.trial_table.add_data(
            trial.task_id,
            trial.trial_id,
            trial.score,
            trial.success,
            not trial.success,
            trial.duration,
            trial.token_usage.get("prompt_tokens") if trial.token_usage else None,
            trial.token_usage.get("completion_tokens") if trial.token_usage else None,
            trial.token_usage.get("total_tokens") if trial.token_usage else None,
            tool_stats.get("total_calls"),
            tool_stats.get("successful_calls"),
            tool_stats.get("failed_calls"),
            tool_duration,
        )

    def log_final_results(self, result: BenchmarkResult) -> None:
        """
        Logs final benchmark results, including overall and per-task metrics, to wandb.
        Uses the flexible metrics system - logs only the metrics that are registered
        in the BenchmarkResult's metric registry.

        Args:
            result: The BenchmarkResult containing all metrics and trial data
        """
        if not self.run:
            return

        # Calculate all metrics using the registry
        try:
            calculated_metrics = result.calculate_metrics()
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            if self.run:
                self.run.log({"metrics_calculation_error": str(e)})
            return

        # Prepare overall metrics for WandB
        overall_metrics = {}
        task_level_metrics = {}

        # Iterate through calculated metrics and categorize them
        for metric_name, metric_value in calculated_metrics.items():
            try:
                # Check if this is a task-level metric (returns dict with task_ids as keys)
                if isinstance(metric_value, dict) and any(
                    task_id in metric_value for task_id in result.all_task_ids
                ):
                    # This is a task-level breakdown metric
                    # Store it for task-level logging
                    task_level_metrics[metric_name] = metric_value
                else:
                    # This is an overall metric
                    # Use metric name (not display name) for WandB consistency
                    overall_metrics[f"overall/{metric_name}"] = metric_value

            except Exception as e:
                logger.warning(f"Error processing metric '{metric_name}': {e}")
                continue

        # Add non-metric fields
        if result.total_duration:
            overall_metrics["overall/total_benchmark_duration_s"] = (
                result.total_duration
            )

        # Log overall metrics
        if overall_metrics:
            try:
                self.run.log(overall_metrics)
                self.run.summary.update(overall_metrics)
                logger.info(f"Logged {len(overall_metrics)} overall metrics to wandb")
            except Exception as e:
                logger.error(f"Error logging overall metrics: {e}")
                if self.run:
                    self.run.log({"overall_metrics_error": str(e)})

        # Log task-level metrics
        logger.info("Logging task-level metrics to wandb")
        for task_id in result.all_task_ids:
            if not result.task_results[task_id].trials:
                logger.warning(
                    f"No trials recorded for task {task_id}, skipping task-level metric logging."
                )
                continue

            try:
                task_metrics = {}

                # Add metrics from the task-level breakdown
                for metric_name, task_values in task_level_metrics.items():
                    if task_id in task_values:
                        task_metrics[f"task_{task_id}/{metric_name}"] = task_values[
                            task_id
                        ]

                if task_metrics:
                    self.run.log(task_metrics)

            except Exception as e:
                logger.error(
                    f"Error calculating or logging task {task_id} metrics: {e}"
                )
                if self.run:
                    self.run.log({f"task_{task_id}/metrics_error": str(e)})

        # Log trial table
        if self.trial_table and self.trial_table.data:
            logger.info("Logging trial details table to wandb")
            try:
                self.run.log({"trial_details": self.trial_table})
            except Exception as e:
                logger.error(f"Error logging trial details table: {e}")

    def log_agent_artifacts(self, verbose: bool) -> None:
        """
        Logs agent message files as wandb artifacts if verbose is True.
        This preserves detailed agent logs for later inspection, aiding in debugging and experiment transparency.
        """
        if not self.run or not verbose:
            return

        logger.info("Logging agent message logs as wandb artifacts")
        log_dir = Path("./agent_logs")
        if log_dir.exists():
            try:
                artifact = wandb.Artifact("agent_logs", type="logs")
                artifact.add_dir(str(log_dir))
                self.run.log_artifact(artifact)
                logger.info(f"Logged agent logs from {log_dir} as artifact")
            except Exception as e:
                logger.error(f"Error logging agent logs artifact: {e}")
        else:
            logger.warning(
                f"Agent log directory {log_dir} not found, skipping log artifact logging"
            )

    def finish(self) -> None:
        """
        Finishes the wandb run and cleans up the run ID file.
        This ensures that the run is properly closed and prevents accidental resumption in future sessions.
        """
        if self.run:
            self.run.finish()
            logger.info("Wandb run finished")
            self._remove_run_id()
