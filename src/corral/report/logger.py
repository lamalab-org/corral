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

    def __init__(
        self,
        project: str = "corral",
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.name = name
        self.run = None
        self.trial_table = None

    def start_logging(self, config: dict[str, Any]) -> None:
        """Initialize wandb run"""
        run_name = self.name or f"{config['agent_type']}-{config['session_id'][:8]}"

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.group,
            name=run_name,
            config=config,
        )

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
        """Log a single trial result"""
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

    def log_final_results(self, result: BenchmarkResult, k_values: list[int]) -> None:
        """Log final benchmark results"""
        if not self.run:
            return

        # Overall metrics
        overall_metrics = {}

        try:
            overall_metrics.update(
                {
                    "overall/average_score": result.average_score(),
                    "overall/overall_success_rate": result.overall_success_rate(),
                    "overall/total_tasks": result.total_tasks,
                }
            )

            # Add timing metrics
            if result.total_duration:
                overall_metrics["overall/total_benchmark_duration_s"] = (
                    result.total_duration
                )

            total_trial_duration = result.overall_total_duration()
            if total_trial_duration:
                overall_metrics["overall/total_trial_duration_s"] = total_trial_duration

            avg_duration = result.overall_average_duration()
            if avg_duration:
                overall_metrics["overall/average_trial_duration_s"] = avg_duration

            overall_metrics["overall/total_tool_execution_duration_s"] = (
                result.total_tool_execution_duration()
            )

            # Add pass@k and pass^k metrics
            for k_val in k_values:
                try:
                    overall_metrics[f"overall/pass@{k_val}"] = result.overall_pass_at_k(
                        k_val
                    )
                    overall_metrics[f"overall/pass^{k_val}"] = (
                        result.overall_pass_hat_k(k_val)
                    )
                except Exception as e:
                    logger.warning(f"Error calculating pass@{k_val}: {e}")

            # Add token usage
            for key, value in result.total_token_usage().items():
                overall_metrics[f"overall/token_usage/{key}"] = value

            # Add tool stats
            for key, value in result.total_tool_calls().items():
                overall_metrics[f"overall/tool_calls/{key}"] = value

            self.run.log(overall_metrics)
            self.run.summary.update(overall_metrics)

        except Exception as e:
            logger.error(f"Error calculating or logging overall metrics: {e}")
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
                task_metrics = {
                    f"task_{task_id}/average_score": result._calculate_task_average_score(
                        task_id
                    ),
                    f"task_{task_id}/success_rate": result.task_success_rate(task_id),
                }

                task_avg_duration = result.task_average_duration(task_id)
                if task_avg_duration:
                    task_metrics[f"task_{task_id}/average_duration_s"] = (
                        task_avg_duration
                    )

                # Add task token usage
                for key, value in result.task_total_token_usage(task_id).items():
                    task_metrics[f"task_{task_id}/token_usage/{key}"] = value

                # Add task pass@k and pass^k
                for k_val in k_values:
                    try:
                        task_metrics[f"task_{task_id}/pass@{k_val}"] = (
                            result.task_pass_at_k(task_id, k_val)
                        )
                        task_metrics[f"task_{task_id}/pass^{k_val}"] = (
                            result.task_pass_hat_k(task_id, k_val)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Not enough trials for task {task_id} to calculate pass@{k_val}: {e}"
                        )

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
        """Log agent message files if verbose was True"""
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
        """Finish wandb run"""
        if self.run:
            self.run.finish()
            logger.info("Wandb run finished")
