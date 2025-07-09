import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests
import wandb
from loguru import logger

from corral.ablations import ToolVerbosity
from corral.report import (
    BenchmarkResult,
    InsufficientTrialsError,
    NoResultsError,
    TaskTrailResult,
    TaskTrialResults,
    ToolResponse,
)


class BenchmarkInterface:
    """General interface for interacting with benchmark server"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_verbosity: str | None = ToolVerbosity.FULL,
    ):
        self.base_url = base_url
        self.current_verbosity = default_verbosity

    def set_verbosity(self, verbosity: str):
        """Set the verbosity level for subsequent requests"""
        self.current_verbosity = verbosity
        logger.info(f"Set tool verbosity to: {verbosity}")

    def get_available_tasks(self) -> list[str]:
        """Get list of available task IDs"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def supports_dependency_chain(self) -> bool:
        """Check if the environment supports dependency chaining"""
        try:
            response = requests.get(f"{self.base_url}/dependency_chain")
            response.raise_for_status()
            return response.json()["dependency_chain"]
        except Exception:
            return False

    def get_available_tools_for_task(
        self, task_id: str, verbosity: str | None = None
    ) -> dict[str, Any]:
        """Get list of available tools for a task with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(f"{self.base_url}/tasks/{task_id}/tools", params=params)
        response.raise_for_status()
        return response.json()

    def get_task_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get complete guide for task including tools with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(f"{self.base_url}/tasks/{task_id}/guide", params=params)
        response.raise_for_status()
        return response.json()["prompt"]

    def get_tools_guide(self, task_id: str, verbosity: str | None = None) -> str:
        """Get tools guide for task with specified verbosity"""
        verbosity = verbosity or self.current_verbosity

        params = {"verbosity": verbosity}
        response = requests.get(
            f"{self.base_url}/tasks/{task_id}/tools/guide", params=params
        )
        response.raise_for_status()
        return response.json()["prompt"]

    def get_task_prompt(self, task_id: str) -> str | list[dict]:
        """Get task prompt without tools description"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/prompt")
        response.raise_for_status()
        return response.json()["prompt"]

    def execute_tool(
        self, task_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResponse:
        """Execute a tool and get result"""
        try:
            logger.info(f"Agent calling tool {tool_name} with args {arguments}")
            response = requests.post(
                f"{self.base_url}/tasks/{task_id}/tools/execute",
                json={"tool_name": tool_name, "arguments": arguments},
            )
            response.raise_for_status()
            data = response.json()
            return ToolResponse(success=True, result=data["result"], error=None)
        except Exception as e:
            return ToolResponse(success=False, result=None, error=str(e))

    def submit_answer(self, task_id: str, answer: str) -> TaskTrailResult:
        """Submit final answer for a task"""
        logger.info(f"Agent submitting answer {answer} for task {task_id}")
        response = requests.post(
            f"{self.base_url}/tasks/{task_id}/submit", json={"answer": answer}
        )
        response.raise_for_status()
        data = response.json()
        return TaskTrailResult(
            task_id=task_id,
            trial_id=data["trial_id"],
            score=data["score"],
            state=data["state"],
            tool_statistics=data["state"]["tool_statistics"],
        )

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get current status of a task"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/status")
        response.raise_for_status()
        return response.json()

    def get_trial_state(self, task_id: str, trial_id: str) -> dict[str, Any]:
        """Get specific trial state"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/trials/{trial_id}")
        response.raise_for_status()
        return response.json()["trial_state"]


class Agent(Protocol):
    """Protocol defining what an agent must implement"""

    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
        """Solve a task and return the answer"""
        ...


class MatAgentBenchmark:
    """Runs benchmarks using an agent implementation"""

    def __init__(
        self,
        interface: BenchmarkInterface,
        agent: Agent,
        checkpoint_dir: str = "./benchmark_checkpoints",
        checkpoint_name: str | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        wandb_name: str | None = None,
    ):
        self.interface = interface
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = (
            checkpoint_name
            if checkpoint_name is not None
            else f"checkpoint_{self.agent.__class__.__name__}"
        )
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self._wandb_run = None  # To hold the wandb run object
        self._trial_details_table = None  # Table to accumulate trial results

    def _get_checkpoint_path(self, session_id: str) -> Path:
        """Get path for checkpoint file"""
        return self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"

    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool | None = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark"""
        if tool_verbosity is not None:
            self.interface.set_verbosity(tool_verbosity)
        if task_ids is None:
            task_ids = self.interface.get_available_tasks()

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        if session_id is None:
            session_id = (
                f"session_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            )

        # Handle k_values
        if k_values is None:
            k_values = list(range(1, trials_per_task + 1))
        elif isinstance(k_values, int):
            k_values = [k_values]
        elif isinstance(k_values, list) and max(k_values) > trials_per_task:
            raise ValueError("k value is greater than the number of trials")

        # Check dependency chain setting
        dependency_chain = self.interface.supports_dependency_chain()

        # Initialize or load task results from checkpoint
        task_results = self._initialize_task_results(task_ids, session_id)

        logger.info(
            f"Running benchmark: {len(task_ids)} tasks with {trials_per_task} trials each"
        )
        logger.info(f"Dependency chain: {dependency_chain}")

        # --- Wandb Initialization ---
        if self.wandb_project:
            run_name = (
                self.wandb_name
                or f"{self.agent.__class__.__name__}-{getattr(self.agent, 'model', 'unknown_model')}-{self.interface.current_verbosity}-{session_id[:8]}"
            )
            logger.info(f"Initializing wandb run: {run_name}")

            # Define initial config
            initial_config = {
                "agent_type": self.agent.__class__.__name__,
                "model": getattr(self.agent, "model", "unknown_model"),
                "tool_verbosity": self.interface.current_verbosity,
                "trials_per_task": trials_per_task,
                "k_values": k_values,
                "dependency_chain": dependency_chain,
                "session_id": session_id,
                "checkpoint_name": self.checkpoint_name,
                "task_ids": task_ids,
                "agent_max_iterations": getattr(self.agent, "max_iterations", None),
                "agent_temperature": getattr(self.agent, "temperature", None),
            }

            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                group=self.wandb_group,
                name=run_name,
                config=initial_config,
            )
            logger.info(f"Wandb run initialized: {self._wandb_run.url}")

            # Initialize the wandb Table for trial details
            self._trial_details_table = wandb.Table(
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
        else:
            logger.warning(
                "Wandb project not specified or wandb not installed, skipping wandb logging."
            )

        # --- Start Benchmark Execution ---
        start_time = datetime.now(tz=timezone.utc)
        try:
            if dependency_chain:
                self._run_chained_execution(
                    task_ids, trials_per_task, task_results, session_id, verbose
                )
            else:
                self._run_independent_execution(
                    task_ids, trials_per_task, task_results, session_id, verbose
                )
        except Exception as e:
            logger.error(
                f"An error occurred during benchmark execution: {e!s}", exc_info=True
            )
            # Log the error itself to wandb if possible
            if self._wandb_run:
                self._wandb_run.log({"benchmark_execution_error": str(e)})
            raise e

        finally:
            # Ensure wandb finish is called even if an error occurred
            end_time = datetime.now(tz=timezone.utc)
            total_duration = (end_time - start_time).total_seconds()

            # --- Generate and Log Report/Metrics ---
            # Use the collected task_results to create the final BenchmarkResult
            final_benchmark_result = BenchmarkResult(
                task_results=task_results,
                k=k_values,
                verbosity=self.interface.current_verbosity,
                total_duration=total_duration,  # Pass total duration
            )

            if self._wandb_run:
                logger.info("Logging final benchmark metrics to wandb")
                overall_metrics = {}

                try:
                    overall_metrics.update(
                        {
                            "overall/average_score": final_benchmark_result.average_score(),
                            "overall/overall_success_rate": final_benchmark_result.overall_success_rate(),
                            "overall/total_tasks": final_benchmark_result.total_tasks,
                            "overall/total_trial_duration_s": final_benchmark_result.overall_total_duration(),
                            "overall/average_trial_duration_s": final_benchmark_result.overall_average_duration(),
                            "overall/total_tool_execution_duration_s": final_benchmark_result.total_tool_execution_duration(),
                            "overall/total_benchmark_duration_s": total_duration,
                        }
                    )

                    # Add pass@k and pass^k
                    for k_val in k_values:
                        try:
                            overall_metrics[f"overall/pass@{k_val}"] = (
                                final_benchmark_result.overall_pass_at_k(k_val)
                            )
                            overall_metrics[f"overall/pass^{k_val}"] = (
                                final_benchmark_result.overall_pass_hat_k(k_val)
                            )
                        except InsufficientTrialsError:
                            logger.warning(
                                f"Not enough trials to calculate overall pass@{k_val} or pass^{k_val}"
                            )
                        except NoResultsError:
                            logger.warning(
                                f"No results to calculate overall pass@{k_val} or pass^{k_val}"
                            )

                    # Add total token usage
                    total_token_usage = final_benchmark_result.total_token_usage()
                    for key, value in total_token_usage.items():
                        overall_metrics[f"overall/token_usage/{key}"] = value

                    # Add total tool call stats
                    tool_call_stats = final_benchmark_result.total_tool_calls()
                    for key, value in tool_call_stats.items():
                        overall_metrics[f"overall/tool_calls/{key}"] = value

                    self._wandb_run.log(overall_metrics)

                except Exception as e:
                    logger.error(
                        f"Error calculating or logging overall metrics: {e!s}",
                        exc_info=True,
                    )
                    if self._wandb_run:
                        self._wandb_run.log({"overall_metrics_error": str(e)})

                # Log task-level metrics
                logger.info("Logging task-level metrics to wandb")
                for task_id in task_results:  # Iterate over task_results keys to ensure we only process tasks that were run
                    if not task_results[task_id].trials:
                        logger.warning(
                            f"No trials recorded for task {task_id}, skipping task-level metric logging."
                        )
                        continue
                    try:
                        task_metrics = {
                            f"task_{task_id}/average_score": final_benchmark_result._calculate_task_average_score(
                                task_id
                            ),
                            f"task_{task_id}/success_rate": final_benchmark_result.task_success_rate(
                                task_id
                            ),
                            f"task_{task_id}/average_duration_s": final_benchmark_result.task_average_duration(
                                task_id
                            ),
                        }
                        task_token_usage = (
                            final_benchmark_result.task_total_token_usage(task_id)
                        )
                        for key, value in task_token_usage.items():
                            task_metrics[f"task_{task_id}/token_usage/{key}"] = value

                        for k_val in k_values:
                            try:
                                task_metrics[f"task_{task_id}/pass@{k_val}"] = (
                                    final_benchmark_result.task_pass_at_k(
                                        task_id, k_val
                                    )
                                )
                                task_metrics[f"task_{task_id}/pass^{k_val}"] = (
                                    final_benchmark_result.task_pass_hat_k(
                                        task_id, k_val
                                    )
                                )
                            except InsufficientTrialsError:
                                logger.warning(
                                    f"Not enough trials ({len(task_results[task_id].trials)}) for task {task_id} to calculate pass@{k_val} or pass^{k_val}"
                                )
                            except NoResultsError:
                                # This shouldn't happen if task_results[task_id].trials is not empty, but as a safeguard
                                logger.warning(
                                    f"No results for task {task_id}, cannot calculate pass@k or pass^k"
                                )

                        self._wandb_run.log(task_metrics)
                    except Exception as e:
                        logger.error(
                            f"Error calculating or logging task {task_id} metrics: {e!s}",
                            exc_info=True,
                        )
                        if self._wandb_run:
                            self._wandb_run.log(
                                {f"task_{task_id}/metrics_error": str(e)}
                            )

                # Log the trial details table if it has data
                if self._trial_details_table and self._trial_details_table.data:
                    logger.info("Logging trial details table to wandb")
                    try:
                        self._wandb_run.log(
                            {"trial_details": self._trial_details_table}
                        )
                    except Exception as e:
                        logger.error(
                            f"Error logging trial details table: {e!s}", exc_info=True
                        )

                # Log agent message files if verbose was True
                if verbose:
                    logger.info("Logging agent message logs as wandb artifacts")
                    # Assumes agent logs are saved in './agent_logs' as per save_agent_messages default
                    log_dir = Path("./agent_logs")
                    if log_dir.exists():
                        try:
                            artifact = wandb.Artifact("agent_logs", type="logs")
                            artifact.add_dir(str(log_dir))
                            self._wandb_run.log_artifact(artifact)
                            logger.info(f"Logged agent logs from {log_dir} as artifact")
                        except Exception as e:
                            logger.error(
                                f"Error logging agent logs artifact: {e!s}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            f"Agent log directory {log_dir} not found, skipping log artifact logging."
                        )

            if self._wandb_run:
                logger.info("Finishing wandb run")
                self._wandb_run.finish()

        return final_benchmark_result

    def _initialize_task_results(
        self, task_ids: list[str], session_id: str
    ) -> dict[str, TaskTrialResults]:
        """Initialize task results, loading from checkpoint if available"""
        checkpoint = self._load_checkpoint(session_id)

        if checkpoint and "task_results" in checkpoint:
            task_results = checkpoint["task_results"]
            logger.info(
                f"Loaded existing results from checkpoint for {len(task_results)} tasks"
            )

            # Ensure all requested tasks have entries
            for task_id in task_ids:
                if task_id not in task_results:
                    task_results[task_id] = TaskTrialResults(task_id=task_id)
        else:
            # Initialize fresh results
            task_results = {
                task_id: TaskTrialResults(task_id=task_id) for task_id in task_ids
            }

        if self._wandb_run:
            task_initial_trials = {
                task_id: len(task_trials.trials)
                for task_id, task_trials in task_results.items()
            }
            logger.info(
                f"Initial trial counts loaded/initialized: {task_initial_trials}"
            )

        return task_results

    def _run_independent_execution(
        self,
        task_ids: list[str],
        trials_per_task: int,
        task_results: dict,
        session_id: str,
        verbose: bool = False,
    ) -> None:
        """Run trials independently - complete all trials for each task"""

        # Filter tasks that still need trials
        remaining_tasks = [
            task_id
            for task_id in task_ids
            if len(task_results[task_id].trials) < trials_per_task
        ]

        logger.info(
            f"Independent execution: {
                len(task_ids) - len(remaining_tasks)} tasks fully completed, {
                len(remaining_tasks)} tasks with trials remaining"
        )

        for task_id in remaining_tasks:
            current_trials_count = len(task_results[task_id].trials)
            remaining_trials = trials_per_task - current_trials_count

            logger.info(
                f"=== Running {remaining_trials} remaining trials for task {task_id} (Starting from Trial {current_trials_count + 1}) ==="
            )

            # Run remaining trials for this task
            # Use a while loop to handle potential re-runs if checkpointing happens mid-task trials
            while len(task_results[task_id].trials) < trials_per_task:
                trial_index = len(
                    task_results[task_id].trials
                )  # Index of the trial about to be run
                logger.info(
                    f"Starting trial {trial_index + 1}/{trials_per_task} for task {task_id}"
                )

                # _run_single_trial will attempt to execute and append the result (success or failure)
                success = self._run_single_trial(
                    task_id, task_results[task_id], verbose
                )

                # _run_single_trial appends the result; we log the appended result
                if task_results[task_id].trials:
                    latest_trial = task_results[task_id].trials[-1]
                    # Check if this trial was just run or loaded from checkpoint
                    # Avoid logging the same trial multiple times if resuming mid-task trial sequence
                    # A robust check might involve timestamp or run_id, but trial_id should suffice if unique
                    if latest_trial.state.get("trial_id") not in [
                        t.state.get("trial_id")
                        for t in task_results[task_id].trials[:-1]
                    ]:
                        self._log_trial_result(latest_trial)
                    else:
                        logger.debug(
                            f"Skipping logging for trial {latest_trial.state.get('trial_id')} of task {task_id}, already logged or loaded."
                        )

                # Save progress after each trial completes or fails within a task
                self._save_checkpoint(session_id, task_results)

                if not success:
                    logger.error(
                        f"Task {task_id} failed on trial {trial_index + 1}. Stopping subsequent trials for this task."
                    )
                    break  # Stop trials for this specific task if one fails (configurable behavior)

    def _run_chained_execution(
        self,
        task_ids: list[str],
        trials_per_task: int,
        task_results: dict,
        session_id: str,
        verbose: bool = False,
    ) -> None:
        """Run trials in lockstep - complete one trial across all tasks before next trial"""

        checkpoint = self._load_checkpoint(session_id)
        completed_trial_rounds = (
            checkpoint.get("completed_trials", 0) if checkpoint else 0
        )  # Number of full trial rounds completed across all tasks

        logger.info(
            f"Chained execution: starting from trial round {completed_trial_rounds + 1}/{trials_per_task}"
        )

        # Run trials in lockstep across all tasks
        for trial_round_num in range(completed_trial_rounds, trials_per_task):
            logger.info(
                f"=== Starting trial round {trial_round_num + 1}/{trials_per_task} for ALL tasks ==="
            )

            current_round_success = (
                True  # Flag for the current trial round across all tasks
            )

            # Run this trial round for each task in sequence
            for task_id in task_ids:
                # Check if this specific task/trial in this round was already completed (e.g., after a crash and resume)
                if len(task_results[task_id].trials) > trial_round_num:
                    logger.info(
                        f"Task {task_id} trial {trial_round_num + 1} appears completed from checkpoint. Skipping execution for this task in this round."
                    )
                    # Log the loaded trial result if not already in the table
                    loaded_trial = task_results[task_id].trials[trial_round_num]
                    # This check is a bit hacky; relies on trial_id uniqueness within the run.
                    # A better way is to query the wandb run if the trial_id exists, but that adds complexity.
                    # For now, rely on adding to the table which should handle duplicates implicitly if trial_id is consistent.
                    # Or, ensure _log_trial_result checks if the row already exists in the table based on task_id and trial_id.
                    # Let's modify _log_trial_result to check if the row exists before adding.
                    self._log_trial_result(
                        loaded_trial
                    )  # Always attempt to add, _log_trial_result will handle duplicates

                    continue

                logger.info(f"Running trial {trial_round_num + 1} for task {task_id}")
                # _run_single_trial will attempt to execute and append the result (success or failure)
                success = self._run_single_trial(
                    task_id, task_results[task_id], verbose
                )

                # _run_single_trial appends the result; we log the appended result
                if task_results[task_id].trials:
                    latest_trial = task_results[task_id].trials[-1]
                    # In chained mode, the latest trial should always be for the current round if execution ran
                    self._log_trial_result(latest_trial)

                if not success:
                    logger.error(
                        f"CRITICAL: Task {task_id} failed in trial round {trial_round_num + 1}. Stopping chained execution."
                    )
                    current_round_success = False
                    break  # Stop this trial round across tasks

            # Save progress after the trial round (either successful or failed)
            # If current_round_success is True, increment completed_trials.
            # If False, completed_trials stays the same, allowing resumption from the failed trial round.
            self._save_checkpoint(
                session_id,
                task_results,
                completed_trials=trial_round_num + 1
                if current_round_success
                else trial_round_num,
            )

            if not current_round_success:
                logger.error(
                    f"Trial round {trial_round_num + 1} failed. Chained execution halted."
                )
                break  # Stop the main trial round loop

    def _save_checkpoint(
        self, session_id: str, task_results: dict, **extra_data
    ) -> None:
        """Save unified checkpoint format"""
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            # completed_trials (int) for chained
            **extra_data,
        }
        checkpoint_path = self._get_checkpoint_path(session_id)
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                # Use a protocol compatible across Python versions, or handle versioning
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(checkpoint_path)  # Atomic rename
            logger.info(f"Checkpoint saved for session {session_id}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint {checkpoint_path}: {e!s}", exc_info=True
            )
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)  # Use missing_ok for robustness
            # Don't re-raise here; failing to save checkpoint shouldn't necessarily stop the benchmark

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """Load checkpoint from file"""
        checkpoint_path = self._get_checkpoint_path(session_id)
        if checkpoint_path.exists():
            try:
                with checkpoint_path.open("rb") as f:
                    checkpoint_data = pickle.load(f)
                    logger.info(
                        f"Checkpoint loaded successfully from {checkpoint_path}"
                    )
                    return checkpoint_data
            except Exception as e:
                logger.warning(
                    f"Error loading checkpoint {checkpoint_path}, starting fresh: {e!s}",
                    exc_info=True,
                )
                # Optionally delete the corrupted checkpoint
                # checkpoint_path.unlink(missing_ok=True)
        return None

    def _run_single_trial(
        self,
        task_id: str,
        task_trials: TaskTrialResults,
        verbose: bool = False,
    ) -> bool:
        """Run a single trial for a task. Returns True if submission was attempted, False otherwise."""
        # Determine the index of this trial within the task's attempts
        trial_index = len(task_trials.trials)
        logger.info(f"Attempting trial {trial_index + 1} for task {task_id}")

        failed_result = None  # Initialize failed result placeholder

        try:
            # The agent handles its internal message history reset/management per run_agent call
            # The interface call will get the current task prompt and tools based on the server state
            # If dependency_chain is true, the server manages the state transitions between tasks.
            # The agent just needs the current task prompt.

            # Agent run_agent returns answer string and token_usage dict
            answer, token_usage = self.agent.run_agent(
                self.interface, task_id, verbose=verbose
            )

            # Attempt to submit the answer to the server
            try:
                result = self.interface.submit_answer(task_id, answer)
                # submit_answer returns TaskTrailResult with server-side trial_id and state
                # Manually add client-side data (token_usage)
                result.token_usage = token_usage
                # The server-side duration is already in result.state['duration'] and copied to result.duration by the interface

                task_trials.trials.append(result)

                logger.info(
                    f"Trial {result.state.get('trial_id', trial_index + 1)} for task {task_id} completed. Score: {result.score:.3f},  Tokens: {token_usage.get('total', 'N/A')}"
                )

                # Logging to wandb Table happens in the calling loop (_run_independent_execution or _run_chained_execution)
                # after the result is appended to task_trials.trials

                return True  # Indicate successful submission attempt

            except Exception as submit_error:
                logger.error(
                    f"Error submitting answer for task {task_id}, trial attempt {trial_index + 1}: {submit_error!s}",
                    exc_info=True,
                )
                # Submission failed, create a failed result entry
                failed_result = TaskTrailResult(
                    task_id=task_id,
                    trial_id=f"attempt_{trial_index + 1}",  # Use attempt index as ID since server ID is not available
                    score=0.0,  # Assume 0 score on submission failure
                    state={
                        "error": str(submit_error),
                        "attempt": trial_index + 1,
                    },  # Capture error info
                    tool_statistics={
                        "error": str(submit_error)
                    },  # Minimal tool stats on failure
                    duration=None,  # Duration to submission unknown
                    token_usage=token_usage,  # Capture token usage up to the point before submission
                    error_message=f"Submission Error: {submit_error!s}",  # Capture submission error
                )
                task_trials.trials.append(
                    failed_result
                )  # Append the failed result to record it

                # Log the failed result immediately
                if self._wandb_run:
                    self._log_trial_result(failed_result, is_failure=True)

                return False  # Indicate submission failure/error

        except KeyboardInterrupt:
            logger.info("Benchmark trial interrupted by user")
            # On interruption, capture state and log a failed trial
            interrupted_result = TaskTrailResult(
                task_id=task_id,
                trial_id=f"attempt_{trial_index + 1}_interrupted",
                score=0.0,
                state={"error": "KeyboardInterrupt", "attempt": trial_index + 1},
                tool_statistics={"error": "KeyboardInterrupt"},
                duration=None,
                token_usage=self.agent.get_total_token_usage(),  # Capture usage up to interrupt
                error_message="KeyboardInterrupt by user",
            )
            task_trials.trials.append(interrupted_result)
            if self._wandb_run:
                self._log_trial_result(interrupted_result, is_failure=True)
                self._wandb_run.log(
                    {"status": "interrupted"}
                )  # Log overall interruption status

            raise  # Re-raise the exception to stop the benchmark

        except Exception as agent_run_error:
            logger.error(
                f"Error during agent run for task {task_id}, trial attempt {trial_index + 1}: {agent_run_error!s}",
                exc_info=True,
            )
            # Agent run failed before submission attempt
            failed_result = TaskTrailResult(
                task_id=task_id,
                trial_id=f"attempt_{trial_index + 1}",  # Use attempt index as ID
                score=0.0,  # Assume 0 score on agent failure
                state={
                    "error": str(agent_run_error),
                    "attempt": trial_index + 1,
                },  # Capture error info
                tool_statistics=self.agent.get_total_token_usage(),  # Use token usage as minimal stats
                duration=None,
                token_usage=self.agent.get_total_token_usage(),  # Capture token usage up to failure
                error_message=f"Agent Run Error: {agent_run_error!s}",  # Capture agent error
            )
            task_trials.trials.append(
                failed_result
            )  # Append the failed result to record it

            # Log the failed result immediately
            if self._wandb_run:
                self._log_trial_result(failed_result, is_failure=True)

            return False  # Indicate agent run failure

    def _log_trial_result(
        self, trial_result: TaskTrailResult, is_failure: bool = False
    ) -> None:
        """Adds individual trial results to the wandb Table."""
        if not self._trial_details_table:
            return  # Only log if the table was initialized

        logger.info(
            f"Adding trial {trial_result.state.get('trial_id', trial_result.trial_id)} for task {trial_result.task_id} to wandb table."
        )

        # Check if this row already exists in the table based on task_id and trial_id
        # This is a basic check and might not be perfect if trial_id generation is inconsistent
        trial_identifier = trial_result.state.get(
            "trial_id", trial_result.trial_id
        )  # Use server ID if available, else local
        if any(
            row[0] == trial_result.task_id and row[1] == trial_identifier
            for row in self._trial_details_table.data
        ):
            logger.debug(
                f"Trial {trial_identifier} for task {trial_result.task_id} already exists in table. Skipping addition."
            )
            return

        tool_stats = (
            trial_result.tool_statistics
            if isinstance(trial_result.tool_statistics, dict)
            else {}
        )

        # Calculate tool execution duration if available in detailed calls
        trial_tool_exec_duration = 0.0
        if "tool_calls" in tool_stats and isinstance(tool_stats["tool_calls"], list):
            for tool_call in tool_stats["tool_calls"]:
                if (
                    isinstance(tool_call, dict)
                    and tool_call.get("duration") is not None
                ):
                    trial_tool_exec_duration += tool_call["duration"]

        self._trial_details_table.add_data(
            trial_result.task_id,
            trial_identifier,  # Use the selected identifier
            trial_result.score,
            trial_result.success,
            is_failure
            or not trial_result.success,  # Mark as failure if is_failure flag is true OR if success property is false
            trial_result.duration,
            trial_result.token_usage.get("prompt_tokens")
            if trial_result.token_usage
            else None,
            trial_result.token_usage.get("completion_tokens")
            if trial_result.token_usage
            else None,
            trial_result.token_usage.get("total_tokens")
            if trial_result.token_usage
            else None,
            tool_stats.get("total_calls"),
            tool_stats.get("successful_calls"),
            tool_stats.get("failed_calls"),
            trial_tool_exec_duration,
        )
