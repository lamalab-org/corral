import pickle
import re
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from corral.agents import BaseAgent
from corral.report import (
    BenchmarkResult,
    CorralWandbLogger,
    TaskTrialResult,
    TaskTrialResults,
)
from corral.router import CorralRouter


def create_session_id() -> str:
    """Create a unique session ID"""
    return f"session_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def validate_k_values(
    k_values: int | list[int] | None, trials_per_task: int
) -> list[int]:
    """Validate and normalize k_values"""
    if k_values is None:
        return list(range(1, trials_per_task + 1))
    elif isinstance(k_values, int):
        return [k_values]
    elif isinstance(k_values, list) and max(k_values) > trials_per_task:
        raise ValueError("k value is greater than the number of trials")
    return k_values


def initialize_task_results(task_ids: list[str]) -> dict[str, TaskTrialResults]:
    """Initialize empty task results"""
    return {task_id: TaskTrialResults(task_id=task_id) for task_id in task_ids}


def filter_incomplete_tasks(
    task_results: dict[str, TaskTrialResults], trials_per_task: int
) -> list[str]:
    """Get list of tasks that still need trials"""
    return [
        task_id
        for task_id, results in task_results.items()
        if len(results.trials) < trials_per_task
    ]


def execute_single_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    agent: BaseAgent,
    verbose: bool = False,
    tool_verbosity: str | None = None,
    configure_timeout: float | None = None,
    enable_surrender: bool = False,
) -> TaskTrialResult:
    """Execute a single trial - pure function"""
    try:
        status = interface.configure_additional_apps(task_id, timeout=configure_timeout)
        logger.info(f"Task {task_id} additional apps/services configured: {status}")

        answer, token_usage = agent.run_agent(
            interface,
            task_id,
            verbose=verbose,
            tool_verbosity=tool_verbosity or "brief",
            enable_surrender=enable_surrender,
        )

        # Check if agent decided to surrender
        if answer == "SURRENDER":
            try:
                result = interface.surrender_task(task_id)
                result.token_usage = token_usage
                return result
            except Exception as surrender_error:
                return TaskTrialResult(
                    task_id=task_id,
                    trial_id=f"attempt_{trial_index + 1}",
                    score=0.0,
                    state={"error": str(surrender_error), "attempt": trial_index + 1},
                    tool_statistics={"error": str(surrender_error)},
                    duration=None,
                    token_usage=token_usage,
                    error_message=f"Surrender Error: {surrender_error}",
                    surrendered=True,
                )

        # Submit answer
        try:
            result = interface.submit_answer(task_id, answer)
            result.token_usage = token_usage
            return result
        except Exception as submit_error:
            return TaskTrialResult(
                task_id=task_id,
                trial_id=f"attempt_{trial_index + 1}",
                score=0.0,
                state={"error": str(submit_error), "attempt": trial_index + 1},
                tool_statistics={"error": str(submit_error)},
                duration=None,
                token_usage=token_usage,
                error_message=f"Submission Error: {submit_error}",
            )
    except Exception as agent_error:
        return TaskTrialResult(
            task_id=task_id,
            trial_id=f"attempt_{trial_index + 1}",
            score=0.0,
            state={"error": str(agent_error), "attempt": trial_index + 1},
            tool_statistics=agent.get_total_token_usage(),
            duration=None,
            token_usage=agent.get_total_token_usage(),
            error_message=f"Agent Error: {agent_error}",
        )


def run_independent_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults]], None],
) -> None:
    """Run trials independently for each task"""
    logger.info(
        f"Running independent trials for {len(task_ids)} tasks with {trials_per_task} trials each"
    )
    remaining_tasks = filter_incomplete_tasks(task_results, trials_per_task)

    for task_id in remaining_tasks:
        while len(task_results[task_id].trials) < trials_per_task:
            trial_index = len(task_results[task_id].trials)
            logger.info(f"Starting trial {trial_index + 1} for task {task_id}")

            result = trial_executor(task_id, trial_index)
            task_results[task_id].trials.append(result)

            logger.info(f"Trial {result.trial_id} completed. Score: {result.score:.3f}")

            # Save checkpoint after each trial
            checkpoint_saver(task_results)
            if not result.success:
                logger.error(f"Trial failed for task {task_id}")


def run_chained_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults], int], None],
    completed_rounds: int = 0,
) -> None:
    """Run trials in lockstep across all tasks"""
    for trial_round in range(completed_rounds, trials_per_task):
        logger.info(f"Starting trial round {trial_round + 1}/{trials_per_task}")

        success = True
        for task_id in task_ids:
            if len(task_results[task_id].trials) > trial_round:
                continue  # Already completed

            logger.info(f"Running trial {trial_round + 1} for task {task_id}")
            result = trial_executor(task_id, trial_round)
            task_results[task_id].trials.append(result)

            if not result.success:
                logger.error(
                    f"Trial failed for task {task_id}: {getattr(result, 'error_message', 'No error message')}"
                )
                logger.error(f"Full result: {result}")
                success = False
        # Save checkpoint after each round
        checkpoint_saver(task_results, trial_round + 1 if success else trial_round)


def create_wandb_config(agent: BaseAgent, session_id: str, **kwargs) -> dict[str, Any]:
    """Create wandb configuration"""
    return {
        "agent_type": agent.__class__.__name__,
        "model": getattr(agent, "model", "unknown_model"),
        "session_id": session_id,
        "agent_max_iterations": getattr(agent, "max_iterations", None),
        "agent_temperature": getattr(agent, "temperature", None),
        **kwargs,
    }


class CorralRunner:
    """Simplified benchmark runner with functional approach"""

    def __init__(
        self,
        interface: CorralRouter,
        agent: BaseAgent,
        checkpoint_dir: str = "./benchmark_checkpoints",
        checkpoint_name: str | None = None,
        logger: CorralWandbLogger | None = None,
        enable_surrender: bool = False,
    ):
        self.interface = interface
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = (
            checkpoint_name or f"checkpoint_{agent.__class__.__name__}"
        )
        self.logger = logger
        self.enable_surrender = enable_surrender

    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
    ) -> BenchmarkResult:
        """Run benchmark with functional approach"""

        # Setup
        if tool_verbosity:
            self.interface.set_verbosity(tool_verbosity)

        task_ids = task_ids or self.interface.get_available_tasks()
        session_id = session_id or create_session_id()
        k_values = validate_k_values(k_values, trials_per_task)

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        # Initialize or load results
        task_results = self._load_or_initialize_results(task_ids, session_id)

        # Create execution functions
        def trial_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            return execute_single_trial(
                task_id=task_id,
                trial_index=trial_index,
                interface=self.interface,
                agent=self.agent,
                verbose=verbose,
                tool_verbosity=tool_verbosity,
                configure_timeout=configure_timeout,
                enable_surrender=self.enable_surrender,
            )

        checkpoint_saver = partial(self._save_checkpoint, session_id)

        # Setup logging
        if self.logger:
            config = create_wandb_config(
                self.agent,
                session_id,
                trials_per_task=trials_per_task,
                k_values=k_values,
                tool_verbosity=self.interface.current_verbosity,
                task_ids=task_ids,
                dependency_chain=self.interface.supports_dependency_chain(),
                enable_surrender=self.enable_surrender,
            )
            self.logger.start_logging(config)

        # Execute benchmark
        start_time = datetime.now(tz=timezone.utc)

        try:
            if self.interface.supports_dependency_chain():
                checkpoint = self._load_checkpoint(session_id)
                completed_rounds = (
                    checkpoint.get("completed_trials", 0) if checkpoint else 0
                )

                run_chained_trials(
                    task_ids,
                    trials_per_task,
                    task_results,
                    self._make_logging_trial_executor(trial_executor),
                    self._make_chained_checkpoint_saver(checkpoint_saver),
                    completed_rounds,
                )
            else:
                run_independent_trials(
                    task_ids,
                    trials_per_task,
                    task_results,
                    self._make_logging_trial_executor(trial_executor),
                    checkpoint_saver,
                )

            # Create final result
            end_time = datetime.now(tz=timezone.utc)
            total_duration = (end_time - start_time).total_seconds()

            result = BenchmarkResult(
                task_results=task_results,
                k=k_values,
                verbosity=self.interface.current_verbosity,
                total_duration=total_duration,
            )

            # Log final results
            if self.logger:
                self.logger.log_final_results(result, k_values)

            # Save final checkpoint with finished suffix and remove original
            self._save_finished_checkpoint(session_id, task_results)
            self._remove_original_checkpoint(session_id)

            return result

        finally:
            if self.logger:
                self.logger.finish()

    def _load_or_initialize_results(
        self, task_ids: list[str], session_id: str
    ) -> dict[str, TaskTrialResults]:
        """Load existing results or initialize new ones"""
        checkpoint = self._load_checkpoint(session_id)

        if checkpoint and "task_results" in checkpoint:
            task_results = checkpoint["task_results"]
            logger.info(f"Loaded existing results for {len(task_results)} tasks")

            # Ensure all requested tasks have entries
            for task_id in task_ids:
                if task_id not in task_results:
                    task_results[task_id] = TaskTrialResults(task_id=task_id)
        else:
            task_results = initialize_task_results(task_ids)

        return task_results

    def _make_logging_trial_executor(self, trial_executor: Callable) -> Callable:
        """Wrap trial executor with logging"""

        def wrapped_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            result = trial_executor(task_id, trial_index)
            if self.logger:
                self.logger.log_trial(result)
            return result

        return wrapped_executor

    def _make_chained_checkpoint_saver(self, checkpoint_saver: Callable) -> Callable:
        """Create checkpoint saver for chained execution"""

        def wrapped_saver(
            task_results: dict[str, TaskTrialResults], completed_trials: int
        ) -> None:
            checkpoint_saver(task_results, completed_trials=completed_trials)

        return wrapped_saver

    def _write_checkpoint_file(
        self,
        checkpoint_path: Path,
        checkpoint: dict,
        session_id: str,
        checkpoint_type: str = "checkpoint",
    ) -> None:
        """
        Write checkpoint data to file using an atomic write pattern.

        This method first writes the checkpoint to a temporary file and then renames it to the target path.
        This approach ensures that the checkpoint file is never left in a partially written or corrupted state,
        even if the process crashes or is interrupted during the write. The atomic rename operation guarantees
        that readers will either see the old file or the fully written new file, but never a half-written file.
        If an error occurs, the temporary file is cleaned up to avoid clutter.
        """
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(checkpoint_path)
            logger.info(f"{checkpoint_type.title()} saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save {checkpoint_type}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _save_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults], **extra_data
    ) -> None:
        """
        Save checkpoint to file, including all relevant session and task state.

        This method centralizes the logic for checkpoint creation, ensuring that all necessary
        metadata (such as session ID and timestamp) is included. It delegates the actual file
        writing to an atomic method to guarantee data integrity. This design allows for robust
        recovery and resumption of long-running or multi-step processes.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **extra_data,
        }

        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            checkpoint_path, checkpoint, session_id, "checkpoint"
        )

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """
        Load checkpoint from file, or recover the most recent one if not found.

        This method attempts to load a checkpoint for the given session. If the specific
        checkpoint file does not exist (e.g., due to interruption or cleanup), it searches
        for the most recent available checkpoint with the same naming pattern. This design
        increases robustness and allows for recovery from unexpected interruptions or missing files.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        if not checkpoint_path.exists():
            # Search for the most recent checkpoint with same checkpoint_name
            return self._find_most_recent_checkpoint()

        try:
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info("Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            return None

    def _find_most_recent_checkpoint(self) -> dict | None:
        """
        Find and load the most recent checkpoint file with the same checkpoint_name.

        This method scans the checkpoint directory for files matching the session pattern,
        excluding those marked as finished. It sorts the files by timestamp and loads the most
        recent one. This enables recovery from interruptions and ensures that progress is not lost
        if the latest checkpoint file is missing or incomplete. It is a fallback mechanism for robust
        checkpoint management.
        """

        # Pattern to match checkpoint files with same name but different session IDs
        # excluding "finished" files
        pattern = f"{self.checkpoint_name}_session_*.pkl"

        matching_files = []
        for file_path in self.checkpoint_dir.glob(pattern):
            file_name = file_path.name
            # Skip finished checkpoints
            if "finished" in file_name:
                continue

            # Extract session ID from filename
            match = re.search(r"session_(\d{8}_\d{6}_\d{6})", file_name)
            if match:
                session_timestamp = match.group(1)
                matching_files.append((file_path, session_timestamp))

        if not matching_files:
            logger.info("No existing checkpoints found")
            return None

        # Sort by session timestamp (most recent first)
        matching_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_file = matching_files[0][0]

        try:
            with most_recent_file.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded most recent checkpoint: {most_recent_file.name}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading most recent checkpoint: {e}")
            return None

    def _save_finished_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults]
    ) -> None:
        """
        Save the final checkpoint with a 'finished' suffix to mark completion.

        This method creates a checkpoint file that is clearly marked as finished, making it easy
        to distinguish between in-progress and completed runs. This helps prevent accidental
        resumption of already completed sessions and provides a clear audit trail for completed
        benchmarks. The atomic write pattern is used for reliability.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "finished",
        }

        finished_checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_finished_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            finished_checkpoint_path, checkpoint, session_id, "final checkpoint"
        )

    def _remove_original_checkpoint(self, session_id: str) -> None:
        """
        Remove the original (unfinished) checkpoint file after completion.

        This method deletes the in-progress checkpoint file once a finished checkpoint has been
        written. This prevents confusion between incomplete and completed runs, and helps keep
        the checkpoint directory clean. The try/except block ensures that errors during removal
        do not interrupt the main workflow.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Original checkpoint removed for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to remove original checkpoint: {e}")
