import pickle
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import requests
from loguru import logger

from corral.ablations import ToolVerbosity
from corral.report import (
    BenchmarkResult,
    CorralWandbLogger,
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
    interface: "BenchmarkInterface",
    agent: "Agent",
    verbose: bool = False,
) -> TaskTrailResult:
    """Execute a single trial - pure function"""
    try:
        # Run agent
        answer, token_usage = agent.run_agent(interface, task_id, verbose=verbose)

        # Submit answer
        try:
            result = interface.submit_answer(task_id, answer)
            result.token_usage = token_usage
            return result
        except Exception as submit_error:
            return TaskTrailResult(
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
        return TaskTrailResult(
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
    trial_executor: Callable[[str, int], TaskTrailResult],
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
    trial_executor: Callable[[str, int], TaskTrailResult],
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
                logger.error(f"Trial failed for task {task_id}")
                success = False
        # Save checkpoint after each round
        checkpoint_saver(task_results, trial_round + 1 if success else trial_round)


def create_wandb_config(agent: "Agent", session_id: str, **kwargs) -> dict[str, Any]:
    """Create wandb configuration"""
    return {
        "agent_type": agent.__class__.__name__,
        "model": getattr(agent, "model", "unknown_model"),
        "session_id": session_id,
        "agent_max_iterations": getattr(agent, "max_iterations", None),
        "agent_temperature": getattr(agent, "temperature", None),
        **kwargs,
    }


class MatAgentBenchmark:
    """Simplified benchmark runner with functional approach"""

    def __init__(
        self,
        interface: "BenchmarkInterface",
        agent: "Agent",
        checkpoint_dir: str = "./benchmark_checkpoints",
        checkpoint_name: str | None = None,
        logger: CorralWandbLogger | None = None,
    ):
        self.interface = interface
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = (
            checkpoint_name or f"checkpoint_{agent.__class__.__name__}"
        )
        self.logger = logger

    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
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
        trial_executor = partial(
            execute_single_trial,
            interface=self.interface,
            agent=self.agent,
            verbose=verbose,
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

        def wrapped_executor(task_id: str, trial_index: int) -> TaskTrailResult:
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

    def _save_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults], **extra_data
    ) -> None:
        """Save checkpoint to file"""
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **extra_data,
        }

        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(checkpoint_path)
            logger.info(f"Checkpoint saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """Load checkpoint from file"""
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        if not checkpoint_path.exists():
            return None

        try:
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info("Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            return None
