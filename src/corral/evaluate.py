import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests
from loguru import logger

from corral.report import (
    BenchmarkResult,
    TaskTrailResult,
    TaskTrialResults,
    ToolResponse,
)


class BenchmarkInterface:
    """General interface for interacting with benchmark server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

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

    def get_available_tools_for_task(self, task_id: str) -> dict[str, Any]:
        """Get list of available tools for a task"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/tools")
        response.raise_for_status()
        return response.json()

    def get_task_guide(self, task_id: str) -> str:
        """Get complete guide for task including tools"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/guide")
        response.raise_for_status()
        return response.json()["prompt"]

    def get_tools_guide(self, task_id: str) -> str:
        """Get tools guide for task"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/tools/guide")
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
    ) -> BenchmarkResult:
        """Run benchmark"""

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

        if dependency_chain:
            self._run_chained_execution(
                task_ids, trials_per_task, task_results, session_id, verbose
            )
        else:
            self._run_independent_execution(
                task_ids, trials_per_task, task_results, session_id, verbose
            )

        return BenchmarkResult(task_results=task_results, k=k_values)

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

        checkpoint = self._load_checkpoint(session_id)
        completed_tasks = (
            set(checkpoint.get("completed_tasks", [])) if checkpoint else set()
        )

        remaining_tasks = [
            task_id
            for task_id in task_ids
            if task_id not in completed_tasks
            or len(task_results[task_id].trials) < trials_per_task
        ]

        logger.info(
            f"Independent execution: {len(completed_tasks)} completed, {len(remaining_tasks)} remaining"
        )

        for task_id in remaining_tasks:
            current_trials = len(task_results[task_id].trials)
            remaining_trials = trials_per_task - current_trials

            logger.info(
                f"=== Running {remaining_trials} remaining trials for task {task_id} ==="
            )

            # Complete all remaining trials for this task
            for trial_num in range(current_trials, trials_per_task):
                success = self._run_single_trial(
                    task_id, task_results[task_id], verbose
                )
                if not success:
                    logger.error(
                        f"Task {task_id} failed on trial {trial_num + 1}, stopping this task"
                    )
                    break

            # Mark task as completed if we finished all trials
            if len(task_results[task_id].trials) == trials_per_task:
                completed_tasks.add(task_id)

            # Save progress after each task
            self._save_checkpoint(
                session_id, task_results, completed_tasks=list(completed_tasks)
            )

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
        completed_trials = checkpoint.get("completed_trials", 0) if checkpoint else 0

        logger.info(f"Chained execution: starting from trial {completed_trials + 1}")

        # Run trials in lockstep across all tasks
        for trial_num in range(completed_trials, trials_per_task):
            logger.info(
                f"=== Starting trial {trial_num + 1}/{trials_per_task} for ALL tasks ==="
            )

            trial_success = True

            # Run this trial for each task in sequence
            for task_id in task_ids:
                success = self._run_single_trial(
                    task_id, task_results[task_id], verbose
                )
                if not success:
                    logger.error(
                        f"CRITICAL: Task {task_id} failed in trial {trial_num + 1}"
                    )
                    trial_success = False
                    break  # Stop this trial round

            if trial_success:
                # All tasks completed this trial successfully
                self._save_checkpoint(
                    session_id, task_results, completed_trials=trial_num + 1
                )
                logger.info(
                    f"Trial {trial_num + 1} completed successfully for all tasks"
                )
            else:
                # Save progress but don't increment completed_trials (will restart this trial)
                self._save_checkpoint(
                    session_id, task_results, completed_trials=trial_num
                )
                logger.error(
                    f"Trial {trial_num + 1} failed, will restart from this trial"
                )
                break

    def _save_checkpoint(
        self, session_id: str, task_results: dict, **extra_data
    ) -> None:
        """Save unified checkpoint format"""
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **extra_data,  # completed_tasks (list) for independent, completed_trials (int) for chained
        }
        self._save_checkpoint_file(session_id, checkpoint)

    def _save_checkpoint_file(self, session_id: str, checkpoint: dict) -> None:
        """Save checkpoint to file with atomic write"""
        checkpoint_path = self._get_checkpoint_path(session_id)
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                pickle.dump(checkpoint, f)
            temp_path.rename(checkpoint_path)  # Atomic rename
            logger.info(f"Checkpoint saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """Load checkpoint from file"""
        checkpoint_path = self._get_checkpoint_path(session_id)
        if checkpoint_path.exists():
            try:
                with checkpoint_path.open("rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Corrupted checkpoint, starting fresh: {e}")
        return None

    def _run_single_trial(
        self,
        task_id: str,
        task_trials: TaskTrialResults,
        verbose: bool = False,
    ) -> bool:
        """Run a single trial for a task. Returns True if successful, False otherwise."""
        try:
            logger.info(
                f"Running trial {len(task_trials.trials) + 1} for task {task_id}"
            )
            answer, token_usage = self.agent.run_agent(
                self.interface, task_id, verbose=verbose
            )
            result = self.interface.submit_answer(task_id, answer)
            duration = result.state.get("duration")
            result.duration = duration
            result.token_usage = token_usage

            task_trials.trials.append(result)

            logger.info(
                f"Trial completed for {task_id}, score: {result.score}, duration: {duration:.2f}s"
            )
            return True

        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
            raise

        except Exception as e:
            logger.error(f"Error during trial for task {task_id}: {e}")
            return False
