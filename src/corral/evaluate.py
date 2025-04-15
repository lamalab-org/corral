from __future__ import annotations

import pickle
from dataclasses import dataclass
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
from corral.utils import save_agent_messages


@dataclass
class TaskResult:
    """Result of a task submission"""

    score: float
    state: dict[str, Any]
    tool_statistics: dict[str, Any]


class BenchmarkInterface:
    """General interface for interacting with benchmark server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_available_tasks(self) -> list[str]:
        """Get list of available task IDs"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def get_available_tools_for_task(self, task_id: str) -> str:
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

    def _save_checkpoint(
        self,
        session_id: str,
        task_results: dict,
        current_task: str,
        completed_trials: int,
    ) -> None:
        """Save checkpoint with current benchmark progress"""
        checkpoint = {
            "task_results": task_results,
            "current_task": current_task,
            "completed_trials": completed_trials,
            "session_id": session_id,
        }
        with self._get_checkpoint_path(session_id).open("wb") as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved for session {session_id}")

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """Load checkpoint if available"""
        path = self._get_checkpoint_path(session_id)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        return None

    def _resume_checkpoint(
        self, session_id: str, task_ids: list[str], trials_per_task: int
    ) -> tuple[dict, int, int]:
        """Resume from checkpoint if available"""
        checkpoint = self._load_checkpoint(session_id)
        task_results = {}
        start_task_idx = 0
        start_trial = 0

        if checkpoint:
            logger.info(f"Resuming from checkpoint for session {session_id}")
            task_results = checkpoint["task_results"]
            current_task = checkpoint["current_task"]
            completed_trials = checkpoint["completed_trials"]
            if current_task in task_ids:
                start_task_idx = task_ids.index(current_task)
                start_trial = completed_trials
                if (
                    completed_trials >= trials_per_task
                    and start_task_idx < len(task_ids) - 1
                ):
                    start_task_idx += 1
                    start_trial = 0
        return task_results, start_task_idx, start_trial

    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool | None = False,
        session_id: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark on specified tasks or all available tasks with checkpointing

        Args:
            task_ids: list of task_ids to run, or None for all tasks
            trials_per_task: Number of trials per task, Default to k=1 to number of trials
            k_values: list of k values, for which pass metrics are calculated. Default to [1, 2, 3, ..., trials_per_task]
            verbose: Whether to save agent messages
            session_id: Unique ID for this benchmark session, used for checkpointing.
                If None, it will be an empty string, and the checkpoint will be saved as checkpoint_{agent_name}_.pkl
        """
        if task_ids is None:
            task_ids = self.interface.get_available_tasks()

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        if session_id is None:
            session_id = ""

        if k_values is None:
            k_values = list(range(1, trials_per_task + 1))
        elif isinstance(k_values, int):
            k_values = [k_values]
        elif isinstance(k_values, list) and max(k_values) > trials_per_task:
            raise ValueError("k value is greater than the number of trials")

        task_results, start_task_idx, start_trial = self._resume_checkpoint(
            session_id, task_ids, trials_per_task
        )
        logger.info(
            f"Running benchmark on tasks: {task_ids} with {trials_per_task} trials per task"
        )

        for _i, task_id in enumerate(task_ids[start_task_idx:], start=start_task_idx):
            logger.info(f"Running task {task_id}")
            if task_id not in task_results:
                task_trials = TaskTrialResults(task_id=task_id)
                task_results[task_id] = task_trials
            else:
                task_trials = task_results[task_id]

            for j in range(start_trial, trials_per_task):
                try:
                    answer, messages = self.agent.run_agent(self.interface, task_id)
                    result = self.interface.submit_answer(task_id, answer)
                    task_trials.trials.append(result)
                    if verbose:
                        save_agent_messages(
                            messages, task_id, self.agent.__class__.__name__
                        )
                    self._save_checkpoint(session_id, task_results, task_id, j + 1)
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
                    self._save_checkpoint(session_id, task_results, task_id, j)
                    raise
            start_trial = 0

        return BenchmarkResult(task_results=task_results, k=k_values)
