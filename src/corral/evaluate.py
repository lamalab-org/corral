from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Protocol

import requests
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


class BenchmarkInterface:
    """General interface for interacting with benchmark server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_available_tasks(self) -> list[str]:
        """Get list of available task IDs"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def get_task_guide(self, task_id: str) -> str:
        """Get complete guide for task including tools"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/guide")
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

    def submit_answer(self, task_id: str, answer: str) -> TaskResult:
        """Submit final answer for a task"""
        logger.info(f"Agent submitting answer {answer} for task {task_id}")
        response = requests.post(
            f"{self.base_url}/tasks/{task_id}/submit", json={"answer": answer}
        )
        response.raise_for_status()
        data = response.json()
        return TaskResult(
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

    def __init__(self, interface: BenchmarkInterface, agent: Agent, k: int = 5):
        self.interface = interface
        self.agent = agent
        self.k = k

    def bench(
        self, task_ids: list[str] | None = None, trials_per_task: int | None = None
    ) -> BenchmarkResult:
        """Run benchmark on specified tasks or all available tasks

        Args:
            task_ids: list of task IDs to run, or None for all tasks
            trials_per_task: Number of trials per task, defaults to k if not specified

        """
        if task_ids is None:
            task_ids = self.interface.get_available_tasks()

        if trials_per_task is None:
            trials_per_task = self.k

        logger.info(
            f"Running benchmark on tasks: {task_ids} with {trials_per_task} trials per task"
        )

        results: dict[str, TaskTrials] = defaultdict(TaskTrials)

        for task_id in task_ids:
            for _ in range(trials_per_task):
                # Get answer from agent
                answer = self.agent.solve_task(self.interface, task_id)

                # Submit and store result
                result = self.interface.submit_answer(task_id, answer)
                results[task_id].trials.append(result)

        return BenchmarkResult(
            task_results=dict(results), k=self.k, total_tasks=len(task_ids)
        )
