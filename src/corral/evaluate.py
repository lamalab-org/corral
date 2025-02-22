from collections import defaultdict
from typing import Any, Protocol

import requests
from loguru import logger

from corral.report import BenchmarkResult, TaskResult, TaskTrials, ToolResponse


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
