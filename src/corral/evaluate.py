from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import requests
from loguru import logger

from corral.base import Tool


@dataclass
class TaskResult:
    """Result of a task submission"""

    score: float
    state: Dict[str, Any]
    tool_statistics: Dict[str, Any]


@dataclass
class ToolResponse:
    """Response from a tool execution"""

    success: bool
    result: str | None
    error: str | None


class BenchmarkInterface:
    """General interface for interacting with benchmark server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def get_task_guide(self, task_id: str) -> str:
        """Get complete guide for task including tools"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/guide")
        response.raise_for_status()
        return response.json()["prompt"]

    def add_tool_to_environment(self, task_id: str, tool: Tool) -> Dict[str, Any]:
        """Add a new tool to an environment

        Args:
            task_id: ID of the task/environment
            name: Name of the tool
            description: Description of the tool
            arguments: List of argument dictionaries with keys: name, type, description, required, default, choices
            execute_code: Python code as string that will be executed when the tool is called

        Returns:
            Dictionary with status information
        """
        logger.info(f"Adding tool {tool.name} to environment {task_id}")

        tool_data = {
            "name": name,
            "description": description,
            "arguments": arguments,
            "execute_code": execute_code,
        }

        try:
            response = requests.post(
                f"{self.base_url}/tasks/{task_id}/tools/add", json=tool_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add tool: {e!s}")
            raise

    def execute_tool(
        self, task_id: str, tool_name: str, arguments: Dict[str, Any]
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

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}/status")
        response.raise_for_status()
        return response.json()


class Agent(Protocol):
    """Protocol defining what an agent must implement"""

    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
        """Solve a task and return the answer"""
        ...


@dataclass
class BenchmarkResult:
    """Results from running benchmark"""

    # TODO: think about the report
    task_results: Dict[str, TaskResult]
    average_score: float
    total_tasks: int
    successful_tasks: int


class MatAgentBenchmark:
    """Runs benchmarks using an agent implementation"""

    def __init__(self, interface: BenchmarkInterface, agent: Agent):
        self.interface = interface
        self.agent = agent

    def bench(self, task_ids: List[str] | None = None) -> BenchmarkResult:
        """Run benchmark on specified tasks or all available tasks"""
        if task_ids is None:
            task_ids = self.interface.get_available_tasks()
        logger.info(f"Running benchmark on tasks: {task_ids}")

        results = {}
        for task_id in task_ids:
            # Get answer from agent
            answer = self.agent.solve_task(self.interface, task_id)

            # Submit and store result
            result = self.interface.submit_answer(task_id, answer)
            results[task_id] = result

        # Calculate statistics
        scores = [r.score for r in results.values()]
        successful = len([s for s in scores if s > 0])

        return BenchmarkResult(
            task_results=results,
            average_score=sum(scores) / len(scores),
            total_tasks=len(scores),
            successful_tasks=successful,
        )
