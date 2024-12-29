import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
from loguru import logger

import requests


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
            f"{self.base_url}/tasks/{task_id}/submit",
            json={"answer": answer}
        )
        response.raise_for_status()
        data = response.json()
        return TaskResult(
            score=data["score"],
            state=data["state"],
            tool_statistics=data["state"]["tool_statistics"]
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
