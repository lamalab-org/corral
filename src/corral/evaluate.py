import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Optional
from loguru import logger
import requests

from corral.report import (
    BenchmarkResult,
    TaskTrailResult,
    TaskTrialResults,
    ToolResponse,
)
import os
import modal


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

    def solve_task(self, interface: BenchmarkInterface, task_id: str, directory_path : str) -> str:
        """Solve a task and return the answer"""
        ...

class MatAgentBenchmark:
    """Runs benchmarks using an agent implementation"""

    def __init__(self, interface: BenchmarkInterface, agent: Agent, results_dir: str = "./results", local_results_dir: str = "./results/", app = "simagent", bash_command = "run_bash_command", dir_command = "change_directory"):
        self.interface = interface
        self.agent = agent
        self.app = app
        self.bash_command = bash_command
        self.dir_command = dir_command

        # Convert to absolute path (relative to the script's location)
        self.results_dir = results_dir
        self.local_results_dir = local_results_dir

        logger.info(f"Results will be saved in: {self.results_dir}")

    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
    ) -> BenchmarkResult:
        """Run benchmark on specified tasks or all available tasks

        Args:
            task_ids: list of task_ids to run, or None for all tasks
            trials_per_task: Number of trials per task, Default to k=1 to number of trials

        """

        if task_ids is None:
            task_ids = self.interface.get_available_tasks()

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        # Validate and set k_values
        if k_values is None:
            k_values = list(range(1, trials_per_task + 1))
        elif isinstance(k_values, int):
            k_values = [k_values]
        elif isinstance(k_values, list) and max(k_values) > trials_per_task:
            raise ValueError("k value is greater than the number of trials")

        logger.info(f"Running benchmark on tasks: {task_ids} with {trials_per_task} trials per task")

        task_results: dict[str, TaskTrialResults] = {}

        # Step 1: Pre-create all required directories before benchmarking
        run_bash_command = modal.Function.lookup(self.app, self.bash_command)
        run_directories = {}  # Store the directories for each task_id
        local_run_directories = {}
        for task_id in task_ids:
            run_directories[task_id] = []
            local_run_directories[task_id] = []
            for run_number in range(1, trials_per_task + 1):
                run_directory = os.path.join(self.results_dir, str(task_id), str(run_number))
                # os.makedirs(run_directory, exist_ok=True)  # Create directory
                run_bash_command.remote("mkdir", ["-p", run_directory])
                run_directories[task_id].append(run_directory)
                local_run_directory = os.path.join(self.local_results_dir, str(task_id), str(run_number))
                os.makedirs(local_run_directory, exist_ok=True)
                local_run_directories[task_id].append(local_run_directory)

        # Step 2: Run benchmark using pre-created directories
        for task_id in task_ids:
            logger.info(f"Running task {task_id}")
            task_trials = TaskTrialResults(task_id=task_id)
            for run_directory, local_directory in zip(run_directories[task_id], local_run_directories[task_id]):
            # for run_index, run_directory in enumerate(run_directories[task_id], start=1):
                # Solve task
                answer = self.agent.solve_task(self.interface, task_id, run_directory, local_directory)
                result = self.interface.submit_answer(task_id, answer)
                task_trials.trials.append(result)
                # # Save answer in the pre-created directory
                # answer_file = os.path.join(run_directory, "answer.txt")
                # with open(answer_file, "w") as f:
                #     f.write(answer)

                # logger.info(f"Saved answer for {task_id} (run_{run_index}) in {answer_file}")

                # Submit and store result
                result = self.interface.submit_answer(task_id, answer)
                task_trials.trials.append(result)

            # Store all trials for this task
            task_results[task_id] = task_trials

        return BenchmarkResult(task_results=task_results, k=k_values)
