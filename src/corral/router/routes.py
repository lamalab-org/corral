from typing import Any

import requests
from loguru import logger

from corral.report.results import TaskTrialResult
from corral.router.verbosity import ToolVerbosity
from corral.types import ToolResponse


class CorralRouter:
    """General interface for interacting with benchmark server"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_verbosity: str | None = ToolVerbosity.FULL.value,
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

    def submit_answer(self, task_id: str, answer: str) -> TaskTrialResult:
        """Submit final answer for a task"""
        logger.info(f"Agent submitting answer {answer} for task {task_id}")
        response = requests.post(
            f"{self.base_url}/tasks/{task_id}/submit", json={"answer": answer}
        )
        response.raise_for_status()
        data = response.json()
        return TaskTrialResult(
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
