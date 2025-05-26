import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable[[dict | str], float]
    submission_format: dict[str, str]
    # Either use output from another task or custom input
    input_from_tasks: list[str] = field(
        default_factory=list
    )  # input required from some of the previous tasks in the group
    initial_input: dict[str, Any] = field(
        default_factory=dict
    )  # initial input for the task, if required

    # Helper method to check if task has dependencies
    def has_dependencies(self) -> bool:
        return len(self.input_from_tasks) > 0


@dataclass
class TaskGroup:
    """Container for related tasks"""

    group_id: str
    tasks: dict[str, TaskDefinition]
    results: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    chained_tasks: bool = field(default=False)  # Whether tasks can depend on each other

    def get_task_input(self, task_id: str) -> dict[str, Any]:
        """Get input for a task either from other tasks or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return {}

        # Start with the initial input
        combined_input = deepcopy(task.initial_input) if task.initial_input else {}

        # Add inputs from dependent tasks
        for dep_task_id in task.input_from_tasks:
            if dep_task_id in self.results:
                # Add the dependent task's result to the input
                dep_result = self.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    # Extract the relevant part of the result
                    combined_input[f"result_from_{dep_task_id}"] = dep_result["answer"]
                else:
                    combined_input[f"result_from_{dep_task_id}"] = dep_result

        return combined_input

    def store_result(self, task_id: str, result: dict[str, Any], score: float) -> None:
        """Store task result and score"""
        self.results[task_id] = result
        self.scores[task_id] = score

    def get_task_dependencies(self) -> dict[str, list[str]]:
        """Get dictionary of task dependencies"""
        return {task_id: task.input_from_tasks for task_id, task in self.tasks.items()}

    def get_ordered_tasks(self) -> list[str]:
        """Return tasks in dependency order"""
        # Simple topological sort
        dependencies = self.get_task_dependencies()
        visited = set()
        ordered = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in dependencies.get(task_id, []):
                visit(dep)
            ordered.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        return ordered

    def check_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        return all(
            dep_task_id in self.results and self.results[dep_task_id] is not None
            for dep_task_id in task.input_from_tasks
        )


class WorkspaceStrategy(Enum):
    """Different workspace isolation strategies"""

    NONE = "none"
    SHARED = "shared"
    TASK_LEVEL = "task_level"
    TRIAL_LEVEL = "trial_level"
    CHAIN_AWARE = "chain_aware"

    @classmethod
    def from_env(cls, default=None) -> "WorkspaceStrategy":
        """Get workspace strategy from environment variable"""
        env_value = os.environ.get("CORRAL_WORKSPACE_STRATEGY", "").lower()

        strategy_map = {
            "none": cls.NONE,
            "shared": cls.SHARED,
            "task": cls.TASK_LEVEL,
            "task_level": cls.TASK_LEVEL,
            "trial": cls.TRIAL_LEVEL,
            "trial_level": cls.TRIAL_LEVEL,
            "chain_aware": cls.CHAIN_AWARE,
            "auto": cls.CHAIN_AWARE,
        }

        if default is None:
            default = cls.CHAIN_AWARE
        return strategy_map.get(env_value, default)
