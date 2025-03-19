import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger

from score import get_score

from corral.base import Environment, Tool

from corral.server import create_benchmark_server


class GroundTruth:
    def __init__(self, type: str, val: str, threshold: str):
        self.type = type
        self.val = val
        self.threshold = threshold

@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable
    submission_format: dict[str, str]
    # Either use output from another task or custom input
    ground_truth: GroundTruth
    input_from_tasks: list[str] = field(default_factory=list)
    initial_input: dict[str, Any] = field(default_factory=dict)

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

    def get_task_input(self, task_id: str) -> dict[str, Any]:
        """Get input for a task either from other tasks or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return {}

        # Start with the initial input
        combined_input = task.initial_input.copy() if task.initial_input else {}

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

        return all(dep_task_id in self.results for dep_task_id in task.input_from_tasks)


class TaskEnvironment(Environment):
    """Environment that works with a task group"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        available_tools: dict[str, Tool],
        # fs_tools: dict[str, Tool] | None = None,
    ):
        self.task_group = task_group
        self.task_id = task_id
        self.available_tools = available_tools
        # self.fs_tools = fs_tools or {}

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize tools and environment
        self.tools = {}
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools for the task
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

        # Add all file system tools
        # for tool in self.fs_tools.values():
        #     self.add_tool(tool)

    def get_task_prompt(self) -> str:
        _combined_input = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
"""
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

        prompt += "\nAvailable input data:\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            if dep_task_id in self.task_group.results:
                dep_result = self.task_group.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from {dep_task_id}: {dep_result['answer']}\n"
                else:
                    prompt += f"- Input from {dep_task_id}: {dep_result}\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                prompt += f"- {key}: {value}\n"

        prompt += "Whatever potentials you need to run the simulation, you can find them at /potentials/EAM/. Do not change your working directory, it has already been set. If the task is to give the final output as a scalar, only return the numerical value, without any units."

        # # Add IO tools description for saving results
        # prompt += "\nIMPORTANT: You have access to filesystem tools which allow you to read and write files. Also ypu can retry many times to get the correct answer. "
        # prompt += "Since some task results will be used in subsequent tasks, make sure to save your results using appropriate filenames. "
        # prompt += (
        #     "This will help you reference and retrieve these files in later tasks."
        # )

        # Add note about dependencies
        if self.current_task.input_from_tasks:
            status = []
            for dep_id in self.current_task.input_from_tasks:
                status_text = (
                    "available"
                    if dep_id in self.task_group.results
                    else "not yet available"
                )
                status.append(f"{dep_id} ({status_text})")

            prompt += f"\n\nThis task uses output from tasks: {', '.join(status)}"

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")
        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission - take only the numerical answer part
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")

            # Try to parse as JSON first
            try:
                submission = json.loads(submission_str)
            except json.JSONDecodeError:
                # If not valid JSON, try to create a simple answer dict
                submission = {"answer": submission_str}

            # Store result in task group
            logger.info(f"Parsed submission: {submission}")

            # Extract the answer field for scoring
            answer = submission.get("answer", submission)
            score = self.current_task.scoring_fn(self.current_task.ground_truth, answer)

            self.task_group.store_result(self.task_id, submission, score)

            return score
        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_catalysis_environments() ->dict[str, Environment]:
    """Create environments for catalysis tasks"""

    task_group = TaskGroup(
        group_id = "energy_minimisation", 
        tasks = {
            "initialisation" : TaskDefinition(
                name = "initialisation",
                description = "Initialise a face-centered cubic (FCC) lattice structure for Aluminum (Al) using LAMMPS with a lattice constant of 4.05 angstrom. Since this is just an initial setup, there is no need to perform energy minimisation. The simulation should use metal units, define a 5x5x5 simulation box and set periodic boundary conditions to True. Save this structure using write_data command in a file in xyz format. As the final output, give the absolute path of the structurel file.",
                tools = ["run_lammps", "run_bash_command"],
                scoring_fn = get_score,
                submission_format={"answer" : "/path/to/structure/file"},
                ground_truth = GroundTruth(type = "structure", val = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/ground_truth/energy_minimisation/initial_structure.xyz", threshold = "0")
            ),
            "simulation_settings" : TaskDefinition(
                name = "simulation_settings",
                description = "You are given the absolute path of the file of the initial lattice structure for Aluminum fcc lattice. Read that structure file and perform energy minimisation using Embedded Atom Method (EAM) with conjugate gradient method in lammps, with force tolerance of 1e-10, energy tolerance of 1e-10, maximum iterations of 1e+5 and maximum evaluations of 1e+5. After energy minimisation, save the final energy after minimisation in a text file, and as the final output give the absolute path of the text file.",
                tools = ["run_lammps", "run_bash_command"],
                scoring_fn = get_score,
                submission_format={"answer" : "/path/to/textfile"},
                ground_truth = GroundTruth(type = "textfile", val = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/tasks/lammps/lammps/ground_truth/energy_minimisation/total_energy.txt", threshold = "1e-1"),
                input_from_tasks=['initialisation']
            )
        }
    )
    # Print task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Print ordering of tasks
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("\nTask Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i+1}. {task_id}")

    # Create all available tools
    # This assumes you have a function to create the tools
    from tools import create_tools

    available_tools = create_tools()

    # Create environments for all tasks
    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            available_tools=available_tools,
            # fs_tools=fs_tools,
        )

    return environments


if __name__ == "__main__":
    # Create file system manager and tools

    # Create all environments with file system tools
    environments = create_catalysis_environments()

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
