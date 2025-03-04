import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from loguru import logger
from tools import create_tools

from corral.base import Environment, Tool
from corral.server import create_benchmark_server


@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable[[dict], float]
    submission_format: dict[str, str]
    # Either use output from another task or custom input
    input_from_task: str | None = None
    initial_input: dict[str, Any] | None = None


@dataclass
class TaskGroup:
    """Container for related tasks"""

    group_id: str
    tasks: dict[str, TaskDefinition]
    results: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def get_task_input(self, task_id: str) -> dict[str, Any] | None:
        """Get input for a task either from another task or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        if task.input_from_task and task.input_from_task in self.results:
            return {"result": self.results[task.input_from_task]}
        return task.initial_input

    def store_result(self, task_id: str, result: dict[str, Any], score: float) -> None:
        """Store task result and score"""
        self.results[task_id] = result
        self.scores[task_id] = score

    def get_task_dependencies(self) -> dict[str, list[str]]:
        """Get dictionary of task dependencies"""
        dependencies = {}
        for task_id, task in self.tasks.items():
            deps = []
            if task.input_from_task:
                deps.append(task.input_from_task)
            dependencies[task_id] = deps
        return dependencies


class TaskEnvironment(Environment):
    """Environment that works with a task group"""

    def __init__(
        self, task_id: str, task_group: TaskGroup, available_tools: dict[str, Tool]
    ):
        self.task_group = task_group
        self.task_id = task_id  # Individual task ID within the group
        self.available_tools = available_tools

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize tools and environment
        self.tools = {}
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

    def get_task_prompt(self) -> str:
        _input_data = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
"""
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

        prompt += "\nAvailable input data:\n"

        if (
            self.current_task.input_from_task
            and self.current_task.input_from_task in self.task_group.results
        ):
            # If this task depends on a previous task, show its result
            previous_result = self.task_group.results[self.current_task.input_from_task]
            if isinstance(previous_result, dict) and "answer" in previous_result:
                prompt += f"Previous task result: {previous_result['answer']}\n"
            else:
                prompt += f"Previous task result: {previous_result}\n"

        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                prompt += f"- {key}: {value}\n"

        if self.current_task.input_from_task:
            status = (
                "available"
                if self.current_task.input_from_task in self.task_group.results
                else "not yet available"
            )
            prompt += f"\nThis task uses output from task: {self.current_task.input_from_task} ({status})"

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")

        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission - take only the numerical answer part
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")  # Debug logger.info

            # Try to parse as JSON first
            try:
                submission = json.loads(submission_str)
            except json.JSONDecodeError:
                # If not valid JSON, try to create a simple answer dict
                submission = {"answer": submission_str}

            # Store result in task group
            logger.info(f"Parsed submission: {submission}")  # Debug logger.info
            score = self.current_task.scoring_fn(submission["answer"])

            self.task_group.store_result(self.task_id, submission, score)

            return score
        except Exception as e:
            logger.info(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.info(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_catalysis_environments() -> dict[str, Environment]:
    """Create environments for catalysis tasks"""

    def score_addition(result: dict) -> float:
        score = 0.0
        if "answer" in result:
            try:
                _answer = float(result["answer"])
                score = 1.0
            except ValueError:
                pass
        return score

    def check_structure(path) -> float:
        """Check if the is a valid path to a cif file"""

        from pymatgen.core import Structure

        strucutre = Structure.from_file(path)

        # if strecture is not None
        if strucutre:
            return 1.0
        return 0

    # Create task group
    task_group = TaskGroup(
        group_id="catalyst",
        tasks={
            "task_1": TaskDefinition(
                name="Retrieve structure",
                description="Retrieve structure of Si from Materials Project and save it as a cif file and submit the path to the cif file",
                tools=["get_structure_from_mp"],
                scoring_fn=check_structure,
                submission_format={"answer": "/path/to/ciffile"},
                initial_input={
                    "mp_id": "mp-149",
                    "path_to_write_dir": "/Users/n0w0f/git/n0w0f/mat-agent-bench/tasks/catalyst/temp",
                },
            ),
            "task_2": TaskDefinition(
                name="Create slab",
                description="Create a slab from the structure of Si and save it as a cif file and submit the path to the cif file",
                tools=[
                    # "create_pymatgen_structure_from_cif",
                    "create_slab_from_structure",
                ],
                scoring_fn=check_structure,
                submission_format={"answer": "/path/to/ciffile"},
                input_from_task="task_1",
            ),
            "task_3": TaskDefinition(
                name="Add CO2",
                description="Add CO2 to the slab and save it as a cif file and submit the path to the cif file",
                tools=["add_molecule_to_slab"],
                scoring_fn=check_structure,
                submission_format={"answer": "/path/to/ciffile"},
                input_from_task="task_2",
            ),
        },
        # # retrive strucutre of co2 molecule
        # "task_3": TaskDefinition(
        #     name="Retrieve structure",
        #     description="Retrieve structure of Si molecule from Materials Project and save it as a pickle file.",
        #     tools=["get_structure_from_mp"],
        #     scoring_fn=check_structure,
        #     submission_format={"answer": "/path/to/picklefile"},
        #     initial_input={"mp_id": "mp-2204849"},
        # ),
        #     "task_4": TaskDefinition(
        #         name="Create slab",
        #         description=f"""Create a slab from Si slab with CO2 molecule adsorbed on it.
        #         Find the adsorption sites possible and create a list of possible configurations save the list of strucutre as pickle file""",
        #         tools=["pymatgen_slab_creator"],
        #         scoring_fn=check_structure,
        #         submission_format={"answer": "/path/to/picklefile"},
        #         input_from_task="task_2"
        #     ),
        # },
    )

    # logger.info task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Create environments for all tasks
    available_tools = create_tools()
    environments = {}

    for task_id in task_group.tasks:
        # All environments share the same task group instance
        # task_id = f"{task_group.group_id}_{_id}"
        environments[task_id] = TaskEnvironment(
            task_id=task_id, task_group=task_group, available_tools=available_tools
        )

    return environments


if __name__ == "__main__":
    # Create all environments
    environments = create_catalysis_environments()

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_task:
            logger.info(f"  Depends on: {env.current_task.input_from_task}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
