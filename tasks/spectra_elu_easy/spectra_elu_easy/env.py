import json
import os
from pathlib import Path

import uvicorn
from loguru import logger
from score import (
    score_molecule_similarity,
)
from tools import (
    create_tools,
)

from corral.base import Environment, Tool
from corral.server import create_benchmark_server
from corral.task import TaskDefinition, TaskGroup

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


class TaskEnvironment(Environment):
    """Environment that works with a task group

    Args:
        task_id (str): ID of the task to work on
        task_group (TaskGroup): Task group containing all the subtasks
        available_tools (dict[str, Tool]): All tools available in the environment (including file system tools)

    Raises:
        ValueError: If task ID is not found in the task group
    """

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        available_tools: dict[str, Tool],
    ):
        self.task_id = task_id
        self.task_group = task_group
        self.available_tools = available_tools

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize environment
        super().__init__(f"{task_group.group_id}_{task_id}")

        # Add required tools for the task
        for tool_name in self.current_task.tools:
            if tool_name in available_tools:
                self.add_tool(available_tools[tool_name])

    def get_task_prompt(self) -> str:
        prompt = (
            f"Task: {self.current_task.name}\n"
            f"Description\nThe characterization of the compound is: {self.current_task.description}\n\n"
            "Required submission format:\n"
        )
        for key, desc in self.current_task.submission_format.items():
            prompt += f"- {key}: {desc}\n"

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
            try:
                answer = submission.get("answer", submission)
            except Exception:
                answer = submission_str

            # Pass additional scoring inputs as keyword arguments
            score = self.current_task.scoring_fn(
                answer, **self.current_task.scoring_inputs
            )

            self.task_group.store_result(self.task_id, submission, score)

            return score

        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_spectra_elu_environments(
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
    """Create environments for the spectra elucidation benchmark tasks."""
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    tasks_path = Path(__file__).parent / "tasks_mcq.json"

    with tasks_path.open() as f:
        tasks = json.load(f)

    # Create all available tools
    available_tools = create_tools()

    # Create task definitions for each task in TASKS
    task_definitions = {}
    for task in tasks:
        task_id = task["name"]
        task_definitions[task_id] = TaskDefinition(
            name="Organic Compound Elucidation",
            description=f"Return the SMILES string for the compound with the following characterization data: {task['spectra']}. Return only the SMILES without any additional text.",
            tools=list(available_tools.keys()),  # Use all available tools
            scoring_fn=score_molecule_similarity,
            scoring_inputs={"ground_truth": task["smiles"]},
            submission_format={"answer": "smiles_or_name_of_the_compound"},
            initial_input={},
        )

    # Create task group with all tasks
    task_group = TaskGroup(
        group_id="spectro_elu",
        tasks=task_definitions,
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

    # Create environments for all tasks
    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            available_tools=available_tools,
        )

    return environments


if __name__ == "__main__":
    # Create all environments with file system tools
    environments = create_spectra_elu_environments()

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
