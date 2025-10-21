"""
Retrosynthesis Benchmark Server

Command-line arguments:
    --host: Host address to run the server (default: value of CORRAL_HOST env var or '0.0.0.0').
    --port: Port to run the server (default: value of CORRAL_PORT env var or 8000).
    --level: Level of the environment (1, 2, or 3) (default: 1).
    --subtask_level: Whether to use subtask-level tasks (default: False).
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger
from retrosynthesis.score import check_reactants, score_final
from retrosynthesis.tools import create_tools

from corral.backend.env import Environment
from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "CORRAL_WORK_DIR/rethrosynthesis")

SCORING_FUNCTIONS = {"score_final": score_final, "check_reactants": check_reactants}


def load_tasks_from_json(
    json_path: Path, work_dir: str = BASE_WORK_DIR
) -> dict[str, TaskDefinition]:
    task_files = json_path.glob("*.json")
    tasks = {}
    for task_file in task_files:
        if not Path(task_file).exists():
            raise ValueError(f"Task file {task_file} is not a valid file.")

        with task_file.open() as f:
            task_data = json.load(f)
        task_id = task_data["id"]
        initial_input = task_data.get("initial_input", {"work_dir": work_dir})
        input_from_tasks = task_data.get("input", {}).get("input_from_task", [])
        if not isinstance(input_from_tasks, list):
            input_from_tasks = []

        tasks[task_id] = TaskDefinition(
            name=task_data["name"],
            description=task_data["input"]["prompt"],
            tools=task_data.get("tools", []),
            scoring_fn=SCORING_FUNCTIONS[str(task_data["scoring_fn"])],
            scoring_inputs=task_data["output"][0]["target"],
            submission_format=task_data.get("submission_format", ""),
            input_from_tasks=input_from_tasks,
            initial_input=initial_input,
        )
    return tasks


class RetroEnvironment(Environment):
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
        work_dir: str,
    ):
        self.task_id = task_id
        self.task_group = task_group
        self.available_tools = create_tools()
        self.work_dir = work_dir

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize environment
        super().__init__(f"{task_group.group_id}_{task_id}", base_work_dir=work_dir)

        logger.info(f"Initializing environment for task {self.task_id}")
        logger.info(f"Task name: {self.current_task}")
        self._add_task_tools()

    def _add_task_tools(self):
        """Add tools required for the current task to the environment"""
        for tool_name in self.current_task.tools:
            if tool_name in self.available_tools:
                self.add_tool(self.available_tools[tool_name])
            else:
                logger.warning(
                    f"Tool {tool_name} not found in available tools for task {self.task_id}"
                )
        if "subtask" not in self.current_task.name:
            # Add file system tools if not already included
            for tool in self.available_tools.values():
                self.add_tool(self.available_tools[tool.name])

    def get_task_prompt(self) -> str:
        prompt = (
            f"Task {self.current_task.name}:\n"
            f"{self.current_task.description}\n\n"
            "Required submission format:\n"
            f"{self.current_task.submission_format}\n\n"
        )

        if self.current_task.input_from_tasks:
            prompt += "\nAvailable input data:\n"

            # Display input data from dependencies
            for dep_task_id in self.current_task.input_from_tasks:
                dep_key = f"{self.task_group.group_id}_{dep_task_id}"
                if dep_key in self.task_group.results:
                    dep_result = self.task_group.results[dep_key]
                    task_prompt = self.task_group.tasks[dep_task_id].description
                    if isinstance(dep_result, dict) and "answer" in dep_result:
                        prompt += f"- Input from '{dep_task_id}' with question: '{task_prompt}' and answer: '{dep_result['answer']}'\n"
                    else:
                        prompt += f"- Input from '{dep_task_id}' with description: '{task_prompt}' and answer: '{dep_result}'\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        logger.info(f"Task prompt for {self.task_id}:\n{prompt}")
        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            return 0.0

        try:
            # Clean the submission
            submitted_answer = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submitted_answer}")
            score = self.current_task.scoring_fn(
                prediction=submitted_answer, target=self.current_task.scoring_inputs
            )
            self.task_group.store_result(
                self.task_id, {"answer": submitted_answer}, score
            )
            logger.info(f"Score for task {self.task_id}: {score}")
            return score

        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_rethrosynthesis_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
    level: int = 1,
) -> dict[str, Environment]:
    """Create environments for the rethrosynthesis benchmark tasks."""
    logger.info("Creating environments for rethrosynthesis tasks...")
    if subtask_level:
        json_path = (
            Path(__file__).parent.parent
            / "environments"
            / f"level_{level}"
            / "subtasks"
        )
    else:
        json_path = (
            Path(__file__).parent.parent / "environments" / f"level_{level}" / "tasks"
        )
    if not json_path.exists():
        raise ValueError(f"Task file {json_path} does not exist.")

    logger.info(f"Loading tasks from {json_path}")

    tasks = load_tasks_from_json(json_path, work_dir=work_dir)

    group_id = "rethrosynthesis" if not subtask_level else "rethrosynthesis_subtasks"
    logger.info(f"Creating task group {group_id} with {len(tasks)} tasks")
    task_group = TaskGroup(
        group_id=group_id,
        tasks=tasks,
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

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = RetroEnvironment(
            task_id=task_id,
            task_group=task_group,
            work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rethrosynthesis Benchmark Server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("CORRAL_HOST", "0.0.0.0"),
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CORRAL_PORT", "8000")),
        help="Port to run the server on",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Level of the environment (1,2, or 3)",
    )

    parser.add_argument(
        "--subtask_level",
        type=bool,
        default=False,
        help="Whether to use subtask level",
    )
    args = parser.parse_args()

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_rethrosynthesis_environments(
        work_dir=BASE_WORK_DIR, subtask_level=args.subtask_level
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    run_server(
        environments=environments,
        host=args.host,
        port=args.port,
    )
