"""
Spectra Elucidation Benchmark Server

Command-line arguments:
    --host: Host address to run the server (default: value of CORRAL_HOST env var or '0.0.0.0').
    --port: Port to run the server (default: value of CORRAL_PORT env var or 8000).
    --subtask_level: Whether to use subtask-level tasks (default: False).
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger
from spectra_elucidation.score import (
    score_formula_match,
    score_isotopic_distribution,
    score_molecule,
    score_molecule_fragments,
    score_num_aromatic_carbons,
    score_num_carbon_symmetry_classes,
    score_num_carbonyl_groups,
    score_num_ch3_groups,
    score_num_hydrogen_symmetry_classes,
    validate_dbe_consistency,
)
from spectra_elucidation.tools import (
    create_tools,
)

from corral.base import Environment
from corral.server import run_server
from corral.task import TaskDefinition, TaskGroup

BASE_WORK_DIR = os.environ.get(
    "CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/spectra_elucidation"
)

SCORING_FUNCTIONS = {
    "1": score_formula_match,
    "2": validate_dbe_consistency,
    "3": score_isotopic_distribution,
    "4": score_num_hydrogen_symmetry_classes,
    "5": score_num_carbon_symmetry_classes,
    "6": score_num_aromatic_carbons,
    "7": score_num_ch3_groups,
    "8": score_num_carbonyl_groups,
    "9": score_molecule_fragments,
    "10": score_molecule,
    "score_molecule": score_molecule,
}


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
        for data in task_data:
            task_id = data["id"]
            initial_input = data.get("initial_input", {"work_dir": work_dir})
            input_from_tasks = data.get("input", {}).get("input_from_task", [])
            if not isinstance(input_from_tasks, list):
                input_from_tasks = []

            tasks[task_id] = TaskDefinition(
                name=data["name"],
                description=data["input"]["prompt"],
                tools=data.get("tools", []),
                scoring_fn=SCORING_FUNCTIONS[str(data["scoring_fn"])],
                scoring_inputs=data["output"][0]["target"],
                submission_format=data.get("submission_format", ""),
                input_from_tasks=input_from_tasks,
                initial_input=initial_input,
            )
    return tasks


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

        self.hidden_args = {"h_smiles": self.current_task.scoring_inputs}

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

        prompt += "\nAvailable input data:\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            dep_key = f"{self.task_group.group_id}_{dep_task_id}"
            if dep_key in self.task_group.results:
                dep_result = self.task_group.results[dep_key]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from {dep_task_id}: {dep_result['answer']}\n"
                else:
                    prompt += f"- Input from {dep_task_id}: {dep_result}\n"

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
            submission_str = self.state.submitted_answer.strip()
            logger.info(f"Raw submission: {submission_str}")
            score = self.current_task.scoring_fn(
                prediction=submission_str, ground_truth=self.current_task.scoring_inputs
            )
            self.task_group.store_result(
                self.task_id, {"answer": submission_str}, score
            )
            logger.info(f"Score for task {self.task_id}: {score}")
            return score

        except Exception as e:
            logger.error(f"Error scoring submission for task {self.task_id}: {e!s}")
            logger.error(f"Submission was: {self.state.submitted_answer}")
            return 0.0


def create_spectra_elu_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
) -> dict[str, Environment]:
    """Create environments for the spectra elucidation benchmark tasks."""
    logger.info("Creating environments for spectra elucidation tasks...")
    if subtask_level:
        json_path = Path(__file__).parent / "subtasks_json"
    else:
        json_path = Path(__file__).parent / "tasks_json"
    if not json_path.exists():
        raise ValueError(f"Task file {json_path} does not exist.")

    logger.info(f"Loading tasks from {json_path}")

    tasks = load_tasks_from_json(json_path, work_dir=work_dir)

    group_id = "spectra_elucidation"
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
        environments[task_id] = TaskEnvironment(
            task_id=task_id,
            task_group=task_group,
            work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectra Elucidation Benchmark Server")
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
        "--subtask_level",
        type=bool,
        default=False,
        help="Whether to use subtask level",
    )
    args = parser.parse_args()

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_spectra_elu_environments(
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
