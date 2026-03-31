import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from tools import calculator, number_converter

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.task_group import TaskGroupEnvironment

# Base working directory
if "CORRAL_WORK_DIR" not in os.environ:
    raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


def addition_score(expected_answer: float | None = None):
    """Factory function that returns a scoring function for addition tasks"""

    def score_fn(result: str) -> float:
        """Score an addition task with expected answer validation"""
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                # Try to parse as JSON first
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "answer" in parsed_result:
                        answer = float(parsed_result["answer"])
                    else:
                        # If not a dict with answer, treat the whole thing as the answer
                        answer = float(result)
                except json.JSONDecodeError:
                    # If not JSON, treat as direct numerical answer
                    answer = float(result)
            else:
                # If already parsed
                if isinstance(result, dict) and "answer" in result:
                    answer = float(result["answer"])
                else:
                    answer = float(result)

            if expected_answer is not None:
                # Check if answer matches expected value (within tolerance)
                return 1.0 if abs(answer - expected_answer) < 0.001 else 0.0
            else:
                # Just check if it's a valid number
                return 1.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for addition_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


def multiplication_score(expected_answer: float | None = None):
    """Factory function that returns a scoring function for multiplication tasks"""

    def score_fn(result: str) -> float:
        """Score a multiplication task with expected answer validation"""
        try:
            # Handle string submissions
            if isinstance(result, str):
                result = result.strip()
                # Try to parse as JSON first
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "answer" in parsed_result:
                        answer = float(parsed_result["answer"])
                    else:
                        # If not a dict with answer, treat the whole thing as the answer
                        answer = float(result)
                except json.JSONDecodeError:
                    # If not JSON, treat as direct numerical answer
                    answer = float(result)
            else:
                # If already parsed
                if isinstance(result, dict) and "answer" in result:
                    answer = float(result["answer"])
                else:
                    answer = float(result)

            if expected_answer is not None:
                # Check if answer matches expected value (within tolerance)
                return 1.0 if abs(answer - expected_answer) < 0.001 else 0.0
            else:
                # Just check if it's a valid number
                return 1.0

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error parsing result for multiplication_score: {e}, result was: {result}"
            )
            return 0.0

    return score_fn


# Registry of scoring functions
SCORING_FUNCTIONS = {
    "addition_score": addition_score,
    "multiplication_score": multiplication_score,
}


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    # If it's a factory function (i.e., takes arguments), call with params
    if params:
        try:
            return fn(**params)
        except Exception as e:
            raise ValueError(
                f"Error initializing scoring function '{name}' with params {params}: {e}"
            ) from e
    else:
        return fn


def load_tasks_from_json(
    json_path: str | Path, work_dir: str
) -> dict[str, TaskDefinition]:
    """Load task definitions from a JSON file.

    Args:
        json_path: Path to the JSON file containing task definitions
        work_dir: Working directory to use for task execution

    Returns:
        dictionary of task definitions keyed by task ID
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Task definition file not found: {json_path}")

    with Path(json_path).open() as f:
        task_data = json.load(f)

    tasks = {}
    for task_id, task_info in task_data.items():
        # Get the scoring function by name from the registry
        scoring_fn_name = task_info.get("scoring_function", "default")
        scoring_params = task_info.get("scoring_params", {})
        scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)

        # Add work_dir to initial input if not already present
        initial_input = task_info.get("initial_input", {}).copy()
        if "work_dir" not in initial_input:
            initial_input["work_dir"] = work_dir

        tasks[task_id] = TaskDefinition(
            name=task_info["name"],
            description=task_info["description"],
            tools=task_info.get("tools", []),
            scoring_fn=scoring_fn,
            submission_format=task_info.get("submission_format", ""),
            input_from_tasks=task_info.get("input_from_tasks", []),
            initial_input=initial_input,
        )

    return tasks


def create_environments(
    task_json_path: str | Path,
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, TaskGroupEnvironment]:
    """Create environments for tasks defined in a JSON file

    Args:
        task_json_path: Path to the JSON file with task definitions
        taskgroup_common_tools: dictionary of Tools which are common for subtasks, for example file system tools
        work_dir: Working directory for task execution

    Returns:
        dictionary of environments keyed by task ID
    """

    logger.info(f"Creating environments from {task_json_path} with work_dir {work_dir}")

    # Load tasks from JSON
    tasks = load_tasks_from_json(task_json_path, work_dir)

    # Create task group
    group_id = Path(task_json_path).stem  # Use filename (without extension) as group ID
    logger.info(f"Creating task group with ID: {group_id}")
    task_group = TaskGroup(group_id=group_id, tasks=tasks)

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
        environments[task_id] = TaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools={
                "calculator": calculator,
                "number_converter": number_converter,
            },
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    # Determine tasks file path
    if len(sys.argv) > 1:
        tasks_json_path = sys.argv[1]
    else:
        tasks_json_path = os.environ.get(
            "CORRAL_TASKS_PATH",
            Path(__file__).parent / "tasks" / "catalysis_tasks.json",
        )

    # Get server settings from environment if provided
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    # Create environments
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Run server
    run_server(environments, host, port)
