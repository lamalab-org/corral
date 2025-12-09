import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from resistor_network.score import (
    check_complete_circuit_solution,
    check_resistance_measurements,
    check_resistor_topology,
    check_resistor_values_only,
    check_valid_circuit_json,
)
from resistor_network.tools import create_tools

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.task_group import TaskGroupEnvironment

# Base working directory
if "CORRAL_WORK_DIR" not in os.environ:
    BASE_WORK_DIR = tempfile.mkdtemp(prefix="resistor_network_")
    logger.info(f"CORRAL_WORK_DIR not set, using temporary directory: {BASE_WORK_DIR}")
else:
    BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


# Registry of scoring functions
SCORING_FUNCTIONS = {
    # Resistor network scoring functions
    "resistor_topology": check_resistor_topology,
    "resistance_measurements": check_resistance_measurements,
    "complete_circuit_solution": check_complete_circuit_solution,
    "resistor_values_only": check_resistor_values_only,
    "valid_circuit_json": check_valid_circuit_json,
}


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    # If it's a factory function (i.e., takes arguments), call with params
    if params:
        try:
            logger.info(f"Initializing scoring function '{name}' with params: {params}")
            return fn(**params)
        except Exception as e:
            raise ValueError(
                f"Error initializing scoring function '{name}' with params {params}: {e}"
            ) from e
    else:
        logger.info(f"Using scoring function '{name}' without params")
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
        logger.info(f"{i + 1}. {task_id}")

    # Create all available tools
    subtask_specific_tools = create_tools()

    # Create environments for all tasks

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    # --- Argument Parsing ---

    # Determine tasks file path
    if len(sys.argv) > 1:
        # First argument (sys.argv[1]) is the tasks file path
        tasks_json_path = sys.argv[1]
    else:
        # Default: Try environment variable, then hardcoded path
        tasks_json_path = os.environ.get(
            "CORRAL_TASKS_PATH",
            Path(__file__).parent / "tasks" / "catalysis_tasks.json",
        )

    # Determine port number
    if len(sys.argv) > 2:
        # Second argument (sys.argv[2]) is the port number
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(
                f"Error: Invalid port number provided: {sys.argv[2]}. Using default port."
            )
            port = int(os.environ.get("CORRAL_PORT", "8000"))
    else:
        # Default: Try environment variable, then default 8000
        port = int(os.environ.get("CORRAL_PORT", "8000"))

    # Get server settings from environment if provided (Host and Work Dir remain env/default)
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # --- Environment Creation ---

    # taskgroup_common_tools = {
    #     "execute_python_code": execute_python_code,
    #     "execute_python_script": execute_python_script,
    # }
    taskgroup_common_tools = None
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
        taskgroup_common_tools=taskgroup_common_tools,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # --- Run Server ---
    logger.info(f"Running server on {host}:{port}")
    run_server(environments, host, port)
