import os
import sys
from collections.abc import Callable
from pathlib import Path

from catalyst.score import (
    BASE_WORK_DIR,
    check_adsorption_sites,
    check_adsorption_structure,
    check_mp_structure,
    check_slab_structure,
    check_slabs_json,
    check_valid_json_file,
)
from catalyst.tools import create_tools
from loguru import logger

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.task_group import TaskGroupEnvironment
from corral.utils.task_loader import (
    load_task_entries,
    load_task_entries_from_env_package,
)

# Registry of scoring functions
SCORING_FUNCTIONS = {
    "mp_structure": check_mp_structure,
    "slabs_json": check_slabs_json,
    "slab_structure": check_slab_structure,
    "adsorption_sites": check_adsorption_sites,
    "adsorption_structure": check_adsorption_structure,
    "file_exists": check_valid_json_file,
}

logger.info(f"Using BASE_WORK_DIR: {BASE_WORK_DIR}")


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

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


def entries_to_task_definitions(
    entries: list[dict], work_dir: str
) -> dict[str, TaskDefinition]:
    """Convert standardized task entries to TaskDefinition objects.

    Args:
        entries: List of task entry dicts from task_loader.
        work_dir: Working directory for task execution.

    Returns:
        Dictionary of TaskDefinition objects keyed by task ID.
    """
    tasks = {}
    for entry in entries:
        task_id = entry["id"]
        scoring_fn_name = entry.get("scoring_function", "default")
        scoring_params = entry.get("scoring_params", {})
        scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)

        initial_input = entry.get("initial_input", {}).copy()
        if "work_dir" not in initial_input:
            initial_input["work_dir"] = work_dir

        tasks[task_id] = TaskDefinition(
            name=entry["name"],
            description=entry["description"],
            tools=entry.get("tools", []),
            scoring_fn=scoring_fn,
            submission_format=entry.get("submission_format", ""),
            input_from_tasks=entry.get("input_from_tasks", []),
            initial_input=initial_input,
        )

    return tasks


def create_environments(
    *,
    local_dir: str | Path | None = None,
    environment: str | None = None,
    level: int = 1,
    task_type: str = "task",
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str = BASE_WORK_DIR,
    group_id: str = "catalyst",
) -> dict[str, TaskGroupEnvironment]:
    """Create environments from HuggingFace or local task configs.

    Args:
        local_dir: Path to local JSON directory (mutually exclusive with environment).
        environment: HF environment name (mutually exclusive with local_dir).
        level: Level number (for HF loading or env-package resolution).
        task_type: "task" or "subtask".
        taskgroup_common_tools: Tools common to all subtasks.
        work_dir: Working directory for task execution.
        group_id: Task group identifier.

    Returns:
        Dictionary of environments keyed by task ID.
    """
    if local_dir is not None:
        entries = load_task_entries(local_dir=local_dir)
    elif environment is not None:
        entries = load_task_entries(
            environment=environment, level=level, task_type=task_type
        )
    else:
        # Default: load from standard directory layout relative to this package
        entries = load_task_entries_from_env_package(
            Path(__file__).parent, level=level, subtask=(task_type == "subtask")
        )

    tasks = entries_to_task_definitions(entries, work_dir)

    logger.info(f"Creating task group '{group_id}' with {len(tasks)} tasks")
    task_group = TaskGroup(group_id=group_id, tasks=tasks)

    logger.info("Task Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("Task Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i + 1}. {task_id}")

    subtask_specific_tools = create_tools()

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

    # Determine tasks file path (optional: for backwards compat with direct JSON path)
    local_dir = None
    if len(sys.argv) > 1:
        local_dir = sys.argv[1]

    # Determine port number
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            logger.error(
                f"Error: Invalid port number provided: {sys.argv[2]}. Using default port."
            )
            port = int(os.environ.get("CORRAL_PORT", "8000"))
    else:
        port = int(os.environ.get("CORRAL_PORT", "8000"))

    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    environments = create_environments(
        local_dir=local_dir,
        work_dir=work_dir,
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
