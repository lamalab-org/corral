import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from resistor_network.score import (
    BASE_WORK_DIR,
    check_complete_circuit_solution,
    check_resistance_measurements,
    check_resistor_topology,
    check_resistor_values_only,
    check_valid_circuit_json,
)
from resistor_network.tools import create_tools

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import Tool

logger.info(f"Using BASE_WORK_DIR: {BASE_WORK_DIR}")
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
    """Load task definitions from a directory of JSON files.

    Args:
        json_path: Path to a directory containing JSON files with task definitions.
                   Each file contains a list of task objects with an "id" field.
        work_dir: Working directory to use for task execution

    Returns:
        dictionary of task definitions keyed by task ID
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Task definition path not found: {json_path}")

    task_files = sorted(json_path.glob("*.json")) if json_path.is_dir() else [json_path]

    tasks = {}
    for task_file in task_files:
        with task_file.open() as f:
            task_data = json.load(f)

        # Support both list format (new) and dict format (legacy)
        if isinstance(task_data, list):
            items = {task["id"]: task for task in task_data}
        else:
            items = task_data

        for task_id, task_info in items.items():
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
                input_map={
                    dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])
                },
                initial_input=initial_input,
            )

    return tasks


def create_environments(
    task_json_path: str | Path,
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
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

    task_json_path = Path(task_json_path)
    name = task_json_path.name if task_json_path.is_dir() else task_json_path.stem
    logger.info(f"Creating linked task environments for group: {name}")

    # Create environments for all tasks; grouping is derived from the graph
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=name,
        toolset=Toolset(
            pool=create_tools(),
            common=taskgroup_common_tools or {},
        ),
    )


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Resistor Network Benchmark Server")
    parser.add_argument(
        "tasks_json_path",
        nargs="?",
        default=None,
        help="Path to tasks JSON file (optional if --mode is provided)",
    )
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
        "--mode",
        type=str,
        choices=["single", "chained"],
        default=None,
        help="Task mode (auto-discovers config/{mode}/{mode}.json)",
    )
    args = parser.parse_args()

    # Resolve tasks JSON path
    if args.tasks_json_path:
        tasks_json_path = args.tasks_json_path
    elif args.mode:
        if args.mode == "single":
            tasks_json_path = (
                Path(__file__).resolve().parents[2]
                / "environments"
                / "level_1"
                / "tasks_json"
            )
        elif args.mode == "chained":
            tasks_json_path = (
                Path(__file__).resolve().parents[2]
                / "environments"
                / "level_1"
                / "subtasks_json"
            )
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        if not Path(tasks_json_path).exists():
            logger.error(f"Task config not found: {tasks_json_path}")
            sys.exit(1)
    else:
        tasks_json_path = (
            Path(__file__).resolve().parents[2]
            / "environments"
            / "level_1"
            / "tasks_json"
        )
        if not Path(tasks_json_path).exists():
            logger.error(f"Task config not found: {tasks_json_path}")
            sys.exit(1)

    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

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
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")

    # --- Run Server ---
    logger.info(f"Running server on {args.host}:{args.port}")
    run_server(environments, args.host, args.port)
