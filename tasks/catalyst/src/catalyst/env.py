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

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import Tool
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
            input_map={dep: InputRef(dep) for dep in entry.get("input_from_tasks", [])},
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
    name: str = "catalyst",
) -> dict[str, Environment]:
    """Create environments from HuggingFace or local task configs.

    Args:
        local_dir: Path to local JSON directory (mutually exclusive with environment).
        environment: HF environment name (mutually exclusive with local_dir).
        level: Level number (for HF loading or env-package resolution).
        task_type: "task" or "subtask".
        taskgroup_common_tools: Tools common to all subtasks.
        work_dir: Working directory for task execution.
        name: Benchmark label used for tracing/LaTeX.

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

    logger.info(f"Creating linked task environments '{name}' with {len(tasks)} tasks")

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

    parser = _argparse.ArgumentParser(description="Catalyst Benchmark Server")
    parser.add_argument(
        "tasks_json_path",
        nargs="?",
        default=None,
        help="Path to tasks JSON file or directory (optional if --mode is provided)",
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
        help="Task mode (auto-discovers environments/level_1/{tasks_json|subtasks_json})",
    )
    args = parser.parse_args()

    # Resolve tasks JSON path
    if args.tasks_json_path:
        local_dir = args.tasks_json_path
    elif args.mode:
        if args.mode == "single":
            local_dir = (
                Path(__file__).resolve().parents[2]
                / "environments"
                / "level_1"
                / "tasks_json"
            )
        elif args.mode == "chained":
            local_dir = (
                Path(__file__).resolve().parents[2]
                / "environments"
                / "level_1"
                / "subtasks_json"
            )
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        if not Path(local_dir).exists():
            logger.error(f"Task config not found: {local_dir}")
            sys.exit(1)
    else:
        local_dir = (
            Path(__file__).resolve().parents[2]
            / "environments"
            / "level_1"
            / "tasks_json"
        )
        if not Path(local_dir).exists():
            logger.error(f"Task config not found: {local_dir}")
            sys.exit(1)

    host = args.host
    port = args.port
    # Absolute so the workspace path the server reports to a sandbox-running
    # agent (e.g. Codex) resolves the same in that agent's process (a different
    # cwd) instead of silently missing and losing files. BASE_WORK_DIR is already
    # absolute; resolve() also normalizes a relative CORRAL_WORK_DIR override.
    work_dir = str(Path(os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)).resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    environments = create_environments(
        local_dir=local_dir,
        work_dir=work_dir,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")

    logger.info(f"Running server on {host}:{port}")
    run_server(environments, host, port)
