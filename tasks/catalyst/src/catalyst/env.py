import os
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

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

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.logging import event, exception_fields
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

event(
    "DEBUG",
    "environment.configuration",
    subsystem="runtime",
    benchmark="catalyst",
    work_dir=BASE_WORK_DIR,
)


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    if params:
        try:
            event(
                "DEBUG",
                "environment.scorer_initializing",
                subsystem="runtime",
                benchmark="catalyst",
                scorer=name,
                arguments=params,
            )
            return fn(**params)
        except Exception as e:
            raise ValueError(
                f"Error initializing scoring function '{name}' with params {params}: {e}"
            ) from e
    else:
        event(
            "DEBUG",
            "environment.scorer_selected",
            subsystem="runtime",
            benchmark="catalyst",
            scorer=name,
        )
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
    started = perf_counter()
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
    )
    try:
        if local_dir is not None:
            entries = load_task_entries(local_dir=local_dir)
        elif environment is not None:
            entries = load_task_entries(
                environment=environment, level=level, task_type=task_type
            )
        else:
            entries = load_task_entries_from_env_package(
                Path(__file__).parent, level=level, subtask=(task_type == "subtask")
            )

        tasks = entries_to_task_definitions(entries, work_dir)
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(
                pool=create_tools(),
                common=taskgroup_common_tools or {},
            ),
        )
    except Exception as exc:
        event(
            "ERROR",
            "environment.failed",
            subsystem="runtime",
            benchmark=name,
            operation="create",
            status="failed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            **exception_fields(exc),
        )
        raise

    event(
        "INFO",
        "environment.completed",
        subsystem="runtime",
        benchmark=name,
        operation="create",
        status="completed",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        environment_count=len(environments),
    )
    return environments


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Inspect Catalyst environments")
    parser.add_argument(
        "tasks_json_path",
        nargs="?",
        default=None,
        help="Path to tasks JSON file or directory (optional if --mode is provided)",
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
            error = FileNotFoundError(f"Task config not found: {local_dir}")
            event(
                "ERROR",
                "environment.configuration_failed",
                subsystem="runtime",
                benchmark="catalyst",
                status="failed",
                **exception_fields(error),
            )
            sys.exit(1)
    else:
        local_dir = (
            Path(__file__).resolve().parents[2]
            / "environments"
            / "level_1"
            / "tasks_json"
        )
        if not Path(local_dir).exists():
            error = FileNotFoundError(f"Task config not found: {local_dir}")
            event(
                "ERROR",
                "environment.configuration_failed",
                subsystem="runtime",
                benchmark="catalyst",
                status="failed",
                **exception_fields(error),
            )
            sys.exit(1)

    # Resolve the configured workspace root before constructing definitions.
    work_dir = str(Path(os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)).resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    environments = create_environments(
        local_dir=local_dir,
        work_dir=work_dir,
    )

    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="catalyst",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
