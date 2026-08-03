import json
import os
from pathlib import Path

from discover_physics.score import check_physics_law
from discover_physics.tools import create_tools
from loguru import logger

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, with_fixed_inputs

BASE_WORK_DIR = os.environ.get(
    "CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/discover_physics"
)

logger.info(f"Using BASE_WORK_DIR: {BASE_WORK_DIR}")


def _expose_world_config(env: Environment) -> str:
    """Expose this trial's hidden world config to tools (e.g. `run_experiment`)
    via `env.hidden_args`, without it ever appearing in the agent-visible
    prompt or tool schema. Mirrors spectra_elucidation's `_expose_ground_truth`."""
    env.hidden_args = {"world_config": env.current_task.scoring_inputs}
    return "Hidden simulator configured for this trial."


def load_tasks_from_json(
    json_path: str | Path, work_dir: str
) -> dict[str, TaskDefinition]:
    """Load task definitions from a directory of JSON files.

    Each task JSON must include a `world_config` dict (`world`, `engine`,
    `noise_std`, `noise_seed`) — used both to score the submission
    (`check_physics_law`) and, via `setup_fn`/`hidden_args`, to let the
    `run_experiment` tool reach the same hidden simulator without exposing
    its configuration to the agent.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Task definition path not found: {json_path}")

    task_files = sorted(json_path.glob("*.json")) if json_path.is_dir() else [json_path]

    tasks: dict[str, TaskDefinition] = {}
    for task_file in task_files:
        with task_file.open() as f:
            task_data = json.load(f)

        items = (
            {task["id"]: task for task in task_data}
            if isinstance(task_data, list)
            else task_data
        )

        for task_id, task_info in items.items():
            world_config = task_info["world_config"]

            initial_input = task_info.get("initial_input", {}).copy()
            if "work_dir" not in initial_input:
                initial_input["work_dir"] = work_dir

            tasks[task_id] = TaskDefinition(
                name=task_info["name"],
                description=task_info["description"],
                tools=task_info.get("tools", ["run_experiment"]),
                scoring_fn=with_fixed_inputs(check_physics_law, **world_config),
                scoring_inputs=world_config,
                submission_format=task_info.get("submission_format", ""),
                initial_input=initial_input,
                setup_fn=_expose_world_config,
                resolve_answer=False,
            )

    return tasks


def create_environments(
    task_json_path: str | Path,
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
    """Create environments for the discover_physics tasks defined at `task_json_path`."""
    logger.info(f"Creating environments from {task_json_path} with work_dir {work_dir}")

    tasks = load_tasks_from_json(task_json_path, work_dir)

    task_json_path = Path(task_json_path)
    name = task_json_path.name if task_json_path.is_dir() else task_json_path.stem
    logger.info(f"Creating task environments for group: {name}")

    # No filesystem workspace tools: submissions are self-contained Python
    # source, not files, and the only capability the agent needs is
    # run_experiment.
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=name,
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
    )


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="DiscoverPhysics Benchmark Server")
    parser.add_argument(
        "tasks_json_path",
        nargs="?",
        default=None,
        help="Path to tasks JSON file or directory (defaults to environments/level_1/tasks_json)",
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
    args = parser.parse_args()

    if args.tasks_json_path:
        tasks_json_path = args.tasks_json_path
    else:
        tasks_json_path = (
            Path(__file__).resolve().parents[2]
            / "environments"
            / "level_1"
            / "tasks_json"
        )
        if not Path(tasks_json_path).exists():
            logger.error(f"Task config not found: {tasks_json_path}")
            raise SystemExit(1)

    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    environments = create_environments(task_json_path=tasks_json_path, work_dir=work_dir)

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")

    logger.info(f"Running server on {args.host}:{args.port}")
    run_server(environments, args.host, args.port)
