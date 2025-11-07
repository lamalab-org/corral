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

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.utils.task_group import TaskGroupEnvironment

BASE_WORK_DIR = os.environ.get(
    "CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/spectra_elucidation"
)

SCORING_FUNCTIONS = {
    "1": score_formula_match,
    "2": validate_dbe_consistency,
    "3": score_isotopic_distribution,
    "4": score_num_carbon_symmetry_classes,
    "5": score_num_hydrogen_symmetry_classes,
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


def create_spectra_elu_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
) -> dict[str, TaskGroupEnvironment]:
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

    # Create all available tools
    subtask_specific_tools = create_tools()

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            base_work_dir=work_dir,
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
