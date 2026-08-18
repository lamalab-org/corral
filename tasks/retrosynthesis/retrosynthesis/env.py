"""Retrosynthesis environment definitions."""

import argparse
import json
import os
from pathlib import Path

from loguru import logger
from retrosynthesis.checks import check_database
from retrosynthesis.score import (
    check_apply_template,
    check_list_molecules,
    check_reactants,
    check_template,
    score_final,
    score_final_without_price,
)
from retrosynthesis.tools import create_tools

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import State
from corral.core.task import InputRef, TaskDefinition, with_fixed_inputs

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "CORRAL_WORK_DIR/rethrosynthesis")

SCORING_FUNCTIONS = {
    "score_final": score_final,
    "check_reactants": check_reactants,
    "check_template": check_template,
    "check_apply_template": check_apply_template,
    "score_final_without_price": score_final_without_price,
    "check_list_molecules": check_list_molecules,
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
        for task in task_data:
            task_id = task["id"]
            initial_input = task.get("initial_inputs", {})
            initial_input["work_dir"] = work_dir
            input_from_tasks = task.get("input", {}).get("input_from_task", [])
            if not isinstance(input_from_tasks, list):
                input_from_tasks = []

            raw_target = task["output"][0]["target"]
            scoring_target = (
                sorted(set(input_from_tasks))
                if raw_target == "input_from_task"
                else raw_target
            )
            tasks[task_id] = TaskDefinition(
                name=task["name"],
                description=task["input"]["prompt"],
                tools=task.get("tools", []),
                scoring_fn=with_fixed_inputs(
                    SCORING_FUNCTIONS[str(task["scoring_function"])],
                    target=scoring_target,
                ),
                scoring_inputs=raw_target,
                submission_format=task.get("submission_format", ""),
                input_map={dep: InputRef(dep) for dep in input_from_tasks},
                initial_input=initial_input,
                prompt_fn=_retro_prompt,
                resolve_answer=False,
            )
    return tasks


def _retro_prompt(env: Environment, state: State) -> str:
    """Task prompt that echoes each dependency's question and answer."""
    task = env.current_task
    prompt = (
        f"Task {task.name}:\n"
        f"{task.description}\n\n"
        "Required submission format:\n"
        f"{task.submission_format}\n\n"
    )

    if task.input_map:
        logger.info(f"Available input tasks for {env.task_id}:")
        logger.info(sorted(task.dependencies()))
        prompt += "\nAvailable input data:\n"

        # Display resolved inputs from dependencies
        resolved = env.resolve_inputs(state)
        for input_name, ref in task.input_map.items():
            logger.info(f"Checking dependency: {ref.task_id}")
            dep_prompt = env.group_tasks[ref.task_id].description
            prompt += f"- Input from '{ref.task_id}' with question: '{dep_prompt}' and answer: '{resolved[input_name]}'\n"

    # Display initial input data
    if task.initial_input:
        for key, value in task.initial_input.items():
            if key != "work_dir":
                prompt += f"- {key}: {value}\n"

    logger.info(f"Task prompt for {env.task_id}:\n{prompt}")
    return prompt


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
            / "subtasks_json"
        )
    else:
        json_path = (
            Path(__file__).parent.parent
            / "environments"
            / f"level_{level}"
            / "tasks_json"
        )
    if not json_path.exists():
        raise ValueError(f"Task file {json_path} does not exist.")

    logger.info(f"Loading tasks from {json_path}")

    tasks = load_tasks_from_json(json_path, work_dir=work_dir)

    name = "rethrosynthesis" if not subtask_level else "rethrosynthesis_subtasks"
    logger.info(f"Creating linked task environments {name} with {len(tasks)} tasks")

    # Top-level tasks get the full toolset; subtasks get only their named tools.
    tool_pool = create_tools()
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=name,
        toolset=Toolset(
            pool=tool_pool,
            common={} if subtask_level else tool_pool,
            workspace_factory=None,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect retrosynthesis environments")

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

    # Check database availability and schema before starting the environment
    logger.info("Performing database checks before starting the environment...")
    try:
        check_database()
    except (ConnectionError, ValueError) as e:
        logger.error(f"Database check failed: {e}")
        logger.error(
            "Please ensure the database is properly set up before starting the environment."
        )
        logger.error(
            "You may need to run: python tasks/retrosynthesis/database_config/phase_b_production.py"
        )
        raise SystemExit(1) from e

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_rethrosynthesis_environments(
        work_dir=BASE_WORK_DIR, subtask_level=args.subtask_level, level=args.level
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")
