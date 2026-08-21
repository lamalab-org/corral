"""Retrosynthesis environment definitions."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

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
from corral.report.logging import event, exception_fields

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
        event(
            "DEBUG",
            "environment.dependencies_resolved",
            subsystem="runtime",
            benchmark="retrosynthesis",
            task_id=env.task_id,
            dependencies=sorted(task.dependencies()),
        )
        prompt += "\nAvailable input data:\n"

        # Display resolved inputs from dependencies
        resolved = env.resolve_inputs(state)
        for input_name, ref in task.input_map.items():
            dep_prompt = env.group_tasks[ref.task_id].description
            prompt += f"- Input from '{ref.task_id}' with question: '{dep_prompt}' and answer: '{resolved[input_name]}'\n"

    # Display initial input data
    if task.initial_input:
        for key, value in task.initial_input.items():
            if key != "work_dir":
                prompt += f"- {key}: {value}\n"

    event(
        "DEBUG",
        "environment.prompt_generated",
        subsystem="runtime",
        benchmark="retrosynthesis",
        task_id=env.task_id,
        prompt=prompt,
    )
    return prompt


def create_rethrosynthesis_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
    level: int = 1,
) -> dict[str, Environment]:
    """Create environments for the rethrosynthesis benchmark tasks."""
    started = perf_counter()
    name = "rethrosynthesis" if not subtask_level else "rethrosynthesis_subtasks"
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
    )
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
    event(
        "DEBUG",
        "environment.tasks_loading",
        subsystem="runtime",
        benchmark=name,
        task_source=str(json_path),
    )
    try:
        if not json_path.exists():
            raise ValueError(f"Task file {json_path} does not exist.")
        tasks = load_tasks_from_json(json_path, work_dir=work_dir)
        tool_pool = create_tools()
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(
                pool=tool_pool,
                common={} if subtask_level else tool_pool,
                workspace_factory=None,
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
    check_started = perf_counter()
    event(
        "INFO",
        "environment.database_check_started",
        subsystem="runtime",
        benchmark="retrosynthesis",
    )
    try:
        check_database()
    except (ConnectionError, ValueError) as exc:
        event(
            "ERROR",
            "environment.database_check_failed",
            subsystem="runtime",
            benchmark="retrosynthesis",
            status="failed",
            duration_ms=round((perf_counter() - check_started) * 1000, 3),
            remediation="Run the retrosynthesis database setup script.",
            **exception_fields(exc),
        )
        raise SystemExit(1) from exc
    event(
        "INFO",
        "environment.database_check_completed",
        subsystem="runtime",
        benchmark="retrosynthesis",
        status="completed",
        duration_ms=round((perf_counter() - check_started) * 1000, 3),
    )

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_rethrosynthesis_environments(
        work_dir=BASE_WORK_DIR, subtask_level=args.subtask_level, level=args.level
    )

    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="retrosynthesis",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
