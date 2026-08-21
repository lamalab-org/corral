"""Spectra-elucidation environment definitions."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

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

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import State
from corral.core.task import (
    EnvironmentSetup,
    InputRef,
    TaskDefinition,
    with_fixed_inputs,
)
from corral.report.logging import event, exception_fields

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

            target = data["output"][0]["target"]
            tasks[task_id] = TaskDefinition(
                name=data["name"],
                description=data["input"]["prompt"],
                tools=data.get("tools", []),
                scoring_fn=with_fixed_inputs(
                    SCORING_FUNCTIONS[str(data["scoring_function"])],
                    ground_truth=target,
                ),
                scoring_inputs=target,
                submission_format=data.get("submission_format", ""),
                input_map={dep: InputRef(dep) for dep in input_from_tasks},
                initial_input=initial_input,
                prompt_fn=_spectra_prompt,
                setup_fn=_expose_ground_truth,
                resolve_answer=False,
            )
    return tasks


def _spectra_prompt(env: Environment, state: State) -> str:
    """Task prompt that echoes each dependency's question and answer."""
    task = env.current_task
    prompt = (
        f"Task {task.name}:\n"
        f"{task.description}\n\n"
        "Required submission format:\n"
        f"{task.submission_format}\n\n"
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
        benchmark="spectra_elucidation",
        task_id=env.task_id,
        prompt=prompt,
    )
    return prompt


def _expose_ground_truth(env: Environment, state: State) -> EnvironmentSetup:
    """Expose the target molecule to tools as a hidden `h_smiles` argument."""
    del state
    return EnvironmentSetup(
        hidden_arguments={"h_smiles": env.current_task.scoring_inputs},
        status="Ground-truth molecule exposed to tools.",
    )


def create_spectra_elu_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
    level: int = 1,
) -> dict[str, Environment]:
    """Create environments for the spectra elucidation benchmark tasks."""
    started = perf_counter()
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
    name = "spectra_elucidation"
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
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
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(pool=create_tools(), workspace_factory=None),
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
    parser = argparse.ArgumentParser(
        description="Inspect spectra-elucidation environments"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Level of the benchmark to run (1-2)",
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
        work_dir=BASE_WORK_DIR, subtask_level=args.subtask_level, level=args.level
    )

    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="spectra_elucidation",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
