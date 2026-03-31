import json
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from score import (
    check_log,
    check_msd,
    check_numerical,
    check_potential_file,
    check_structure,
)
from tools import (
    convert_structure_to_lammps_data,
    execute_python_script,
    get_nth_run_log,
    get_potential_metadata,
    get_structure_from_mp_text,
    keyword_log_extractor,
    run_lammps,
    visualisation_tool,
)

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.context7_tools import get_library_documentation
from corral.utils.io_tools import GrepTool
from corral.utils.task_group import TaskGroupEnvironment as _BaseTaskGroupEnvironment

SCORING_FUNCTIONS = {
    "check_numerical": check_numerical,
    "check_potential_file": check_potential_file,
    "check_structure": check_structure,
    "check_log": check_log,
    "check_msd": check_msd,
}


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Get a scoring function by name from the registry, with optional parameters"""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    # If it's a factory function (i.e., takes arguments), call with params
    if params:
        try:
            return fn(**params)
        except Exception as e:
            raise ValueError(
                f"Error initializing scoring function '{name}' with params {params}: {e}"
            ) from e
    else:
        return fn


def load_tasks_from_json(json_path: Path, work_dir: str) -> dict[str, TaskDefinition]:
    task_files = json_path.glob("*.json")

    if not task_files:
        raise FileNotFoundError(f"No task definition files found in: {json_path}")

    logger.info(f"Loading tasks from JSON files in {json_path}")
    tasks = {}
    for task_file in task_files:
        with task_file.open() as f:
            task_data = json.load(f)
        for task_id, task_info in task_data.items():
            # Get the scoring function by name from the registry
            scoring_fn_name = task_info.get("scoring_function", "default")
            scoring_params = task_info.get("scoring_params", {})

            # Resolve 'target' if it looks like a relative path
            target = scoring_params.get("target")
            scoring_params["target"] = target

            # Optionally reassign if task_info is reused later
            task_info["scoring_params"] = scoring_params

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


class TaskGroupEnvironment(_BaseTaskGroupEnvironment):
    """MD-specific environment with domain-specific prompts and scoring."""

    def get_task_prompt(self) -> str:
        """Generate the task prompt with MD-specific guidelines."""

        prompt = f"""\nTask: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
{self.current_task.submission_format}

"""

        prompt += "\nAvailable input data:\n"

        prompt += "All the potentials, can be found at /potentials/.\n\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            if dep_task_id in self.task_group.results:
                dep_result = self.task_group.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from {dep_task_id}: {dep_result['answer']}\n"
                else:
                    prompt += f"- Input from {dep_task_id}: {dep_result}\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.current_work_dir:
            prompt += (
                f"\nYour current workspace directory is: {self.current_work_dir}\n"
                "All files you generate should be saved in this directory.\n\n"
                "### Important Resource and File Access Guidelines ###\n"
                "1. **Potential Files**:\n"
                "   - These files are *fully verified and correct*.\n"
                "   - You must **not attempt to read or parse them directly**.\n"
                "   - Reading them is unnecessary and will waste important computational resources.\n\n"
                "2. **Simulation Log Files**:\n"
                "   - These files are *very large* and should **not be directly parsed**.\n"
                "   - Direct parsing would cause excessive cost and resource usage.\n\n"
                "Important: Files in /structures and /potentials should not be modified at any cost, including operations like copying or moving them. Doing this will immediately return in error.\n\n"
                "### Simulation Logging Requirements ###\n"
                "For every simulation run involving any ensemble (e.g., NVT, NPT, NVE, etc.), if applicable, the log file **must** record the following quantities:\n"
                "   - Step\n"
                "   - Temperature\n"
                "   - Pressure\n"
                "   - Density\n"
                "These quantities should be written at an appropriate, user-configurable frequency (typically 1000 timesteps) suitable for monitoring equilibration and production behavior.\n"
            )

        # Add note about dependencies
        if self.current_task.input_from_tasks:
            status = []
            for dep_id in self.current_task.input_from_tasks:
                status_text = (
                    "available"
                    if dep_id in self.task_group.results
                    else "not yet available"
                )
                status.append(f"{dep_id} ({status_text})")

            prompt += f"\n\nThis task uses output from tasks: {', '.join(status)}"

        logger.info(f"PROMPT : {prompt}")

        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0

        try:
            # Get and log the raw submission
            answer_value = self.state.submitted_answer.strip()
            logger.info(f"Raw submission for {self.task_id}: {answer_value!r}")

            # Call the scoring function with the raw answer
            score = self.current_task.scoring_fn(answer_value)

            # Store result in task group
            self.task_group.store_result(self.task_id, {"answer": answer_value}, score)
            logger.info(f"Task {self.task_id} scored: {score}")

            return score

        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0


def create_environments(
    work_dir: str,
    subtask_level: bool,
    environment: str,
    level: str,
    taskgroup_common_tools: dict[str, Tool] | None = None,
) -> dict[str, TaskGroupEnvironment]:
    logger.info("Creating environments for MD")

    if subtask_level:
        logger.info("Creating environments with subtask level enabled")
        json_path = (
            Path(__file__).parent.parent.parent
            / "environments"
            / environment
            / level
            / "subtasks"
        )
    else:
        json_path = (
            Path(__file__).parent.parent.parent
            / "environments"
            / environment
            / level
            / "tasks"
        )

    # Load tasks from JSON
    tasks = load_tasks_from_json(json_path, work_dir)

    # Create task group
    group_id = f"MD-{environment}"
    logger.info(f"Creating task group {group_id} with {len(tasks)} tasks")
    task_group = TaskGroup(group_id=group_id, tasks=tasks)

    # Print task dependencies for reference
    logger.info("\nTask Dependencies:")
    for task_id, deps in task_group.get_task_dependencies().items():
        logger.info(f"- {task_id}: depends on {deps}")

    # Print ordering of tasks
    ordered_tasks = task_group.get_ordered_tasks()
    logger.info("\nTask Execution Order:")
    for i, task_id in enumerate(ordered_tasks):
        logger.info(f"{i+1}. {task_id}")

    # Create environments for all tasks
    subtask_specific_tools = {
        "convert_structure_to_lammps_data": convert_structure_to_lammps_data,
        "get_potential_metadata": get_potential_metadata,
        "get_structure_from_mp_text": get_structure_from_mp_text,
        "run_lammps": run_lammps,
        "get_nth_run_log": get_nth_run_log,
        "keyword_log_extractor": keyword_log_extractor,
        "visualisation_tool": visualisation_tool,
    }

    extra_file_tools = {
        "library_docs": get_library_documentation,
        "execute_python_script": execute_python_script,
    }

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
            fsmanager_app="simagent",
            extra_file_tools=extra_file_tools,
            extra_fsmanager_tool_classes={"grep": GrepTool},
        )

    return environments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--subtask_level", type=lambda x: x.lower() == "true", required=True
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--level", required=True)
    args = parser.parse_args()

    # Create all environments with file system tools
    environments = create_environments(
        work_dir=args.dir,
        subtask_level=args.subtask_level,
        environment=args.environment,
        level=args.level,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    run_server(
        environments=environments,
        port=args.port,
    )
