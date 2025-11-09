import json
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from score import check_numerical, check_potential_file, check_structure
from tools import (
    convert_structure_to_lammps_data,
    execute_python_script,
    get_potential_metadata,
    get_structure_from_mp_text,
    run_lammps,
)

from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.context7_tools import get_library_documentation
from corral.utils.io_tools import (
    FSManager,
    GrepTool,
)
from corral.utils.task_group import TaskGroupEnvironment

SCORING_FUNCTIONS = {
    "check_numerical": check_numerical,
    "check_potential_file": check_potential_file,
    "check_structure": check_structure,
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
            if isinstance(target, str) and (target.endswith(".data")):
                json_dir = Path(task_file).resolve().parent.parent
                abs_target_path = Path(json_dir, target).resolve()
                logger.info(f"Resolving target path: {abs_target_path}")

                if not abs_target_path.is_file():
                    raise FileNotFoundError(
                        f"[{task_id}] Target path does not exist: {abs_target_path}"
                    )

                scoring_params["target"] = abs_target_path

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


class MDTaskGroupEnvironment(TaskGroupEnvironment):
    """MD-specific environment that extends TaskGroupEnvironment with custom file tools"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        subtask_specific_tools: dict[str, Tool],
        base_work_dir: str,
        taskgroup_common_tools: dict[str, Tool] | None = None,
    ):
        super().__init__(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            base_work_dir=base_work_dir,
            taskgroup_common_tools=taskgroup_common_tools,
        )

    def _setup_file_tools(self):
        """Setup file tools for current workspace with MD-specific additions"""
        super()._setup_file_tools()

        if self.current_work_dir:
            # Create FSManager for current workspace
            fs_manager = FSManager(
                "file", base_path=self.current_work_dir, app="simagent"
            )

            # Add MD-specific tools
            self.tools.update(
                {
                    "grep": GrepTool(fs_manager),
                    "library_docs": get_library_documentation,
                    "execute_python_script": execute_python_script,
                }
            )
            logger.info(
                f"Added MD-specific file tools for workspace: {self.current_work_dir}"
            )

    def get_task_prompt(self) -> str:
        """Generate the task prompt with MD-specific additions"""
        # Get the base prompt from parent class
        base_prompt = super().get_task_prompt()

        # Build MD-specific additions
        md_additions = []

        # Add potentials location info
        md_additions.append("All the potentials, can be found at /potentials/.")

        # Add workspace path instruction if workspace exists
        if self.current_work_dir:
            md_additions.append(
                f"Save all the files in {self.current_work_dir} when using tools use this path."
            )

        # Insert MD additions after "Available input data:" section
        if md_additions and "Available input data:\n" in base_prompt:
            parts = base_prompt.split("Available input data:\n", 1)
            if len(parts) == 2:
                prompt = (
                    parts[0]
                    + "Available input data:\n"
                    + "\n".join(md_additions)
                    + "\n\n"
                    + parts[1]
                )
            else:
                prompt = base_prompt
        else:
            # If we can't insert in the expected location, append at the end
            prompt = base_prompt + "\n\n" + "\n".join(md_additions)

        logger.info(f"PROMPT : {prompt}")
        return prompt


def create_environments(
    work_dir: str,
    subtask_level: bool,
    environment: str,
    level: str,
    taskgroup_common_tools: dict[str, Tool] | None = None,
) -> dict[str, MDTaskGroupEnvironment]:
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
        logger.info(f"{i + 1}. {task_id}")

    # Create environments for all tasks
    subtask_specific_tools = {
        "convert_structure_to_lammps_data": convert_structure_to_lammps_data,
        "get_potential_metadata": get_potential_metadata,
        "get_structure_from_mp_text": get_structure_from_mp_text,
        "run_lammps": run_lammps,
    }

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = MDTaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
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
