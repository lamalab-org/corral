import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from ml.score import (
    composition_list_quality,
    ml_dataset_preparation_quality_binary,
    ml_pipeline_score,
    model_evaluation_completeness_binary,
    model_training_success_binary,
    polymorph_retrieval_success,
    score_polymorph_dataset,
)
from ml.tools import create_ml_tools

from corral.base import Environment, Tool
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import run_server
from corral.task import TaskDefinition, TaskGroup

# Base working directory
if "CORRAL_WORK_DIR" not in os.environ:
    BASE_WORK_DIR = tempfile.mkdtemp(prefix="catalyst_")
    logger.info(f"CORRAL_WORK_DIR not set, using temporary directory: {BASE_WORK_DIR}")
else:
    BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]
    logger.info(f"Using CORRAL_WORK_DIR: {BASE_WORK_DIR}")


# Registry of scoring functions
SCORING_FUNCTIONS = {
    "ml_pipeline_score": ml_pipeline_score,
    "model_training_success_binary": model_training_success_binary,
    "model_evaluation_completeness_binary": model_evaluation_completeness_binary,
    "polymorph_retrieval_success": polymorph_retrieval_success,
    "score_polymorph_dataset": score_polymorph_dataset,
    "ml_dataset_preparation_quality_binary": ml_dataset_preparation_quality_binary,
    "composition_list_quality": composition_list_quality,
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


def load_tasks_from_json(
    json_path: str | Path, work_dir: str
) -> dict[str, TaskDefinition]:
    """Load task definitions from a JSON file.

    Args:
        json_path: Path to the JSON file containing task definitions
        work_dir: Working directory to use for task execution

    Returns:
        dictionary of task definitions keyed by task ID
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Task definition file not found: {json_path}")

    with Path(json_path).open() as f:
        task_data = json.load(f)

    tasks = {}
    for task_id, task_info in task_data.items():
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
            input_from_tasks=task_info.get("input_from_tasks", []),
            initial_input=initial_input,
        )

    return tasks


class TaskGroupEnvironment(Environment):
    """Environment that works with a task group - simple composition approach"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        subtask_specific_tools: dict[str, Tool],
        base_work_dir: str,
        taskgroup_common_tools: dict[str, Tool] | None = None,
    ):
        self.task_group = task_group
        self.subtask_specific_tools = subtask_specific_tools
        self.taskgroup_common_tools = taskgroup_common_tools or {}

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        super().__init__(f"{task_id}", base_work_dir=base_work_dir)

        # Add tools
        self._add_task_tools()
        self._setup_file_tools()

    def _add_task_tools(self):
        """Add required tools for the task"""
        for tool_name in self.current_task.tools:
            if tool_name in self.subtask_specific_tools:
                self.add_tool(self.subtask_specific_tools[tool_name])
            else:
                logger.warning(
                    f"Tool {tool_name} required for task {self.task_id} not found"
                )

        for tool in self.taskgroup_common_tools.values():
            self.add_tool(tool)

    def _setup_file_tools(self):
        """Setup file tools for current workspace"""
        if self.current_work_dir:
            logger.info(
                f"DEBUG: Setting up FSManager with base_path: {self.current_work_dir}"
            )
            # Create new FSManager for current workspace
            fs_manager = FSManager("file", base_path=self.current_work_dir)

            # Add/update file tools
            self.tools.update(
                {
                    "list_files": ListFilesTool(fs_manager),
                    "read_file": ReadFileTool(fs_manager),
                    "write_file": WriteFileTool(fs_manager),
                    "file_info": FileInfoTool(fs_manager),
                    "cat_files": CatFilesTool(fs_manager),
                    "copy_file": CopyFileTool(fs_manager),
                }
            )
            logger.info(
                f"DEBUG: File tools setup complete for workspace: {self.current_work_dir}"
            )
        else:
            logger.warning("DEBUG: No current_work_dir set, skipping file tools setup")

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        return trial_id

    def get_task_prompt(self) -> str:
        """Generate the task prompt for the current task"""
        _combined_input = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
{self.current_task.submission_format}

"""

        prompt += "\nAvailable input data:\n"

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
            prompt += f"\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n Save all the files in {self.current_work_dir}. when using tools use this path\n"

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
    task_json_path: str | Path,
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str | Path = BASE_WORK_DIR,
) -> dict[str, TaskGroupEnvironment]:
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

    # Create task group
    group_id = Path(task_json_path).stem  # Use filename (without extension) as group ID
    logger.info(f"Creating task group with ID: {group_id}")
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

    # Create all available tools
    subtask_specific_tools = create_ml_tools()

    # Create environments for all tasks

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = TaskGroupEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    # Determine tasks file path
    if len(sys.argv) > 1:
        tasks_json_path = sys.argv[1]
    else:
        tasks_json_path = os.environ.get(
            "CORRAL_TASKS_PATH",
            str(Path(__file__).parent.parent.parent / "config" / "dataset.json"),
        )

    # Get server settings from environment if provided
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    work_dir = os.environ.get("CORRAL_WORK_DIR", BASE_WORK_DIR)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    # Create environments
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"  Depends on: {env.current_task.input_from_tasks}")

    # Run server
    run_server(environments, host, port)
