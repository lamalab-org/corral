import json
import os
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from score import check_numerical
from tools import (
    derive_rate_law,
    fit_kinetic_parameters,
    generate_ode_system,
    setup_reaction_network,
)

from corral.base import Environment, Tool
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import run_server
from corral.task import TaskDefinition, TaskGroup

# if "CORRAL_WORK_DIR" not in os.environ:
#     raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
# if "ENVIRONMENT" not in os.environ:
#     raise OSError("MD Environment not specified.")


# CORRAL_WORK_DIR = os.environ["CORRAL_WORK_DIR"]
# ENVIRONMENT = os.environ["ENVIRONMENT"]
# TASK_TYPE = os.environ["TASK_TYPE"]


SCORING_FUNCTIONS = {"check_numerical": check_numerical}


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
            task_info["scoring_params"] = scoring_params

            scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)
            # Add work_dir to initial input if not already present
            initial_input = task_info.get("initial_input", {}).copy()
            if "experimental_data_path" in initial_input:
                exp_data_path = initial_input["experimental_data_path"]
                if isinstance(exp_data_path, str) and exp_data_path.endswith(".csv"):
                    json_dir = Path(task_file).resolve().parent.parent
                    abs_exp_data_path = Path(
                        json_dir, "kinetic_modelling", exp_data_path
                    ).resolve()
                    logger.info(
                        f"Resolving experimental data path: {abs_exp_data_path}"
                    )

                    # if not abs_exp_data_path.is_file():
                    #     raise FileNotFoundError(
                    #         f"[{task_id}] Experimental data path does not exist: {abs_exp_data_path}"
                    #     )

                    initial_input["experimental_data_path"] = str(abs_exp_data_path)
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

        super().__init__(
            f"{task_id}",
            base_work_dir=base_work_dir,
            fs_manager=FSManager("file", base_path=base_work_dir),
        )

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
                    "grep": GrepTool(fs_manager),
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

    def _describe_experimental_data(self) -> str:
        """Generate description of available experimental data."""
        # data = self.experimental_data
        # description = [
        #     f"- Time points: {len(data)} measurements",
        #     f"- Time range: {data['time'].min():.1f} to {data['time'].max():.1f} time units",
        #     f"- Species monitored: {', '.join([col for col in data.columns if col != 'time'])}"
        # ]

        # # Add data quality information
        # missing_data = data.isnull().sum().sum()
        # if missing_data > 0:
        #     description.append(f"- Missing data points: {missing_data}")

        # return "\n".join(description)
        return ""

    def get_task_prompt(self) -> str:
        """Generate the task prompt for the current task"""

        data_description = self._describe_experimental_data()

        prompt = f"""\nTask: {self.current_task.name}
Description: {self.current_task.description}
**Experimental Data Available:** {data_description}

**Your Goal:**
Develop a kinetic model that accurately describes the experimental observations. Your final model should include:
1. A reaction mechanism (if mechanism discovery is required)
2. Rate laws for each reaction
3. Fitted kinetic parameters
4. Statistical validation of the model fit

**Success Criteria:**
- Model achieves R² > 0.95 on experimental data
- Physical reasonableness of parameters
- Statistical validation passes standard tests

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
            prompt += f"\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n Save all the files in {self.current_work_dir} when using tools use this path.\n"

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
    subtask_level: bool = False,
    taskgroup_common_tools: dict[str, Tool] | None = None,
) -> dict[str, TaskGroupEnvironment]:
    logger.info("Creating environments for MD")

    if subtask_level:
        logger.info("Creating environments with subtask level enabled")
        json_path = Path(__file__).parent.parent / "subtasks"
    else:
        json_path = Path(__file__).parent.parent / "tasks"

    # Load tasks from JSON
    tasks = load_tasks_from_json(json_path, work_dir)

    # Create task group
    group_id = "KINETIC-MODELLING"
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
        "setup_reaction_network": setup_reaction_network,
        "derive_rate_law": derive_rate_law,
        "generate_ode_system": generate_ode_system,
        "fit_kinetic_parameters": fit_kinetic_parameters,
    }

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
    import argparse

    parser = argparse.ArgumentParser(description="MD Benchmark Server")
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

    # Create all environments with file system tools

    CORRAL_WORK_DIR = "/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_6/mat-agent-bench/tasks/kinetic_modeling/corral_work_dir/"

    environments = create_environments(
        work_dir=CORRAL_WORK_DIR, subtask_level=args.subtask_level
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
