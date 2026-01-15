#!/usr/bin/env python
import gc
import json
import os
from collections.abc import Callable
from pathlib import Path
import platform
import nanosurf
from loguru import logger

# ----------------------------------------------------------
# Safe pythoncom import (Windows only)
# ----------------------------------------------------------
if platform.system() == "Windows":
    import pythoncom
else:
    pythoncom = None

from corral.backend.env import Environment
from corral.backend.server import run_server
from corral.backend.task import TaskDefinition, TaskGroup
from corral.backend.tool import Tool
from corral.utils.code_tools import execute_python_code
from corral.utils.io_tools import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from score import (
    check_equation,
    check_file_exists,
    check_image_quality,
    check_numerical,
    check_params_function,
    check_roughness_function,
)
from tools import (
    Code_Executor,
    Document_Retrieval,
    Image_Analyzer,
    Image_optimizer,
    scan_grain_area,
    visualize_grain_boxes,
)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt_4o").strip()
logger.info(f"[SERVER] Using LLM_MODEL={LLM_MODEL}")
ENVIRONMENT = "enviroment"
TASK_TYPE = "subtasks_2"  # "single_task" or "subtasks"
BASE_WORK_DIR = rf"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\{LLM_MODEL}\{ENVIRONMENT}\{TASK_TYPE}"

SCORING_FUNCTIONS = {
    "check_numerical": check_numerical,
    "check_image_quality": check_image_quality,
    "check_params_function": check_params_function,
    "check_file_exists": check_file_exists,
    "check_roughness_function": check_roughness_function,
    "check_mathematical_eq": check_equation,
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
    logger.info(f"Loading task definitions from {json_path}")
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Task definition file not found: {json_path}")

    with Path(json_path).open() as f:
        task_data = json.load(f)

    tasks = {}
    for task_id, task_info in task_data.items():
        scoring_fn_name = task_info.get(
            "scoring_function", "default"
        )  # Get the scoring function by name from the registry
        scoring_params = task_info.get("scoring_params", {})

        scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)
        initial_input = task_info.get(
            "initial_input", {}
        ).copy()  # Add work_dir to initial input if not already present
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


class AFMEnvironment(Environment):
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
        self.base_work_dir = base_work_dir

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        self.initial_params = self.current_task.initial_input["params"]

        self.afm_dir = self.base_work_dir

        super().__init__(f"{task_id}", base_work_dir=base_work_dir)

        # Add tools
        self._add_task_tools()
        self._setup_file_tools()

    def _add_task_tools(self):
        """Add required tools for the task"""
        for tool_name in self.current_task.tools:
            if tool_name in self.subtask_specific_tools:
                # logger.info(f"tool name : {tool_name}")
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

    def reset_params(self) -> None:
        if pythoncom:
            pythoncom.CoInitialize()
        spm = nanosurf.SPM()
        application = spm.application
        application.SetGalleryHistoryDirectoryPath(self.current_work_dir)
        scan = application.Scan
        zcontrol = application.ZController
        head = application.ScanHead
        opmode = application.OperatingMode

        # Access initial parameters
        params = self.initial_params

        # Helper to safely set attributes
        def safe_set(obj, attr, key, transform=lambda x: x):
            if key in params:
                setattr(obj, attr, transform(params[key]))

        # Apply scan parameters (converted to meters and seconds)
        safe_set(scan, "ImageHeight", "image_height", lambda x: x * 1e-9)
        safe_set(scan, "ImageWidth", "image_width", lambda x: x * 1e-9)
        safe_set(scan, "Scantime", "times_per_line")
        safe_set(scan, "Points", "points_per_line")
        safe_set(scan, "Rotation", "rotation")
        safe_set(scan, "Lines", "lines_per_frame")
        safe_set(scan, "CenterPosX", "centre_x", lambda x: x * 1e-9)
        safe_set(scan, "CenterPosY", "centre_y", lambda x: x * 1e-9)

        # Z-control parameters
        safe_set(zcontrol, "PGain", "pgain")
        safe_set(zcontrol, "IGain", "igain")
        safe_set(zcontrol, "DGain", "dgain")
        # safe_set(zcontrol, "SetPoint", "setpoint")  # Uncomment if needed
        safe_set(opmode, "OperatingMode", "mode")

        # Head and operating mode
        safe_set(head, "CantileverByGUID", "tip")
        # if "mode" in params:
        #     opmode.OperatingMode = getattr(spm.OperatingMode, params["mode"])

        logger.info(
            f"AFM parameters have been reset to initial values. "
            f"AFM images will be saved at {application.GetGalleryHistoryDirectoryPath}. "
            f"Corral's current working directory is {self.afm_dir}."
        )

        # Cleanup
        del zcontrol
        del scan
        del application
        del spm
        gc.collect()
        if pythoncom:
            pythoncom.CoInitialize()

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        # os.path.join(self.base_work_dir, trial_id)
        self._setup_file_tools()
        # self.reset_params()
        return trial_id

    def get_task_prompt(self) -> str:
        _combined_input = self.task_group.get_task_input(self.task_id)
        prompt = "You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API."
        prompt += f"""\nTask: {self.current_task.name}
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
                if key not in ["work_dir", "params"]:
                    prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.afm_dir:
            prompt += f"\nIMPORTANT: You have access to filesystem tools. All image scans will automatically be saved in your isolated workspace which is {self.afm_dir}."

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

        # logger.info(f"PROMPT : {prompt}")
        return prompt

    def configure_additional_apps(self):
        logger.info("configuration taking place!!!!!!!")
        self.reset_params()
        return "No external object configuration needed for this trial."

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
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, AFMEnvironment]:
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
    group_id = Path(
        task_json_path
    ).parent  # Use filename (without extension) as group ID
    logger.info(f"Creating task group with ID: {ENVIRONMENT}")
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

    subtask_specific_tools = {
        "visualize_grain_boxes": visualize_grain_boxes,
        "scan_grain_area": scan_grain_area,
        "Document_Retrieval": Document_Retrieval,
        "Image_optimizer": Image_optimizer,
        "Code_Executor": Code_Executor,
        "Image_Analyzer": Image_Analyzer,
        "execute_python_code": execute_python_code,
    }

    environments = {}
    for task_id in task_group.tasks:
        environments[task_id] = AFMEnvironment(
            task_id=task_id,
            task_group=task_group,
            subtask_specific_tools=subtask_specific_tools,
            taskgroup_common_tools=taskgroup_common_tools,
            base_work_dir=work_dir,
        )

    return environments


if __name__ == "__main__":
    tasks_json_path = (
        Path(__file__).parent.parent.parent
        / "afm"
        / "src"
        / ENVIRONMENT
        / f"{TASK_TYPE}.json"
    )
    logger.info(f"task directory {tasks_json_path}")
    work_dir = BASE_WORK_DIR
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )
    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_from_tasks:
            logger.info(f"Depends on: {env.current_task.input_from_tasks}")

    # Run server
    run_server(environments, host, port)
