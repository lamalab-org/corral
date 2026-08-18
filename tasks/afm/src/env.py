#!/usr/bin/env python
import gc
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path

import nanosurf
from loguru import logger

# ----------------------------------------------------------
# Safe pythoncom import (Windows only)
# ----------------------------------------------------------
if platform.system() == "Windows":
    import pythoncom
else:
    pythoncom = None

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import State
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.utils.code_tools import execute_python_code
from score import (
    check_file_exists,
    check_image_quality,
    check_mathematical_eq,
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
TASK_TYPE = "subtasks_1"  # "single_task" or "subtasks"
BASE_WORK_DIR = rf"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\{LLM_MODEL}\{ENVIRONMENT}\{TASK_TYPE}"

SCORING_FUNCTIONS = {
    "check_numerical": check_numerical,
    "check_image_quality": check_image_quality,
    "check_params_function": check_params_function,
    "check_file_exists": check_file_exists,
    "check_roughness_function": check_roughness_function,
    "check_mathematical_eq": check_mathematical_eq,
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
            input_map={
                dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])
            },
            initial_input=initial_input,
            resolve_answer=False,
        )

    return tasks


class AFMEnvironment(Environment):
    """Environment for AFM tasks (escape hatch: drives the Nanosurf API).

    Behaviour beyond the stateful hardware reset is shared with the generic
    `Environment`; only the prompt and per-task instrument reset are
    specialised here. Evaluation uses the task's scoring callable directly.
    """

    @property
    def initial_params(self) -> dict:
        return self.current_task.initial_input["params"]

    @property
    def afm_dir(self) -> str | None:
        return self.workspace_path

    def reset_params(self) -> None:
        if pythoncom:
            pythoncom.CoInitialize()
        spm = nanosurf.SPM()
        application = spm.application
        application.SetGalleryHistoryDirectoryPath(self.workspace_path)
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
            pythoncom.CoUninitialize()

    def get_task_prompt(self, state: State) -> str:
        prompt = "You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API."
        prompt += f"""\nTask: {self.current_task.name}
        Description: {self.current_task.description}
        Required submission format:
        {self.current_task.submission_format}

        """

        prompt += "\nAvailable input data:\n"

        # Display resolved inputs from dependencies
        resolved = self.resolve_inputs(state)
        for input_name, ref in self.current_task.input_map.items():
            prompt += f"- {input_name} (from {ref.task_id}): {resolved[input_name]}\n"

        # Display initial input data
        for key, value in self.current_task.initial_input.items():
            if key not in ["work_dir", "params"]:
                prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.afm_dir:
            prompt += f"\nIMPORTANT: You have access to filesystem tools. All image scans will automatically be saved in your isolated workspace which is {self.afm_dir}."

        # logger.info(f"PROMPT : {prompt}")
        return prompt

    def configure(self, state: State) -> tuple[State, str]:
        logger.info("configuration taking place!!!!!!!")
        self.reset_params()
        return state, "No external object configuration needed for this task."


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

    logger.info(f"Creating linked task environments with ID: {ENVIRONMENT}")

    subtask_specific_tools = {
        "visualize_grain_boxes": visualize_grain_boxes,
        "scan_grain_area": scan_grain_area,
        "Document_Retrieval": Document_Retrieval,
        "Image_optimizer": Image_optimizer,
        "Code_Executor": Code_Executor,
        "Image_Analyzer": Image_Analyzer,
        "execute_python_code": execute_python_code,
    }

    # Create environments for all tasks; grouping is derived from the graph
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=ENVIRONMENT,
        toolset=Toolset(
            pool=subtask_specific_tools,
            common=taskgroup_common_tools or {},
        ),
        env_cls=AFMEnvironment,
    )


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
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )
    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"Depends on: {sorted(env.current_task.dependencies())}")
