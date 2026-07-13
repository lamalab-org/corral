import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from corral_md.score import (
    check_log,
    check_msd,
    check_numerical,
    check_potential_file,
    check_structure,
)
from corral_md.tools import (
    convert_structure_to_lammps_data,
    execute_python_script,
    get_nth_run_log,
    get_potential_metadata,
    get_structure_from_mp_text,
    keyword_log_extractor,
    run_lammps,
    visualisation_tool,
)
from loguru import logger

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.server import run_server
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import Tool
from corral.utils.context7_tools import get_library_documentation
from corral.utils.io_tools import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/corral_md")

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
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Task definition path not found: {json_path}")

    task_files = sorted(json_path.glob("*.json")) if json_path.is_dir() else [json_path]

    if not task_files:
        raise FileNotFoundError(f"No task definition files found in: {json_path}")

    logger.info(f"Loading tasks from JSON files in {json_path}")
    tasks = {}
    for task_file in task_files:
        with task_file.open() as f:
            task_data = json.load(f)

        items = {task["id"]: task for task in task_data}

        for task_id, task_info in items.items():
            # Get the scoring function by name from the registry
            scoring_fn_name = task_info.get("scoring_function", "default")
            scoring_params = task_info.get("scoring_params", {})

            # Resolve 'target' if it looks like a relative path
            target = scoring_params.get("target")
            # if isinstance(target, str) and (target.endswith(".data")):
            # json_dir = Path(task_file).resolve().parent.parent
            # abs_target_path = Path(json_dir, target).resolve()
            # logger.info(f"Resolving target path: {abs_target_path}")

            # if not abs_target_path.is_file():
            #     raise FileNotFoundError(
            #         f"[{task_id}] Target path does not exist: {abs_target_path}"
            #     )

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
                input_map={
                    dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])
                },
                initial_input=initial_input,
                prompt_fn=_md_task_prompt,
                resolve_answer=False,
            )

    return tasks


def _md_file_tools(workspace: str) -> dict[str, Tool]:
    """MD-specific filesystem tools backed by the simagent FSManager."""
    fs_manager = FSManager("file", base_path=workspace, app="simagent")
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
        "grep": GrepTool(fs_manager),
        "library_docs": get_library_documentation,
        "execute_python_script": execute_python_script,
    }


def _md_task_prompt(env: Environment) -> str:
    """Generate the MD task prompt with resource and logging guidance."""

    prompt = f"""\nTask: {env.current_task.name}
Description: {env.current_task.description}

Required submission format:
{env.current_task.submission_format}

"""

    prompt += "\nAvailable input data:\n"

    prompt += "All the potentials, can be found at /potentials/.\n\n"

    # Display resolved inputs from dependencies
    for input_name, ref in env.current_task.input_map.items():
        if env.state.is_completed(ref.task_id):
            value = env.state.get_output(ref.task_id, ref.key)
            prompt += f"- {input_name} (from {ref.task_id}): {value}\n"

    # Display initial input data
    for key, value in env.current_task.initial_input.items():
        if key != "work_dir":
            prompt += f"- {key}: {value}\n"

    # Add workspace info
    if env.state.workspace:
        prompt += (
            f"\nYour current workspace directory is: {env.state.workspace}\n"
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

    logger.info(f"PROMPT : {prompt}")

    return prompt


def create_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
    level: int = 1,
    taskgroup_common_tools: dict[str, Tool] | None = None,
) -> dict[str, Environment]:
    logger.info("Creating environments for MD")

    if subtask_level:
        logger.info("Creating environments with subtask level enabled")
        json_path = (
            Path(__file__).parent.parent.parent
            / "environments"
            / f"level_{level}"
            / "subtasks_json"
        )
    else:
        json_path = (
            Path(__file__).parent.parent.parent
            / "environments"
            / f"level_{level}"
            / "tasks_json"
        )

    if not json_path.exists():
        logger.error(f"Task config not found: {json_path}")
        sys.exit(1)

    # Load tasks from JSON
    tasks = load_tasks_from_json(json_path, work_dir)

    name = f"MD-level_{level}"
    logger.info(f"Creating linked task environments {name} with {len(tasks)} tasks")

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

    # Create environments for all tasks; grouping is derived from the graph.
    # MD uses a simagent-backed FSManager for workspaces and file tools.
    return build_environments(
        tasks,
        base_work_dir=work_dir,
        name=name,
        toolset=Toolset(
            pool=subtask_specific_tools,
            common=taskgroup_common_tools or {},
            workspace_factory=_md_file_tools,
        ),
        fs_manager=FSManager("file", base_path=work_dir, app="simagent"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corral MD Benchmark Server")
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
        "--level",
        type=int,
        default=1,
        help="Level of the benchmark to run",
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
    environments = create_environments(
        work_dir=BASE_WORK_DIR,
        subtask_level=args.subtask_level,
        level=args.level,
    )

    logger.info("\nCreated Environments:")
    for env_id, env in environments.items():
        logger.info(f"- {env_id}")
        logger.info(f"  Task: {env.current_task.name}")
        if env.current_task.input_map:
            logger.info(f"  Depends on: {sorted(env.current_task.dependencies())}")

    run_server(
        environments=environments,
        host=args.host,
        port=args.port,
    )
