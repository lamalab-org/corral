import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import ClassVar

from corral_md.score import (
    check_log,
    check_msd,
    check_numerical,
    check_potential_file,
    check_structure,
)
from corral_md.tools import (
    build_run_lammps_tool,
    convert_structure_to_lammps_data,
    execute_python_script,
    get_nth_run_log,
    get_potential_metadata,
    get_structure_from_mp_text,
    keyword_log_extractor,
    run_lammps,
    visualisation_tool,
)

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import ExecutionState
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.report.logging import event, exception_fields
from corral.utils.context7_tools import get_library_documentation
from corral.workspace import (
    WorkspaceFilesystem,
    build_workspace_tools,
    confine_workspace_path,
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

    event(
        "DEBUG",
        "environment.tasks_loading",
        subsystem="runtime",
        benchmark="corral_md",
        task_source=str(json_path),
        task_file_count=len(task_files),
    )
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
    """MD filesystem and LAMMPS tools bound to one local workspace."""
    tools = build_workspace_tools(WorkspaceFilesystem(workspace))
    return {
        **{
            name: tools[name]
            for name in (
                "list_files",
                "read_file",
                "write_file",
                "file_info",
                "cat_files",
                "copy_file",
                "grep",
            )
        },
        "library_docs": get_library_documentation,
        "execute_python_script": execute_python_script,
        # Overrides the statically selected tool with a workspace-aware variant
        # that handles Modal upload and download internally.
        "run_lammps": build_run_lammps_tool(workspace),
    }


class MolecularDynamicsEnvironment(Environment):
    """Confine every agent-controlled domain-tool path to this task workspace."""

    _PATH_ARGUMENTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "get_structure_from_mp_text": ("file_path",),
        "convert_structure_to_lammps_data": ("structure_path", "output_file"),
        "run_lammps": ("input_file",),
        "get_nth_run_log": ("path", "save"),
        "keyword_log_extractor": ("path",),
        "execute_python_script": ("script_path", "working_dir"),
        "visualisation_tool": ("path",),
    }

    def preprocess_arguments(self, tool_name: str, args: dict) -> dict:
        parsed = super().preprocess_arguments(tool_name, args)
        if not self.workspace_path:
            return parsed
        for argument in self._PATH_ARGUMENTS.get(tool_name, ()):
            value = parsed.get(argument)
            if isinstance(value, str) and value:
                parsed[argument] = str(
                    confine_workspace_path(
                        self.workspace_path,
                        value,
                        allow_root=argument == "working_dir",
                    )
                )
        return parsed


def _md_task_prompt(env: Environment, state: ExecutionState) -> str:
    """Generate the MD task prompt with resource and logging guidance."""
    prompt = f"""\nTask: {env.current_task.name}
Description: {env.current_task.description}

Required submission format:
{env.current_task.submission_format}

"""

    prompt += "\nAvailable input data:\n"

    prompt += (
        "All potentials are mounted read-only below /potentials/. Use these exact "
        "catalog paths in LAMMPS inputs: /potentials/SW/Si.sw, "
        "/potentials/TERSOFF/2007_SiO.tersoff, "
        "/potentials/EAM/Al99.eam.alloy, "
        "/potentials/EAM/Cu_Zhou04.eam.alloy, "
        "/potentials/EAM/Mg_Zhou04.eam.alloy, "
        "/potentials/EAM/Fe-C_Hepburn_Ackland.eam.fs, and "
        "/potentials/BKS/pot.mod.\n\n"
    )

    # Display resolved inputs from dependencies
    resolved = env.resolve_inputs(state)
    for input_name, ref in env.current_task.input_map.items():
        prompt += f"- {input_name} (from {ref.task_id}): {resolved[input_name]}\n"

    # Display initial input data
    for key, value in env.current_task.initial_input.items():
        if key != "work_dir":
            prompt += f"- {key}: {value}\n"

    # Add workspace info
    if env.workspace_path:
        prompt += (
            "\nYou have an isolated task workspace. Filesystem and domain tools "
            "resolve paths against it automatically. Always pass workspace-relative "
            "POSIX paths such as `input/run.in`; never pass an absolute host path "
            "to a workspace tool.\n\n"
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

    event(
        "DEBUG",
        "environment.prompt_generated",
        subsystem="runtime",
        benchmark="corral_md",
        task_id=env.task_id,
        prompt=prompt,
    )

    return prompt


def create_environments(
    work_dir: str = BASE_WORK_DIR,
    subtask_level: bool = False,
    level: int = 1,
    taskgroup_common_tools: dict[str, Tool] | None = None,
) -> dict[str, Environment]:
    started = perf_counter()
    name = f"MD-level_{level}"
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
    )
    # Modal path rewriting needs one stable spelling of the local workspace.
    # Advertising an absolute path also keeps paths written into LAMMPS inputs
    # directly mappable to the isolated /results/corral/jobs/... directory.
    work_dir = str(Path(work_dir).expanduser().resolve())

    if subtask_level:
        event(
            "DEBUG",
            "environment.subtasks_enabled",
            subsystem="runtime",
            benchmark=name,
        )
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

    try:
        if not json_path.exists():
            raise FileNotFoundError(f"Task config not found: {json_path}")
        tasks = load_tasks_from_json(json_path, work_dir)
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(
                pool=subtask_specific_tools,
                common=taskgroup_common_tools or {},
                workspace_factory=_md_file_tools,
            ),
            env_cls=MolecularDynamicsEnvironment,
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
    parser = argparse.ArgumentParser(description="Inspect Corral MD environments")
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

    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="corral_md",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
