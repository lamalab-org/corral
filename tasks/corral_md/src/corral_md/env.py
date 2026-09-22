import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from corral_md.modal_workspace import (
    pinned_release,
    recovery_snapshot,
)
from corral_md.score import (
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.submission import resolve_submission
from corral_md.submission_examples import example_prompt, seed_examples
from corral_md.tools import (
    build_execute_python_script_tool,
    build_md_terminal_tool,
    build_run_lammps_tool,
    build_run_verified_md_tool,
    convert_structure_to_lammps_data,
    get_nth_run_log,
    get_potential_metadata,
    get_structure_from_mp_text,
    keyword_log_extractor,
    run_lammps,
    visualisation_tool,
)
from corral_md.workspace import MDWorkspaceFilesystem, local_path

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.events import ExecutionStarted
from corral.core.state import ExecutionState
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.core.workspace import WorkspaceState
from corral.report.logging import event, exception_fields
from corral.utils.context7_tools import get_library_documentation
from corral.workspace import (
    build_workspace_tools,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/corral_md")
PACKAGE_DATA_ROOT = Path(__file__).resolve().parents[2] / "environments"

SCORING_FUNCTIONS = {
    "check_level1_workflow": check_level1_workflow,
    "check_level2_workflow": check_level2_workflow,
}


def get_scoring_function(name: str, params: dict | None = None) -> Callable:
    """Validate and initialize a configured scoring factory."""
    fn = SCORING_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Scoring function '{name}' not found in the registry")

    try:
        return fn(**({} if params is None else params))
    except Exception as e:
        raise ValueError(
            f"Error initializing scoring function '{name}' with params {params}: {e}"
        ) from e


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

            try:
                scoring_fn = get_scoring_function(scoring_fn_name, scoring_params)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid task '{task_id}' in {task_file}: {exc}"
                ) from exc
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
                submission_resolver=resolve_submission,
            )

    return tasks


def _md_file_tools(workspace: str) -> dict[str, Tool]:
    """MD filesystem and LAMMPS tools bound to one local workspace."""
    tools = build_workspace_tools(MDWorkspaceFilesystem(workspace))
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
        "terminal": build_md_terminal_tool(workspace),
        "execute_python_script": build_execute_python_script_tool(workspace),
        # Overrides the statically selected tool with a workspace-aware variant
        # that handles Modal upload and download internally.
        "run_lammps": build_run_lammps_tool(workspace),
        "run_verified_md": build_run_verified_md_tool(workspace),
    }


class MolecularDynamicsEnvironment(Environment):
    """Confine every agent-controlled domain-tool path to this task workspace."""

    def _ensure_seed_directories(self) -> None:
        if self.workspace_path:
            seed = json.loads(
                Path(__file__).with_name("base_workspace.json").read_text()
            )
            for name in seed["directories"]:
                (Path(self.workspace_path) / name).mkdir(parents=True, exist_ok=True)
            scorer = self.current_task.scoring_fn
            if isinstance(scorer, WorkflowScorer):
                seed_examples(self.workspace_path, scorer.task_number)

    def initial_event(self, **kwargs: Any) -> ExecutionStarted:
        """Create the standard local MD workspace before the first tool call."""
        self._ensure_seed_directories()
        return super().initial_event(**kwargs)

    def prepare_workspace(self, workspace: WorkspaceState) -> None:
        super().prepare_workspace(workspace)
        if not workspace.files:
            self._ensure_seed_directories()

    def execute_tool(
        self, state: ExecutionState, tool: Tool, arguments: dict[str, Any]
    ) -> Any:
        release = state.runtime.metadata.get("corral_md_release_id")
        volume_name = state.runtime.metadata.get("corral_md_volume_name")
        with (
            pinned_release(
                release if isinstance(release, str) else None,
                volume_name if isinstance(volume_name, str) else None,
            ),
            recovery_snapshot(state.workspace, self.workspace_manager),
        ):
            result = super().execute_tool(state, tool, arguments)
            if isinstance(result, str) and self.workspace_path:
                return result.replace(
                    str(Path(self.workspace_path).resolve()), "/workspace"
                )
            return result

    _PATH_ARGUMENTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "get_structure_from_mp_text": ("file_path",),
        "convert_structure_to_lammps_data": ("structure_path", "output_file"),
        "get_nth_run_log": ("path", "save"),
        "keyword_log_extractor": ("path",),
        "visualisation_tool": ("path",),
    }

    def preprocess_arguments(self, tool_name: str, args: dict) -> dict:
        parsed = super().preprocess_arguments(tool_name, args)
        if not self.workspace_path:
            return parsed
        for argument in self._PATH_ARGUMENTS.get(tool_name, ()):
            value = parsed.get(argument)
            if value is not None:
                parsed[argument] = str(
                    local_path(
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

    scorer = env.current_task.scoring_fn
    if isinstance(scorer, WorkflowScorer):
        prompt += example_prompt(scorer.task_number, workspace=bool(env.workspace_path))

    prompt += "\nAvailable input data:\n"

    prompt += ""

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
            "accept only absolute POSIX paths under /workspace, such as "
            "`/workspace/input/run.in`. Relative paths (including `input/run.in`), "
            "parent traversal, symbolic links and controller paths are forbidden. "
            "Use /workspace/input for inputs, /workspace/scripts for code and "
            "/workspace/output for results.\n\n"
            "### Important Resource and File Access Guidelines ###\n"
            "1. **Potential Files**:\n"
            "   - All potentials are mounted read-only below /workspace/potentials/. Hence whenever working with potential files, always use absolute paths (eg. /workspace/potentials/SW/Si.sw) as all potential files are mounted at fixed locations. Otherwise, the simulation will fail due to wrong path for the potential.\n"
            "   - These files are *fully verified and correct*.\n"
            "   - You must **not attempt to read or parse them directly**.\n"
            "   - Reading them is unnecessary and will waste important computational resources.\n\n"
            "2. **Simulation Log Files**:\n"
            "   - These files are *very large* and should **not be directly parsed**.\n"
            "   - Direct parsing would cause excessive cost and resource usage.\n\n"
            "Shared assets in /workspace/structures, /workspace/models and "
            "/workspace/potentials are read-only. You may copy a supplied structure "
            "into /workspace/input to work on it. Models are available to Python "
            "in the Modal GPU runtime at /workspace/models/teacher.model and "
            "/workspace/models/student.model. Copy any structure or potential needed "
            "by a local CPU script into the writable workspace first.\n\n"
            "### Choosing a Simulation Engine ###\n"
            "If the task calls for a MACE-family potential/model, always conduct the "
            "MD simulation via an ASE Python script (execute_python_script with "
            "use_gpu=True), not LAMMPS. "
            "For all other potentials (SW, Tersoff, EAM, BKS, etc.), always use LAMMPS "
            "via run_lammps.\n\n"
            "### Local and GPU Execution ###\n"
            "The single execute_python_script tool routes lightweight analysis, plotting, "
            "and file conversion to the local CPU by default. Set use_gpu=True only for "
            "scripts that construct an ASE Calculator backed by a MACE model or otherwise "
            "require CUDA; those calls run on an A100 in Modal. Inside local CPU scripts "
            "and terminal commands, use paths "
            "relative to the current workspace (for example `output/results.json`). "
            "Structured tool path arguments still require absolute /workspace paths. "
            "Modal GPU calls use /workspace as their working directory.\n\n"
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
        json_path = PACKAGE_DATA_ROOT / f"level_{level}" / "subtasks_json"
    else:
        json_path = PACKAGE_DATA_ROOT / f"level_{level}" / "tasks_json"

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
            if not subtask_level:
                raise FileNotFoundError(f"Task config not found: {json_path}")
            tasks = {}
        else:
            tasks = load_tasks_from_json(json_path, work_dir)
        environments = (
            build_environments(
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
            if tasks
            else {}
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
