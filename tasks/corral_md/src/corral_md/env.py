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
from corral.runtime.tool_execution import PreparedToolCall
from corral.utils.context7_tools import get_library_documentation
from corral.workspace import (
    build_workspace_tools,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/corral_md")
# Wheels bundle task definitions inside the package; editable checkouts keep
# their canonical copies at the project root.
PACKAGE_DATA_ROOT = Path(__file__).with_name("environments")
if not PACKAGE_DATA_ROOT.is_dir():
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


def load_tasks_from_json(json_path: Path) -> dict[str, TaskDefinition]:
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
            tasks[task_id] = TaskDefinition(
                name=task_info["name"],
                description=task_info["description"],
                tools=task_info.get("tools", []),
                scoring_fn=scoring_fn,
                submission_format=task_info.get("submission_format", ""),
                input_map={
                    dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])
                },
                initial_input=task_info.get("initial_input", {}),
                prompt_fn=_md_task_prompt,
                resolve_answer=False,
                submission_resolver=resolve_submission,
            )

    return tasks


def _md_file_tools(workspace: str) -> dict[str, Tool]:
    """MD filesystem and LAMMPS tools bound to one local workspace."""
    tools = build_workspace_tools(MDWorkspaceFilesystem(workspace))
    # Preserve MD's read-only asset namespace in every file operation. Generic
    # worker filesystem tools do not know about MD's virtual asset catalogs.
    # These confined operations only read/write data; they never execute it.
    for name in (
        "list_files",
        "read_file",
        "write_file",
        "file_info",
        "copy_file",
        "cat_files",
        "grep",
    ):
        tools[name].trusted = True
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
        "get_library_documentation": get_library_documentation,
        "terminal": build_md_terminal_tool(workspace),
        "execute_python_script": build_execute_python_script_tool(workspace),
        # Simulation tools manage Modal upload and download for this workspace.
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
                seed_examples(
                    self.workspace_path, scorer.task_number, level=scorer.level
                )

    def initial_event(self, **kwargs: Any) -> ExecutionStarted:
        """Create the standard local MD workspace before the first tool call."""
        self._ensure_seed_directories()
        return super().initial_event(**kwargs)

    def prepare_workspace(self, workspace: WorkspaceState) -> None:
        super().prepare_workspace(workspace)
        if not workspace.files:
            self._ensure_seed_directories()

    def execute_controller_tool(
        self, state: ExecutionState, prepared: PreparedToolCall
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
            if prepared.trusted and prepared.tool_name in self._PATH_ARGUMENTS:
                arguments = prepared.arguments
                for argument in self._PATH_ARGUMENTS[prepared.tool_name]:
                    if arguments.get(argument) is not None:
                        arguments[argument] = str(
                            local_path(self.workspace_path, arguments[argument])
                        )
                return prepared.tool.execute(**arguments)
            return super().execute_controller_tool(state, prepared)

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
                # Keep the public path in prepared calls and durable actions.
                # Restricted workers mount /workspace; the shared executor
                # translates these values only for the local fallback.
                local_path(self.workspace_path, value)
        return parsed


def _md_task_prompt(env: Environment, state: ExecutionState) -> str:
    """Generate the MD task prompt with resource and logging guidance."""
    prompt = env._default_task_prompt(state)

    scorer = env.current_task.scoring_fn
    if isinstance(scorer, WorkflowScorer):
        prompt += example_prompt(
            scorer.task_number, workspace=bool(env.workspace_path), level=scorer.level
        )

    if env.workspace_path:
        prompt += (
            "\nShared MD assets in /workspace/structures, /workspace/potentials, "
            "and /workspace/models are read-only catalogs available through the "
            "file tools and Modal runtime. They are not mounted in the local "
            "terminal or CPU runtime. Use copy_file to copy structures and "
            "potentials into /workspace/input before local analysis. GPU scripts "
            "can use /workspace/models/teacher.model and "
            "/workspace/models/student.model directly.\n\n"
            "Run MACE simulations with ASE through execute_python_script(use_gpu=True) "
            "on Modal's A100. Use run_lammps for other potentials, including SW, "
            "Tersoff, EAM, and BKS. Leave use_gpu=False for CPU analysis, plotting, "
            "and conversion. Inside local scripts and terminal commands, use paths "
            "relative to the current workspace, such as output/results.json. "
            "Modal GPU scripts start in /workspace.\n\n"
            "Record Step, Temperature, Pressure, and Density for each simulation "
            "where applicable, at a sampling interval suitable for the task "
            "(typically 1000 timesteps). Use get_nth_run_log and "
            "keyword_log_extractor to inspect large LAMMPS logs.\n"
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
    # The shared environment loader can still pass the legacy selector.
    if subtask_level:
        raise ValueError("Corral MD has no subtasks; select level 1 or 2.")
    started = perf_counter()
    name = f"MD-level_{level}"
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark=name,
        operation="create",
    )
    # Keep controller storage stable even when callers change their directory.
    work_dir = str(Path(work_dir).expanduser().resolve())
    json_path = PACKAGE_DATA_ROOT / f"level_{level}" / "tasks_json"

    # Create environments for all tasks
    domain_tools = {
        "convert_structure_to_lammps_data": convert_structure_to_lammps_data,
        "get_potential_metadata": get_potential_metadata,
        "get_structure_from_mp_text": get_structure_from_mp_text,
        "get_nth_run_log": get_nth_run_log,
        "keyword_log_extractor": keyword_log_extractor,
        "visualisation_tool": visualisation_tool,
    }

    try:
        tasks = load_tasks_from_json(json_path)
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(
                pool=domain_tools,
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
    args = parser.parse_args()

    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)

    # Create all environments with file system tools
    environments = create_environments(
        work_dir=BASE_WORK_DIR,
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
