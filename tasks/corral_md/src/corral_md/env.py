import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from corral_md.local_runtime import assets_directory
from corral_md.modal_workspace import (
    authorized_resources,
    configured_asset_volume_name,
    pinned_release,
    recovery_snapshot,
)
from corral_md.resources import asset_descriptors, asset_relative_paths
from corral_md.scoring.evaluator import (
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.submission import resolve_local_submission, resolve_submission
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
    with_task1_recording,
)
from corral_md.workspace import MDWorkspaceFilesystem, local_path

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.events import ExecutionStarted
from corral.core.resources import (
    RESOURCE_CATALOG_METADATA_KEY,
    FileResourceDescriptor,
    ResourceCatalogMismatchError,
    resource_file_matches,
)
from corral.core.state import ExecutionState
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.core.workspace import WorkspaceState
from corral.report.logging import event, exception_fields
from corral.runtime import permissions
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


def load_tasks_from_json(
    json_path: Path, *, local: bool = False, assets_dir: str | None = None
) -> dict[str, TaskDefinition]:
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
            if local:
                scoring_params = {
                    **scoring_params,
                    "verification_backend": "local",
                    "assets_dir": assets_dir,
                }

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
                submission_resolver=(
                    resolve_local_submission if local else resolve_submission
                ),
            )

    return tasks


def _md_file_tools(
    workspace: str, *, local: bool = False, assets_dir: str | None = None
) -> dict[str, Tool]:
    """MD filesystem and LAMMPS tools bound to one local workspace."""
    tools = build_workspace_tools(MDWorkspaceFilesystem(workspace))
    resources = tuple(asset_descriptors())
    for name in (
        "list_files",
        "read_file",
        "file_info",
        "copy_file",
        "cat_files",
        "grep",
    ):
        tools[name].resources = resources
    selected = {
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
        "terminal": build_md_terminal_tool(workspace, local=local),
        "execute_python_script": build_execute_python_script_tool(
            workspace, local=local, assets_dir=assets_dir
        ),
        # Simulation tools use the backend selected for this workspace.
        "run_lammps": build_run_lammps_tool(
            workspace, local=local, assets_dir=assets_dir
        ),
        **({} if local else {"run_verified_md": build_run_verified_md_tool(workspace)}),
    }
    for name in ("terminal", "execute_python_script", "run_lammps", "run_verified_md"):
        if name in selected:
            selected[name].resources = resources
    return selected


class MolecularDynamicsEnvironment(Environment):
    """Confine every agent-controlled domain-tool path to this task workspace."""

    local_mode = False

    def ensure_file_resources_durable(
        self, names: tuple[str, ...] | None = None
    ) -> None:
        """Cache pinned Modal assets once, then use ordinary Corral resources."""
        selected = tuple(self.file_resources) if names is None else names
        relative_paths = asset_relative_paths()
        with self._resource_materialization_lock:
            for name in selected:
                if (
                    name not in self.file_resources
                    or name in self.file_resource_sources
                ):
                    continue
                descriptor = self.file_resources[name]
                store = self.workspace_manager.artifact_store
                if asyncio.run(store.contains(descriptor.blob_ref)):
                    continue
                destination = (
                    Path(self.base_work_dir)
                    / ".corral"
                    / "md-assets"
                    / descriptor.sha256
                    / descriptor.filename
                )
                if not resource_file_matches(destination, descriptor):
                    import modal

                    kind, relative = relative_paths[name].split("/", 1)
                    volume = modal.Volume.from_name(configured_asset_volume_name(kind))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        dir=destination.parent
                    ) as temporary:
                        downloaded = Path(temporary) / descriptor.filename
                        with downloaded.open("wb") as stream:
                            for chunk in volume.read_file("/" + relative):
                                stream.write(chunk)
                        if not resource_file_matches(downloaded, descriptor):
                            raise ResourceCatalogMismatchError(
                                f"MD asset {name!r} does not match its pinned descriptor"
                            )
                        downloaded.replace(destination)
                    destination.chmod(0o444)
                self.file_resource_sources[name] = destination
        super().ensure_file_resources_durable(names)

    def for_task(self, task_execution_id: str, **kwargs):
        bound = super().for_task(task_execution_id, **kwargs)
        bound.local_mode = self.local_mode
        return bound

    def _resolve_tools(self, workspace: str | None) -> dict[str, Tool]:
        tools = super()._resolve_tools(workspace)
        scorer = self.current_task.scoring_fn
        if (
            workspace
            and isinstance(scorer, WorkflowScorer)
            and scorer.task_number == 1
            and "run_lammps" in tools
        ):
            tools["run_lammps"] = with_task1_recording(
                tools["run_lammps"], workspace, level=scorer.level
            )
        return tools

    def _ensure_seed_directories(self) -> None:
        if self.workspace_path:
            seed = json.loads(
                Path(__file__).with_name("base_workspace.json").read_text()
            )
            for name in seed["directories"]:
                (Path(self.workspace_path) / name).mkdir(parents=True, exist_ok=True)

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
        # Preserve the authority frozen by central preparation. Resource names
        # alone do not authorize mounts, and the live catalog may have changed.
        catalog = state.task.environment.get(RESOURCE_CATALOG_METADATA_KEY, {})
        descriptors = {}
        for name, (_source, public_path) in prepared.resource_mounts.items():
            if name not in prepared.resources or name not in catalog:
                raise ResourceCatalogMismatchError(
                    f"Prepared MD resource {name!r} has no pinned authorization"
                )
            descriptor = FileResourceDescriptor.model_validate(catalog[name])
            if descriptor.public_path != public_path:
                raise ResourceCatalogMismatchError(
                    f"Prepared MD resource {name!r} has a different public path"
                )
            descriptors[name] = descriptor
        with (
            authorized_resources(descriptors),
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
                if (
                    tool_name == "convert_structure_to_lammps_data"
                    and argument == "structure_path"
                ):
                    authorized = getattr(self.tools.get(tool_name), "resources", ())
                    if any(
                        descriptor.public_path == value
                        for name, descriptor in self.file_resources.items()
                        if name in authorized
                    ):
                        continue
                # Keep the public path in prepared calls and durable actions.
                # Restricted workers mount /workspace; the shared executor
                # translates these values only for the local fallback.
                local_path(self.workspace_path, value)
        return parsed


def _md_task_prompt(env: Environment, state: ExecutionState) -> str:
    """Generate the MD task prompt with resource and logging guidance."""
    prompt = env._default_task_prompt(state)

    scorer = env.current_task.scoring_fn
    if env.workspace_path:
        prompt += (
            "\nShared MD assets are immutable declared resources listed under "
            "/workspace/resources. Their public paths are available to file tools "
            "and sandboxed execution. Use copy_file when an editable copy is needed. "
            "Run MACE simulations with ASE through execute_python_script(use_gpu=True); "
            "use run_lammps for SW, Tersoff, EAM, and BKS potentials. "
            "Leave use_gpu=False for CPU analysis, plotting, and conversion.\n\n"
        )
        if getattr(scorer, "task_number", None) not in (6, 7, 9):
            prompt += (
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
    local: bool = True,
    assets_dir: str | None = None,
) -> dict[str, Environment]:
    """Build MD runs with local simulation and scoring unless Modal is selected."""
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
    if type(local) is not bool:
        raise ValueError("local must be a boolean")
    if permissions.enabled() and Path("/usr/local/share/corral-md-assets").exists():
        raise ValueError(
            "Rebuild the MD image with assets under /opt/corral/assets/corral_md "
            "and remove /usr/local/share/corral-md-assets: the legacy /usr catalog "
            "is visible to workers without resource authorization."
        )
    configured_assets = assets_dir or os.getenv("CORRAL_MD_ASSETS_DIR")
    local_assets = (
        str(assets_directory(configured_assets)) if local or configured_assets else None
    )
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
        tasks = load_tasks_from_json(json_path, local=local, assets_dir=local_assets)
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=name,
            toolset=Toolset(
                pool=domain_tools,
                common=taskgroup_common_tools or {},
                workspace_factory=partial(
                    _md_file_tools, local=local, assets_dir=local_assets
                ),
            ),
            env_cls=MolecularDynamicsEnvironment,
            file_resources=asset_descriptors(),
            file_resource_sources={
                name: Path(local_assets) / relative
                for name, relative in asset_relative_paths().items()
            }
            if local_assets
            else None,
        )
        for environment in environments.values():
            environment.local_mode = local
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
