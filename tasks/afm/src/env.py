#!/usr/bin/env python3
import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from afm_contract import CONTRACT_VERSION, INSTRUCTIONS, reference
from afm_runtime import InstrumentUnavailable, runtime
from corral.core.environment import Environment, Toolset, build_environments
from corral.core.events import ExecutionStarted, TaskConfigured
from corral.core.state import ExecutionState
from corral.core.task import InputRef, TaskDefinition
from corral.core.tool import Tool
from corral.report.logging import event, exception_fields
from corral.utils.code_tools import execute_python_code
from score import (
    score_friction,
    score_roughness,
    score_roughness_and_friction,
    score_topography,
)
from submission import resolve_submission
from tools import (
    Code_Executor,
    Document_Retrieval,
    Image_Analyzer,
    Image_optimizer,
    scan_grain_area,
    visualize_grain_boxes,
)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt_4o").strip()
event(
    "DEBUG",
    "environment.configuration",
    subsystem="runtime",
    benchmark="afm",
    llm_model=LLM_MODEL,
)
ENVIRONMENT = "enviroment"
BASE_WORK_DIR = rf"C:\Users\Admin\Desktop\corral\corral\tasks\afm\src\afm\{LLM_MODEL}\{ENVIRONMENT}\tasks"

# Every task and trial in this process controls the same physical instrument.
_INSTRUMENT_LOCK = threading.Lock()

SCORING_FUNCTIONS = {
    "score_topography": score_topography,
    "score_roughness": score_roughness,
    "score_friction": score_friction,
    "score_roughness_and_friction": score_roughness_and_friction,
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
    event(
        "DEBUG",
        "environment.tasks_loading",
        subsystem="runtime",
        benchmark="afm",
        task_source=str(json_path),
    )
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Task definition file not found: {json_path}")

    json_path = Path(json_path)
    if json_path.is_dir() and (json_path / "tasks_json").is_dir():
        json_path = json_path / "tasks_json"
    task_files = sorted(json_path.glob("*.json")) if json_path.is_dir() else [json_path]
    task_data = {}
    for task_file in task_files:
        with task_file.open(encoding="utf-8") as f:
            entries = json.load(f)
        if isinstance(entries, list):
            entries = {entry["id"]: entry for entry in entries}
        task_data.update(entries)

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
            submission_resolver=resolve_submission,
            execution_version=CONTRACT_VERSION,
        )

    return tasks


class AFMEnvironment(Environment):
    """Environment for AFM tasks (escape hatch: drives the Nanosurf API).

    Behaviour beyond the stateful hardware reset is shared with the generic
    `Environment`; only the prompt and per-task instrument reset are
    specialised here. Evaluation uses the task's scoring callable directly.
    """

    def initial_event(self, **kwargs) -> ExecutionStarted:
        # Setup happens after execution.started. Defer the prompt until the
        # configuration event contains the verified machine identity.
        started = super().initial_event(**kwargs)
        metadata = dict(started.task)
        metadata.pop("prompt", None)
        return ExecutionStarted(**{**started.model_dump(), "task": metadata})

    def capture_environment(self, environment):
        captured = dict(super().capture_environment(environment))
        if hasattr(self, "_initial_snapshot"):
            captured["afm_initial_snapshot"] = self._initial_snapshot
        return captured

    @asynccontextmanager
    async def task_execution_guard(self) -> AsyncIterator[None]:
        """Allow one AFM task at a time, regardless of runner concurrency."""
        # Nonblocking acquisition keeps the event loop responsive and cannot
        # leave a worker thread acquiring an abandoned lock after cancellation.
        while not _INSTRUMENT_LOCK.acquire(blocking=False):  # noqa: ASYNC110
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            try:
                # The owner serializes this after any in-flight instrument call.
                # Quarantine prevents cleanup from issuing more commands after
                # an uncertain/hung call; the operator must recover in that case.
                cleanup = asyncio.create_task(
                    asyncio.to_thread(runtime().call, "finish")
                )
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # Retain task ownership until cleanup has completed/failed.
                    # A command timeout quarantines the worker instead of freeing
                    # the instrument to a second task while a COM call is live.
                    await cleanup
                    raise
            except InstrumentUnavailable as exc:
                event(
                    "ERROR",
                    "afm.cleanup_blocked",
                    subsystem="runtime",
                    benchmark="afm",
                    error=str(exc),
                )
                raise
            finally:
                _INSTRUMENT_LOCK.release()

    @property
    def initial_params(self) -> dict:
        return self.current_task.initial_input["params"]

    @property
    def afm_dir(self) -> str | None:
        return self.workspace_path

    def reset_params(self) -> None:
        # Validate the local reference before touching hardware; no provider call.
        reference("preflight")
        params = runtime().call(
            "begin",
            self.initial_params,
            self.workspace_path,
            Path(self.workspace_path).name,
        )
        self._initial_snapshot = params
        event(
            "DEBUG",
            "environment.configuration_completed",
            subsystem="runtime",
            benchmark="afm",
            task_id=self.task_id,
            workspace=self.afm_dir,
            image_directory=params["Current Working directory"],
            machine_id=params["machine_id"],
            hostname=params["hostname"],
            afm_contract_version=CONTRACT_VERSION,
        )

    def get_task_prompt(self, state: ExecutionState) -> str:
        prompt = "You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API."
        prompt += f"""\nTask Description: {self.current_task.description}
        Required submission format:
        {self.current_task.submission_format}

        """

        prompt += "\n" + INSTRUCTIONS
        snapshot = state.environment.values.get("afm_initial_snapshot")
        if snapshot is not None:
            prompt += (
                f"\nInstrument: {snapshot['machine_id']} "
                f"(host: {snapshot['hostname']}).\n"
            )
        else:
            prompt += "\nInstrument setup has not completed; no verified state is available.\n"
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

        return prompt

    def configure(self, state: ExecutionState) -> TaskConfigured:
        event(
            "DEBUG",
            "environment.configuration_started",
            subsystem="runtime",
            benchmark="afm",
            task_id=self.task_id,
        )
        self.reset_params()
        configured = super().configure(state)
        return TaskConfigured(
            **{
                **configured.model_dump(),
                "status": f"AFM instrument {runtime().config.machine_id} parameters configured.",
            }
        )


def create_environments(
    task_json_path: str | Path,
    taskgroup_common_tools: dict[str, Tool] | None = None,
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, AFMEnvironment]:
    """Create environments for tasks defined in a JSON file

    Args:
        task_json_path: Path to the JSON file with task definitions
        taskgroup_common_tools: dictionary of Tools shared by tasks, for example file system tools
        work_dir: Working directory for task execution

    Returns:
        dictionary of environments keyed by task ID
    """

    started = perf_counter()
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark="afm",
        operation="create",
    )
    event(
        "DEBUG",
        "environment.creation_details",
        subsystem="runtime",
        benchmark="afm",
        task_source=str(task_json_path),
        work_dir=work_dir,
    )

    task_specific_tools = {
        "visualize_grain_boxes": visualize_grain_boxes,
        "scan_grain_area": scan_grain_area,
        "Document_Retrieval": Document_Retrieval,
        "Image_optimizer": Image_optimizer,
        "Code_Executor": Code_Executor,
        "Image_Analyzer": Image_Analyzer,
        "execute_python_code": execute_python_code,
    }

    try:
        tasks = load_tasks_from_json(task_json_path, work_dir)
        environments = build_environments(
            tasks,
            base_work_dir=work_dir,
            name=ENVIRONMENT,
            toolset=Toolset(
                pool=task_specific_tools,
                common=taskgroup_common_tools or {},
            ),
            env_cls=AFMEnvironment,
        )
    except Exception as exc:
        event(
            "ERROR",
            "environment.failed",
            subsystem="runtime",
            benchmark="afm",
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
        benchmark="afm",
        operation="create",
        status="completed",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        environment_count=len(environments),
    )
    return environments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create AFM task environments.")
    parser.add_argument(
        "--level",
        choices=("level_1", "level_2"),
        default="level_1",
        help="Task level to load (default: level_1).",
    )
    args = parser.parse_args()
    tasks_json_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / args.level
        / "tasks_json"
    )
    event(
        "DEBUG",
        "environment.task_source_resolved",
        subsystem="runtime",
        benchmark="afm",
        task_source=str(tasks_json_path),
    )
    work_dir = BASE_WORK_DIR
    environments = create_environments(
        task_json_path=tasks_json_path,
        work_dir=work_dir,
    )
    for env_id, env in environments.items():
        event(
            "DEBUG",
            "environment.created",
            subsystem="runtime",
            benchmark="afm",
            task_id=env_id,
            task_name=env.current_task.name,
            dependencies=sorted(env.current_task.dependencies()),
        )
