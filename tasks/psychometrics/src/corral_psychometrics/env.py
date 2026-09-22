"""Corral environment for the psychometrics tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import ExecutionState
from corral.core.task import EnvironmentSetup, InputRef, TaskDefinition, with_fixed_inputs
from corral.core.transition import ToolExecutionResult
from corral.report.logging import event, exception_fields
from corral.runtime import permissions
from corral.tools.python_repl import PythonREPLTool

from corral_psychometrics import paths, score
from corral_psychometrics.tools import workspace_tools

SCORING_FUNCTIONS: dict[str, Callable[..., dict]] = {
    name: getattr(score, name)
    for name in (
        "score_model_criteria",
        "score_behavioral_validity",
        "score_misfit_replication",
        "score_adaptive_bank_choice",
        "score_model_identification",
        "score_population_classification",
        "score_gender_item_integrity",
    )
}


def _verify_checksums(
    task_id: str, task_info: dict[str, Any], scoring_params: dict[str, Any], root: Path
) -> None:
    """Check a task's files against the checksums recorded for them.

    Verifies that every public input hashes to its recorded value, and that the definition and the answer key were built from the same dataset.
    """
    declared = task_info.get("initial_input", {}).get("data_sha256")
    data_dir = paths.resolve(scoring_params["data_dir"], root=root)
    checksums = dict(task_info.get("public_sha256", {}))
    if "public_sha256" in task_info:
        if set(checksums) != set(task_info["initial_input"]["public_inputs"]):
            raise ValueError(f"{task_id}: checksums must cover every public input")
    if declared:
        if isinstance(declared, dict):
            dataset_checksums = declared
        else:
            inputs = task_info["initial_input"]
            filename = scoring_params.get("dataset") or inputs.get("datasets", ["data.csv"])[0]
            dataset_checksums = {filename: declared}
        for filename, expected in dataset_checksums.items():
            if filename in checksums and checksums[filename] != expected:
                raise RuntimeError(f"{task_id}: inconsistent checksums for {filename}")
            checksums[filename] = expected
    for filename, expected in checksums.items():
        file = paths.resolve(filename, root=data_dir)
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{task_id}: checksum mismatch for {filename}; rebuild the task")
    truth_path = scoring_params.get("truth_path")
    if not declared or not truth_path:
        return
    recorded = json.loads(paths.resolve(truth_path, root=root).read_text())
    answered = recorded.get("provenance", {}).get("data_sha256")
    if not answered:
        return
    # A task with several datasets records one checksum per file; its answer
    # key records the checksum of the single dataset it was built from.
    declared_checksums = set(declared.values()) if isinstance(declared, dict) else {declared}
    if answered not in declared_checksums:
        raise RuntimeError(
            f"{task_id}: the task definition and its answer key were built from "
            f"different datasets (key records {answered[:12]}, definition has "
            f"{', '.join(sorted(item[:12] for item in declared_checksums))}). "
            "Rebuild with `python build.py`."
        )


def _copy_public_inputs(
    env: Environment, data_dir: Path, declared: tuple[str, ...]
) -> EnvironmentSetup:
    """Copy a task's public inputs into its isolated workspace (usually the dataset and codebook).

    Fails unless `data_dir` holds exactly the declared files.
    """
    if not env.workspace_path:
        raise RuntimeError("psychometrics tasks require a workspace")
    workspace = Path(env.workspace_path).resolve()
    data_dir = data_dir.resolve()

    declared = set(declared)
    present = {entry.name for entry in data_dir.iterdir() if entry.is_file()}
    if undeclared := present - declared:
        raise ValueError(
            f"{data_dir} holds files the task does not declare as inputs: "
            f"{sorted(undeclared)}. Public data and the answer key must stay "
            "in separate trees."
        )
    if missing := declared - present:
        raise FileNotFoundError(
            f"declared task inputs are missing from {data_dir}: {sorted(missing)}"
        )

    for name in sorted(declared):
        source = (data_dir / name).resolve()
        destination = (workspace / name).resolve()
        if source.parent != data_dir or destination.parent != workspace:
            raise ValueError(f"public task input must be a filename: {name!r}")
        shutil.copyfile(source, destination)
    return EnvironmentSetup(
        hidden_arguments={"work_dir": str(workspace), "analysis_session": None},
        status="Public task inputs copied into the isolated workspace.",
    )


class PsychometricsEnvironment(Environment):
    """Dispatches PythonREPL, keeping its session checkpoint in task state."""

    def execute_tool(self, state: ExecutionState, tool: Any, arguments: dict[str, Any]) -> Any:
        if tool.name != "PythonREPL":
            return super().execute_tool(state, tool, arguments)
        if not isinstance(tool, PythonREPLTool):
            raise TypeError("PythonREPL must use Corral's PythonREPLTool")
        if not self.workspace_path:
            raise RuntimeError("psychometrics tasks require a workspace")

        environment = dict(state.environment.values)
        hidden = dict(environment.get("hidden_arguments", {}))
        checkpoint = hidden.get("analysis_session")
        code = arguments["input_code"]

        if permissions.enabled():
            result = tool.execute_repl(
                code=code, checkpoint=checkpoint, workspace=self.workspace_path
            )
            output, checkpoint = result.output, result.checkpoint
        else:
            session = tool.create_session()
            try:
                session.restore(checkpoint)
                output = session.execute(code)
                checkpoint = session.snapshot()
            finally:
                session.close()

        hidden["analysis_session"] = checkpoint
        environment["hidden_arguments"] = hidden
        return ToolExecutionResult(
            content=output,
            environment=self.capture_environment(environment),
        )


def _task_prompt(env: Environment, state: ExecutionState) -> str:
    """The task text the agent sees, with its workspace file list."""
    del state
    task = env.current_task
    files = sorted(task.initial_input.get("public_inputs", ()))
    prompt = (
        f"Task: {task.name}\n\n{task.description}\n\n"
        f"Required submission format:\n{task.submission_format}\n\n"
        "Available workspace files:\n" + "\n".join(f"- {name}" for name in files)
    )
    return prompt


def _scoring_function(name: str, params: dict[str, Any], root: Path) -> Callable[[Any], dict]:
    """Bind a scorer to its task's rules and the root its paths resolve against."""
    try:
        fn = SCORING_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown psychometrics scoring function: {name}") from exc
    return with_fixed_inputs(fn, params=params, base_dir=root)


def load_tasks_from_json(
    json_path: str | Path, *, data_root: str | Path | None = None
) -> dict[str, TaskDefinition]:
    """Load generated task definitions and bind their scorers.

    Pass `data_root` to load definitions from a dataset other than the default.
    """
    root = paths.task_root(data_root)
    source = Path(json_path)
    if list(source.glob("*.building")):
        raise RuntimeError(f"unfinished task publication in {source}; rebuild the affected task")
    if not source.is_dir() or not list(source.glob("*.json")):
        raise FileNotFoundError(f"no task definitions found in {source}")
    tasks: dict[str, TaskDefinition] = {}
    for task_file in sorted(source.glob("*.json")):
        for task_info in json.loads(task_file.read_text()):
            task_id = task_info["id"]
            if task_id in tasks:
                raise ValueError(f"duplicate psychometrics task id: {task_id}")
            scoring_params = task_info.get("scoring_params", {})
            data_dir = paths.resolve(scoring_params["data_dir"], root=root)
            declared = tuple(task_info["initial_input"]["public_inputs"])
            _verify_checksums(task_id, task_info, scoring_params, root)
            tasks[task_id] = TaskDefinition(
                name=task_info["name"],
                description=task_info["description"],
                tools=[],
                scoring_fn=_scoring_function(task_info["scoring_function"], scoring_params, root),
                submission_format=task_info.get("submission_format", ""),
                input_map={dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])},
                initial_input=task_info.get("initial_input", {}),
                prompt_fn=_task_prompt,
                setup_fn=lambda env, state, path=data_dir, names=declared: _copy_public_inputs(
                    env, path, names
                ),
                resolve_answer=False,
            )
    return tasks


def create_environments(
    level: int = 1,
    work_dir: str | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Environment]:
    """Create one isolated psychometrics environment per generated task.

    `data_root`  which built dataset to read; defaults to the source checkout
    `work_dir`   where the agents' writable workspaces go
    """
    started = perf_counter()
    root = paths.task_root(data_root)
    work_dir = work_dir or os.environ.get("CORRAL_WORK_DIR", str(root / "CORRAL_WORK_DIR"))
    task_path = paths.tasks_json_dir(level, root=root)
    if not task_path.is_dir():
        raise FileNotFoundError(
            f"task definitions not found: {task_path}. Generate them with "
            "`python build.py` from the task root."
        )
    try:
        tasks = load_tasks_from_json(task_path, data_root=root)
        return build_environments(
            tasks,
            base_work_dir=str(Path(work_dir).expanduser().resolve()),
            name=f"psychometrics-level-{level}",
            toolset=Toolset(pool={}, workspace_factory=workspace_tools),
            env_cls=PsychometricsEnvironment,
        )
    except Exception as exc:
        event(
            "ERROR",
            "environment.failed",
            subsystem="runtime",
            benchmark="psychometrics",
            operation="create",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            **exception_fields(exc),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect psychometrics environments")
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    environments = create_environments(args.level, args.work_dir, args.data_root)
    for task_id, environment in environments.items():
        print(
            task_id,
            [tool["function"]["name"] for tool in environment.get_available_tools()],
        )
