"""Corral environment for the synthetic psychometrics tasks.

Each task exposes its public input files and a persistent analysis worker.
Generators, scorers, checks, and hidden answers stay outside the workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

from corral.core.environment import Environment, Toolset, build_environments
from corral.core.state import ExecutionState
from corral.core.task import EnvironmentSetup, InputRef, TaskDefinition, with_fixed_inputs
from corral.core.transition import ToolExecutionResult
from corral.report.logging import event, exception_fields
from corral.runtime import permissions

from psychometrics import score
from psychometrics.tools import _analysis_step, local_analysis_step, workspace_tools

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/psychometrics")
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


def _public_files(value: Any) -> Iterator[str]:
    """Yield public input filenames embedded in a task's initial input."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _public_files(item)
    elif isinstance(value, list):
        for item in value:
            yield from _public_files(item)
    elif isinstance(value, str) and value.lower().endswith((".csv", ".md")):
        yield value


def _copy_public_inputs(env: Environment, artifact_dir: Path) -> EnvironmentSetup:
    """Copy only task-declared public inputs into the isolated workspace."""
    if not env.workspace_path:
        raise RuntimeError("psychometrics tasks require a workspace")
    workspace = Path(env.workspace_path).resolve()
    names = sorted(set(_public_files(env.current_task.initial_input)))
    for name in names:
        source = (artifact_dir / name).resolve()
        destination = (workspace / name).resolve()
        if source.parent != artifact_dir.resolve() or destination.parent != workspace:
            raise ValueError(f"public task input must be a filename: {name!r}")
        if not source.is_file():
            raise FileNotFoundError(f"public task input is missing: {source}")
        shutil.copyfile(source, destination)
    return EnvironmentSetup(
        hidden_arguments={"work_dir": str(workspace), "analysis_session": None},
        status="Public task inputs copied into the isolated workspace.",
    )


class PsychometricsEnvironment(Environment):
    """Run Python analysis in a public-data-only persistent worker session."""

    def execute_tool(self, state: ExecutionState, tool: Any, arguments: dict[str, Any]) -> Any:
        if tool.name != "PythonREPL":
            return super().execute_tool(state, tool, arguments)
        if not self.workspace_path:
            raise RuntimeError("psychometrics tasks require a workspace")

        environment = dict(state.environment.values)
        hidden = dict(environment.get("hidden_arguments", {}))
        checkpoint = hidden.get("analysis_session")
        code = arguments.get("input_code")
        if permissions.enabled():
            result = permissions.run_worker(
                "tool",
                (_analysis_step, {"code": code, "checkpoint": checkpoint}),
                self.workspace_path,
            )["content"]
        else:
            result = local_analysis_step(code, checkpoint)
        hidden["analysis_session"] = result["checkpoint"]
        environment["hidden_arguments"] = hidden
        return ToolExecutionResult(
            content=result["output"],
            environment=self.capture_environment(environment),
        )


def _task_prompt(env: Environment, state: ExecutionState) -> str:
    """Add concise workspace guidance without exposing private paths."""
    del state
    task = env.current_task
    files = sorted(set(_public_files(task.initial_input)))
    prompt = (
        f"Task: {task.name}\n\n{task.description}\n\n"
        f"Required submission format:\n{task.submission_format}\n\n"
        "Available workspace files:\n"
        + "\n".join(f"- {name}" for name in files)
        + "\n\nYou may write analysis scripts and intermediate results there."
    )
    return prompt


def _scoring_function(name: str, params: dict[str, Any]) -> Callable[[Any], dict]:
    try:
        fn = SCORING_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown psychometrics scoring function: {name}") from exc
    return with_fixed_inputs(fn, params=params)


def load_tasks_from_json(json_path: str | Path) -> dict[str, TaskDefinition]:
    """Load generated task definitions and bind their scorers."""
    source = Path(json_path)
    tasks: dict[str, TaskDefinition] = {}
    for task_file in sorted(source.glob("*.json")):
        for task_info in json.loads(task_file.read_text()):
            task_id = task_info["id"]
            scoring_params = task_info.get("scoring_params", {})
            artifact_dir = (
                ROOT
                / "artifacts"
                / f"level_{task_info['level']}"
                / (f"task_{int(task_id.rsplit('_t', 1)[1].split('_', 1)[0]):02d}")
            )
            tasks[task_id] = TaskDefinition(
                name=task_info["name"],
                description=task_info["description"],
                tools=[],
                scoring_fn=_scoring_function(task_info["scoring_function"], scoring_params),
                submission_format=task_info.get("submission_format", ""),
                input_map={dep: InputRef(dep) for dep in task_info.get("input_from_tasks", [])},
                initial_input=task_info.get("initial_input", {}),
                prompt_fn=_task_prompt,
                setup_fn=lambda env, state, path=artifact_dir: _copy_public_inputs(env, path),
                resolve_answer=False,
            )
    return tasks


def create_environments(
    level: int = 1,
    work_dir: str = DEFAULT_WORK_DIR,
) -> dict[str, Environment]:
    """Create one isolated psychometrics environment per generated task."""
    started = perf_counter()
    task_path = ROOT / "environments" / f"level_{level}" / "tasks_json"
    if not task_path.is_dir():
        raise FileNotFoundError(f"task definitions not found: {task_path}")
    try:
        tasks = load_tasks_from_json(task_path)
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
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    for task_id, environment in create_environments(args.level, args.work_dir).items():
        print(
            task_id,
            [tool["function"]["name"] for tool in environment.get_available_tools()],
        )
