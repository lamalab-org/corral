"""Build Corral environments for inference-time policy optimization."""

from __future__ import annotations

import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from corral.core import ToolExecutionResult
from corral.core.environment import (
    Environment,
    Toolset,
    default_file_tools,
)
from corral.core.task import EnvironmentSetup, TaskDefinition
from corral.report.logging import event, exception_fields
from inference_opt.task_prompts import task_prompt
from inference_opt.score import policy_score
from inference_opt.tools import create_tools

if TYPE_CHECKING:
    from corral.core.state import ExecutionState

__all__ = ["InferenceOptEnvironment", "create_environments", "load_tasks_from_json"]

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", ".corral/workspaces")

_PACKAGE_ROOT = Path(__file__).resolve().parent


def load_tasks_from_json(
    json_path: str | Path, work_dir: str
) -> dict[str, TaskDefinition]:
    """Load task definitions and bind each to its scorer."""
    path = Path(json_path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no task definitions found at {path}")

    entries: list[dict[str, Any]] = []
    for file in files:
        value = json.loads(file.read_text(encoding="utf-8"))
        entries.extend(value if isinstance(value, list) else [value])

    tasks: dict[str, TaskDefinition] = {}
    for entry in entries:
        config = dict(entry["initial_input"])
        config["base_urls"] = _bind_endpoints(
            config.get("models", []), config.get("base_urls")
        )
        tasks[entry["id"]] = TaskDefinition(
            name=entry["name"],
            description=entry["description"],
            tools=list(entry.get("tools", [])),
            scoring_fn=policy_score(config, work_dir),
            submission_format=entry.get("submission_format", "submission.json"),
            initial_input=config,
            prompt_fn=task_prompt,
            setup_fn=_prepare_workspace,
            # The scorer needs the path resolved against the re-materialised
            # workspace; that resolution is the only channel telling it where
            # the agent's files ended up.
            resolve_answer=True,
        )
    return tasks


def _bind_endpoints(models: list[str], configured: Any) -> dict[str, str]:
    """Resolve task endpoints from config, falling back to startup environment."""
    base_urls = dict(configured or {})
    default = os.environ.get("CORRAL_VLLM_URL", "http://127.0.0.1:8000")
    for model in models:
        key = "CORRAL_VLLM_URL_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
        base_urls.setdefault(model, os.environ.get(key, default))
    return base_urls


def _seed_workspace(root: Path) -> None:
    """Put the guide, a runnable starter policy, and the state dir in place.

    The agent starts from something that already runs. Without it the first two
    experiments go on discovering the contract rather than on strategy, which is
    not what this environment is trying to measure.
    """
    (root / "revealed").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)

    guide_source = _PACKAGE_ROOT / "guide"
    if guide_source.is_dir():
        destination = root / "guide"
        destination.mkdir(parents=True, exist_ok=True)
        for document in guide_source.glob("*.md"):
            target = destination / document.name
            if not target.exists():
                shutil.copyfile(document, target)

    template = _PACKAGE_ROOT / "templates" / "policy"
    policy_dir = root / "policy"
    if template.is_dir() and not (policy_dir / "policy.py").exists():
        policy_dir.mkdir(parents=True, exist_ok=True)
        for item in template.iterdir():
            if item.is_file():
                shutil.copyfile(item, policy_dir / item.name)

    for name, body in (
        ("notes.md", "# Notes\n\nWhat I have learned about this student so far.\n"),
        (
            "TODO.md",
            "# TODO\n\n- [ ] Read guide/policy_api.md\n- [ ] Dry-run the starter policy\n",
        ),
    ):
        path = root / name
        if not path.exists():
            path.write_text(body, encoding="utf-8")


def _prepare_workspace(env: Environment, state: ExecutionState) -> EnvironmentSetup:
    """Expose the workspace to tools and seed its starting contents."""
    del state
    workspace = env.workspace_path or ""
    if workspace:
        _seed_workspace(Path(workspace))
    config = dict(env.current_task.initial_input)
    budget = dict(config.get("budget") or {})
    inference_state = {
        "experiments": 0,
        "debug_runs": 0,
        "student_calls": 0,
        "reveals": 0,
        "probe_calls": 0,
        "revealed_ids": [],
        "runs": [],
        "best_run_id": None,
        "limits": {
            "max_experiments": int(budget.get("max_experiments", 20)),
            "max_debug_runs": int(budget.get("max_debug_runs", 10)),
            "max_student_calls": int(budget.get("max_student_calls", 250)),
            "max_reveals": int(budget.get("max_reveals", 4)),
            "max_probe_calls": int(budget.get("max_probe_calls", 30)),
            "reveal_batch": int(budget.get("reveal_batch", 5)),
        },
    }
    return EnvironmentSetup(
        hidden_arguments={"work_dir": workspace, "inference_state": inference_state},
        status="Policy workspace prepared with guide, starter policy and state.",
    )


class InferenceOptEnvironment(Environment):
    """Run inference tools and return their updated session state."""

    def execute_tool(self, state: ExecutionState, tool, arguments):
        if "inference_state" not in tool.hidden_args:
            return super().execute_tool(state, tool, arguments)
        session = deepcopy(arguments["inference_state"])
        raw = tool.execute(**{**arguments, "inference_state": session})
        environment = {
            **dict(state.environment.values),
            "hidden_arguments": {
                **dict(state.environment.values.get("hidden_arguments", {})),
                "inference_state": session,
            },
        }
        return ToolExecutionResult(content=raw, environment=environment)


def create_environments(
    *,
    local_dir: str | Path | None = None,
    level: int = 1,
    task_type: str = "task",
    work_dir: str = BASE_WORK_DIR,
) -> dict[str, Environment]:
    """Create the inference-optimization environments for one level."""
    root = Path(__file__).resolve().parents[1]
    if local_dir is not None:
        source = Path(local_dir)
    else:
        kind = "subtasks_json" if task_type == "subtask" else "tasks_json"
        source = root / "environments" / f"level_{level}" / kind

    name = f"inference_opt_level_{level}"
    started = perf_counter()
    event(
        "INFO",
        "environment.started",
        subsystem="runtime",
        benchmark="inference_opt",
        operation="create",
    )
    try:
        tasks = load_tasks_from_json(source, work_dir)
        # Built one at a time rather than through `build_environments`, because
        # each task binds its tools to its own benchmark and student. A single
        # shared pool would bind every task to whichever was configured last.
        environments = {
            task_id: InferenceOptEnvironment(
                task_id=task_id,
                task=task,
                base_work_dir=work_dir,
                component_id=name,
                toolset=Toolset(
                    pool=create_tools(task.initial_input, work_dir),
                    # The agent writes policy.py with these. Without workspace
                    # tools it cannot produce a submission at all.
                    workspace_factory=default_file_tools,
                ),
            )
            for task_id, task in tasks.items()
        }
    except Exception as exc:
        event(
            "ERROR",
            "environment.failed",
            subsystem="runtime",
            benchmark="inference_opt",
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
        benchmark="inference_opt",
        operation="create",
        status="completed",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        environment_count=len(environments),
    )
    return environments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect inference_opt environments")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--task-type", default="task", choices=["task", "subtask"])
    args = parser.parse_args()

    for task_id, environment in create_environments(
        level=args.level, task_type=args.task_type
    ).items():
        task = environment.current_task
        print(
            f"{task_id:24} {task.initial_input['benchmark']:14} "
            f"models={task.initial_input['models']} tools={len(environment.tools)}"
        )
