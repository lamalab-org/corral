"""Build Corral environments for inference-time policy optimization."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from corral.core import EnvironmentResourceAdapter
from corral.core.environment import (
    Environment,
    EnvironmentSetup,
    Toolset,
    default_file_tools,
)
from corral.core.task import TaskDefinition
from corral.report.logging import event, exception_fields
from inference_opt.budget import BudgetSpec, StateLedger
from inference_opt.task_prompts import task_prompt
from inference_opt.score import policy_score
from inference_opt.tools import create_tools

if TYPE_CHECKING:
    from corral.core.state import ExecutionState

__all__ = ["InferenceOptEnvironment", "create_environments", "load_tasks_from_json"]

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", ".corral/workspaces")
LEDGER_RESOURCE = "inference_state"

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
            # workspace so the scorer can find the submitted files.
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
    # The eval runner hands these to Inspect, which expects the ``/v1`` root.
    return {model: _v1_root(url) for model, url in base_urls.items()}


def _v1_root(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _seed_workspace(root: Path, policy_api: str = "primitive") -> None:
    """Create the guide, starter policy, and artifact directories."""
    (root / "revealed").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)

    guide_source = _PACKAGE_ROOT / "guide" / "policy_api.md"
    if guide_source.is_file():
        destination = root / "guide"
        destination.mkdir(parents=True, exist_ok=True)
        core = destination / "policy_api.md"
        if not core.exists():
            guide = guide_source.read_text(encoding="utf-8")
            start = "<!-- enhanced:start -->"
            end = "<!-- enhanced:end -->"
            if policy_api != "enhanced":
                before, _, remainder = guide.partition(start)
                _, _, after = remainder.partition(end)
                guide = before.rstrip() + "\n" + after.lstrip()
            else:
                guide = guide.replace(start, "").replace(end, "")
            core.write_text(guide, encoding="utf-8")

    template = _PACKAGE_ROOT / "templates" / "policy"
    policy_dir = root / "policy"
    if template.is_dir() and not (policy_dir / "policy.py").exists():
        policy_dir.mkdir(parents=True, exist_ok=True)
        for item in template.iterdir():
            if item.is_file():
                shutil.copyfile(item, policy_dir / item.name)

    for name, body in (("notes.md", "# Notes\n\nWhat I have learned about this student so far.\n"),):
        path = root / name
        if not path.exists():
            path.write_text(body, encoding="utf-8")


def _prepare_workspace(env: Environment, state: ExecutionState) -> EnvironmentSetup:
    """Expose the workspace to tools and seed its starting contents."""
    del state
    config = dict(env.current_task.initial_input)
    workspace = env.workspace_path or ""
    if workspace:
        _seed_workspace(Path(workspace), str(config.get("policy_api", "primitive")))
    return EnvironmentSetup(
        hidden_arguments={"work_dir": workspace},
        status="Policy workspace prepared with guide, starter policy and state.",
    )


def _initial_ledger(task: TaskDefinition) -> dict[str, Any]:
    budget = BudgetSpec.from_mapping(task.initial_input.get("budget"))
    return StateLedger({}, budget).state


class InferenceLedgerAdapter(EnvironmentResourceAdapter):
    """Restore and capture the budget ledger as committed JSON state."""

    def restore(self, state: Any) -> object:
        if not isinstance(state, Mapping):
            raise TypeError("inference_state resource must be an object")
        return json.loads(json.dumps(state))

    def capture(self, runtime: object) -> Any:
        if not isinstance(runtime, dict):
            raise TypeError("inference_state runtime must be a dict")
        return runtime


class InferenceOptEnvironment(Environment):
    """One inference-optimization task; the budget ledger is a Corral resource."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        adapters = dict(kwargs.pop("resource_adapters", {}) or {})
        adapters.setdefault(LEDGER_RESOURCE, InferenceLedgerAdapter())
        super().__init__(*args, resource_adapters=adapters, **kwargs)


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
        environments = {
            task_id: InferenceOptEnvironment(
                task_id=task_id,
                task=task,
                base_work_dir=work_dir,
                component_id=name,
                resource_states={LEDGER_RESOURCE: _initial_ledger(task)},
                toolset=Toolset(
                    pool=create_tools(task.initial_input, work_dir),
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
