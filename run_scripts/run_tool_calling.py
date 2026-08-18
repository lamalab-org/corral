#!/usr/bin/env python3
"""Run ToolCallingAgent against a Corral environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dotenv import load_dotenv

from corral import ENVIRONMENT_NAMES
from corral.cli import run_benchmark

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

AGENT_ID = "tool-calling"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--env-kwargs must be valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--env-kwargs must be a JSON object")
    return value


def _optional_timeout(raw: str) -> float | None:
    if raw.casefold() in {"none", "null"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timeout must be a positive number or 'none'"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return value


def _list_tasks(environments: Mapping[str, Any]) -> None:
    lines = []
    for task_id, environment in environments.items():
        task = environment.current_task
        dependencies = ", ".join(sorted(task.dependencies())) or "-"
        description = " ".join(task.description.strip().split())
        if len(description) > 120:
            description = f"{description[:117]}..."
        lines.append(f"{task_id}\tdependencies={dependencies}\t{description}")
    sys.stdout.write("\n".join(lines) + "\n")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return slug[:120] or uuid4().hex


def _print_results(run_id: str, result: Any) -> None:
    lines = [f"Run {run_id} completed:"]
    for task_id, task_results in result.task_results.items():
        for trial in task_results.trials:
            status = "failed" if trial.error_message else "completed"
            output = json.dumps(trial.output, ensure_ascii=False, default=str)
            lines.append(
                f"- {task_id} [{trial.trial_id}]: {status}, "
                f"score={trial.score:g}, output={output}"
            )
    sys.stdout.write("\n".join(lines) + "\n")


async def run(args: argparse.Namespace) -> int:
    """Preserve the legacy ToolCallingAgent command via the shared CLI path."""
    return await run_benchmark(args, agent_name=AGENT_ID)


def build_parser() -> argparse.ArgumentParser:
    root = _repository_root()
    parser = argparse.ArgumentParser(
        description="Run ToolCallingAgent with a Corral environment."
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=[],
        help=(
            "Task ID to run; repeat for multiple. If omitted, all tasks run. "
            "Dependencies are included automatically."
        ),
    )
    parser.add_argument("--list-tasks", action="store_true")

    environment = parser.add_argument_group("environment")
    environment.add_argument(
        "--environment",
        required=True,
        choices=ENVIRONMENT_NAMES,
        help="Registered environment to run.",
    )
    environment.add_argument(
        "--env-kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="JSON object configuring the selected environment.",
    )

    agent = parser.add_argument_group("agent")
    agent.add_argument("--model", default="openai/gpt-5.6-terra")
    agent.add_argument("--api-endpoint")
    agent.add_argument("--temperature", type=float, default=1.0)
    agent.add_argument("--max-iterations", type=int, default=20)
    agent.add_argument("--enable-surrender", action="store_true")

    execution = parser.add_argument_group("execution")
    execution.add_argument("--trials", type=int, default=1)
    execution.add_argument("--max-parallel", type=int, default=1)
    execution.add_argument("--max-parallel-per-task", type=int, default=1)
    execution.add_argument("--no-evaluate", action="store_true")
    execution.add_argument("--run-id")
    execution.add_argument("--task-queue")
    execution.add_argument("--temporal-address", default="localhost:7233")
    execution.add_argument("--temporal-namespace", default="default")
    execution.add_argument(
        "--activity-timeout",
        type=_optional_timeout,
        default=None,
        metavar="SECONDS|none",
        help="Start-to-Close timeout; default: none.",
    )
    execution.add_argument(
        "--heartbeat-timeout",
        type=_optional_timeout,
        default=None,
        metavar="SECONDS|none",
        help="Heartbeat timeout; default: none.",
    )
    execution.add_argument("--max-attempts", type=int, default=3)
    execution.add_argument(
        "--state-file",
        default=str(root / ".corral" / "tool-calling-states.jsonl"),
    )
    execution.add_argument("--report")
    execution.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(_repository_root() / ".env")
    parser = build_parser()
    try:
        return asyncio.run(run(parser.parse_args(argv)))
    except (ImportError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
