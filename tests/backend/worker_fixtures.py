"""Picklable env builders for the process-worker backend tests.

These live in a standalone importable module (not the test file) because the
worker pool spawns child processes: `multiprocessing`'s `spawn` start method
pickles the `build_envs` callable *by reference* and re-imports its module in
the child. A function defined in a `test_*.py` collected under
`--import-mode=importlib` is not importable by a plain module name, so it
could not be unpickled in the worker. The sibling `conftest.py` puts this
directory on `sys.path` (inherited by spawned children), so the child can
`import worker_fixtures` and rebuild the environments.
"""

from typing import Literal

from pydantic import Field

from corral.backend.env import Toolset, build_environments
from corral.backend.task import InputRef, TaskDefinition
from corral.backend.tool import tool


def calc(
    operation: Literal["add", "mul"] = Field(description="operation to perform"),
    x: float = Field(description="first operand"),
    y: float = Field(default=1.0, description="second operand"),
) -> str:
    """Perform a basic math operation."""
    return str(x + y if operation == "add" else x * y)


def boom(message: str = Field(default="boom", description="failure message")) -> str:
    """A tool whose execution always raises (to exercise error handling)."""
    raise ValueError(message)


def _process_task(task_id: str, tools: list[str]) -> TaskDefinition:
    return TaskDefinition(
        name=task_id,
        description="a task",
        tools=tools,
        scoring_fn=lambda answer: 1.0,
        submission_format={},
    )


def _process_toolset() -> Toolset:
    # No workspace tools: the trial runs without a base work dir, so the worker
    # needs nothing on disk to reconstruct.
    return Toolset(
        pool={"calc": tool(calc), "boom": tool(boom)}, workspace_factory=None
    )


def build_process_calc_envs():
    """Two independent `concurrency="process"` tasks exposing calc + boom."""
    tasks = {
        "task_a": _process_task("task_a", ["calc", "boom"]),
        "task_b": _process_task("task_b", ["calc", "boom"]),
    }
    return build_environments(tasks, toolset=_process_toolset(), concurrency="process")


def build_process_chain_envs():
    """A two-task `concurrency="process"` chain: downstream consumes upstream."""
    upstream = TaskDefinition(
        name="upstream",
        description="produce a value",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    downstream = TaskDefinition(
        name="downstream",
        description="consume the value",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        input_map={"from_upstream": InputRef("upstream", "answer")},
    )
    return build_environments(
        {"upstream": upstream, "downstream": downstream},
        toolset=Toolset(workspace_factory=None),
        concurrency="process",
    )
