from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import JsonValue

    from corral.core.environment import Environment
    from corral.core.state import ExecutionState


@dataclass(frozen=True)
class EnvironmentSetup:
    """Serializable values produced while configuring one task execution."""

    hidden_arguments: Mapping[str, JsonValue] = field(default_factory=dict)
    values: Mapping[str, JsonValue] = field(default_factory=dict)
    status: str = "Additional apps/services configured for this task."


@dataclass(frozen=True)
class InputRef:
    """Reference to the output of another task.

    `key` selects which output field the downstream task receives and supports
    dotted paths into structured outputs (e.g. `"answer.smiles"`).
    """

    task_id: str
    key: str = "answer"


@dataclass(frozen=True)
class TaskDefinition:
    """Definition of a task with its requirements and evaluation hook.

    Tasks are immutable configuration describing the static DAG: dependencies
    are declared per input field via `input_map`. Runtime inputs and outputs
    live in commit-materialized :class:`corral.core.state.ExecutionState`
    projections.

    Behaviour is injected through the (immutable, callable) hooks rather than
    by subclassing the environment:

    - `prompt_fn(env, state)` overrides the default task prompt.
    - `setup_fn(env, state)` returns serializable environment values and hidden
      tool arguments for a task execution; it never mutates the environment or
      execution projection.
    - `scoring_fn(answer)` is consumed by an evaluation-layer `TaskScorer`;
      task execution never invokes it.
    - `resolve_answer` controls whether the submitted answer is path-resolved
      for evaluation (off for non-file answers such as numbers, SMILES or
      JSON). Runtime outputs always retain the submitted value itself.
    """

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable[[Any], float]
    submission_format: dict[str, str]
    # Tool names to withhold even if named in `tools` (or in the whole pool when
    # a `Toolset` selects all for an empty `tools` list). Honoured generically
    # by `Toolset.resolve`; defaults to "exclude nothing".
    excluded_tools: tuple[str, ...] = ()
    scoring_inputs: dict[str, Any] = field(default_factory=dict)
    # Static values available to the task
    initial_input: dict[str, Any] = field(default_factory=dict)
    # Named inputs that come from other tasks' outputs
    input_map: dict[str, InputRef] = field(default_factory=dict)
    # Optional behaviour hooks (immutable config, never mutated at runtime)
    prompt_fn: Callable[[Environment, ExecutionState], str | list[dict]] | None = None
    setup_fn: (
        Callable[[Environment, ExecutionState], EnvironmentSetup | None] | None
    ) = None
    resolve_answer: bool = True

    def dependencies(self) -> set[str]:
        return {ref.task_id for ref in self.input_map.values()}


def with_fixed_inputs(
    scoring_fn: Callable[..., float], **fixed: Any
) -> Callable[[Any], float]:
    """Adapt a scoring function to the single-argument form `score(answer)`.

    Extra keyword arguments (e.g. `ground_truth` or `target`) are bound up
    front so an evaluation-layer scorer can always call `scoring_fn(answer)`. The
    original function's name/docstring are preserved so documentation
    generation still reports the underlying scorer.
    """

    @functools.wraps(scoring_fn)
    def _scorer(answer: Any) -> float:
        return scoring_fn(answer, **fixed)

    return _scorer


def build_dependency_graph(
    tasks: Mapping[str, TaskDefinition],
) -> dict[str, list[str]]:
    """Derive the task dependency graph from a collection of task definitions."""
    return {task_id: sorted(task.dependencies()) for task_id, task in tasks.items()}


def assert_dependencies_selected(
    task_ids: list[str], graph: Mapping[str, list[str]]
) -> None:
    """Raise if a selected task depends on one not in the selection.

    A run must be *dependency-closed*: every dependency of every selected task
    is also selected. Otherwise a task could never have its inputs satisfied,
    which previously surfaced as `NOT YET AVAILABLE` text inside a prompt.
    Raising here turns that into a clear, up-front error instead.
    """
    selected = set(task_ids)
    missing = {
        tid: [d for d in graph.get(tid, []) if d not in selected] for tid in task_ids
    }
    missing = {tid: deps for tid, deps in missing.items() if deps}
    if missing:
        lines = "; ".join(f"{t} needs {d}" for t, d in missing.items())
        raise ValueError(
            f"Selected tasks are not dependency-closed: {lines}. "
            "Add the missing dependencies to task_ids or run the full set."
        )


def order_selected(task_ids: list[str], graph: Mapping[str, list[str]]) -> list[str]:
    """Topologically order the selected ids, honouring only intra-selection edges.

    Operates on a plain `{task_id: [deps]}` adjacency dict (the form the
    runner receives over HTTP), so it does not need `TaskDefinition`s. Edges
    pointing outside the selection are ignored. Raises `ValueError` on cycles.
    """
    selected = set(task_ids)
    temporary: set[str] = set()
    permanent: set[str] = set()
    ordered: list[str] = []

    def visit(tid: str) -> None:
        if tid in permanent:
            return
        if tid in temporary:
            raise ValueError(f"Cycle detected involving task {tid!r}")
        temporary.add(tid)
        for dep in graph.get(tid, []):
            if dep in selected:
                visit(dep)
        temporary.discard(tid)
        permanent.add(tid)
        ordered.append(tid)

    for tid in task_ids:
        visit(tid)
    return ordered


def topological_order(tasks: Mapping[str, TaskDefinition]) -> list[str]:
    """Return task ids in dependency order; raises ValueError on cycles."""
    # Delegate to the graph-based implementation so there is one ordering algorithm.
    return order_selected(list(tasks), build_dependency_graph(tasks))


def connected_components(tasks: Mapping[str, TaskDefinition]) -> list[list[str]]:
    """Group task ids into weakly-connected components of the dependency graph.

    A "group" of chained tasks is exactly a connected component; tasks with no
    edges form singleton components (today's "single task"). The framework
    derives this instead of the author declaring a `group_id`.
    """
    parent = {task_id: task_id for task_id in tasks}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for task_id, task in tasks.items():
        for dep_id in task.dependencies():
            if dep_id in parent:
                union(task_id, dep_id)

    order = list(tasks)
    members: dict[str, list[str]] = {}
    for task_id in order:
        members.setdefault(find(task_id), []).append(task_id)

    # Preserve the original task insertion order, both of components and within.
    components: list[list[str]] = []
    seen: set[str] = set()
    for task_id in order:
        root = find(task_id)
        if root not in seen:
            seen.add(root)
            components.append(members[root])
    return components


def validate_task_graph(tasks: Mapping[str, TaskDefinition]) -> None:
    """Check that all dependencies exist and the graph has no cycles."""
    known = set(tasks)

    for task_id, task in tasks.items():
        missing = task.dependencies() - known
        if missing:
            raise ValueError(
                f"Task {task_id!r} depends on unknown task(s): {sorted(missing)}"
            )

    topological_order(tasks)  # raises on cycles
