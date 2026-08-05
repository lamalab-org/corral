from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import anyio

from corral.concurrency import EpisodeTaskCompleted

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from corral.report.results import TaskTrialResult


class NodeExecutor(Protocol):
    """Runs one DAG node and returns its trial result.

    Implementations do the real work — open a trial runtime scoped to the
    episode, run the agent, submit/surrender, close the runtime — offloading the
    blocking part to a worker thread bounded by the global concurrency limiter.
    """

    def __call__(
        self, task_id: str, trial_round: int
    ) -> Awaitable[TaskTrialResult]: ...


async def run_episode_dag(
    *,
    trial_round: int,
    ordered_ids: list[str],
    dep_graph: Mapping[str, list[str]],
    node_executor: NodeExecutor,
    unreachable_result: Callable[[str, int, str], TaskTrialResult],
    emit: Callable[[EpisodeTaskCompleted], Awaitable[None]],
) -> None:
    """Execute one episode's tasks as a concurrent dependency DAG.

    Every task in `ordered_ids` runs in its own child task. A node waits for
    each of its in-selection parents to finish; if any parent did **not**
    succeed the node is recorded as *unreachable* (via `unreachable_result`)
    without calling `node_executor`, which cascades for free to that node's own
    dependents. Otherwise `node_executor` runs the node concurrently with any
    unrelated siblings. Each finished node is reported through `emit`.

    Args:
        trial_round: The trial round this episode represents (tags each event).
        ordered_ids: The episode's task ids in topological order. Ordering is
            not required for correctness (readiness is gated on parent events)
            but keeps deterministic tie-breaking.
        dep_graph: `{task_id: [parent_ids]}` restricted to the selection, so a
            parent id always has its own node/event in this episode.
        node_executor: Async callable running a single node to a result.
        unreachable_result: Builds the result for a node whose upstream chain
            broke, given `(task_id, trial_round, missing_parent)`.
        emit: Async sink receiving one :class:`EpisodeTaskCompleted` per node.
    """
    # One completion event per node; a node's event fires only after its result
    # and success flag are recorded, giving waiters a clean happens-before.
    events: dict[str, anyio.Event] = {tid: anyio.Event() for tid in ordered_ids}
    succeeded: dict[str, bool] = {}

    async def run_node(task_id: str) -> None:
        parents = dep_graph.get(task_id, [])
        for parent in parents:
            # Defensive: only wait on parents that are part of this episode.
            if parent in events:
                await events[parent].wait()

        missing = next(
            (p for p in parents if p in events and not succeeded.get(p, False)),
            None,
        )
        if missing is not None:
            result = unreachable_result(task_id, trial_round, missing)
        else:
            result = await node_executor(task_id, trial_round)

        succeeded[task_id] = result.success
        events[task_id].set()
        await emit(EpisodeTaskCompleted(trial_round, task_id, result))

    async with anyio.create_task_group() as task_group:
        for task_id in ordered_ids:
            task_group.start_soon(run_node, task_id)
