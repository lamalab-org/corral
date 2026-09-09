"""Authorization-aware projections for agent decisions and trace inspection."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, JsonValue

from corral.core._immutable import FrozenModel
from corral.core.state import (
    ActionState,
    AgentRunState,
    ExecutionState,
    TaskState,
    ToolInvocationState,
)


class AgentContext(FrozenModel):
    """The subset of an execution an agent is authorized to observe."""

    execution_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    through_commit_hash: str
    agent_run: AgentRunState
    task: TaskState
    messages: tuple[Mapping[str, JsonValue], ...] = ()
    actions: tuple[ActionState, ...] = ()
    tool_invocations: tuple[ToolInvocationState, ...] = ()
    imported_context: tuple[Mapping[str, JsonValue], ...] = ()


class TraceAccessError(PermissionError):
    """The requesting run cannot access the selected agent trace."""


def is_descendant(
    state: ExecutionState, *, descendant_run_id: str, ancestor_run_id: str
) -> bool:
    current = state.agent_runs.get(descendant_run_id)
    seen: set[str] = set()
    while current is not None and current.parent_run_id is not None:
        if current.parent_run_id == ancestor_run_id:
            return True
        if current.parent_run_id in seen:
            return False
        seen.add(current.parent_run_id)
        current = state.agent_runs.get(current.parent_run_id)
    return False


class AgentContextResolver:
    """Resolve private contexts and enforce parent/descendant trace access."""

    @staticmethod
    def _decision_order_messages(
        state: ExecutionState,
        agent_run_id: str,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Order parallel results for provider context, not ledger chronology."""
        messages = tuple(state.conversations.get(agent_run_id, ()))
        group_order: dict[str, dict[str, int]] = {}
        for action in state.actions.values():
            group = action.parallel_group_id
            if action.requested_by_run_id != agent_run_id or group is None:
                continue
            order = group_order.setdefault(group, {})
            order[action.action.id] = len(order)

        ordered: list[Mapping[str, JsonValue]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            metadata = message.get("metadata")
            group = (
                metadata.get("parallel_group_id")
                if isinstance(metadata, Mapping)
                else None
            )
            if message.get("role") != "tool" or not isinstance(group, str):
                ordered.append(message)
                index += 1
                continue
            batch: list[Mapping[str, JsonValue]] = []
            while index < len(messages):
                candidate = messages[index]
                candidate_metadata = candidate.get("metadata")
                candidate_group = (
                    candidate_metadata.get("parallel_group_id")
                    if isinstance(candidate_metadata, Mapping)
                    else None
                )
                if candidate.get("role") != "tool" or candidate_group != group:
                    break
                batch.append(candidate)
                index += 1
            positions = group_order.get(group, {})
            ordered.extend(
                sorted(
                    batch,
                    key=lambda item: positions.get(
                        str(item.get("tool_call_id")), 2**31
                    ),
                )
            )
        return tuple(ordered)

    def for_agent(
        self,
        state: ExecutionState,
        agent_run_id: str,
        *,
        through_hash: str | None = None,
    ) -> AgentContext:
        if through_hash is not None and through_hash != state.through_commit_hash:
            raise ValueError(
                "the supplied projection was not materialized through through_hash"
            )
        run = state.agent_runs.get(agent_run_id)
        if run is None:
            raise TraceAccessError(f"agent run {agent_run_id!r} does not exist")
        actions = tuple(
            action
            for action in state.actions.values()
            if action.requested_by_run_id == agent_run_id
        )
        action_ids = {action.action.id for action in actions}
        invocations = tuple(
            invocation
            for invocation in state.tool_invocations.values()
            if invocation.action_id in action_ids
        )
        return AgentContext(
            execution_id=state.execution_id,
            branch_id=state.branch_id,
            through_commit_hash=state.through_commit_hash,
            agent_run=run,
            task=state.task,
            messages=self._decision_order_messages(state, agent_run_id),
            actions=actions,
            tool_invocations=invocations,
            imported_context=run.imported_context,
        )

    def inspect(
        self,
        state: ExecutionState,
        *,
        requester_run_id: str,
        target_run_id: str,
    ) -> AgentContext:
        if requester_run_id != target_run_id and not is_descendant(
            state,
            descendant_run_id=target_run_id,
            ancestor_run_id=requester_run_id,
        ):
            raise TraceAccessError(
                "an agent may inspect only its own trace or a descendant trace"
            )
        return self.for_agent(state, target_run_id)


__all__ = [
    "AgentContext",
    "AgentContextResolver",
    "TraceAccessError",
    "is_descendant",
]
