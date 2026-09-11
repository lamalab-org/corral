"""Deterministic materialization of the execution commit log."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.events import (
    SPAWN_SUBAGENT_TOOL_NAME,
    AgentCompleted,
    AgentSpawned,
    AgentStarted,
    AgentStateUpdated,
    AgentTurnRecorded,
    ContextImported,
    EnvironmentOperation,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    ParallelGroupCompleted,
    RuntimeUpdate,
    SubmissionAccepted,
    TaskConfigured,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    UsageDelta,
    WorkspaceDelta,
)
from corral.core.state import (
    ActionState,
    AgentRunState,
    EnvironmentState,
    ExecutionState,
    RuntimeState,
    TaskOutput,
    TaskState,
    ToolInvocationState,
    UsageState,
)
from corral.core.workspace import WorkspaceState

if TYPE_CHECKING:
    from pydantic import JsonValue

    from corral.core.commit import Commit


class ReducerError(ValueError):
    """An event cannot be applied to the selected projection."""


class SharedStateConflictError(ReducerError):
    """A shared environment/workspace precondition is stale."""


def _replace(model: Any, **changes: Any) -> Any:
    values = model.model_dump(mode="python")
    values.update(changes)
    return type(model).model_validate(values)


def _runtime_after(current: RuntimeState, update: RuntimeUpdate | None) -> RuntimeState:
    if update is None:
        return current
    return RuntimeState(
        status=update.status or current.status,
        started_at=update.started_at or current.started_at,
        ended_at=update.ended_at or current.ended_at,
        metadata={**dict(current.metadata), **dict(update.metadata)},
    )


def _usage_after(current: UsageState, delta: UsageDelta) -> UsageState:
    return UsageState(
        input_tokens=current.input_tokens + delta.input_tokens,
        output_tokens=current.output_tokens + delta.output_tokens,
        reasoning_tokens=current.reasoning_tokens + delta.reasoning_tokens,
        llm_calls=current.llm_calls + delta.llm_calls,
        tool_calls=current.tool_calls + delta.tool_calls,
        agent_steps=current.agent_steps + delta.agent_steps,
    )


def _add_usage(
    usage_by_run: Mapping[str, UsageState], run_id: str, delta: UsageDelta
) -> dict[str, UsageState]:
    values = dict(usage_by_run)
    values[run_id] = _usage_after(values.get(run_id, UsageState()), delta)
    return values


def _apply_operation(
    root: MutableMapping[str, Any], operation: EnvironmentOperation
) -> None:
    cursor: MutableMapping[str, Any] = root
    for part in operation.path[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ReducerError(
                f"environment path {operation.path!r} traverses a non-object value"
            )
        cursor = child
    leaf = operation.path[-1]
    if operation.operation == "set":
        cursor[leaf] = json.loads(json.dumps(operation.value, allow_nan=False))
    else:
        cursor.pop(leaf, None)


def _environment_after(
    current: EnvironmentState,
    operations: tuple[EnvironmentOperation, ...],
    expected_revision: int | None,
) -> EnvironmentState:
    if expected_revision is not None and current.revision != expected_revision:
        raise SharedStateConflictError(
            "environment revision conflict: "
            f"expected {expected_revision}, found {current.revision}"
        )
    if not operations:
        return current
    values = json.loads(json.dumps(current.values, allow_nan=False))
    for operation in operations:
        _apply_operation(values, operation)
    return EnvironmentState(revision=current.revision + 1, values=values)


def _workspace_after(
    current: WorkspaceState,
    delta: WorkspaceDelta | None,
    expected_revision: int | None,
) -> WorkspaceState:
    if expected_revision is not None and current.revision != expected_revision:
        raise SharedStateConflictError(
            "workspace revision conflict: "
            f"expected {expected_revision}, found {current.revision}"
        )
    if delta is None:
        return current
    if delta.workspace_id != current.id:
        raise ReducerError("workspace effects cannot replace workspace identity")
    return WorkspaceState(
        id=current.id,
        revision=current.revision + 1,
        files=delta.files,
        artifacts=delta.artifacts,
    )


def _descends_from(
    agent_runs: Mapping[str, AgentRunState], descendant: str, ancestor: str
) -> bool:
    current = agent_runs.get(descendant)
    seen: set[str] = set()
    while current is not None and current.parent_run_id is not None:
        if current.parent_run_id == ancestor:
            return True
        if current.parent_run_id in seen:
            return False
        seen.add(current.parent_run_id)
        current = agent_runs.get(current.parent_run_id)
    return False


class EventReducer:
    """Pure reducer for typed commit events."""

    def apply(self, state: ExecutionState | None, commit: Commit) -> ExecutionState:
        event = commit.event
        if state is None:
            if not isinstance(event, ExecutionStarted):
                raise ReducerError("the first commit must be execution.started")
            if commit.parent_hash is not None:
                raise ReducerError("the first commit cannot have a parent")
            dependency_outputs = {
                task_id: TaskOutput.model_validate(value)
                for task_id, value in event.dependency_outputs.items()
            }
            runtime = _runtime_after(RuntimeState(), event.runtime)
            return ExecutionState(
                through_commit_hash=commit.hash,
                execution_id=commit.execution_id,
                branch_id=commit.branch_id,
                task=TaskState(
                    metadata=event.task,
                    model=event.model,
                    scaffold=event.scaffold,
                    environment=event.environment_metadata,
                    dependency_outputs=dependency_outputs,
                ),
                environment=EnvironmentState(values=event.environment),
                workspace=event.workspace,
                runtime=runtime,
            )
        if state.execution_id != commit.execution_id:
            raise ReducerError("a commit cannot change execution identity")
        if commit.parent_hash != state.through_commit_hash:
            raise ReducerError("commit parent does not match the projection head")
        if state.runtime.metadata.get("execution_completed"):
            raise ReducerError("no commits may follow execution.completed")
        if state.is_terminal:
            if isinstance(event, AgentStateUpdated):
                run = state.agent_runs.get(commit.author.run_id)
                if run is None or run.status not in {"created", "running"}:
                    raise ReducerError("terminal agent state cannot be updated")
            elif isinstance(event, AgentTurnRecorded):
                run = state.agent_runs.get(commit.author.run_id)
                if run is None or run.status not in {"created", "running"}:
                    raise ReducerError("terminal agent turn requires an active run")
                if event.actions:
                    raise ReducerError(
                        "new actions cannot be proposed after a submission is accepted"
                    )
            elif isinstance(event, AgentCompleted):
                run = state.agent_runs.get(event.agent_run_id)
                if run is None or run.status not in {"created", "running"}:
                    raise ReducerError("agent run is already terminal")
            elif isinstance(event, ExecutionCompleted):
                if state.runtime.metadata.get("execution_completed"):
                    raise ReducerError("execution is already completed")
                if state.runtime.status not in {"submitted", "surrendered"}:
                    raise ReducerError(
                        "execution.completed requires an accepted submission"
                    )
                if event.status != state.runtime.status:
                    raise ReducerError(
                        "execution.completed status must match the accepted submission"
                    )
            else:
                raise ReducerError(
                    f"{event.type} cannot be appended after the execution is terminal"
                )

        data = state.model_dump(mode="python")
        data["through_commit_hash"] = commit.hash
        data["branch_id"] = commit.branch_id
        agent_runs = dict(state.agent_runs)
        conversations = {
            key: tuple(value) for key, value in state.conversations.items()
        }
        actions = dict(state.actions)
        invocations = dict(state.tool_invocations)
        usage_by_run = dict(state.usage_by_run)
        runtime = state.runtime
        environment = state.environment
        workspace = state.workspace

        if isinstance(event, ExecutionStarted):
            raise ReducerError("execution.started can only be the first event")
        if isinstance(event, TaskConfigured):
            environment = _environment_after(
                environment,
                event.environment_operations,
                event.expected_environment_revision,
            )
            workspace = _workspace_after(
                workspace, event.workspace_delta, event.expected_workspace_revision
            )
            runtime = _runtime_after(runtime, event.runtime_update)
            runtime = _replace(
                runtime,
                metadata={
                    **dict(runtime.metadata),
                    "configuration_status": event.status,
                },
            )
        elif isinstance(event, AgentStarted):
            existing = agent_runs.get(event.agent_run_id)
            if existing is None:
                agent_runs[event.agent_run_id] = AgentRunState(
                    run_id=event.agent_run_id,
                    actor_id=event.agent_id,
                    parent_run_id=event.parent_run_id,
                    status="running",
                    context_cutoff_hash=event.context_cutoff_hash,
                    handoff=event.handoff,
                    metadata=event.metadata,
                )
            else:
                if existing.status != "created":
                    raise ReducerError("agent run has already been started")
                if existing.actor_id != event.agent_id:
                    raise ReducerError("agent.started cannot change actor identity")
                if existing.parent_run_id != event.parent_run_id:
                    raise ReducerError("agent.started cannot change parent identity")
                if existing.context_cutoff_hash != event.context_cutoff_hash:
                    raise ReducerError("agent.started cannot change context cutoff")
                if existing.handoff != event.handoff:
                    raise ReducerError("agent.started cannot change its handoff")
                if existing.metadata != event.metadata:
                    raise ReducerError("agent.started cannot change agent metadata")
                agent_runs[event.agent_run_id] = _replace(existing, status="running")
            conversations.setdefault(event.agent_run_id, ())
            usage_by_run.setdefault(event.agent_run_id, UsageState())
        elif isinstance(event, AgentSpawned):
            parent = agent_runs.get(commit.author.run_id)
            if parent is None:
                raise ReducerError("the spawning parent run is not registered")
            if parent.status not in {"created", "running"}:
                raise ReducerError("a terminal agent run cannot spawn children")
            if event.child_run_id in agent_runs:
                raise ReducerError("child agent run already exists")
            agent_runs[event.child_run_id] = AgentRunState(
                run_id=event.child_run_id,
                actor_id=event.child_actor_id,
                parent_run_id=commit.author.run_id,
                context_cutoff_hash=event.context_cutoff_hash,
                handoff=event.handoff,
                metadata=event.metadata,
            )
            agent_runs[parent.run_id] = _replace(
                parent, child_run_ids=(*parent.child_run_ids, event.child_run_id)
            )
            conversations[event.child_run_id] = ()
            usage_by_run[event.child_run_id] = UsageState()
            conversations[parent.run_id] = (
                *conversations.get(parent.run_id, ()),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": event.child_run_id,
                            "type": "function",
                            "function": {
                                "name": SPAWN_SUBAGENT_TOOL_NAME,
                                "arguments": json.dumps(
                                    {
                                        "actor_id": event.child_actor_id,
                                        "handoff": event.handoff,
                                    },
                                    allow_nan=False,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                            },
                        }
                    ],
                    "metadata": {
                        "child_run_id": event.child_run_id,
                        "inspectable": True,
                    },
                },
            )
        elif isinstance(event, AgentTurnRecorded):
            run = agent_runs.get(commit.author.run_id)
            if run is None:
                raise ReducerError("agent run is not registered")
            if run.status not in {"created", "running"}:
                raise ReducerError("a terminal agent run cannot record turns")
            conversations[run.run_id] = (
                *conversations.get(run.run_id, ()),
                *event.messages,
            )
            for proposed in event.actions:
                if proposed.id in actions:
                    raise ReducerError(f"action id {proposed.id!r} already exists")
                trusted = Action(
                    id=proposed.id,
                    name=proposed.name,
                    arguments=proposed.arguments,
                    actor_id=commit.author.actor_id,
                    content=proposed.content,
                    metadata=proposed.metadata,
                )
                actions[trusted.id] = ActionState(
                    action=trusted,
                    requested_by_run_id=run.run_id,
                    parallel_group_id=event.parallel_group_id,
                )
            usage_by_run = _add_usage(usage_by_run, run.run_id, event.usage_delta)
        elif isinstance(event, AgentStateUpdated):
            run = agent_runs.get(commit.author.run_id)
            if run is None:
                raise ReducerError("agent run is not registered")
            if run.status not in {"created", "running"}:
                raise ReducerError("a terminal agent run cannot update state")
            algorithm_state = dict(run.algorithm_state)
            algorithm_state[event.namespace] = event.value
            agent_runs[run.run_id] = _replace(run, algorithm_state=algorithm_state)
        elif isinstance(event, ToolStarted):
            action_state = actions.get(event.action_id)
            if action_state is None:
                raise ReducerError("tool invocation refers to an unknown action")
            if action_state.requested_by_run_id != event.requested_by_run_id:
                raise ReducerError("tool invocation requester does not own the action")
            if event.invocation_id in invocations:
                raise ReducerError("tool invocation id already exists")
            if action_state.status != "pending":
                raise ReducerError("only a pending action can start an invocation")
            if action_state.action.name != event.tool_name:
                raise ReducerError("tool invocation name does not match its action")
            invocations[event.invocation_id] = ToolInvocationState(
                invocation_id=event.invocation_id,
                action_id=event.action_id,
                requested_by_run_id=event.requested_by_run_id,
                tool_name=event.tool_name,
                started_commit_hash=commit.hash,
            )
            actions[event.action_id] = _replace(
                action_state,
                status="running",
                invocation_ids=(*action_state.invocation_ids, event.invocation_id),
            )
        elif isinstance(event, ToolCompleted):
            invocation = invocations.get(event.invocation_id)
            action_state = actions.get(event.action_id)
            if invocation is None or action_state is None:
                raise ReducerError("tool completion refers to unknown execution state")
            if invocation.status != "running" or action_state.status != "running":
                raise ReducerError(
                    "tool invocation has already reached a terminal state"
                )
            if (
                invocation.action_id != event.action_id
                or invocation.requested_by_run_id != event.requested_by_run_id
            ):
                raise ReducerError("tool completion correlation ids do not match")
            environment = _environment_after(
                environment,
                event.environment_operations,
                event.expected_environment_revision,
            )
            workspace = _workspace_after(
                workspace, event.workspace_delta, event.expected_workspace_revision
            )
            runtime = _runtime_after(runtime, event.runtime_update)
            invocations[event.invocation_id] = _replace(
                invocation,
                status="completed",
                observation=event.observation,
                duration_ms=event.duration_ms,
            )
            actions[event.action_id] = _replace(action_state, status="completed")
            conversations[event.requested_by_run_id] = (
                *conversations.get(event.requested_by_run_id, ()),
                {
                    "role": "tool",
                    "tool_call_id": event.action_id,
                    "name": action_state.action.name,
                    "content": event.observation,
                    "metadata": {
                        "status": event.status,
                        "success": event.status == "success",
                        "duration_ms": event.duration_ms,
                        "invocation_id": event.invocation_id,
                        "parallel_group_id": action_state.parallel_group_id,
                    },
                },
            )
            usage_by_run = _add_usage(
                usage_by_run, event.requested_by_run_id, event.usage_delta
            )
        elif isinstance(event, ToolFailed):
            invocation = invocations.get(event.invocation_id)
            action_state = actions.get(event.action_id)
            if invocation is None or action_state is None:
                raise ReducerError("tool failure refers to unknown execution state")
            if invocation.status != "running" or action_state.status != "running":
                raise ReducerError(
                    "tool invocation has already reached a terminal state"
                )
            if (
                invocation.action_id != event.action_id
                or invocation.requested_by_run_id != event.requested_by_run_id
            ):
                raise ReducerError("tool failure correlation ids do not match")
            invocations[event.invocation_id] = _replace(
                invocation,
                status="failed",
                error=event.error,
                duration_ms=event.duration_ms,
            )
            actions[event.action_id] = _replace(action_state, status="failed")
            conversations[event.requested_by_run_id] = (
                *conversations.get(event.requested_by_run_id, ()),
                {
                    "role": "tool",
                    "tool_call_id": event.action_id,
                    "name": action_state.action.name,
                    "content": event.error,
                    "metadata": {
                        "status": "failed",
                        "success": False,
                        "duration_ms": event.duration_ms,
                        "invocation_id": event.invocation_id,
                        "error_type": event.error_type,
                        "parallel_group_id": action_state.parallel_group_id,
                    },
                },
            )
            usage_by_run = _add_usage(
                usage_by_run, event.requested_by_run_id, event.usage_delta
            )
        elif isinstance(event, ParallelGroupCompleted):
            missing = [
                action_id for action_id in event.action_ids if action_id not in actions
            ]
            if missing:
                raise ReducerError(
                    f"parallel group contains unknown actions: {missing}"
                )
            invalid = [
                action_id
                for action_id in event.action_ids
                if actions[action_id].parallel_group_id != event.group_id
                or actions[action_id].status not in {"completed", "failed"}
            ]
            if invalid:
                raise ReducerError(
                    f"parallel group contains incomplete or unrelated actions: {invalid}"
                )
        elif isinstance(event, ContextImported):
            target = agent_runs.get(commit.author.run_id)
            if target is None:
                raise ReducerError("importing agent run is not registered")
            if target.status not in {"created", "running"}:
                raise ReducerError("a terminal agent run cannot import context")
            if event.source_run_id != target.run_id and not _descends_from(
                agent_runs, event.source_run_id, target.run_id
            ):
                raise ReducerError("agents may import context only from descendants")
            imported: Mapping[str, JsonValue] = {
                "source_run_id": event.source_run_id,
                "source_commit_ids": list(event.source_commit_ids),
                "representation": event.representation,
            }
            agent_runs[target.run_id] = _replace(
                target, imported_context=(*target.imported_context, imported)
            )
            conversations[target.run_id] = (
                *conversations.get(target.run_id, ()),
                {
                    "role": "system",
                    "name": "context.imported",
                    "content": event.representation,
                    "metadata": {
                        "source_run_id": event.source_run_id,
                        "source_commit_ids": list(event.source_commit_ids),
                    },
                },
            )
        elif isinstance(event, AgentCompleted):
            run = agent_runs.get(event.agent_run_id)
            if run is None:
                raise ReducerError("completed agent run is not registered")
            if run.status not in {"created", "running"}:
                raise ReducerError("agent run is already terminal")
            agent_runs[event.agent_run_id] = _replace(
                run,
                status=event.status,
                result_summary=event.result_summary,
                trace_head=event.trace_head,
                metadata={**dict(run.metadata), **dict(event.metadata)},
            )
            usage_by_run = _add_usage(
                usage_by_run, event.agent_run_id, event.usage_delta
            )
            if run.parent_run_id is not None:
                conversations[run.parent_run_id] = (
                    *conversations.get(run.parent_run_id, ()),
                    {
                        "role": "tool",
                        "tool_call_id": run.run_id,
                        "name": SPAWN_SUBAGENT_TOOL_NAME,
                        "content": {
                            "child_run_id": run.run_id,
                            "status": event.status,
                            "result": event.result_summary,
                        },
                        "metadata": {
                            "status": event.status,
                            "success": event.status in {"completed", "surrendered"},
                            "child_run_id": run.run_id,
                            "trace_head": event.trace_head,
                            "inspectable": True,
                        },
                    },
                )
        elif isinstance(event, SubmissionAccepted):
            action_state = actions.get(event.action_id)
            if action_state is None or action_state.status != "completed":
                raise ReducerError("submission requires a completed action")
            if action_state.requested_by_run_id != event.requested_by_run_id:
                raise ReducerError("submission requester does not own its action")
            requesting_run = agent_runs.get(event.requested_by_run_id)
            if requesting_run is None or requesting_run.parent_run_id is not None:
                raise ReducerError("only a root agent run may submit the execution")
            if action_state.action.name != SUBMIT_ANSWER_TOOL_NAME:
                raise ReducerError("submission requires the canonical submission tool")
            if action_state.action.arguments.get("answer") != event.answer:
                raise ReducerError("accepted answer differs from the committed action")
            data["submission"] = event.answer
            runtime = _replace(
                runtime,
                status="surrendered" if event.surrendered else "submitted",
                ended_at=commit.recorded_at,
            )
        elif isinstance(event, ExecutionFailed):
            runtime = _replace(
                runtime,
                status="failed",
                ended_at=commit.recorded_at,
                metadata={
                    **dict(runtime.metadata),
                    **dict(event.metadata),
                    "error": event.error,
                    "error_type": event.error_type,
                },
            )
        elif isinstance(event, ExecutionCompleted):
            runtime = _replace(
                runtime,
                status=event.status,
                ended_at=runtime.ended_at or commit.recorded_at,
                metadata={
                    **dict(runtime.metadata),
                    **dict(event.metadata),
                    "execution_completed": True,
                },
            )
        else:  # pragma: no cover - exhaustive union guard
            raise ReducerError(f"unsupported event type {type(event).__name__}")

        data.update(
            agent_runs=agent_runs,
            conversations=conversations,
            actions=actions,
            tool_invocations=invocations,
            usage_by_run=usage_by_run,
            runtime=runtime,
            environment=environment,
            workspace=workspace,
        )
        return ExecutionState.model_validate(data)

    def replay(self, commits: list[Commit] | tuple[Commit, ...]) -> ExecutionState:
        state: ExecutionState | None = None
        for commit in commits:
            state = self.apply(state, commit)
        if state is None:
            raise ReducerError("cannot materialize an empty commit stream")
        return state


__all__ = [
    "EventReducer",
    "ReducerError",
    "SharedStateConflictError",
]
