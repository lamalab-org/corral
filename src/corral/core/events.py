"""Typed, immutable deltas accepted by the execution commit ledger."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from corral.core._immutable import FrozenModel
from corral.core.action import Action
from corral.core.workspace import Artifact, FileRef, WorkspaceState

SPAWN_SUBAGENT_TOOL_NAME = "spawn_subagent"


class UsageDelta(FrozenModel):
    """Resource use caused by one event, never a cumulative counter."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    agent_steps: int = Field(default=0, ge=0)
    # Retained only so schema-v1 commits remain hash-compatible on replay.
    # Reducers and exporters deliberately ignore this legacy catch-all.
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class RuntimeUpdate(FrozenModel):
    """Fields changed in the materialized runtime namespace."""

    status: (
        Literal[
            "created",
            "running",
            "submitted",
            "surrendered",
            "terminal",
            "failed",
        ]
        | None
    ) = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class EnvironmentOperation(FrozenModel):
    """One deterministic mutation of the shared environment namespace."""

    operation: Literal["set", "delete"]
    path: tuple[str, ...] = Field(min_length=1)
    value: JsonValue = None

    @model_validator(mode="after")
    def _validate_operation(self) -> EnvironmentOperation:
        if any(not part for part in self.path):
            raise ValueError("environment operation paths cannot contain empty parts")
        if self.operation == "delete" and self.value is not None:
            raise ValueError("delete operations cannot carry a value")
        return self


class WorkspaceDelta(FrozenModel):
    """Complete manifest values produced by one atomic workspace capture."""

    workspace_id: str = Field(min_length=1)
    files: Mapping[str, FileRef] = Field(default_factory=dict)
    artifacts: Mapping[str, Artifact] = Field(default_factory=dict)

    @classmethod
    def from_workspace(cls, workspace: WorkspaceState) -> WorkspaceDelta:
        return cls(
            workspace_id=workspace.id,
            files=workspace.files,
            artifacts=workspace.artifacts,
        )


class ExecutionStarted(FrozenModel):
    type: Literal["execution.started"] = "execution.started"
    task: Mapping[str, JsonValue] = Field(default_factory=dict)
    environment: Mapping[str, JsonValue] = Field(default_factory=dict)
    environment_metadata: Mapping[str, JsonValue] = Field(default_factory=dict)
    workspace: WorkspaceState = Field(default_factory=WorkspaceState)
    scaffold: Mapping[str, JsonValue] = Field(default_factory=dict)
    model: Mapping[str, JsonValue] = Field(default_factory=dict)
    dependency_outputs: Mapping[str, JsonValue] = Field(default_factory=dict)
    runtime: RuntimeUpdate = Field(default_factory=RuntimeUpdate)


class TaskConfigured(FrozenModel):
    type: Literal["task.configured"] = "task.configured"
    status: str
    environment_operations: tuple[EnvironmentOperation, ...] = ()
    workspace_delta: WorkspaceDelta | None = None
    expected_environment_revision: int | None = Field(default=None, ge=0)
    expected_workspace_revision: int | None = Field(default=None, ge=0)
    runtime_update: RuntimeUpdate | None = None

    @model_validator(mode="after")
    def _validate_preconditions(self) -> TaskConfigured:
        if self.environment_operations and self.expected_environment_revision is None:
            raise ValueError("environment operations require an expected revision")
        if (
            self.workspace_delta is not None
            and self.expected_workspace_revision is None
        ):
            raise ValueError("workspace effects require an expected revision")
        return self


class AgentStarted(FrozenModel):
    type: Literal["agent.started"] = "agent.started"
    agent_run_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    parent_run_id: str | None = Field(default=None, min_length=1)
    context_cutoff_hash: str | None = None
    handoff: JsonValue = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class AgentSpawned(FrozenModel):
    type: Literal["agent.spawned"] = "agent.spawned"
    child_run_id: str = Field(min_length=1)
    child_actor_id: str = Field(min_length=1)
    context_cutoff_hash: str | None = None
    handoff: JsonValue
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class AgentTurnRecorded(FrozenModel):
    type: Literal["agent.turn_recorded"] = "agent.turn_recorded"
    messages: tuple[Mapping[str, JsonValue], ...] = ()
    actions: tuple[Action, ...] = ()
    parallel_group_id: str | None = Field(default=None, min_length=1)
    usage_delta: UsageDelta = Field(default_factory=UsageDelta)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parallel_group(self) -> AgentTurnRecorded:
        if len(self.actions) > 1 and self.parallel_group_id is None:
            raise ValueError("multiple actions require a parallel group")
        if self.parallel_group_id is not None and len(self.actions) < 2:
            raise ValueError("a parallel group requires at least two actions")
        return self


class AgentStateUpdated(FrozenModel):
    type: Literal["agent.state_updated"] = "agent.state_updated"
    namespace: str = Field(min_length=1)
    value: Mapping[str, JsonValue] = Field(default_factory=dict)


class ToolStarted(FrozenModel):
    type: Literal["tool.started"] = "tool.started"
    action_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    requested_by_run_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)


class ToolCompleted(FrozenModel):
    type: Literal["tool.completed"] = "tool.completed"
    action_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    requested_by_run_id: str = Field(min_length=1)
    observation: JsonValue
    status: Literal["success", "invalid_tool", "invalid_args", "execution_error"]
    environment_operations: tuple[EnvironmentOperation, ...] = ()
    workspace_delta: WorkspaceDelta | None = None
    usage_delta: UsageDelta = Field(default_factory=lambda: UsageDelta(tool_calls=1))
    runtime_update: RuntimeUpdate | None = None
    expected_environment_revision: int | None = Field(default=None, ge=0)
    expected_workspace_revision: int | None = Field(default=None, ge=0)
    duration_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_preconditions(self) -> ToolCompleted:
        if self.environment_operations and self.expected_environment_revision is None:
            raise ValueError("environment operations require an expected revision")
        if (
            self.workspace_delta is not None
            and self.expected_workspace_revision is None
        ):
            raise ValueError("workspace effects require an expected revision")
        return self


class ToolFailed(FrozenModel):
    type: Literal["tool.failed"] = "tool.failed"
    action_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    requested_by_run_id: str = Field(min_length=1)
    error: str
    error_type: str | None = None
    usage_delta: UsageDelta = Field(default_factory=lambda: UsageDelta(tool_calls=1))
    duration_ms: float | None = Field(default=None, ge=0)


class ParallelGroupCompleted(FrozenModel):
    type: Literal["parallel_group.completed"] = "parallel_group.completed"
    group_id: str = Field(min_length=1)
    action_ids: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_actions(self) -> ParallelGroupCompleted:
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("parallel group action ids must be unique")
        return self


class ContextImported(FrozenModel):
    type: Literal["context.imported"] = "context.imported"
    source_run_id: str = Field(min_length=1)
    source_commit_ids: tuple[str, ...] = Field(min_length=1)
    representation: JsonValue


class AgentCompleted(FrozenModel):
    type: Literal["agent.completed"] = "agent.completed"
    agent_run_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    result_summary: JsonValue = None
    trace_head: str | None = None
    usage_delta: UsageDelta = Field(default_factory=UsageDelta)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class SubmissionAccepted(FrozenModel):
    type: Literal["submission.accepted"] = "submission.accepted"
    action_id: str = Field(min_length=1)
    requested_by_run_id: str = Field(min_length=1)
    answer: str
    surrendered: bool = False


class ExecutionFailed(FrozenModel):
    type: Literal["execution.failed"] = "execution.failed"
    error: str
    error_type: str = Field(min_length=1)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class ExecutionCompleted(FrozenModel):
    type: Literal["execution.completed"] = "execution.completed"
    status: Literal["submitted", "surrendered", "terminal"] = "terminal"
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


StateEvent: TypeAlias = Annotated[
    ExecutionStarted
    | TaskConfigured
    | AgentStarted
    | AgentSpawned
    | AgentTurnRecorded
    | AgentStateUpdated
    | ToolStarted
    | ToolCompleted
    | ToolFailed
    | ParallelGroupCompleted
    | ContextImported
    | AgentCompleted
    | SubmissionAccepted
    | ExecutionFailed
    | ExecutionCompleted,
    Field(discriminator="type"),
]


def event_from_dict(value: Mapping[str, Any]) -> StateEvent:
    """Validate a serialized event through the discriminated union."""

    return TypeAdapter(StateEvent).validate_python(value)


__all__ = [
    "SPAWN_SUBAGENT_TOOL_NAME",
    "AgentCompleted",
    "AgentSpawned",
    "AgentStarted",
    "AgentStateUpdated",
    "AgentTurnRecorded",
    "ContextImported",
    "EnvironmentOperation",
    "ExecutionCompleted",
    "ExecutionFailed",
    "ExecutionStarted",
    "ParallelGroupCompleted",
    "RuntimeUpdate",
    "StateEvent",
    "SubmissionAccepted",
    "TaskConfigured",
    "ToolCompleted",
    "ToolFailed",
    "ToolStarted",
    "UsageDelta",
    "WorkspaceDelta",
    "event_from_dict",
]
