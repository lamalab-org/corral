"""Read-only execution projections reconstructed from authored commits."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, field_validator

from corral.core._immutable import FrozenModel, validate_sha256_hex
from corral.core.action import Action
from corral.core.workspace import WorkspaceState

_USAGE_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "llm_calls",
    "tool_calls",
    "agent_steps",
)


class UsageState(FrozenModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    agent_steps: int = Field(default=0, ge=0)


class RuntimeState(FrozenModel):
    status: Literal[
        "created", "running", "submitted", "surrendered", "terminal", "failed"
    ] = "created"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class TaskOutput(FrozenModel):
    output: Mapping[str, JsonValue] = Field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class TaskState(FrozenModel):
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)
    model: Mapping[str, JsonValue] = Field(default_factory=dict)
    scaffold: Mapping[str, JsonValue] = Field(default_factory=dict)
    environment: Mapping[str, JsonValue] = Field(default_factory=dict)
    dependency_outputs: Mapping[str, TaskOutput] = Field(default_factory=dict)


class EnvironmentState(FrozenModel):
    revision: int = Field(default=0, ge=0)
    values: Mapping[str, JsonValue] = Field(default_factory=dict)


class AgentRunState(FrozenModel):
    run_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    parent_run_id: str | None = Field(default=None, min_length=1)
    status: str = Field(default="created", min_length=1)
    context_cutoff_hash: str | None = None
    handoff: JsonValue = None
    child_run_ids: tuple[str, ...] = ()
    algorithm_state: Mapping[str, Mapping[str, JsonValue]] = Field(default_factory=dict)
    imported_context: tuple[Mapping[str, JsonValue], ...] = ()
    result_summary: JsonValue = None
    trace_head: str | None = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class ActionState(FrozenModel):
    action: Action
    requested_by_run_id: str = Field(min_length=1)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    parallel_group_id: str | None = None
    invocation_ids: tuple[str, ...] = ()


class ToolInvocationState(FrozenModel):
    invocation_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    requested_by_run_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    started_commit_hash: str
    status: Literal["running", "completed", "failed"] = "running"
    observation: JsonValue = None
    error: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)

    @field_validator("started_commit_hash")
    @classmethod
    def _validate_started_hash(cls, value: str) -> str:
        validate_sha256_hex(value, field_name="started_commit_hash")
        return value


class ExecutionState(FrozenModel):
    """Current projection; private histories remain partitioned by agent run."""

    through_commit_hash: str
    execution_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    task: TaskState = Field(default_factory=TaskState)
    environment: EnvironmentState = Field(default_factory=EnvironmentState)
    workspace: WorkspaceState = Field(default_factory=WorkspaceState)
    agent_runs: Mapping[str, AgentRunState] = Field(default_factory=dict)
    conversations: Mapping[str, tuple[Mapping[str, JsonValue], ...]] = Field(
        default_factory=dict
    )
    actions: Mapping[str, ActionState] = Field(default_factory=dict)
    tool_invocations: Mapping[str, ToolInvocationState] = Field(default_factory=dict)
    usage_by_run: Mapping[str, UsageState] = Field(default_factory=dict)
    runtime: RuntimeState = Field(default_factory=RuntimeState)
    submission: str | None = None

    @field_validator("through_commit_hash")
    @classmethod
    def _validate_through_hash(cls, value: str) -> str:
        validate_sha256_hex(value, field_name="through_commit_hash")
        return value

    @property
    def is_terminal(self) -> bool:
        return self.submission is not None or self.runtime.status in {
            "submitted",
            "surrendered",
            "terminal",
            "failed",
        }

    @property
    def usage(self) -> UsageState:
        totals = dict.fromkeys(_USAGE_COUNTERS, 0)
        for usage in self.usage_by_run.values():
            for counter in _USAGE_COUNTERS:
                totals[counter] += getattr(usage, counter)
        return UsageState(**totals)

    @property
    def dependency_outputs(self) -> Mapping[str, TaskOutput]:
        return self.task.dependency_outputs

    @property
    def pending_actions(self) -> tuple[ActionState, ...]:
        return tuple(
            action
            for action in self.actions.values()
            if action.status in {"pending", "running"}
        )

    @property
    def tool_statistics(self) -> dict[str, int]:
        statistics: dict[str, int] = {}
        for action_state in self.actions.values():
            name = action_state.action.name
            statistics[name] = statistics.get(name, 0) + 1
        return statistics


__all__ = [
    "ActionState",
    "AgentRunState",
    "EnvironmentState",
    "ExecutionState",
    "RuntimeState",
    "TaskOutput",
    "TaskState",
    "ToolInvocationState",
    "UsageState",
]
