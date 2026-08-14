"""Immutable, revisioned execution state.

State v2 is deliberately independent of the current ``Environment`` execution
loop.  It establishes the durable boundary that later runtime PRs can adopt
without making persistence depend on live agent or environment objects.
"""

from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, model_validator
from pydantic import JsonValue as JSONValue

from corral.core._immutable import FrozenModel, validate_sha256_hex
from corral.core.workspace import WorkspaceState

if TYPE_CHECKING:
    from datetime import datetime

STATE_SCHEMA_VERSION = 2
_USAGE_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "llm_calls",
    "tool_calls",
    "agent_steps",
)


def new_state_id() -> str:
    """Mint a globally unique identifier for a State revision chain."""
    return str(uuid4())


class StateMetadata(FrozenModel):
    """Durable configuration needed to understand or restore an execution."""

    model: Mapping[str, JSONValue] = Field(default_factory=dict)
    scaffold: Mapping[str, JSONValue] = Field(default_factory=dict)
    environment: Mapping[str, JSONValue] = Field(default_factory=dict)
    task: Mapping[str, JSONValue] = Field(default_factory=dict)
    extra: Mapping[str, JSONValue] = Field(default_factory=dict)


class UsageState(FrozenModel):
    """Cumulative resource use recorded by committed transitions."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    agent_steps: int = Field(default=0, ge=0)
    metadata: Mapping[str, JSONValue] = Field(default_factory=dict)


class RuntimeState(FrozenModel):
    """Serializable progress needed by the next execution step."""

    status: Literal[
        "created", "running", "submitted", "surrendered", "terminal", "failed"
    ] = "created"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: Mapping[str, JSONValue] = Field(default_factory=dict)


RuntimeState.model_rebuild(
    _types_namespace={"datetime": datetime_module.datetime},
)


class TaskOutput(FrozenModel):
    """An upstream task output, intentionally free of benchmark scores."""

    output: Mapping[str, JSONValue] = Field(default_factory=dict)
    metadata: Mapping[str, JSONValue] = Field(default_factory=dict)


def _canonical_payload(state: State) -> dict[str, Any]:
    return state.model_dump(mode="json")


def canonical_state_json(state: State) -> str:
    """Return the stable JSON representation used for hashing and storage."""
    return json.dumps(
        _canonical_payload(state),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class State(FrozenModel):
    """One immutable version of a Corral execution's durable state."""

    schema_version: int = STATE_SCHEMA_VERSION
    id: str = Field(default_factory=new_state_id, min_length=1)
    revision: int = Field(default=0, ge=0)
    parent_revision: int | None = Field(default=None, ge=0)
    parent_hash: str | None = None

    metadata: StateMetadata = Field(default_factory=StateMetadata)
    messages: tuple[Mapping[str, JSONValue], ...] = ()
    environment: Mapping[str, JSONValue] = Field(default_factory=dict)
    workspace: WorkspaceState
    usage: UsageState = Field(default_factory=UsageState)
    runtime: RuntimeState = Field(default_factory=RuntimeState)
    dependency_outputs: Mapping[str, TaskOutput] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _initialize_workspace(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        state_id = data.get("id") or new_state_id()
        data["id"] = state_id
        if "workspace" not in data:
            workspace_id = str(uuid5(NAMESPACE_URL, f"corral:workspace:{state_id}"))
            data["workspace"] = WorkspaceState(id=workspace_id)
        return data

    @model_validator(mode="after")
    def _validate_revision_chain(self) -> State:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"State schema_version must be {STATE_SCHEMA_VERSION}; "
                "other State schemas are not supported"
            )
        if self.revision == 0:
            if self.parent_revision is not None or self.parent_hash is not None:
                raise ValueError("revision 0 cannot have a parent")
        elif self.parent_revision != self.revision - 1:
            raise ValueError("parent_revision must immediately precede revision")
        elif self.parent_hash is None:
            raise ValueError("non-initial revisions require parent_hash")

        if self.parent_hash is not None:
            validate_sha256_hex(self.parent_hash, field_name="parent_hash")
        return self

    @property
    def state_hash(self) -> str:
        """Content hash of this complete revision."""
        return _sha256(canonical_state_json(self))

    def to_json(self) -> str:
        """Serialize with deterministic key ordering."""
        return canonical_state_json(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> State:
        """Validate data in the current State schema."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> State:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("serialized State must be a JSON object")
        return cls.from_dict(raw)

    def fork(
        self,
        *,
        messages: tuple[Mapping[str, JSONValue], ...] | None = None,
        environment: Mapping[str, JSONValue] | None = None,
        workspace: WorkspaceState | Mapping[str, Any] | None = None,
        usage: UsageState | Mapping[str, Any] | None = None,
        runtime: RuntimeState | Mapping[str, Any] | None = None,
        dependency_outputs: Mapping[str, TaskOutput | Mapping[str, Any]] | None = None,
    ) -> State:
        """Create one immutable child while preserving execution identity.

        Any mutable namespace not supplied is inherited unchanged. Messages
        remain append-only and cumulative usage counters cannot decrease.
        Multiple calls on the same parent intentionally create sibling forks.
        """
        next_messages = self.messages if messages is None else tuple(messages)
        if len(next_messages) < len(self.messages):
            raise ValueError("State messages are append-only")
        if tuple(next_messages[: len(self.messages)]) != self.messages:
            raise ValueError("State messages are append-only")

        next_usage = self.usage if usage is None else UsageState.model_validate(usage)
        for counter in _USAGE_COUNTERS:
            if getattr(next_usage, counter) < getattr(self.usage, counter):
                raise ValueError(f"State usage counter {counter!r} cannot decrease")

        next_workspace = (
            self.workspace
            if workspace is None
            else WorkspaceState.model_validate(workspace)
        )
        if next_workspace != self.workspace:
            if next_workspace.id != self.workspace.id:
                raise ValueError("a State fork cannot replace its workspace identity")
            if next_workspace.revision != self.workspace.revision + 1:
                raise ValueError(
                    "a changed workspace must be the next WorkspaceState revision"
                )

        return State(
            id=self.id,
            revision=self.revision + 1,
            parent_revision=self.revision,
            parent_hash=self.state_hash,
            metadata=self.metadata,
            messages=next_messages,
            environment=self.environment if environment is None else environment,
            workspace=next_workspace,
            usage=next_usage,
            runtime=self.runtime if runtime is None else runtime,
            dependency_outputs=(
                self.dependency_outputs
                if dependency_outputs is None
                else dependency_outputs
            ),
        )


class AgentStateView(FrozenModel):
    """Read-only State projection passed to future stateless agents."""

    id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    state_hash: str
    metadata: StateMetadata
    messages: tuple[Mapping[str, JSONValue], ...]
    environment: Mapping[str, JSONValue]
    workspace: WorkspaceState
    usage: UsageState
    runtime: RuntimeState
    dependency_outputs: Mapping[str, TaskOutput]

    @model_validator(mode="after")
    def _validate_state_hash(self) -> AgentStateView:
        validate_sha256_hex(self.state_hash, field_name="state_hash")
        return self

    @classmethod
    def from_state(cls, state: State) -> AgentStateView:
        return cls(
            id=state.id,
            revision=state.revision,
            state_hash=state.state_hash,
            metadata=state.metadata,
            messages=state.messages,
            environment=state.environment,
            workspace=state.workspace,
            usage=state.usage,
            runtime=state.runtime,
            dependency_outputs=state.dependency_outputs,
        )
