"""Backend-independent, failure-isolated task observations.

Observers receive immutable State snapshots at the transition boundaries. They
never participate in persistence, retries, or control flow; a broken observer
is deliberately equivalent to :class:`NoOpObserver` from the runtime's point
of view.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from corral.core.action import Action
    from corral.core.state import State

logger = logging.getLogger(__name__)

ObservationType = Literal[
    "span",
    "agent",
    "generation",
    "tool",
    "evaluator",
]


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Worker-side correlation data that must not be persisted in State."""

    execution_id: str
    benchmark_run_id: str | None = None
    task_id: str | None = None
    temporal_workflow_id: str | None = None
    temporal_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """Description of one operation at a durable State boundary."""

    name: str
    context: ObservationContext
    as_type: ObservationType = "span"
    state_before: State | None = None
    action: Action | None = None
    input: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationSpan(Protocol):
    """One active observation returned by an :class:`Observer`."""

    def update(
        self,
        *,
        state_after: State | None = None,
        action: Action | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def end(self, error: BaseException | None = None) -> None: ...


class Observer(Protocol):
    """Starts passive observations around task operations."""

    def start(self, observation: Observation) -> ObservationSpan: ...

    def flush(self) -> None: ...


class _NoOpSpan:
    def update(
        self,
        *,
        state_after: State | None = None,
        action: Action | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del state_after, action, output, metadata

    def end(self, error: BaseException | None = None) -> None:
        del error


class NoOpObserver:
    """Observer used when tracing is disabled or cannot be configured."""

    def start(self, observation: Observation) -> ObservationSpan:
        del observation
        return _NoOpSpan()

    def flush(self) -> None:
        return None


def _observer_failure(operation: str, exc: BaseException) -> None:
    logger.warning("observer failed during %s: %s", operation, exc)


@contextmanager
def observe_safely(
    observer: Observer,
    observation: Observation,
) -> Iterator[ObservationSpan]:
    """Run an observation without allowing it to change task behavior."""

    try:
        span = observer.start(observation)
    except BaseException as exc:
        _observer_failure("start", exc)
        span = _NoOpSpan()

    try:
        yield span
    except BaseException as exc:
        try:
            span.end(exc)
        except BaseException as observer_exc:
            _observer_failure("end", observer_exc)
        raise
    else:
        try:
            span.end()
        except BaseException as exc:
            _observer_failure("end", exc)


def update_safely(
    span: ObservationSpan,
    *,
    state_after: State | None = None,
    action: Action | None = None,
    output: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Update an observation while preserving the execution outcome."""

    try:
        span.update(
            state_after=state_after,
            action=action,
            output=output,
            metadata=metadata,
        )
    except BaseException as exc:
        _observer_failure("update", exc)


def json_value(value: Any) -> Any:
    """Convert common Corral values into data accepted by trace backends."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def state_snapshot(state: State) -> dict[str, Any]:
    """Return a complete State payload, including its content hash."""

    return {
        "state_hash": state.state_hash,
        **state.model_dump(mode="json"),
    }


def state_changes(before: State, after: State) -> dict[str, Any]:
    """Return explicit before/after values for every changed State field."""

    before_payload = before.model_dump(mode="json")
    after_payload = after.model_dump(mode="json")
    changes: dict[str, Any] = {}
    for field_name in before_payload.keys() | after_payload.keys():
        previous = before_payload.get(field_name)
        current = after_payload.get(field_name)
        if previous != current:
            changes[field_name] = {"before": previous, "after": current}
    return changes


def observation_input(observation: Observation) -> dict[str, Any] | None:
    """Build the full input exported for an observation."""

    payload: dict[str, Any] = {}
    if observation.state_before is not None:
        payload["state"] = state_snapshot(observation.state_before)
    if observation.action is not None:
        payload["action"] = json_value(observation.action)
    if observation.input:
        payload["operation"] = json_value(observation.input)
    return payload or None


def observation_output(
    observation: Observation,
    *,
    state_after: State | None,
    output: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build output with the complete State and an explicit State delta."""

    payload: dict[str, Any] = {}
    if state_after is not None:
        payload["state"] = state_snapshot(state_after)
        if observation.state_before is not None:
            payload["state_changes"] = state_changes(
                observation.state_before,
                state_after,
            )
    if output:
        payload["result"] = json_value(output)
    return payload or None


def observation_metadata(
    observation: Observation,
    *,
    state_after: State | None,
    action: Action | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build searchable correlation and State-revision attributes."""

    context = observation.context
    before = observation.state_before
    state = state_after or before
    current_action = action or observation.action
    result: dict[str, Any] = {
        "execution_id": context.execution_id,
        "trial_id": context.execution_id,
        "benchmark_run_id": context.benchmark_run_id,
        "task_id": context.task_id,
        "temporal_workflow_id": context.temporal_workflow_id,
        "temporal_run_id": context.temporal_run_id,
        "state_revision_before": before.revision if before is not None else None,
        "state_hash_before": before.state_hash if before is not None else None,
        "state_revision_after": (
            state_after.revision if state_after is not None else None
        ),
        "state_hash_after": (
            state_after.state_hash if state_after is not None else None
        ),
    }
    if state is not None:
        result.update(
            {
                "state_id": state.id,
                "model": state.metadata.model.get("name"),
                "agent_type": state.metadata.scaffold.get("name"),
                "environment_type": state.metadata.environment.get("name"),
            }
        )
    if before is not None and state_after is not None:
        changes = state_changes(before, state_after)
        result.update(
            {
                "changed_state_fields": sorted(changes),
                "environment_changed": "environment" in changes,
                "workspace_changed": "workspace" in changes,
            }
        )
    if current_action is not None:
        result.update(
            {
                "action_id": current_action.id,
                "action_name": current_action.name,
            }
        )
    result.update(observation.metadata)
    if metadata:
        result.update(metadata)
    return {
        key: json_value(value) for key, value in result.items() if value is not None
    }


__all__ = [
    "NoOpObserver",
    "Observation",
    "ObservationContext",
    "ObservationSpan",
    "ObservationType",
    "Observer",
    "json_value",
    "observation_input",
    "observation_metadata",
    "observation_output",
    "observe_safely",
    "state_changes",
    "state_snapshot",
    "update_safely",
]
