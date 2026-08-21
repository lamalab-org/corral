"""Failure-isolated observations centered on persisted authored commits."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

from corral.report.logging import event, exception_fields

if TYPE_CHECKING:
    from corral.core.actors import ActorRef
    from corral.core.commit import Commit

ObservationType = Literal["span", "agent", "generation", "tool", "evaluator"]


@dataclass(frozen=True, slots=True)
class ObservationContext:
    execution_id: str
    benchmark_run_id: str | None = None
    task_id: str | None = None
    temporal_workflow_id: str | None = None
    temporal_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """Description of an operation that may eventually persist a commit."""

    name: str
    context: ObservationContext
    as_type: ObservationType = "span"
    based_on_hash: str | None = None
    actor: ActorRef | None = None
    input: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationSpan(Protocol):
    def update(
        self,
        *,
        commit: Commit | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def end(self, error: BaseException | None = None) -> None: ...


class Observer(Protocol):
    def start(self, observation: Observation) -> ObservationSpan: ...

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None: ...

    def flush(self) -> None: ...


class _NoOpSpan:
    def update(
        self,
        *,
        commit: Commit | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del commit, output, metadata

    def end(self, error: BaseException | None = None) -> None:
        del error


class NoOpObserver:
    def start(self, observation: Observation) -> ObservationSpan:
        del observation
        return _NoOpSpan()

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        del commit, context

    def flush(self) -> None:
        return None


class _CompositeSpan:
    def __init__(self, spans: tuple[ObservationSpan, ...]) -> None:
        self._spans = spans

    def update(
        self,
        *,
        commit: Commit | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        for span in self._spans:
            try:
                span.update(commit=commit, output=output, metadata=metadata)
            except BaseException as exc:
                _observer_failure("update", exc)

    def end(self, error: BaseException | None = None) -> None:
        for span in self._spans:
            try:
                span.end(error)
            except BaseException as exc:
                _observer_failure("end", exc)


class CompositeObserver:
    def __init__(self, *observers: Observer) -> None:
        self.observers = tuple(observers)

    def start(self, observation: Observation) -> ObservationSpan:
        spans: list[ObservationSpan] = []
        for observer in self.observers:
            try:
                spans.append(observer.start(observation))
            except BaseException as exc:
                _observer_failure("start", exc)
        return _CompositeSpan(tuple(spans))

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        for observer in self.observers:
            try:
                observer.record_commit(commit, context=context)
            except BaseException as exc:
                _observer_failure("record_commit", exc)

    def flush(self) -> None:
        for observer in self.observers:
            try:
                observer.flush()
            except BaseException as exc:
                _observer_failure("flush", exc)


def _observer_failure(operation: str, exc: BaseException) -> None:
    event(
        "WARNING",
        "observer.failed",
        subsystem="observability",
        message=f"Observer failed during {operation}",
        operation=operation,
        **exception_fields(exc),
    )


@contextmanager
def observe_safely(
    observer: Observer, observation: Observation
) -> Iterator[ObservationSpan]:
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
    commit: Commit | None = None,
    output: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    try:
        span.update(commit=commit, output=output, metadata=metadata)
    except BaseException as exc:
        _observer_failure("update", exc)


def record_commit_safely(
    observer: Observer,
    commit: Commit,
    *,
    context: ObservationContext | None = None,
) -> None:
    """Notify a passive observer only after the append transaction succeeds."""
    try:
        observer.record_commit(commit, context=context)
    except BaseException as exc:
        _observer_failure("record_commit", exc)


def json_value(value: Any) -> Any:
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


def commit_input(commit: Commit) -> dict[str, Any] | None:
    values = {
        key: getattr(commit.event, key, None)
        for key in (
            "action_id",
            "invocation_id",
            "requested_by_run_id",
            "child_run_id",
            "source_run_id",
        )
        if getattr(commit.event, key, None) is not None
    }
    return json_value(values) or None


def commit_output(commit: Commit) -> dict[str, Any]:
    return {"event": commit.event.model_dump(mode="json")}


def commit_metadata(commit: Commit) -> dict[str, Any]:
    metadata = {
        "commit_hash": commit.hash,
        "parent_hash": commit.parent_hash,
        "based_on_hash": commit.based_on_hash,
        "execution_id": commit.execution_id,
        "branch_id": commit.branch_id,
        "sequence": commit.sequence,
        "branch_sequence": commit.branch_sequence,
        "event_type": commit.event.type,
        "author_kind": commit.author.kind,
        "author_id": commit.author.actor_id,
        "author_run_id": commit.author.run_id,
        "parent_run_id": commit.author.parent_run_id,
    }
    for key in (
        "action_id",
        "invocation_id",
        "requested_by_run_id",
        "agent_run_id",
        "child_run_id",
        "source_run_id",
        "source_commit_ids",
        "trace_head",
    ):
        value = getattr(commit.event, key, None)
        if value is not None:
            metadata[key] = json_value(value)
    if getattr(commit.event, "child_run_id", None) is not None:
        metadata["spawned_by"] = commit.author.run_id
    if getattr(commit.event, "action_id", None) is not None:
        metadata["caused_by"] = commit.event.action_id
    if getattr(commit.event, "source_commit_ids", None) is not None:
        metadata["imports"] = json_value(commit.event.source_commit_ids)
    return metadata


__all__ = [
    "CompositeObserver",
    "NoOpObserver",
    "Observation",
    "ObservationContext",
    "ObservationSpan",
    "ObservationType",
    "Observer",
    "commit_input",
    "commit_metadata",
    "commit_output",
    "json_value",
    "observe_safely",
    "record_commit_safely",
    "update_safely",
]
