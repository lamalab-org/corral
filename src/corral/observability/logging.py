"""Local structured logging for operations and authored commits."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from corral.logging import event, exception_fields
from corral.observability.base import (
    Observation,
    ObservationContext,
    ObservationSpan,
    commit_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.core.commit import Commit


class _LoggingSpan:
    def __init__(self, observation: Observation) -> None:
        self._observation = observation
        self._started = time.perf_counter()
        self._commit: Commit | None = None
        self._metadata: dict[str, Any] = {}
        self._ended = False

    def update(
        self,
        *,
        commit: Commit | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del output
        if commit is not None:
            self._commit = commit
        if metadata:
            self._metadata.update(metadata)

    def end(self, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        context = self._observation.context
        fields: dict[str, Any] = {
            "execution_id": context.execution_id,
            "benchmark_run_id": context.benchmark_run_id,
            "task_id": context.task_id,
            "duration_ms": round((time.perf_counter() - self._started) * 1000, 3),
            **dict(self._observation.metadata),
            **self._metadata,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        if self._commit is not None:
            fields.update(commit_metadata(self._commit))
        event(
            "DEBUG" if error is None else "WARNING",
            f"observation.{self._observation.name}.completed",
            subsystem="observability",
            **fields,
            **(exception_fields(error) if error is not None else {}),
        )


class LoggingObserver:
    def __init__(self) -> None:
        self._recorded_hashes: set[str] = set()

    def start(self, observation: Observation) -> ObservationSpan:
        return _LoggingSpan(observation)

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        if commit.hash in self._recorded_hashes:
            return
        self._recorded_hashes.add(commit.hash)
        fields = commit_metadata(commit)
        if context is not None:
            fields.update(
                benchmark_run_id=context.benchmark_run_id,
                task_id=context.task_id,
                workflow_id=context.temporal_workflow_id,
                workflow_run_id=context.temporal_run_id,
            )
        event(
            "DEBUG",
            "state.commit",
            subsystem="persistence",
            **{key: value for key, value in fields.items() if value is not None},
        )

    def flush(self) -> None:
        return None


__all__ = ["LoggingObserver"]
