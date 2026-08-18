"""Local Loguru observer for Corral's existing structured observations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from corral.logging import event, exception_fields

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.core.action import Action
    from corral.core.state import State
    from corral.observability.base import Observation, ObservationSpan


def _correlation_fields(observation: Observation) -> dict[str, Any]:
    context = observation.context
    state = observation.state_before
    action = observation.action
    return {
        key: value
        for key, value in {
            "benchmark_run_id": context.benchmark_run_id,
            "execution_id": context.execution_id,
            "task_id": context.task_id,
            "workflow_id": context.temporal_workflow_id,
            "workflow_run_id": context.temporal_run_id,
            "state_hash": state.state_hash if state is not None else None,
            "action_id": action.id if action is not None else None,
            **dict(observation.metadata),
        }.items()
        if value is not None
    }


def _start_event(observation: Observation) -> tuple[str, str] | None:
    if observation.name == "task.run":
        return (
            "INFO",
            "task.resumed" if observation.metadata.get("resumed") else "task.started",
        )
    if observation.name == "task.evaluate":
        return "INFO", "evaluation.started"
    if observation.name.startswith("tool."):
        return "DEBUG", "tool.started"
    return None


class _LoggingSpan:
    def __init__(self, observation: Observation) -> None:
        self._observation = observation
        self._started = time.perf_counter()
        self._state_after: State | None = None
        self._action: Action | None = None
        self._metadata: dict[str, Any] = {}
        self._ended = False

    def update(
        self,
        *,
        state_after: State | None = None,
        action: Action | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del output  # full results belong in the trace/payload sink, not INFO logs
        if state_after is not None:
            self._state_after = state_after
        if action is not None:
            self._action = action
        if metadata:
            self._metadata.update(metadata)

    def end(self, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        observation = self._observation
        state = self._state_after or observation.state_before
        action = self._action or observation.action
        duration_ms = round((time.perf_counter() - self._started) * 1000, 3)
        fields = {
            **_correlation_fields(observation),
            **self._metadata,
            "duration_ms": duration_ms,
            **(
                {"state_hash": state.state_hash, "status": state.runtime.status}
                if state is not None
                else {}
            ),
            **({"action_id": action.id} if action is not None else {}),
        }

        if observation.name == "task.run":
            state_failed = state is not None and state.runtime.status == "failed"
            if error is not None and not state_failed:
                event(
                    "WARNING",
                    "task.attempt_failed",
                    subsystem="runtime",
                    **fields,
                    **exception_fields(error),
                )
            elif state_failed:
                error_values: dict[str, Any] = {}
                if error is not None:
                    error_values = exception_fields(error)
                elif state is not None:
                    message = state.runtime.metadata.get("error")
                    if message is not None:
                        error_values = {
                            "error_type": state.runtime.metadata.get(
                                "error_type", "TaskFailure"
                            ),
                            "error_message": str(message),
                        }
                event(
                    "ERROR",
                    "task.failed",
                    subsystem="runtime",
                    **fields,
                    **error_values,
                )
            else:
                event(
                    "INFO",
                    "task.completed",
                    subsystem="runtime",
                    **fields,
                )
            return

        if observation.name == "task.evaluate":
            if error is not None:
                event(
                    "WARNING" if observation.metadata.get("recoverable") else "ERROR",
                    (
                        "evaluation.attempt_failed"
                        if observation.metadata.get("recoverable")
                        else "evaluation.failed"
                    ),
                    subsystem="evaluation",
                    **fields,
                    **exception_fields(error),
                )
            else:
                event(
                    "INFO",
                    "evaluation.completed",
                    subsystem="evaluation",
                    **fields,
                )
            return

        if observation.name == "state.commit":
            event(
                "DEBUG",
                "state.committed",
                subsystem="persistence",
                **fields,
            )
            return

        if observation.name.startswith("tool."):
            succeeded = error is None
            event(
                "DEBUG" if succeeded else "WARNING",
                "tool.completed" if succeeded else "tool.failed",
                subsystem="tool",
                tool_name=observation.name.removeprefix("tool."),
                **fields,
                **(exception_fields(error) if error is not None else {}),
            )
            return

        event(
            "DEBUG" if error is None else "WARNING",
            f"observation.{observation.name}.completed",
            subsystem="observability",
            **fields,
            **(exception_fields(error) if error is not None else {}),
        )


class LoggingObserver:
    """Translate task observations into local schema-compliant log events."""

    def start(self, observation: Observation) -> ObservationSpan:
        started = _start_event(observation)
        if started is not None:
            level, name = started
            subsystem = (
                "evaluation"
                if observation.name == "task.evaluate"
                else "tool"
                if observation.name.startswith("tool.")
                else "runtime"
            )
            event(level, name, subsystem=subsystem, **_correlation_fields(observation))
        return _LoggingSpan(observation)

    def flush(self) -> None:
        return None


__all__ = ["LoggingObserver"]
