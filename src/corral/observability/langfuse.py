"""Langfuse implementation of Corral's passive task observer."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from .base import (
    NoOpObserver,
    Observation,
    ObservationSpan,
    Observer,
    observation_input,
    observation_metadata,
    observation_output,
)

if TYPE_CHECKING:
    from langfuse import Langfuse

    from corral.core.action import Action
    from corral.core.state import State

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "credential",
    "hiddenargument",
    "password",
    "privatekey",
    "secret",
)
_SENSITIVE_TOKEN_KEYS = {
    "accesstoken",
    "authtoken",
    "idtoken",
    "refreshtoken",
    "token",
}
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_KEY_PATTERN = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def _sensitive_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return normalized in _SENSITIVE_TOKEN_KEYS or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


def mask_sensitive_data(data: Any = None, **kwargs: Any) -> Any:
    """Redact credentials and common PII before trace data is exported."""

    if data is None and "data" in kwargs:
        data = kwargs["data"]
    if isinstance(data, Mapping):
        return {
            str(key): (_REDACTED if _sensitive_key(key) else mask_sensitive_data(value))
            for key, value in data.items()
        }
    if isinstance(data, tuple | list | set | frozenset):
        return [mask_sensitive_data(value) for value in data]
    if isinstance(data, str):
        value = _BEARER_PATTERN.sub("Bearer [REDACTED]", data)
        value = _KEY_PATTERN.sub(_REDACTED, value)
        return _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", value)
    return data


def deterministic_trace_id(execution_id: str) -> str:
    """Map one Corral task execution onto a stable W3C trace ID."""

    return hashlib.sha256(f"corral:{execution_id}".encode()).hexdigest()[:32]


def _null_attribute_scope(**_kwargs: Any) -> AbstractContextManager[Any]:
    return nullcontext()


class _LangfuseSpan:
    def __init__(
        self,
        observation: Observation,
        span: Any,
        observation_scope: AbstractContextManager[Any],
        attribute_scope: AbstractContextManager[Any],
        *,
        mask: Callable[..., Any],
    ) -> None:
        self._observation = observation
        self._span = span
        self._observation_scope = observation_scope
        self._attribute_scope = attribute_scope
        self._mask = mask
        self._state_after: State | None = None
        self._action: Action | None = None
        self._output: Mapping[str, Any] | None = None
        self._metadata: Mapping[str, Any] | None = None
        self._started = time.perf_counter()
        self._ended = False

    def update(
        self,
        *,
        state_after: State | None = None,
        action: Action | None = None,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if state_after is not None:
            self._state_after = state_after
        if action is not None:
            self._action = action
        if output is not None:
            self._output = output
        if metadata is not None:
            self._metadata = metadata

    def end(self, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        duration_ms = round((time.perf_counter() - self._started) * 1000, 3)
        metadata = observation_metadata(
            self._observation,
            state_after=self._state_after,
            action=self._action,
            metadata={
                **dict(self._metadata or {}),
                "duration_ms": duration_ms,
                **(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    if error is not None
                    else {}
                ),
            },
        )
        update: dict[str, Any] = {
            "output": self._mask(
                data=observation_output(
                    self._observation,
                    state_after=self._state_after,
                    output=self._output,
                )
            ),
            "metadata": self._mask(data=metadata),
        }
        if error is not None:
            update.update(level="ERROR", status_message=str(error))

        state_before = self._observation.state_before
        state_after = self._state_after
        if self._observation.as_type == "generation" and state_after is not None:
            before_usage = state_before.usage if state_before is not None else None
            input_tokens = state_after.usage.input_tokens - (
                before_usage.input_tokens if before_usage is not None else 0
            )
            output_tokens = state_after.usage.output_tokens - (
                before_usage.output_tokens if before_usage is not None else 0
            )
            update["usage_details"] = {
                "input": max(0, input_tokens),
                "output": max(0, output_tokens),
                "total": max(0, input_tokens + output_tokens),
            }

        try:
            self._span.update(**update)
        finally:
            try:
                self._observation_scope.__exit__(None, None, None)
            finally:
                self._attribute_scope.__exit__(None, None, None)


class LangfuseObserver:
    """Export each task transition into one deterministic Langfuse trace."""

    def __init__(
        self,
        client: Langfuse | Any | None = None,
        *,
        mask: Callable[..., Any] = mask_sensitive_data,
        attribute_scope_factory: Callable[..., AbstractContextManager[Any]]
        | None = None,
    ) -> None:
        if client is None:
            from langfuse import Langfuse, propagate_attributes

            client = Langfuse(mask=mask)
            attribute_scope_factory = propagate_attributes
        elif attribute_scope_factory is None:
            attribute_scope_factory = _null_attribute_scope
        self.client = client
        self._mask = mask
        self._attribute_scope_factory = attribute_scope_factory

    def start(self, observation: Observation) -> ObservationSpan:
        context = observation.context
        trace_id = deterministic_trace_id(context.execution_id)
        propagated_metadata = {
            "execution_id": context.execution_id,
            **(
                {"benchmark_run_id": context.benchmark_run_id}
                if context.benchmark_run_id is not None
                else {}
            ),
            **({"task_id": context.task_id} if context.task_id is not None else {}),
        }
        attributes: dict[str, Any] = {
            "trace_name": (
                f"corral.task.{context.task_id}"
                if context.task_id is not None
                else "corral.task"
            ),
            "metadata": propagated_metadata,
        }
        if context.benchmark_run_id is not None:
            attributes["session_id"] = context.benchmark_run_id

        attribute_scope = self._attribute_scope_factory(**attributes)
        attribute_scope.__enter__()
        try:
            current_trace_id = None
            get_current_trace_id = getattr(self.client, "get_current_trace_id", None)
            if get_current_trace_id is not None:
                current_trace_id = get_current_trace_id()

            start_args: dict[str, Any] = {
                "name": observation.name,
                "as_type": observation.as_type,
                "input": self._mask(data=observation_input(observation)),
                "metadata": self._mask(
                    data=observation_metadata(
                        observation,
                        state_after=None,
                        action=None,
                        metadata=None,
                    )
                ),
            }
            if current_trace_id != trace_id:
                start_args["trace_context"] = {"trace_id": trace_id}
            if observation.as_type == "generation":
                state = observation.state_before
                if state is not None:
                    model = state.metadata.model.get("name")
                    if model is not None:
                        start_args["model"] = str(model)

            observation_scope = self.client.start_as_current_observation(**start_args)
            span = observation_scope.__enter__()
        except BaseException:
            attribute_scope.__exit__(None, None, None)
            raise
        return _LangfuseSpan(
            observation,
            span,
            observation_scope,
            attribute_scope,
            mask=self._mask,
        )

    def flush(self) -> None:
        flush = getattr(self.client, "flush", None)
        if flush is not None:
            flush()


def observer_from_env() -> Observer:
    """Create Langfuse only when credentials are present; otherwise no-op."""

    enabled = os.getenv("CORRAL_LANGFUSE_ENABLED", "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return NoOpObserver()
    configured = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )
    if not configured and enabled not in {"1", "true", "yes", "on"}:
        return NoOpObserver()
    try:
        return LangfuseObserver()
    except BaseException:
        return NoOpObserver()


__all__ = [
    "LangfuseObserver",
    "deterministic_trace_id",
    "mask_sensitive_data",
    "observer_from_env",
]
