"""Langfuse projection of conversation deltas authored by commits."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from corral.core.events import (
    AgentStarted,
    AgentTurnRecorded,
    ExecutionStarted,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    UsageDelta,
)
from corral.observability.base import (
    CompositeObserver,
    NoOpObserver,
    Observation,
    ObservationContext,
    ObservationSpan,
    Observer,
)
from corral.observability.logging import LoggingObserver
from corral.report.logging import event, exception_fields, redact_sensitive_data

if TYPE_CHECKING:
    from langfuse import Langfuse

    from corral.core.commit import Commit

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
    if data is None and "data" in kwargs:
        data = kwargs["data"]
    data = redact_sensitive_data(data, include_payloads=True)
    if isinstance(data, Mapping):
        operation_path = data.get("path")
        masks_operation_value = (
            isinstance(operation_path, tuple | list)
            and bool(operation_path)
            and _sensitive_key(operation_path[-1])
        )
        return {
            str(key): (
                _REDACTED
                if _sensitive_key(key) or (key == "value" and masks_operation_value)
                else mask_sensitive_data(value)
            )
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
    return hashlib.sha256(f"corral:{execution_id}".encode()).hexdigest()[:32]


def _null_attribute_scope(**_kwargs: Any) -> AbstractContextManager[Any]:
    return nullcontext()


def _usage_details(delta: UsageDelta) -> dict[str, Any] | None:
    if not any(
        (
            delta.input_tokens,
            delta.output_tokens,
            delta.reasoning_tokens,
            delta.llm_calls,
        )
    ):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": delta.input_tokens,
        "completion_tokens": delta.output_tokens,
        "total_tokens": delta.input_tokens + delta.output_tokens,
    }
    if delta.reasoning_tokens:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": delta.reasoning_tokens
        }
    return usage


def _execution_started_messages(started: ExecutionStarted) -> list[Mapping[str, Any]]:
    prompt = started.task.get("prompt")
    if prompt is None:
        return []
    if isinstance(prompt, list) and all(
        isinstance(message, Mapping) and isinstance(message.get("role"), str)
        for message in prompt
    ):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": prompt}]


class LangfuseObserver:
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
        self._recorded_hashes: set[str] = set()
        self._noop = NoOpObserver()
        self._models: dict[str, str] = {}
        self._agents: dict[tuple[str, str], str] = {}
        self._tools: dict[tuple[str, str], str] = {}
        self._pending_inputs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}

    def start(self, observation: Observation) -> ObservationSpan:
        # Runtime/task/tool spans are intentionally local-only. Persisted model
        # turns below are the sole Langfuse export boundary.
        return self._noop.start(observation)

    def _metadata(self, commit: Commit, *, run_id: str | None = None) -> dict[str, Any]:
        selected_run = run_id or commit.author.run_id
        metadata: dict[str, Any] = {
            "model": self._models.get(commit.execution_id),
            "agent": self._agents.get(
                (commit.execution_id, selected_run),
                commit.author.actor_id if commit.author.kind == "agent" else None,
            ),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _record_delta(
        self,
        commit: Commit,
        context: ObservationContext,
        *,
        name: str,
        as_type: str,
        output: Mapping[str, Any],
        run_id: str,
        input_messages: list[Mapping[str, Any]] | None = None,
        usage: UsageDelta | None = None,
        set_trace_input: bool = False,
    ) -> None:
        trace_id = deterministic_trace_id(context.execution_id)
        attributes: dict[str, Any] = {
            "trace_name": (
                f"corral.task.{context.task_id}" if context.task_id else "corral.task"
            ),
        }
        if context.benchmark_run_id is not None:
            attributes["session_id"] = context.benchmark_run_id
        attribute_scope = self._attribute_scope_factory(**attributes)
        with attribute_scope:
            metadata = self._metadata(commit, run_id=run_id)
            model = self._models.get(commit.execution_id)
            args: dict[str, Any] = {
                "name": name,
                "as_type": as_type,
                "metadata": self._mask(data=metadata),
            }
            if input_messages:
                args["input"] = self._mask(data=input_messages)
            if as_type == "generation" and model is not None:
                args["model"] = model
            get_current_trace_id = getattr(self.client, "get_current_trace_id", None)
            current_trace_id = (
                get_current_trace_id() if get_current_trace_id is not None else None
            )
            if current_trace_id != trace_id:
                args["trace_context"] = {"trace_id": trace_id}
            with self.client.start_as_current_observation(**args) as span:
                if set_trace_input and input_messages:
                    set_trace_io = getattr(span, "set_trace_io", None)
                    if set_trace_io is not None:
                        set_trace_io(input=self._mask(data=input_messages))
                update: dict[str, Any] = {
                    "output": self._mask(data=dict(output)),
                }
                usage_details = _usage_details(usage) if usage is not None else None
                if usage_details is not None:
                    update["usage_details"] = usage_details
                span.update(**update)

    def _record_agent_messages(
        self,
        commit: Commit,
        event: AgentTurnRecorded,
        context: ObservationContext,
    ) -> None:
        if not event.messages:
            return
        key = (commit.execution_id, commit.author.run_id)
        messages = [dict(message) for message in event.messages]
        assistant_indexes = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        if not assistant_indexes:
            self._pending_inputs.setdefault(key, []).extend(messages)
            return
        output_index = assistant_indexes[-1]
        inputs = [
            *self._pending_inputs.pop(key, []),
            *messages[:output_index],
        ]
        output = messages[output_index]
        trailing = messages[output_index + 1 :]
        if event.usage_delta.llm_calls == 0 and output.get("content") is None:
            self._pending_inputs[key] = [*inputs, output, *trailing]
            return
        if trailing:
            self._pending_inputs.setdefault(key, []).extend(trailing)
        self._record_delta(
            commit,
            context,
            name=str(output.get("name") or "agent.turn"),
            as_type="generation",
            input_messages=inputs,
            output=output,
            run_id=commit.author.run_id,
            usage=event.usage_delta,
        )

    def _record_tool_result(
        self,
        commit: Commit,
        event: ToolCompleted | ToolFailed,
        context: ObservationContext,
    ) -> None:
        tool_name = self._tools.get(
            (commit.execution_id, event.invocation_id),
            "tool",
        )
        content = event.observation if isinstance(event, ToolCompleted) else event.error
        message: Mapping[str, Any] = {
            "role": "tool",
            "tool_call_id": event.action_id,
            "name": tool_name,
            "content": content,
        }
        key = (commit.execution_id, event.requested_by_run_id)
        self._pending_inputs.setdefault(key, []).append(message)
        self._record_delta(
            commit,
            context,
            name=tool_name,
            as_type="tool",
            output=message,
            run_id=event.requested_by_run_id,
        )

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        if commit.hash in self._recorded_hashes:
            return
        self._recorded_hashes.add(commit.hash)
        selected_context = context or ObservationContext(
            execution_id=commit.execution_id
        )
        commit_event = commit.event
        if isinstance(commit_event, ExecutionStarted):
            model = commit_event.model.get("name")
            if isinstance(model, str) and model:
                self._models[commit.execution_id] = model
            self._record_delta(
                commit,
                selected_context,
                name=commit_event.type,
                as_type="span",
                input_messages=_execution_started_messages(commit_event),
                output={"status": commit_event.runtime.status or "running"},
                run_id=commit.author.run_id,
                set_trace_input=True,
            )
            return
        if isinstance(commit_event, AgentStarted):
            self._agents[(commit.execution_id, commit_event.agent_run_id)] = (
                commit_event.agent_id
            )
            return
        if isinstance(commit_event, ToolStarted):
            self._tools[(commit.execution_id, commit_event.invocation_id)] = (
                commit_event.tool_name
            )
            return
        if isinstance(commit_event, AgentTurnRecorded):
            self._record_agent_messages(commit, commit_event, selected_context)
            return
        if isinstance(commit_event, ToolCompleted | ToolFailed):
            self._record_tool_result(commit, commit_event, selected_context)

    def flush(self) -> None:
        flush = getattr(self.client, "flush", None)
        if flush is not None:
            flush()


def observer_from_env() -> Observer:
    local = LoggingObserver()
    enabled = os.getenv("CORRAL_LANGFUSE_ENABLED", "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return local
    configured = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )
    if not configured and enabled not in {"1", "true", "yes", "on"}:
        return local
    try:
        remote = LangfuseObserver()
    except BaseException as exc:
        event(
            "WARNING",
            "observability.initialization_failed",
            subsystem="observability",
            backend="langfuse",
            **exception_fields(exc),
        )
        return local
    return CompositeObserver(local, remote)


__all__ = [
    "LangfuseObserver",
    "deterministic_trace_id",
    "mask_sensitive_data",
    "observer_from_env",
]
