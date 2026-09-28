"""Langfuse projection of conversation deltas authored by commits."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from corral.core.events import (
    AgentStarted,
    AgentTurnRecorded,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    SubmissionAccepted,
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
_SELECTED_TRACE: ContextVar[str | None] = ContextVar("corral_trace", default=None)


@lru_cache
def _task_tracer_provider(public_key: str | None) -> Any:
    # Langfuse shares its exporter by public key, so observers must share the
    # provider too. Otherwise only the first observer's spans are exported.
    del public_key
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    class TaskIdGenerator(RandomIdGenerator):
        def generate_trace_id(self) -> int:
            value = _SELECTED_TRACE.get()
            return int(value, 16) if value else super().generate_trace_id()

    return TracerProvider(id_generator=TaskIdGenerator())


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


class _EvaluationSpan:
    """Consume the existing task.evaluate callback without changing Observer."""

    def __init__(self, observer: LangfuseObserver, context: ObservationContext) -> None:
        self.observer = observer
        self.context = context
        self.result: Mapping[str, Any] | None = None

    def update(self, *, commit=None, output=None, metadata=None) -> None:
        del commit, metadata
        self.result = output

    def end(self, error: BaseException | None = None) -> None:
        if error is not None or self.result is None or self.result.get("score") is None:
            return
        result = self.result
        identity = f"{self.context.execution_id}:{result['commit_hash']}:{result['scorer_version']}"
        self.observer.client.create_score(
            trace_id=deterministic_trace_id(self.context.execution_id),
            score_id=hashlib.sha256(identity.encode()).hexdigest(),
            name="benchmark_score",
            value=float(result["score"]),
            data_type="NUMERIC",
            comment=self.observer._mask(data=result.get("feedback")),
            metadata=self.observer._mask(
                data={
                    "commit_hash": result["commit_hash"],
                    "scorer_version": result["scorer_version"],
                }
            ),
            timestamp=self.observer._last_times.get(
                self.context.execution_id, datetime.now(timezone.utc)
            ),
        )
        self.result = None


@dataclass
class _RestoredGeneration:
    commit: Commit
    context: ObservationContext
    started_at: datetime
    arguments: dict[str, Any]


class LangfuseObserver:
    def __init__(
        self,
        client: Langfuse | Any | None = None,
        *,
        mask: Callable[..., Any] = mask_sensitive_data,
        attribute_scope_factory: Callable[..., AbstractContextManager[Any]]
        | None = None,
        tracer_provider: Any = None,
    ) -> None:
        self._public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self._provider = tracer_provider or _task_tracer_provider(self._public_key)
        if client is None:
            from langfuse import Langfuse, propagate_attributes

            client = Langfuse(mask=mask, tracer_provider=self._provider)
            attribute_scope_factory = propagate_attributes
        elif attribute_scope_factory is None:
            attribute_scope_factory = _null_attribute_scope
        self.client = client
        self._mask = mask
        self._attribute_scope_factory = attribute_scope_factory
        self._recorded_hashes: set[str] = set()
        self._restoring = False
        self._restored_steps: set[tuple[str, str]] = set()
        self._restored_generations: dict[tuple[str, str], _RestoredGeneration] = {}
        self._noop = NoOpObserver()
        self._models: dict[str, str] = {}
        self._agents: dict[tuple[str, str], str] = {}
        self._tools: dict[tuple[str, str], str] = {}
        self._pending_inputs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        self._roots: dict[str, Any] = {}
        self._steps: dict[tuple[str, str], Any] = {}
        self._step_numbers: dict[tuple[str, str], int] = {}
        self._generations: dict[
            tuple[str, str], tuple[Any, dict[str, Any], datetime]
        ] = {}
        self._actions: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._tool_starts: dict[tuple[str, str], datetime] = {}
        self._task_inputs: dict[str, list[Mapping[str, Any]]] = {}
        self._last_times: dict[Any, datetime] = {}

    def start(self, observation: Observation) -> ObservationSpan:
        if observation.name == "task.evaluate":
            return _EvaluationSpan(self, observation.context)
        # Runtime/task/tool spans are intentionally local-only. Persisted model
        # turns below are the sole Langfuse export boundary.
        return self._noop.start(observation)

    def _start_span(self, context, *, name, as_type, started_at, parent=None, **kwargs):
        # The SDK's high-level start method cannot set a historical start time.
        # Use its public OTEL provider and public observation wrappers; export,
        # masking, batching, and scoring remain owned by the Langfuse SDK.
        from langfuse import (
            LangfuseAgent,
            LangfuseGeneration,
            LangfuseSpan,
            LangfuseTool,
        )
        from opentelemetry.context import Context
        from opentelemetry.trace import (
            NonRecordingSpan,
            SpanContext,
            TraceFlags,
            set_span_in_context,
        )

        parent_context = Context()
        if parent is not None:
            parent_context = set_span_in_context(
                NonRecordingSpan(
                    SpanContext(
                        trace_id=int(parent.trace_id, 16),
                        span_id=int(parent.id, 16),
                        is_remote=False,
                        trace_flags=TraceFlags(1),
                    )
                ),
                parent_context,
            )
        token = _SELECTED_TRACE.set(deterministic_trace_id(context.execution_id))
        try:
            tracer = self._provider.get_tracer(
                "langfuse-sdk",
                attributes={
                    "public_key": self._public_key,
                },
            )
            otel_span = tracer.start_span(
                name,
                context=parent_context,
                start_time=int(started_at.timestamp() * 1e9),
                attributes={
                    "langfuse.trace.name": f"corral.task.{context.task_id or context.execution_id}",
                    "session.id": context.benchmark_run_id or context.execution_id,
                    "langfuse.trace.metadata.execution_id": context.execution_id,
                },
            )
            wrapper = {
                "agent": LangfuseAgent,
                "span": LangfuseSpan,
                "generation": LangfuseGeneration,
                "tool": LangfuseTool,
            }[as_type]
            return wrapper(otel_span=otel_span, langfuse_client=self.client, **kwargs)
        finally:
            _SELECTED_TRACE.reset(token)

    def _end_step(self, key: tuple[str, str]) -> None:
        self._restored_steps.discard(key)
        self._restored_generations.pop(key, None)
        generation = self._generations.pop(key, None)
        if generation is not None:
            span, output, ended_at = generation
            span.update(output=self._mask(data=output))
            span.end(end_time=int(ended_at.timestamp() * 1e9))
        step = self._steps.pop(key, None)
        if step is not None:
            step.end(end_time=int(self._last_times[key].timestamp() * 1e9))

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
        input_messages: Any = None,
        usage: UsageDelta | None = None,
        set_trace_input: bool = False,
    ) -> None:
        key = (commit.execution_id, run_id)
        if self._restoring:
            if set_trace_input:
                self._last_times[commit.execution_id] = commit.occurred_at
                return
            if as_type == "generation":
                self._restored_steps.discard(key)
                # Only the final, unclosed generation was never exported.
                # Older generations must not be replayed: Langfuse ingestion
                # does not deduplicate repeated observations.
                self._restored_generations[key] = _RestoredGeneration(
                    commit,
                    context,
                    self._last_times.get(key, commit.occurred_at),
                    {
                        "name": name,
                        "as_type": as_type,
                        "output": json.loads(json.dumps(output)),
                        "run_id": run_id,
                        "input_messages": input_messages,
                        "usage": usage,
                    },
                )
            if key not in self._restored_steps:
                self._step_numbers[key] = self._step_numbers.get(key, 0) + 1
                self._restored_steps.add(key)
            if as_type == "tool":
                self._tool_starts.pop(
                    (commit.execution_id, commit.event.invocation_id), None
                )
            self._last_times[key] = commit.occurred_at
            return
        metadata = self._metadata(commit, run_id=run_id)
        root = self._roots.get(commit.execution_id)
        if root is None:
            root = self._start_span(
                context,
                name=f"Task · {context.task_id or commit.execution_id}",
                as_type="agent",
                started_at=self._last_times.get(
                    commit.execution_id, commit.occurred_at
                ),
                input=self._mask(
                    data=input_messages
                    if set_trace_input
                    else self._task_inputs.get(commit.execution_id)
                ),
                metadata=self._mask(data=metadata),
            )
            self._roots[commit.execution_id] = root
            self._last_times[commit.execution_id] = commit.occurred_at
        if set_trace_input:
            return
        if as_type == "generation":
            self._end_step(key)
        if key not in self._steps:
            if key not in self._restored_steps:
                self._step_numbers[key] = self._step_numbers.get(key, 0) + 1
            self._restored_steps.discard(key)
            self._steps[key] = self._start_span(
                context,
                name=f"Step {self._step_numbers[key]:02d}",
                as_type="span",
                parent=root,
                started_at=self._last_times.get(key, commit.occurred_at),
            )
        started_at = self._last_times.get(key, commit.occurred_at)
        if as_type == "tool":
            started_at = self._tool_starts.pop(
                (commit.execution_id, commit.event.invocation_id), commit.occurred_at
            )
            metadata.update(
                execution_duration_ms=commit.event.duration_ms,
                timing_source="tool_start_and_completion_commits",
            )
        else:
            metadata.update(
                timing_source="estimated_between_commits",
                input_source="reconstructed_conversation",
            )
        args: dict[str, Any] = {}
        if as_type == "generation":
            args["model"] = self._models.get(commit.execution_id)
            args["usage_details"] = _usage_details(usage) if usage is not None else None
        span = self._start_span(
            context,
            name=name,
            as_type=as_type,
            parent=self._steps[key],
            started_at=started_at,
            input=self._mask(data=input_messages),
            metadata=self._mask(data=metadata),
            **args,
        )
        if as_type == "generation":
            # A provider response can arrive as multiple action commits. Keep
            # its output editable until the step is complete, then export once.
            self._generations[key] = (
                span,
                json.loads(json.dumps(output)),
                commit.occurred_at,
            )
        else:
            failed = (
                isinstance(commit.event, ToolFailed) or commit.event.status != "success"
            )
            span.update(
                output=self._mask(data=dict(output)),
                level="ERROR" if failed else "DEFAULT",
            )
            span.end(end_time=int(commit.occurred_at.timestamp() * 1e9))
        self._last_times[key] = commit.occurred_at

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
            *self._pending_inputs.get(key, []),
            *messages[:output_index],
        ]
        output = messages[output_index]
        trailing = messages[output_index + 1 :]
        if (
            event.usage_delta.llm_calls == 0
            and event.actions
            and (key in self._steps or key in self._restored_steps)
        ):
            self._pending_inputs[key] = [*inputs, output, *trailing]
            generation = self._generations.get(key)
            restored = self._restored_generations.get(key)
            generation_output = (
                generation[1]
                if generation is not None
                else restored.arguments["output"]
                if restored is not None
                else None
            )
            if generation_output is not None:
                calls = generation_output.setdefault("tool_calls", [])
                known = {call.get("id") for call in calls}
                calls.extend(
                    action.to_tool_call()
                    for action in event.actions
                    if action.id not in known
                )
            return
        self._pending_inputs[key] = [*inputs, output, *trailing]
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
            input_messages=self._actions.pop(
                (commit.execution_id, event.action_id), None
            ),
            run_id=event.requested_by_run_id,
        )

    def restore_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        """Rebuild context, retaining the generation interrupted before export."""
        self._restoring = True
        try:
            self.record_commit(commit, context=context)
        finally:
            self._restoring = False

    def record_commit(
        self, commit: Commit, *, context: ObservationContext | None = None
    ) -> None:
        if commit.hash in self._recorded_hashes:
            return
        if not self._restoring:
            for key, generation in list(self._restored_generations.items()):
                if key[0] != commit.execution_id:
                    continue
                self._restored_generations.pop(key)
                last_time = self._last_times[key]
                self._last_times[key] = generation.started_at
                # _record_delta opens this same step, rather than a new one.
                self._step_numbers[key] -= 1
                self._record_delta(
                    generation.commit, generation.context, **generation.arguments
                )
                self._last_times[key] = last_time
        self._recorded_hashes.add(commit.hash)
        selected_context = context or ObservationContext(
            execution_id=commit.execution_id
        )
        commit_event = commit.event
        if isinstance(commit_event, ExecutionStarted):
            self._task_inputs[commit.execution_id] = _execution_started_messages(
                commit_event
            )
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
            key = (commit.execution_id, commit_event.agent_run_id)
            self._last_times[key] = commit.occurred_at
            self._pending_inputs[key] = list(
                self._task_inputs.get(commit.execution_id, [])
            )
            self._agents[(commit.execution_id, commit_event.agent_run_id)] = (
                commit_event.agent_id
            )
            return
        if isinstance(commit_event, ToolStarted):
            self._tool_starts[(commit.execution_id, commit_event.invocation_id)] = (
                commit.occurred_at
            )
            self._tools[(commit.execution_id, commit_event.invocation_id)] = (
                commit_event.tool_name
            )
            return
        if isinstance(commit_event, AgentTurnRecorded):
            for action in commit_event.actions:
                self._actions[(commit.execution_id, action.id)] = dict(action.arguments)
            self._record_agent_messages(commit, commit_event, selected_context)
            return
        if isinstance(commit_event, ToolCompleted | ToolFailed):
            self._record_tool_result(commit, commit_event, selected_context)
            return
        if isinstance(commit_event, SubmissionAccepted):
            root = self._roots.get(commit.execution_id)
            if root is not None:
                root.update(
                    output=self._mask(
                        data={
                            "answer": commit_event.answer,
                            "status": "surrendered"
                            if commit_event.surrendered
                            else "submitted",
                        }
                    )
                )
            return
        if isinstance(commit_event, ExecutionCompleted | ExecutionFailed):
            for key in self._steps.keys() | self._restored_steps:
                if key[0] == commit.execution_id:
                    self._end_step(key)
            root = self._roots.pop(commit.execution_id, None)
            if root is not None:
                if isinstance(commit_event, ExecutionFailed):
                    root.update(
                        output=self._mask(data={"error": commit_event.error}),
                        level="ERROR",
                    )
                root.end(end_time=int(commit.occurred_at.timestamp() * 1e9))
            self._last_times[commit.execution_id] = commit.occurred_at

    def flush(self) -> None:
        flush = getattr(self.client, "flush", None)
        if flush is not None:
            flush()


def langfuse_enabled() -> bool:
    """Whether environment configuration requests Langfuse export."""
    enabled = os.getenv("CORRAL_LANGFUSE_ENABLED", "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False
    configured = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )
    return configured or enabled in {"1", "true", "yes", "on"}


def observer_from_env() -> Observer:
    local = LoggingObserver()
    if not langfuse_enabled():
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
    "langfuse_enabled",
    "mask_sensitive_data",
    "observer_from_env",
]
