"""Architecture tests for passive, State-centric task observation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.core.transition import ToolExecutionResult
from corral.observability import (
    LangfuseObserver,
    NoOpObserver,
    Observation,
    ObservationContext,
    deterministic_trace_id,
    mask_sensitive_data,
    observe_safely,
    state_changes,
    update_safely,
)
from corral.persistence import JSONLStateStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@dataclass
class RecordedObservation:
    observation: Observation
    state_after: Any = None
    action: Action | None = None
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: BaseException | None = None


class RecordingSpan:
    def __init__(
        self,
        observation: Observation,
        records: list[RecordedObservation],
    ) -> None:
        self.record = RecordedObservation(observation)
        self.records = records

    def update(
        self,
        *,
        state_after=None,
        action=None,
        output=None,
        metadata=None,
    ) -> None:
        self.record.state_after = state_after
        self.record.action = action
        self.record.output = dict(output) if output is not None else None
        self.record.metadata.update(metadata or {})

    def end(self, error=None) -> None:
        self.record.error = error
        self.records.append(self.record)


class RecordingObserver:
    def __init__(self) -> None:
        self.records: list[RecordedObservation] = []

    def start(self, observation: Observation) -> RecordingSpan:
        return RecordingSpan(observation, self.records)

    def flush(self) -> None:
        return None


class BrokenObserver:
    def start(self, observation: Observation):
        del observation
        raise ConnectionError("tracing backend is unavailable")

    def flush(self) -> None:
        raise ConnectionError("tracing backend is unavailable")


class ChangeEnvironmentAgent:
    model = "test-model"

    async def run_session(self, session):
        await session.execute(
            Action(id="change-environment", name="set_temperature", arguments={})
        )
        return AgentOutcome(
            status="agent_failure",
            error="observation fixture completed",
        )


def _environment() -> Environment:
    def set_temperature() -> ToolExecutionResult:
        """Record a structured environment update."""
        return ToolExecutionResult(
            content="temperature set",
            environment={
                "hidden_arguments": {"api_key": "sk-super-secret"},
                "jobs": {},
                "values": {"temperature": 298.15},
            },
        )

    task = TaskDefinition(
        name="observed-task",
        description="change the environment",
        tools=["set_temperature"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    return Environment(
        "observed-task",
        task,
        toolset=Toolset(
            pool={"set_temperature": tool(set_temperature)},
            workspace_factory=None,
        ),
    )


@pytest.mark.anyio()
async def test_runtime_observes_complete_state_and_environment_changes(tmp_path):
    observer = RecordingObserver()
    environment = _environment()
    state_path = tmp_path / "states.jsonl"

    with JSONLStateStore(state_path) as store:
        runtime = TaskRuntime(store, observer)
        observed = await runtime.run(
            ChangeEnvironmentAgent(),
            environment,
            execution_id="observed-execution",
            started_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            max_iterations=1,
            model_metadata={"name": "test-model"},
            scaffold_metadata={"name": "ChangeEnvironmentAgent"},
        )

    tool_record = next(
        record
        for record in observer.records
        if record.observation.name == "tool.set_temperature"
    )
    assert tool_record.observation.state_before.pending_action is not None
    assert tool_record.state_after.environment["values"] == {"temperature": 298.15}
    with JSONLStateStore(state_path) as store:
        persisted_before = await store.load(
            tool_record.observation.state_before.state_hash
        )
        persisted_after = await store.load(tool_record.state_after.state_hash)
    assert persisted_before.environment["values"] == {}
    assert persisted_before.pending_action is not None
    assert persisted_after.environment["values"] == {"temperature": 298.15}
    assert persisted_after.parent_hash == persisted_before.state_hash
    changes = state_changes(
        tool_record.observation.state_before,
        tool_record.state_after,
    )
    assert changes["environment"]["before"]["values"] == {}
    assert changes["environment"]["after"]["values"] == {"temperature": 298.15}
    assert any(
        record.observation.name == "state.commit" and record.state_after == observed
        for record in observer.records
    )
    assert {record.observation.name for record in observer.records} >= {
        "task.run",
        "state.commit",
        "tool.set_temperature",
    }


@pytest.mark.anyio()
async def test_observer_outage_does_not_change_task_execution(tmp_path):
    environment = _environment()
    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        observed = await TaskRuntime(store, BrokenObserver()).run(
            ChangeEnvironmentAgent(),
            environment,
            execution_id="observer-outage",
            started_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert observed.environment["values"] == {"temperature": 298.15}
    assert observed.usage.agent_steps == 1
    assert observed.usage.tool_calls == 1


class FakeLangfuseSpan:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeObservationScope:
    def __init__(self, client, call, span) -> None:
        self.client = client
        self.call = call
        self.span = span
        self.previous_trace_id = None

    def __enter__(self):
        self.previous_trace_id = self.client.current_trace_id
        trace_context = self.call.get("trace_context")
        if trace_context is not None:
            self.client.current_trace_id = trace_context["trace_id"]
        return self.span

    def __exit__(self, *_args):
        self.client.current_trace_id = self.previous_trace_id


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.spans: list[FakeLangfuseSpan] = []
        self.current_trace_id = None

    def get_current_trace_id(self):
        return self.current_trace_id

    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        span = FakeLangfuseSpan()
        self.spans.append(span)
        return FakeObservationScope(self, kwargs, span)


def test_langfuse_uses_deterministic_trace_session_and_masks_state():
    client = FakeLangfuseClient()
    propagated: list[dict[str, Any]] = []

    @contextmanager
    def attributes(**kwargs):
        propagated.append(kwargs)
        yield

    observer = LangfuseObserver(
        client,
        attribute_scope_factory=attributes,
    )
    context = ObservationContext(
        execution_id="benchmark-1:task-a:0",
        benchmark_run_id="benchmark-1",
        task_id="task-a",
        temporal_workflow_id="corral/benchmark-1/task-a/0",
    )
    with observe_safely(
        observer,
        Observation(
            name="state.commit",
            context=context,
            input={"api_key": "sk-1234567890"},
        ),
    ) as span:
        update_safely(
            span,
            output={"environment": {"hidden_arguments": {"password": "do-not-export"}}},
        )

    call = client.calls[0]
    assert call["trace_context"] == {
        "trace_id": deterministic_trace_id(context.execution_id)
    }
    assert call["input"]["operation"]["api_key"] == "[REDACTED]"
    assert propagated[0]["session_id"] == "benchmark-1"
    assert propagated[0]["trace_name"] == "corral.task.task-a"
    result = client.spans[0].updates[0]["output"]["result"]
    assert result["environment"]["hidden_arguments"] == "[REDACTED]"


def test_mask_sensitive_data_recurses_through_state_payloads():
    masked = mask_sensitive_data(
        {
            "environment": {
                "hidden_arguments": {"api_key": "sk-1234567890"},
                "values": {"owner": "person@example.com"},
            },
            "authorization": "Bearer abc123",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }
    )
    assert masked == {
        "environment": {
            "hidden_arguments": "[REDACTED]",
            "values": {"owner": "[EMAIL_REDACTED]"},
        },
        "authorization": "[REDACTED]",
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }


def test_noop_observer_is_a_valid_default():
    observer = NoOpObserver()
    with observe_safely(
        observer,
        Observation(
            name="restore",
            context=ObservationContext(execution_id="task"),
        ),
    ):
        pass
