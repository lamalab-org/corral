from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone

from corral.core import (
    ActorRef,
    AgentStarted,
    AgentTurnRecorded,
    Commit,
    EnvironmentOperation,
    ExecutionStarted,
    ToolCompleted,
    ToolStarted,
    UsageDelta,
)
from corral.observability import (
    LangfuseObserver,
    Observation,
    ObservationContext,
    deterministic_trace_id,
)


class FakeSpan:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.trace_io_updates: list[dict] = []

    def update(self, **values) -> None:
        self.updates.append(values)

    def set_trace_io(self, **values):
        self.trace_io_updates.append(values)
        return self


class FakeScope:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, *args) -> None:
        del args


class FakeLangfuse:
    def __init__(self) -> None:
        self.starts: list[dict] = []
        self.spans: list[FakeSpan] = []

    def get_current_trace_id(self):
        return None

    def start_as_current_observation(self, **values):
        self.starts.append(values)
        span = FakeSpan()
        self.spans.append(span)
        return FakeScope(span)

    def flush(self) -> None:
        return None


def _commit(event, author, sequence) -> Commit:
    occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Commit.create(
        execution_id="execution",
        branch_id="main",
        sequence=sequence,
        branch_sequence=sequence,
        parent_hash=None if sequence == 1 else "a" * 64,
        based_on_hash=None if sequence == 1 else "b" * 64,
        author=author,
        event=event,
        occurred_at=occurred_at,
        recorded_at=occurred_at,
    )


def _contains_state_snapshot(value) -> bool:
    if isinstance(value, dict):
        if "state" in value or "execution_state" in value:
            return True
        return any(_contains_state_snapshot(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_state_snapshot(item) for item in value)
    return False


def test_langfuse_projects_execution_input_and_conversation_deltas():
    client = FakeLangfuse()
    trace_attributes = []
    observer = LangfuseObserver(
        client,
        attribute_scope_factory=lambda **values: (
            trace_attributes.append(values) or nullcontext()
        ),
    )
    context = ObservationContext(
        execution_id="execution",
        benchmark_run_id="benchmark",
        task_id="task",
    )
    runtime = ActorRef(kind="runtime", actor_id="corral", run_id="runtime")
    agent = ActorRef(kind="agent", actor_id="ai-scientist", run_id="agent-run-1")
    tool = ActorRef(kind="tool", actor_id="search", run_id="invocation-1")
    commits = [
        _commit(
            ExecutionStarted(
                task={
                    "prompt": [
                        {"role": "system", "content": "You are a scientist"},
                        {"role": "user", "content": "Form a hypothesis"},
                    ]
                },
                model={"name": "openai/gpt-5.6-terra"},
            ),
            runtime,
            1,
        ),
        _commit(
            AgentStarted(agent_run_id="agent-run-1", agent_id="ai-scientist"),
            runtime,
            2,
        ),
        _commit(
            AgentTurnRecorded(
                messages=(
                    {"role": "system", "content": "Keep sk-sensitive-value private"},
                    {"role": "user", "content": "Form a hypothesis"},
                    {
                        "role": "assistant",
                        "content": "Test dilution first",
                        "name": "task_formulation",
                    },
                ),
                usage_delta=UsageDelta(
                    input_tokens=70,
                    output_tokens=30,
                    reasoning_tokens=20,
                    llm_calls=1,
                ),
            ),
            agent,
            3,
        ),
        _commit(
            ToolStarted(
                action_id="action-1",
                invocation_id="invocation-1",
                requested_by_run_id="agent-run-1",
                tool_name="search",
            ),
            runtime,
            4,
        ),
        _commit(
            ToolCompleted(
                action_id="action-1",
                invocation_id="invocation-1",
                requested_by_run_id="agent-run-1",
                observation={"api_key": "sk-sensitive-value", "result": "found"},
                status="success",
                environment_operations=(
                    EnvironmentOperation(
                        operation="set",
                        path=("credentials", "access_token"),
                        value="secret-value",
                    ),
                ),
                expected_environment_revision=0,
            ),
            tool,
            5,
        ),
    ]

    observer.start(
        Observation(name="task.run", context=context, input={"large": "state"})
    ).end()
    for commit in commits:
        observer.record_commit(commit, context=context)
    observer.record_commit(commits[-1], context=context)

    assert len(client.starts) == 3
    execution_start, generation, tool_start = client.starts
    execution_messages = [
        {"role": "system", "content": "You are a scientist"},
        {"role": "user", "content": "Form a hypothesis"},
    ]
    assert execution_start == {
        "name": "execution.started",
        "as_type": "span",
        "input": execution_messages,
        "metadata": {"model": "openai/gpt-5.6-terra"},
        "trace_context": {"trace_id": deterministic_trace_id("execution")},
    }
    assert client.spans[0].trace_io_updates == [{"input": execution_messages}]
    assert client.spans[0].updates[-1] == {"output": {"status": "running"}}
    assert generation == {
        "name": "task_formulation",
        "as_type": "generation",
        "input": [
            {"role": "system", "content": "Keep [REDACTED] private"},
            {"role": "user", "content": "Form a hypothesis"},
        ],
        "metadata": {
            "model": "openai/gpt-5.6-terra",
            "agent": "ai-scientist",
        },
        "model": "openai/gpt-5.6-terra",
        "trace_context": {"trace_id": deterministic_trace_id("execution")},
    }
    assert client.spans[1].updates[-1] == {
        "output": {
            "role": "assistant",
            "content": "Test dilution first",
            "name": "task_formulation",
        },
        "usage_details": {
            "prompt_tokens": 70,
            "completion_tokens": 30,
            "total_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }
    assert tool_start["name"] == "search"
    assert tool_start["as_type"] == "tool"
    assert tool_start["metadata"] == {
        "model": "openai/gpt-5.6-terra",
        "agent": "ai-scientist",
    }
    assert client.spans[2].updates[-1] == {
        "output": {
            "role": "tool",
            "tool_call_id": "action-1",
            "name": "search",
            "content": {"api_key": "[REDACTED]", "result": "found"},
        }
    }
    assert trace_attributes == [
        {"trace_name": "corral.task.task", "session_id": "benchmark"},
        {"trace_name": "corral.task.task", "session_id": "benchmark"},
        {"trace_name": "corral.task.task", "session_id": "benchmark"},
    ]
    assert not any(_contains_state_snapshot(value) for value in client.starts)
    assert not any(
        _contains_state_snapshot(update)
        for span in client.spans
        for update in span.updates
    )


def test_langfuse_wraps_plain_execution_prompt_as_user_message():
    client = FakeLangfuse()
    observer = LangfuseObserver(client)
    context = ObservationContext(execution_id="execution", task_id="task")
    runtime = ActorRef(kind="runtime", actor_id="corral", run_id="runtime")

    observer.record_commit(
        _commit(
            ExecutionStarted(
                task={"prompt": "Measure the sample"},
                model={"name": "openai/gpt-5.6-terra"},
            ),
            runtime,
            1,
        ),
        context=context,
    )

    messages = [{"role": "user", "content": "Measure the sample"}]
    assert client.starts[0]["input"] == messages
    assert client.spans[0].trace_io_updates == [{"input": messages}]
