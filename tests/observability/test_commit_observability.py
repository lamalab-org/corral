from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from corral.core import (
    Action,
    ActorRef,
    AgentStarted,
    AgentTurnRecorded,
    Commit,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    SubmissionAccepted,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    UsageDelta,
)
from corral.observability import (
    LangfuseObserver,
    Observation,
    ObservationContext,
    deterministic_trace_id,
)


@pytest.fixture
def observer(monkeypatch):
    # Test the real SDK's exported parent contexts, not just its call arguments.
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client_type = langfuse.Langfuse
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-test-{uuid4()}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-only")
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(
        langfuse,
        "Langfuse",
        lambda **kwargs: client_type(span_exporter=exporter, **kwargs),
    )
    instance = LangfuseObserver()
    scores = []
    monkeypatch.setattr(
        instance.client, "create_score", lambda **kwargs: scores.append(kwargs)
    )
    yield instance, exporter, scores
    instance.client.shutdown()


def _commit(event, sequence, *, execution_id="execution", run_id="agent-run"):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sequence)
    return Commit.create(
        execution_id=execution_id,
        branch_id="main",
        sequence=sequence,
        branch_sequence=sequence,
        parent_hash=None,
        based_on_hash=None,
        author=ActorRef(kind="agent", actor_id="tool-calling", run_id=run_id),
        event=event,
        occurred_at=timestamp,
        recorded_at=timestamp,
    )


def _events():
    search = Action(
        id="one", name="search", arguments={"query": "ions", "api_key": "secret"}
    )
    inspect = Action(id="two", name="inspect", arguments={})
    return [
        ExecutionStarted(
            task={"prompt": "Identify ions"}, model={"name": "test-model"}
        ),
        AgentStarted(agent_run_id="agent-run", agent_id="tool-calling"),
        AgentTurnRecorded(
            messages=(search.to_message(),),
            actions=(search,),
            usage_delta=UsageDelta(llm_calls=1, input_tokens=10, output_tokens=5),
        ),
        ToolStarted(
            action_id="one",
            invocation_id="i-one",
            requested_by_run_id="agent-run",
            tool_name="search",
        ),
        ToolCompleted(
            action_id="one",
            invocation_id="i-one",
            requested_by_run_id="agent-run",
            observation={"result": "found", "api_key": "secret"},
            status="success",
            duration_ms=120,
        ),
        AgentTurnRecorded(messages=(inspect.to_message(),), actions=(inspect,)),
        ToolStarted(
            action_id="two",
            invocation_id="i-two",
            requested_by_run_id="agent-run",
            tool_name="inspect",
        ),
        ToolCompleted(
            action_id="two",
            invocation_id="i-two",
            requested_by_run_id="agent-run",
            observation="confirmed",
            status="success",
            duration_ms=50,
        ),
        AgentTurnRecorded(
            messages=({"role": "assistant", "content": "Answer"},),
            usage_delta=UsageDelta(llm_calls=1, input_tokens=20, output_tokens=3),
        ),
        SubmissionAccepted(
            action_id="answer", requested_by_run_id="agent-run", answer="Fe+3"
        ),
        ExecutionCompleted(status="submitted"),
    ]


def _record(instance, events, *, execution_id="execution"):
    context = ObservationContext(
        execution_id, benchmark_run_id="benchmark", task_id="task"
    )
    commits = [
        _commit(event, index, execution_id=execution_id)
        for index, event in enumerate(events)
    ]
    for commit in commits:
        instance.record_commit(commit, context=context)
    instance.record_commit(commits[-1], context=context)
    instance.flush()
    return commits, context


def _json(span, attribute):
    return json.loads(span.attributes[f"langfuse.observation.{attribute}"])


def test_task_tree_has_one_real_root_and_correct_step_parents(observer):
    instance, exporter, _ = observer
    _record(instance, _events())
    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 1
    root = roots[0]
    assert f"{root.context.trace_id:032x}" == deterministic_trace_id("execution")
    by_id = {span.context.span_id: span for span in spans}
    assert len(spans) == 7  # task, two steps, two generations, two tools
    for span in spans:
        assert span.context.trace_id == root.context.trace_id
        assert span.start_time <= span.end_time
        assert span.attributes["session.id"] == "benchmark"
        if span.parent:
            parent = by_id[span.parent.span_id]
            assert (
                parent.start_time <= span.start_time <= span.end_time <= parent.end_time
            )
    tools = [
        span for span in spans if span.attributes["langfuse.observation.type"] == "tool"
    ]
    assert len({span.parent.span_id for span in tools}) == 1
    assert by_id[tools[0].parent.span_id].name == "Step 01"
    assert _json(root, "input") == [{"role": "user", "content": "Identify ions"}]
    assert _json(root, "output") == {"answer": "Fe+3", "status": "submitted"}
    assert root.end_time - root.start_time == 10_000_000_000


def test_tools_include_masked_arguments_results_and_recorded_duration(observer):
    instance, exporter, _ = observer
    _record(instance, _events())
    tools = {
        span.name: span
        for span in exporter.get_finished_spans()
        if span.name in {"search", "inspect"}
    }
    assert _json(tools["search"], "input") == {"query": "ions", "api_key": "[REDACTED]"}
    assert _json(tools["inspect"], "input") == {}
    assert _json(tools["search"], "output")["content"]["api_key"] == "[REDACTED]"
    assert tools["search"].end_time - tools["search"].start_time == 1_000_000_000
    assert (
        tools["search"].attributes[
            "langfuse.observation.metadata.execution_duration_ms"
        ]
        == "120.0"
    )


def test_model_continuations_share_one_response_and_context_is_retained(observer):
    instance, exporter, _ = observer
    _record(instance, _events())
    models = sorted(
        (
            span
            for span in exporter.get_finished_spans()
            if span.attributes["langfuse.observation.type"] == "generation"
        ),
        key=lambda span: span.start_time,
    )
    assert len(models) == 2
    assert len(_json(models[0], "output")["tool_calls"]) == 2
    assert _json(models[1], "input")[0] == {"role": "user", "content": "Identify ions"}
    assert (
        len(
            [
                message
                for message in _json(models[1], "input")
                if message["role"] == "tool"
            ]
        )
        == 2
    )
    assert (
        models[0].attributes["langfuse.observation.metadata.timing_source"]
        == "estimated_between_commits"
    )
    assert models[0].end_time - models[0].start_time == 1_000_000_000


def test_existing_evaluation_callback_attaches_zero_score_after_root_ends(observer):
    instance, exporter, scores = observer
    commits, context = _record(instance, _events())
    before = exporter.get_finished_spans()
    evaluation = instance.start(Observation(name="task.evaluate", context=context))
    evaluation.update(
        output={
            "score": 0.0,
            "commit_hash": commits[-1].hash,
            "scorer_version": "test-v1",
        }
    )
    evaluation.end()
    evaluation.end()
    assert len(scores) == 1
    assert scores[0]["value"] == 0.0
    assert scores[0]["name"] == "benchmark_score"
    assert scores[0]["trace_id"] == deterministic_trace_id("execution")
    assert exporter.get_finished_spans() == before


def test_failed_evaluation_does_not_invent_a_zero_score(observer):
    instance, _, scores = observer
    evaluation = instance.start(
        Observation(name="task.evaluate", context=ObservationContext("execution"))
    )
    evaluation.end(RuntimeError("scoring failed"))
    assert scores == []


def test_interleaved_executions_keep_distinct_roots(observer):
    instance, exporter, _ = observer
    for index, event in enumerate(_events()):
        for execution_id in ("one", "two"):
            instance.record_commit(
                _commit(event, index, execution_id=execution_id),
                context=ObservationContext(execution_id),
            )
    instance.flush()
    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 2
    by_id = {span.context.span_id: span for span in spans}
    for span in spans:
        if span.parent:
            assert by_id[span.parent.span_id].context.trace_id == span.context.trace_id


def test_tool_and_execution_failure_end_connected_spans(observer):
    instance, exporter, _ = observer
    events = [
        *_events()[:4],
        ToolFailed(
            action_id="one",
            invocation_id="i-one",
            requested_by_run_id="agent-run",
            error="broken",
            duration_ms=5,
        ),
        ExecutionFailed(error="agent failed", error_type="RuntimeError"),
    ]
    _record(instance, events)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert {
        span.name
        for span in spans
        if span.attributes.get("langfuse.observation.level") == "ERROR"
    } == {"Task · task", "search"}


def test_repeated_observers_share_the_sdk_exporter(observer):
    first, exporter, _ = observer
    second = LangfuseObserver()
    _record(first, _events(), execution_id="first")
    _record(second, _events(), execution_id="second")
    roots = [span for span in exporter.get_finished_spans() if span.parent is None]
    assert {f"{span.context.trace_id:032x}" for span in roots} == {
        deterministic_trace_id("first"),
        deterministic_trace_id("second"),
    }


@pytest.mark.parametrize("cut", [4, 5, 6, 7, 8])
def test_resume_recovers_unexported_generation_without_replaying_history(observer, cut):
    instance, exporter, _ = observer
    context = ObservationContext(
        "execution", benchmark_run_id="benchmark", task_id="task"
    )
    commits = [_commit(event, index) for index, event in enumerate(_events())]
    for _ in range(2):
        for commit in commits[:cut]:
            instance.restore_commit(commit, context=context)
    instance.flush()
    assert exporter.get_finished_spans() == ()

    for commit in commits[cut:]:
        instance.record_commit(commit, context=context)
    instance.flush()
    spans = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    assert all(
        f"{span.context.trace_id:032x}" == deterministic_trace_id("execution")
        and span.attributes["session.id"] == "benchmark"
        for span in spans
    )
    models = sorted(
        (
            span
            for span in spans
            if span.attributes["langfuse.observation.type"] == "generation"
        ),
        key=lambda span: span.start_time,
    )
    assert len(models) == 2
    assert all(
        span.attributes["langfuse.observation.model.name"] == "test-model"
        for span in models
    )
    assert by_id[models[0].parent.span_id].name == "Step 01"
    assert by_id[models[1].parent.span_id].name == "Step 02"
    assert [call["id"] for call in _json(models[0], "output")["tool_calls"]] == [
        "one",
        "two",
    ]
    assert models[0].end_time - models[0].start_time == 1_000_000_000
    inputs = _json(models[1], "input")
    assert inputs[0] == {"role": "user", "content": "Identify ions"}
    assert [message["name"] for message in inputs if message["role"] == "tool"] == [
        "search",
        "inspect",
    ]
    tools = {
        span.name: span
        for span in spans
        if span.attributes["langfuse.observation.type"] == "tool"
    }
    assert set(tools) == (
        {"search", "inspect"} if cut == 4 else {"inspect"} if cut < 8 else set()
    )
    for name, span in tools.items():
        assert by_id[span.parent.span_id].name == "Step 01"
        assert span.end_time - span.start_time == 1_000_000_000
        assert _json(span, "input") == (
            {"query": "ions", "api_key": "[REDACTED]"} if name == "search" else {}
        )
    for commit in commits:
        instance.restore_commit(commit, context=context)
    instance.flush()
    assert exporter.get_finished_spans() == spans


@pytest.mark.parametrize("cut", range(1, 12))
def test_crash_resume_exports_each_generation_and_usage_once(observer, cut):
    before, exporter, _ = observer
    context = ObservationContext(
        "execution", benchmark_run_id="benchmark", task_id="task"
    )
    commits = [_commit(event, index) for index, event in enumerate(_events())]
    for commit in commits[:cut]:
        before.record_commit(commit, context=context)
    before.flush()
    exported_before = exporter.get_finished_spans()

    # A killed process loses open spans. Its completed exports remain available.
    resumed = LangfuseObserver()
    for _ in range(2):
        for commit in commits[:cut]:
            resumed.restore_commit(commit, context=context)
    resumed.flush()
    assert exporter.get_finished_spans() == exported_before
    for commit in commits[cut:]:
        resumed.record_commit(commit, context=context)
    resumed.flush()

    models = sorted(
        (
            span
            for span in exporter.get_finished_spans()
            if span.attributes["langfuse.observation.type"] == "generation"
        ),
        key=lambda span: span.start_time,
    )
    assert len(models) == 2
    assert _json(models[0], "input") == [{"role": "user", "content": "Identify ions"}]
    assert [call["id"] for call in _json(models[0], "output")["tool_calls"]] == [
        "one",
        "two",
    ]
    assert _json(models[1], "output") == {"role": "assistant", "content": "Answer"}
    assert [_json(span, "usage_details") for span in models] == [
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
    ]
    assert models[0].end_time - models[0].start_time == 1_000_000_000


def test_multiple_resumes_keep_pending_generations_separate(observer):
    initial, exporter, _ = observer
    histories = {
        execution: [
            _commit(event, index, execution_id=execution)
            for index, event in enumerate(_events())
        ]
        for execution in ("one", "two")
    }
    for commits in histories.values():
        for commit in commits[:4]:
            initial.record_commit(commit)
    initial.flush()
    assert exporter.get_finished_spans() == ()
    interrupted = LangfuseObserver()
    for commits in histories.values():
        for commit in commits[:4]:
            interrupted.restore_commit(commit)
        for commit in commits[4:8]:
            interrupted.record_commit(commit)
    interrupted.flush()
    resumed = LangfuseObserver()
    for commits in histories.values():
        for commit in commits[:8]:
            resumed.restore_commit(commit)
        for commit in commits[8:]:
            resumed.record_commit(commit)
    resumed.flush()
    models = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes["langfuse.observation.type"] == "generation"
    ]
    assert len(models) == 4
    for execution in histories:
        selected = [
            span
            for span in models
            if f"{span.context.trace_id:032x}" == deterministic_trace_id(execution)
        ]
        assert len(selected) == 2
        assert (
            len(
                _json(min(selected, key=lambda span: span.start_time), "output")[
                    "tool_calls"
                ]
            )
            == 2
        )


def test_resumed_tool_failure_keeps_the_interrupted_generation(observer):
    instance, exporter, _ = observer
    events = [
        *_events()[:4],
        ToolFailed(
            action_id="one",
            invocation_id="i-one",
            requested_by_run_id="agent-run",
            error="failed after resume",
            duration_ms=5,
        ),
        ExecutionFailed(error="agent failed", error_type="RuntimeError"),
    ]
    commits = [_commit(event, index) for index, event in enumerate(events)]
    for commit in commits[:4]:
        instance.restore_commit(commit)
    for commit in commits[4:]:
        instance.record_commit(commit)
    instance.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 4  # task, step, generation, failed tool
    generation = next(
        span
        for span in spans
        if span.attributes["langfuse.observation.type"] == "generation"
    )
    assert _json(generation, "usage_details")["total_tokens"] == 15
    assert _json(generation, "output")["tool_calls"][0]["id"] == "one"
    assert {
        span.name
        for span in spans
        if span.attributes.get("langfuse.observation.level") == "ERROR"
    } == {"Task · execution", "search"}


def test_restore_finished_trace_only_attaches_evaluation(observer):
    instance, exporter, scores = observer
    context = ObservationContext(
        "execution", benchmark_run_id="benchmark", task_id="task"
    )
    commits = [_commit(event, index) for index, event in enumerate(_events())]
    for commit in commits:
        instance.restore_commit(commit, context=context)
    evaluation = instance.start(Observation(name="task.evaluate", context=context))
    evaluation.update(
        output={
            "score": 1.0,
            "commit_hash": commits[-1].hash,
            "scorer_version": "test-v1",
        }
    )
    evaluation.end()
    instance.flush()
    assert exporter.get_finished_spans() == ()
    assert scores[0]["trace_id"] == deterministic_trace_id("execution")
    assert scores[0]["timestamp"] == commits[-1].occurred_at
