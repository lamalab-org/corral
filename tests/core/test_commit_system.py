from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from itertools import pairwise

import pytest

from corral.core import (
    Action,
    ActorRef,
    AgentCompleted,
    AgentContextResolver,
    AgentSpawned,
    AgentStarted,
    AgentTurnRecorded,
    CommitRequest,
    ContextImported,
    EnvironmentOperation,
    ExecutionCompleted,
    ExecutionStarted,
    RuntimeUpdate,
    SubmissionAccepted,
    ToolCompleted,
    ToolStarted,
    TraceAccessError,
)
from corral.persistence import (
    AuthorPermissionError,
    CommitConflictError,
    SQLiteCommitStore,
)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def runtime_actor(execution_id: str) -> ActorRef:
    return ActorRef(kind="runtime", actor_id="corral", run_id=f"runtime:{execution_id}")


async def append_runtime(store, execution_id, request_id, event, based_on=None):
    actor = runtime_actor(execution_id)
    return await store.append(
        CommitRequest(
            request_id=request_id,
            execution_id=execution_id,
            branch_id="main",
            based_on_hash=based_on,
            author=actor,
            event=event,
        )
    )


async def initialized_store(tmp_path, execution_id="execution"):
    store = SQLiteCommitStore(
        tmp_path / f"{execution_id}.sqlite3",
        execution_id,
        snapshot_interval=50,
    )
    root = await append_runtime(
        store,
        execution_id,
        "execution:started",
        ExecutionStarted(
            runtime=RuntimeUpdate(
                status="running", started_at=datetime.now(timezone.utc)
            )
        ),
    )
    return store, root


@pytest.mark.anyio()
async def test_linear_stale_appends_and_replay_use_small_events(tmp_path):
    store, root = await initialized_store(tmp_path)
    started = await append_runtime(
        store,
        "execution",
        "agent:main:started",
        AgentStarted(agent_run_id="main-run", agent_id="main"),
        root.hash,
    )
    actor = ActorRef(kind="agent", actor_id="main", run_id="main-run")
    bound = store.bind(actor)
    previous = started
    for index in range(100):
        previous = await bound.append(
            CommitRequest(
                request_id=f"message:{index}",
                execution_id="execution",
                branch_id="main",
                based_on_hash=started.hash if index > 0 else previous.hash,
                author=actor,
                event=AgentTurnRecorded(
                    messages=({"role": "assistant", "content": str(index)},)
                ),
            )
        )

    commits = [commit async for commit in store.iter_commits("main")]
    assert all(
        commit.parent_hash == parent.hash for parent, commit in pairwise(commits)
    )
    assert commits[-1].based_on_hash == started.hash
    replayed = await store.materialize("main")
    assert len(replayed.conversations["main-run"]) == 100
    assert 0 < store.snapshot_count() < len(commits)

    connection = sqlite3.connect(store.path)
    payloads = [row[0] for row in connection.execute("SELECT event_json FROM commits")]
    assert all('"conversations"' not in payload for payload in payloads)
    assert all('"agent_runs"' not in payload for payload in payloads)
    connection.close()
    store.close()

    reopened = SQLiteCommitStore(
        tmp_path / "execution.sqlite3", "execution", snapshot_interval=50
    )
    assert await reopened.materialize("main") == replayed
    reopened.close()


@pytest.mark.anyio()
async def test_idempotency_explicit_branching_and_author_binding(tmp_path):
    store, root = await initialized_store(tmp_path)
    main_started = await append_runtime(
        store,
        "execution",
        "agent:main:started",
        AgentStarted(agent_run_id="main-run", agent_id="main"),
        root.hash,
    )
    actor = ActorRef(kind="agent", actor_id="main", run_id="main-run")
    request = CommitRequest(
        request_id="turn:one",
        execution_id="execution",
        branch_id="main",
        based_on_hash=main_started.hash,
        author=actor,
        event=AgentTurnRecorded(
            messages=({"role": "assistant", "content": "x"},),
            actions=(
                Action(
                    id="authored-action",
                    name="noop",
                    actor_id="spoofed-model-author",
                ),
            ),
        ),
    )
    first = await store.bind(actor).append(request)
    assert await store.bind(actor).append(request) == first
    projected = await store.materialize("main")
    assert projected.actions["authored-action"].action.actor_id == "main"
    with pytest.raises(AuthorPermissionError):
        await store.append(request)

    await store.create_branch(branch_id="experiment", from_hash=main_started.hash)
    experiment_request = CommitRequest(
        request_id="experiment:turn",
        execution_id="execution",
        branch_id="experiment",
        based_on_hash=main_started.hash,
        author=actor,
        event=AgentTurnRecorded(messages=({"role": "assistant", "content": "y"},)),
    )
    experiment = await store.bind(actor, branch_id="experiment").append(
        experiment_request
    )
    assert experiment.parent_hash == main_started.hash
    assert (await store.head("main")).hash == first.hash
    assert (await store.head("experiment")).hash == experiment.hash
    store.close()


@pytest.mark.anyio()
async def test_shared_preconditions_reject_stale_tool_effects(tmp_path):
    store, root = await initialized_store(tmp_path)
    started = await append_runtime(
        store,
        "execution",
        "agent:main:started",
        AgentStarted(agent_run_id="main-run", agent_id="main"),
        root.hash,
    )
    actor = ActorRef(kind="agent", actor_id="main", run_id="main-run")
    action_a = Action(id="action-a", name="mutate")
    action_b = Action(id="action-b", name="mutate")
    proposed = await store.bind(actor).append(
        CommitRequest(
            request_id="turn:actions",
            execution_id="execution",
            branch_id="main",
            based_on_hash=started.hash,
            author=actor,
            event=AgentTurnRecorded(
                actions=(action_a, action_b), parallel_group_id="mutation-group"
            ),
        )
    )
    runtime = runtime_actor("execution")
    first_started = await store.append(
        CommitRequest(
            request_id="tool:first:started",
            execution_id="execution",
            branch_id="main",
            based_on_hash=proposed.hash,
            author=runtime,
            event=ToolStarted(
                action_id="action-a",
                invocation_id="invocation-a",
                requested_by_run_id="main-run",
                tool_name="mutate",
            ),
        )
    )
    second_started = await store.append(
        CommitRequest(
            request_id="tool:second:started",
            execution_id="execution",
            branch_id="main",
            based_on_hash=proposed.hash,
            author=runtime,
            event=ToolStarted(
                action_id="action-b",
                invocation_id="invocation-b",
                requested_by_run_id="main-run",
                tool_name="mutate",
            ),
        )
    )

    operation = (EnvironmentOperation(operation="set", path=("value",), value=1),)
    tool_a = ActorRef(kind="tool", actor_id="mutate", run_id="invocation-a")
    completed_a = await store.bind(tool_a).append(
        CommitRequest(
            request_id="tool:first:completed",
            execution_id="execution",
            branch_id="main",
            based_on_hash=first_started.hash,
            author=tool_a,
            event=ToolCompleted(
                action_id="action-a",
                invocation_id="invocation-a",
                requested_by_run_id="main-run",
                observation="ok",
                status="success",
                environment_operations=operation,
                expected_environment_revision=0,
            ),
        )
    )
    assert completed_a.parent_hash == second_started.hash

    tool_b = ActorRef(kind="tool", actor_id="mutate", run_id="invocation-b")
    with pytest.raises(CommitConflictError, match="environment revision conflict"):
        await store.bind(tool_b).append(
            CommitRequest(
                request_id="tool:second:completed",
                execution_id="execution",
                branch_id="main",
                based_on_hash=second_started.hash,
                author=tool_b,
                event=ToolCompleted(
                    action_id="action-b",
                    invocation_id="invocation-b",
                    requested_by_run_id="main-run",
                    observation="ok",
                    status="success",
                    environment_operations=operation,
                    expected_environment_revision=0,
                ),
            )
        )
    store.close()


@pytest.mark.anyio()
async def test_submission_allows_only_cleanup_turns_until_execution_completion(
    tmp_path,
):
    store, root = await initialized_store(tmp_path)
    started = await append_runtime(
        store,
        "execution",
        "agent:main:started",
        AgentStarted(agent_run_id="main-run", agent_id="main"),
        root.hash,
    )
    agent = ActorRef(kind="agent", actor_id="main", run_id="main-run")
    proposed = await store.bind(agent).append(
        CommitRequest(
            request_id="turn:submit",
            execution_id="execution",
            branch_id="main",
            based_on_hash=started.hash,
            author=agent,
            event=AgentTurnRecorded(
                actions=(
                    Action(
                        id="submit-action",
                        name="submit_answer",
                        arguments={"answer": "42"},
                    ),
                )
            ),
        )
    )
    runtime = runtime_actor("execution")
    tool_started = await store.append(
        CommitRequest(
            request_id="submit:started",
            execution_id="execution",
            branch_id="main",
            based_on_hash=proposed.hash,
            author=runtime,
            event=ToolStarted(
                action_id="submit-action",
                invocation_id="submit-invocation",
                requested_by_run_id="main-run",
                tool_name="submit_answer",
            ),
        )
    )
    tool = ActorRef(kind="tool", actor_id="submit_answer", run_id="submit-invocation")
    completed = await store.bind(tool).append(
        CommitRequest(
            request_id="submit:completed",
            execution_id="execution",
            branch_id="main",
            based_on_hash=tool_started.hash,
            author=tool,
            event=ToolCompleted(
                action_id="submit-action",
                invocation_id="submit-invocation",
                requested_by_run_id="main-run",
                observation="accepted",
                status="success",
            ),
        )
    )
    accepted = await store.append(
        CommitRequest(
            request_id="submission:accepted",
            execution_id="execution",
            branch_id="main",
            based_on_hash=completed.hash,
            author=runtime,
            event=SubmissionAccepted(
                action_id="submit-action",
                requested_by_run_id="main-run",
                answer="42",
            ),
        )
    )
    cleanup = await store.bind(agent).append(
        CommitRequest(
            request_id="turn:cleanup",
            execution_id="execution",
            branch_id="main",
            based_on_hash=accepted.hash,
            author=agent,
            event=AgentTurnRecorded(
                messages=({"role": "assistant", "content": "finished"},)
            ),
        )
    )
    with pytest.raises(CommitConflictError, match="new actions cannot be proposed"):
        await store.bind(agent).append(
            CommitRequest(
                request_id="turn:too-late",
                execution_id="execution",
                branch_id="main",
                based_on_hash=cleanup.hash,
                author=agent,
                event=AgentTurnRecorded(
                    actions=(Action(id="late-action", name="other"),)
                ),
            )
        )
    terminal = await store.append(
        CommitRequest(
            request_id="execution:completed",
            execution_id="execution",
            branch_id="main",
            based_on_hash=cleanup.hash,
            author=runtime,
            event=ExecutionCompleted(status="submitted"),
        )
    )
    with pytest.raises(CommitConflictError, match="no commits may follow"):
        await store.bind(agent).append(
            CommitRequest(
                request_id="turn:after-terminal",
                execution_id="execution",
                branch_id="main",
                based_on_hash=terminal.hash,
                author=agent,
                event=AgentTurnRecorded(
                    messages=({"role": "assistant", "content": "too late"},)
                ),
            )
        )
    store.close()


@pytest.mark.anyio()
async def test_agent_contexts_are_private_until_explicit_import(tmp_path):
    store, root = await initialized_store(tmp_path)
    started = await append_runtime(
        store,
        "execution",
        "agent:parent:started",
        AgentStarted(agent_run_id="parent", agent_id="parent"),
        root.hash,
    )
    parent = ActorRef(kind="agent", actor_id="parent", run_id="parent")
    private = await store.bind(parent).append(
        CommitRequest(
            request_id="parent:private",
            execution_id="execution",
            branch_id="main",
            based_on_hash=started.hash,
            author=parent,
            event=AgentTurnRecorded(
                messages=({"role": "assistant", "content": "parent private"},)
            ),
        )
    )
    spawned = await store.bind(parent).append(
        CommitRequest(
            request_id="parent:spawn",
            execution_id="execution",
            branch_id="main",
            based_on_hash=private.hash,
            author=parent,
            event=AgentSpawned(
                child_run_id="child",
                child_actor_id="child",
                context_cutoff_hash=private.hash,
                handoff="research",
            ),
        )
    )
    await append_runtime(
        store,
        "execution",
        "agent:child:started",
        AgentStarted(
            agent_run_id="child",
            agent_id="child",
            parent_run_id="parent",
            context_cutoff_hash=private.hash,
            handoff="research",
        ),
        spawned.hash,
    )
    child = ActorRef(
        kind="agent", actor_id="child", run_id="child", parent_run_id="parent"
    )
    child_turn = await store.bind(child).append(
        CommitRequest(
            request_id="child:private",
            execution_id="execution",
            branch_id="main",
            based_on_hash=spawned.hash,
            author=child,
            event=AgentTurnRecorded(
                messages=({"role": "assistant", "content": "child private"},)
            ),
        )
    )
    child_completed = await store.bind(child).append(
        CommitRequest(
            request_id="child:completed",
            execution_id="execution",
            branch_id="main",
            based_on_hash=child_turn.hash,
            author=child,
            event=AgentCompleted(
                agent_run_id="child",
                status="completed",
                result_summary={"answer": "child summary"},
                trace_head=child_turn.hash,
            ),
        )
    )
    state = await store.materialize("main")
    resolver = AgentContextResolver()
    parent_messages = resolver.for_agent(state, "parent").messages
    assert "child private" not in str(parent_messages)
    assert "child summary" in str(parent_messages)
    spawn_call = next(
        message for message in parent_messages if message.get("tool_calls")
    )
    spawn_result = next(
        message for message in parent_messages if message.get("tool_call_id") == "child"
    )
    assert spawn_call["tool_calls"][0]["id"] == "child"
    assert spawn_result["name"] == "spawn_subagent"
    assert "parent private" not in str(resolver.for_agent(state, "child").messages)
    assert "child private" in str(
        resolver.inspect(
            state, requester_run_id="parent", target_run_id="child"
        ).messages
    )
    with pytest.raises(TraceAccessError):
        resolver.inspect(state, requester_run_id="child", target_run_id="parent")

    imported = await store.bind(parent).append(
        CommitRequest(
            request_id="parent:import",
            execution_id="execution",
            branch_id="main",
            based_on_hash=child_completed.hash,
            author=parent,
            event=ContextImported(
                source_run_id="child",
                source_commit_ids=(child_turn.hash,),
                representation="selected summary",
            ),
        )
    )
    state = await store.materialize("main", imported.hash)
    assert "selected summary" in str(resolver.for_agent(state, "parent").messages)
    store.close()
