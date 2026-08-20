from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from corral.agents.schema import AgentOutcome
from corral.agents.session import (
    INSPECT_SUBAGENT_TOOL_NAME,
    AgentSession,
    AgentSessionCapabilities,
)
from corral.core import (
    Action,
    ActorRef,
    AgentStarted,
    AgentTurnRecorded,
    CommitRequest,
    ToolStarted,
)
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.observability import NoOpObserver
from corral.persistence import SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def environment_with_tools(**functions) -> Environment:
    return Environment(
        "task",
        TaskDefinition(
            name="task",
            description="Use the tools and submit an answer.",
            tools=list(functions),
            scoring_fn=lambda answer: float(bool(answer)),
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
        toolset=Toolset(
            pool={
                name: tool(function, concurrency="read_only")
                for name, function in functions.items()
            },
            workspace_factory=None,
        ),
    )


class ParallelAgent:
    def __init__(self) -> None:
        self.tool_message_order: list[str] = []

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        results = await session.execute_many(
            (
                Action(id="slow-action", name="slow"),
                Action(id="fast-action", name="fast"),
            )
        )
        assert [result.result for result in results] == ["slow", "fast"]
        self.tool_message_order = [
            str(message["tool_call_id"])
            for message in session.messages
            if message.get("role") == "tool"
        ]
        submitted = await session.execute(
            Action(id="submit-action", name="submit_answer", arguments={"answer": "ok"})
        )
        assert submitted.success
        return AgentOutcome(status="completed", answer="ok")


@pytest.mark.anyio()
async def test_parallel_results_keep_chronology_and_decision_order(tmp_path):
    def slow() -> str:
        """Finish after the fast tool."""
        time.sleep(0.05)
        return "slow"

    def fast() -> str:
        """Finish immediately."""
        return "fast"

    store = SQLiteCommitStore(tmp_path / "parallel.sqlite3")
    agent = ParallelAgent()
    state = await TaskRuntime(store, NoOpObserver()).run(
        agent,
        environment_with_tools(slow=slow, fast=fast),
        execution_id="parallel",
        started_at=datetime.now(timezone.utc),
        max_iterations=4,
    )

    commits = [
        commit async for commit in store.for_execution("parallel").iter_commits("main")
    ]
    completed_actions = [
        commit.event.action_id
        for commit in commits
        if commit.event.type == "tool.completed"
        and commit.event.action_id != "submit-action"
    ]
    assert completed_actions == ["fast-action", "slow-action"]
    assert agent.tool_message_order == ["slow-action", "fast-action"]
    parallel_turn = next(
        message
        for conversation in state.conversations.values()
        for message in conversation
        if len(message.get("tool_calls", [])) == 2
    )
    assert [call["id"] for call in parallel_turn["tool_calls"]] == [
        "slow-action",
        "fast-action",
    ]
    assert state.runtime.status == "submitted"
    assert state.runtime.metadata["execution_completed"] is True
    store.close()


class ChildAgent:
    async def run_session(self, session: AgentSession) -> AgentOutcome:
        await session.record_message(
            {"role": "assistant", "content": "private child result"}
        )
        return AgentOutcome(status="completed", answer="child answer")


class ParentAgent:
    session_capabilities = AgentSessionCapabilities(inspect_subagents=True)

    def __init__(self) -> None:
        self.before_inspection = ""
        self.after_inspection = ""
        self.after_import = ""
        self.inspection = {}

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        await session.record_message({"role": "assistant", "content": "parent only"})
        child_id = await session.spawn_subagent(
            ChildAgent(), handoff="investigate", actor_id="researcher"
        )
        await session.wait_for_subagent(child_id)
        self.before_inspection = str(session.messages)
        inspected = await session.execute(
            Action(
                name=INSPECT_SUBAGENT_TOOL_NAME,
                arguments={
                    "child_run_id": child_id,
                    "include_commits": True,
                    "limit": 10,
                },
            )
        )
        assert inspected.success
        self.inspection = json.loads(inspected.result or "{}")
        self.after_inspection = str(session.messages)
        trace = await session.inspect_subagent(child_id)
        assert "private child result" in str(trace.context.messages)
        await session.import_subagent_context(
            child_id,
            representation="approved child summary",
            source_commit_ids=(trace.commits[-1].hash,),
        )
        self.after_import = str(session.messages)
        submitted = await session.execute(
            Action(name="submit_answer", arguments={"answer": "done"})
        )
        assert submitted.success
        return AgentOutcome(status="completed", answer="done")


@pytest.mark.anyio()
async def test_subagent_trace_is_private_until_imported(tmp_path):
    store = SQLiteCommitStore(tmp_path / "subagent.sqlite3")
    agent = ParentAgent()
    state = await TaskRuntime(store, NoOpObserver()).run(
        agent,
        environment_with_tools(),
        execution_id="subagent",
        started_at=datetime.now(timezone.utc),
        max_iterations=5,
    )

    assert "private child result" not in agent.before_inspection
    assert "child answer" in agent.before_inspection
    assert "private child result" in agent.after_inspection
    assert agent.inspection["child_run_id"]
    assert agent.inspection["commits"]
    assert "approved child summary" in agent.after_import
    assert len(state.agent_runs) == 2
    commits = [
        commit async for commit in store.for_execution("subagent").iter_commits("main")
    ]
    assert sum(commit.event.type == "agent.spawned" for commit in commits) == 1
    assert {commit.branch_id for commit in commits} == {"main"}
    parent_run = next(
        run for run in state.agent_runs.values() if run.parent_run_id is None
    )
    assert parent_run.metadata["session_capabilities"] == {"inspect_subagents": True}
    child_run_id = parent_run.child_run_ids[0]
    inspection_action = next(
        action
        for action in state.actions.values()
        if action.action.name == INSPECT_SUBAGENT_TOOL_NAME
    )
    assert inspection_action.requested_by_run_id == parent_run.run_id
    assert inspection_action.status == "completed"
    inspection_invocation = state.tool_invocations[inspection_action.invocation_ids[0]]
    assert inspection_invocation.status == "completed"
    assert inspection_invocation.observation["child_run_id"] == child_run_id
    parent_messages = state.conversations[parent_run.run_id]
    spawn_call = next(
        message for message in parent_messages if message.get("tool_calls")
    )
    spawn_result = next(
        message
        for message in parent_messages
        if message.get("tool_call_id") == child_run_id
    )
    assert spawn_call["tool_calls"][0]["id"] == child_run_id
    assert spawn_result["content"] == {
        "child_run_id": child_run_id,
        "status": "completed",
        "result": {"answer": "child answer", "error": None},
    }
    assert spawn_result["metadata"]["inspectable"] is True
    recovered_session = AgentSession(
        environment_with_tools(),
        state,
        actor=ActorRef(
            kind="agent", actor_id=parent_run.actor_id, run_id=parent_run.run_id
        ),
        runtime_actor=ActorRef(
            kind="runtime", actor_id="corral", run_id="runtime:subagent"
        ),
        state_store=store.for_execution("subagent"),
        max_iterations=5,
    )
    recovered = await recovered_session.wait_for_subagent(child_run_id)
    assert recovered.answer == "child answer"
    assert INSPECT_SUBAGENT_TOOL_NAME in {
        tool["function"]["name"] for tool in recovered_session.tools
    }
    recovered_session.close()
    store.close()


class RecoveryAgent:
    def __init__(self) -> None:
        self.tool_names: set[str] = set()

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        self.tool_names = {tool["function"]["name"] for tool in session.tools}
        submitted = await session.execute(
            Action(name="submit_answer", arguments={"answer": "recovered"})
        )
        assert submitted.success
        return AgentOutcome(status="completed", answer="recovered")


class SubmitThenRaiseAgent:
    async def run_session(self, session: AgentSession) -> AgentOutcome:
        submitted = await session.execute(
            Action(name="submit_answer", arguments={"answer": "durable"})
        )
        assert submitted.success
        raise RuntimeError("harness cleanup failed")


@pytest.mark.anyio()
async def test_accepted_submission_survives_cleanup_failure_and_retry(tmp_path):
    store = SQLiteCommitStore(tmp_path / "accepted.sqlite3")
    environment = environment_with_tools()
    state = await TaskRuntime(store, NoOpObserver()).run(
        SubmitThenRaiseAgent(),
        environment,
        execution_id="accepted",
        started_at=datetime.now(timezone.utc),
        max_iterations=2,
    )

    assert state.submission == "durable"
    assert state.runtime.status == "submitted"
    assert state.runtime.metadata["execution_completed"] is True
    root_run = next(
        run for run in state.agent_runs.values() if run.parent_run_id is None
    )
    assert root_run.status == "completed"
    assert root_run.metadata["cleanup_error_type"] == "RuntimeError"

    retried = await TaskRuntime(store, NoOpObserver()).run(
        RecoveryAgent(),
        environment,
        execution_id="accepted",
        started_at=datetime.now(timezone.utc),
        max_iterations=2,
    )
    assert retried.through_commit_hash == state.through_commit_hash
    store.close()


@pytest.mark.anyio()
async def test_running_tool_resumes_with_stable_invocation_id(tmp_path):
    calls: list[int] = []

    def increment(value: int) -> str:
        """Record one recovered execution."""
        calls.append(value)
        return str(value + 1)

    execution_id = "recovery"
    environment = environment_with_tools(increment=increment)
    store = SQLiteCommitStore(tmp_path / "recovery.sqlite3", execution_id)
    runtime_actor = ActorRef(
        kind="runtime", actor_id="corral", run_id=f"runtime:{execution_id}"
    )
    agent = RecoveryAgent()
    agent_actor = ActorRef(
        kind="agent",
        actor_id=type(agent).__name__,
        run_id=str(uuid5(NAMESPACE_URL, f"corral:agent:{execution_id}:main")),
    )
    root = await store.append(
        CommitRequest(
            request_id="execution:started",
            branch_id="main",
            based_on_hash=None,
            author=runtime_actor,
            event=environment.initial_event(
                execution_id=execution_id,
                actor_id=agent_actor.actor_id,
                started_at=datetime.now(timezone.utc),
            ),
        )
    )
    configured = await store.append(
        CommitRequest(
            request_id="task:configured",
            branch_id="main",
            based_on_hash=root.hash,
            author=runtime_actor,
            event=environment.configure(await store.materialize("main")),
        )
    )
    started = await store.append(
        CommitRequest(
            request_id="agent:started",
            branch_id="main",
            based_on_hash=configured.hash,
            author=runtime_actor,
            event=AgentStarted(
                agent_run_id=agent_actor.run_id,
                agent_id=agent_actor.actor_id,
            ),
        )
    )
    action = Action(id="stable-action", name="increment", arguments={"value": 1})
    proposed = await store.bind(agent_actor).append(
        CommitRequest(
            request_id="agent:proposal",
            branch_id="main",
            based_on_hash=started.hash,
            author=agent_actor,
            event=AgentTurnRecorded(actions=(action,)),
        )
    )
    invocation_id = str(uuid5(NAMESPACE_URL, f"corral:tool:{execution_id}:{action.id}"))
    await store.append(
        CommitRequest(
            request_id=f"tool:{invocation_id}:started",
            branch_id="main",
            based_on_hash=proposed.hash,
            author=runtime_actor,
            event=ToolStarted(
                action_id=action.id,
                invocation_id=invocation_id,
                requested_by_run_id=agent_actor.run_id,
                tool_name=action.name,
            ),
        )
    )

    state = await TaskRuntime(store, NoOpObserver()).run(
        agent,
        environment,
        execution_id=execution_id,
        started_at=datetime.now(timezone.utc),
        max_iterations=3,
    )

    assert calls == [1]
    assert INSPECT_SUBAGENT_TOOL_NAME not in agent.tool_names
    assert state.tool_invocations[invocation_id].status == "completed"
    commits = [commit async for commit in store.iter_commits("main")]
    assert (
        sum(
            commit.event.type == "tool.started"
            and commit.event.invocation_id == invocation_id
            for commit in commits
        )
        == 1
    )
    store.close()


class FailingObserver:
    def start(self, observation):
        del observation
        raise RuntimeError("observer unavailable")

    def record_commit(self, commit, *, context=None):
        del commit, context
        raise RuntimeError("observer unavailable")

    def flush(self):
        raise RuntimeError("observer unavailable")


@pytest.mark.anyio()
async def test_observer_failure_never_rolls_back_commits(tmp_path):
    store = SQLiteCommitStore(tmp_path / "observer.sqlite3")
    state = await TaskRuntime(store, FailingObserver()).run(
        RecoveryAgent(),
        environment_with_tools(),
        execution_id="observer",
        started_at=datetime.now(timezone.utc),
        max_iterations=2,
    )

    assert state.submission == "recovered"
    assert (await store.for_execution("observer").head("main")).hash == (
        state.through_commit_hash
    )
    store.close()
