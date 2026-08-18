"""Tests for transition-by-transition State persistence in the runtime."""

import json
from datetime import datetime, timezone
from itertools import pairwise
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import pytest

from corral.agents.schema import AgentOutcome, AgentUsage
from corral.core import Action
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME
from corral.core.environment import Environment, Toolset
from corral.core.state import RuntimeState, State, StateMetadata
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.core.tool_catalog import (
    MissingToolCatalogError,
    ToolCatalogMismatchError,
)
from corral.persistence import JSONLStateStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class ToolThenSubmitAgent:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    async def run_session(self, session):
        self.calls += 1
        response = await session.execute(Action(name="increment", arguments={}))
        assert response.success is True
        submitted = await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        assert submitted.success is True
        return AgentOutcome(status="completed", answer="42")


class OwningSessionAgent:
    """Small agent that owns its complete tool-use loop."""

    model = "test-model"

    async def run_session(self, session):
        await session.record_message(
            {"role": "assistant", "content": "I will measure it."}
        )
        response = await session.execute(Action(name="increment", arguments={}))
        assert response.success is True
        await session.record_message(
            {"role": "assistant", "content": "The submit-ready answer is 42."}
        )
        submitted = await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        assert submitted.success is True
        return AgentOutcome(
            status="completed",
            answer="42",
            usage=AgentUsage(input_tokens=7, output_tokens=3, llm_calls=1),
        )


class OutcomeSessionAgent:
    model = "test-model"

    def __init__(self, outcome: AgentOutcome) -> None:
        self.outcome = outcome

    async def run_session(self, session):
        await session.record_message(
            {"role": "assistant", "content": "session finished"}
        )
        if self.outcome.status == "completed":
            assert self.outcome.answer is not None
            await session.execute(
                Action(
                    name=SUBMIT_ANSWER_TOOL_NAME,
                    arguments={"answer": self.outcome.answer},
                )
            )
        elif self.outcome.status == "surrendered":
            await session.execute(
                Action(
                    name=SUBMIT_ANSWER_TOOL_NAME,
                    arguments={"answer": "SURRENDER"},
                )
            )
        return self.outcome


class ResumeAwareAgent:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    async def run_session(self, session):
        self.calls += 1
        if session.state.tool_statistics.get("increment", 0) == 0:
            await session.execute(Action(name="increment", arguments={}))
        await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        return AgentOutcome(status="completed", answer="42")


class InterruptAfterBeforeStore(JSONLStateStore):
    """Simulate losing the worker after the before-tool commit is durable."""

    def __init__(self, path) -> None:
        super().__init__(path)
        self.interrupt_once = True

    async def save(
        self,
        state,
        transition_id=None,
        *,
        advance_head=False,
    ):
        committed = await super().save(
            state,
            transition_id,
            advance_head=advance_head,
        )
        if (
            self.interrupt_once
            and isinstance(transition_id, str)
            and transition_id.endswith(":before")
        ):
            self.interrupt_once = False
            raise RuntimeError("simulated worker loss after before-tool commit")
        return committed


def _environment(counter: dict[str, Any]) -> Environment:
    def increment(corral_action_id: str | None = None) -> str:
        """Increment the test counter."""
        counter["calls"] += 1
        counter["action_ids"].append(corral_action_id)
        return str(counter["calls"])

    task = TaskDefinition(
        name="task",
        description="increment once, then answer",
        tools=["increment"],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    return Environment(
        "task",
        task,
        toolset=Toolset(
            pool={
                "increment": tool(
                    increment,
                    hidden_args=["corral_action_id"],
                )
            },
            workspace_factory=None,
        ),
    )


def _records(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _historical_state_without_catalog(
    *,
    state_id: str,
    status: Literal["running", "terminal"] = "running",
) -> State:
    """Model an unreleased pre-snapshot State without rewriting its metadata."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return State(
        id=state_id,
        metadata=StateMetadata(
            environment={
                "name": "Environment",
                # Even an older partial list is not an authoritative snapshot.
                "tools": [],
            },
            task={"id": "task", "prompt": "historical prompt"},
        ),
        runtime=RuntimeState(
            status=status,
            started_at=now,
            ended_at=now if status == "terminal" else None,
        ),
    )


@pytest.mark.anyio()
async def test_runtime_persists_each_tool_boundary_and_reuses_completion(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ToolThenSubmitAgent()
    path = tmp_path / "states.jsonl"

    with JSONLStateStore(path) as store:
        runtime = TaskRuntime(store)
        final = await runtime.run(
            agent,
            environment,
            execution_id="execution-1",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=2,
        )
        retried = await runtime.run(
            agent,
            environment,
            execution_id="execution-1",
            # Workflow retries reuse the root by execution identity; rebuilding
            # caller metadata must not create another task checkpoint chain.
            started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            max_iterations=2,
        )

    assert retried.state_hash == final.state_hash
    assert final.runtime.status == "submitted"
    assert final.submission == "42"
    assert final.usage.agent_steps == 1
    assert final.usage.tool_calls == 2
    assert agent.calls == 1
    assert counter["calls"] == 1

    records = _records(path)
    states = [
        State.from_dict(record["state"])
        for record in records
        if record["record"] == "state"
    ]
    transitions = [
        record["transition_id"]
        for record in records
        if record["record"] == "transition"
    ]
    heads = [record for record in records if record["record"] == "head"]

    assert [state.revision for state in states] == list(range(7))
    assert all(
        child.parent_hash == parent.state_hash for parent, child in pairwise(states)
    )
    assert [state.pending_action.name for state in states if state.pending_action] == [
        "increment",
        "submit_answer",
    ]
    assert transitions[0] == "task:configured"
    assert transitions[-1] == "task:completed"
    assert [value.rsplit(":", 1)[-1] for value in transitions[1:-1]] == [
        "before",
        "after",
        "before",
        "after",
    ]
    assert len(heads) == len(states)
    assert heads[-1]["state_hash"] == final.state_hash


@pytest.mark.anyio()
async def test_restore_rejects_environment_catalog_drift_without_advancing_state(
    tmp_path,
):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ToolThenSubmitAgent()
    execution_id = "catalog-drift"
    state_id = str(uuid5(NAMESPACE_URL, f"corral:execution:{execution_id}"))
    initial = environment.initial_state(
        state_id=state_id,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    environment.tools["increment"].description = "A changed increment schema."

    with JSONLStateStore(tmp_path / "catalog-drift.jsonl") as store:
        await store.save(initial, advance_head=True)

        with pytest.raises(ToolCatalogMismatchError, match="stored fingerprint"):
            await TaskRuntime(store).run(
                agent,
                environment,
                execution_id=execution_id,
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                max_iterations=2,
            )

        assert await store.load_head(state_id) == initial
        assert await store.children(initial.state_hash) == ()

    assert agent.calls == 0
    assert counter["calls"] == 0


@pytest.mark.anyio()
async def test_nonterminal_state_without_catalog_requires_restart_or_exact_rebuild(
    tmp_path,
):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ToolThenSubmitAgent()
    execution_id = "missing-catalog"
    state_id = str(uuid5(NAMESPACE_URL, f"corral:execution:{execution_id}"))
    historical = _historical_state_without_catalog(state_id=state_id)

    with JSONLStateStore(tmp_path / "missing-catalog.jsonl") as store:
        await store.save(historical, advance_head=True)

        with pytest.raises(
            MissingToolCatalogError,
            match="will not derive one from the current Environment",
        ):
            await TaskRuntime(store).run(
                agent,
                environment,
                execution_id=execution_id,
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                max_iterations=2,
            )

        assert await store.load_head(state_id) == historical
        assert await store.children(historical.state_hash) == ()

    assert agent.calls == 0
    assert counter["calls"] == 0


@pytest.mark.anyio()
async def test_terminal_state_without_catalog_is_rejected(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ToolThenSubmitAgent()
    execution_id = "terminal-without-catalog"
    state_id = str(uuid5(NAMESPACE_URL, f"corral:execution:{execution_id}"))
    historical = _historical_state_without_catalog(
        state_id=state_id,
        status="terminal",
    )

    with JSONLStateStore(tmp_path / "terminal-without-catalog.jsonl") as store:
        await store.save(historical, advance_head=True)
        assert await store.load(historical.state_hash) == historical

        with pytest.raises(MissingToolCatalogError):
            await TaskRuntime(store).run(
                agent,
                environment,
                execution_id=execution_id,
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                max_iterations=2,
            )

        assert await store.load_head(state_id) == historical
        assert await store.children(historical.state_hash) == ()

    assert agent.calls == 0
    assert counter["calls"] == 0


@pytest.mark.anyio()
async def test_runtime_ignores_legacy_task_run_transition_without_a_head(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ToolThenSubmitAgent()
    execution_id = "legacy-task-run"
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state_id = str(uuid5(NAMESPACE_URL, f"corral:execution:{execution_id}"))
    initial = environment.initial_state(
        state_id=state_id,
        started_at=started_at,
    )
    legacy = initial.fork(
        runtime=RuntimeState(
            status="terminal",
            started_at=started_at,
            ended_at=started_at,
            metadata={"legacy": True},
        )
    )

    with JSONLStateStore(tmp_path / "legacy.jsonl") as store:
        await store.save(initial)
        await store.save(legacy, transition_id="task-run")

        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id=execution_id,
            started_at=started_at,
            max_iterations=2,
        )

    assert final.state_hash != legacy.state_hash
    assert final.runtime.status == "submitted"
    assert final.submission == "42"
    assert agent.calls == 1
    assert counter["calls"] == 1


@pytest.mark.anyio()
async def test_before_snapshot_includes_agent_history_and_after_has_observation(
    tmp_path,
):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = OwningSessionAgent()
    path = tmp_path / "states.jsonl"

    with JSONLStateStore(path) as store:
        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id="session-1",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=4,
        )

    assert final.runtime.status == "submitted"
    assert final.submission == "42"
    assert final.usage.agent_steps == 1
    assert final.usage.tool_calls == 2
    assert final.tool_statistics == {"increment": 1, "submit_answer": 1}
    assert counter["calls"] == 1
    states = [
        State.from_dict(record["state"])
        for record in _records(path)
        if record["record"] == "state"
    ]
    before_increment = next(
        state
        for state in states
        if state.pending_action is not None and state.pending_action.name == "increment"
    )
    after_increment = states[states.index(before_increment) + 1]
    assert before_increment.messages[-2]["content"] == "I will measure it."
    assert after_increment.parent_hash == before_increment.state_hash
    assert after_increment.pending_action is None
    assert after_increment.messages[-1]["role"] == "tool"
    assert after_increment.messages[-1]["name"] == "increment"


@pytest.mark.anyio()
async def test_retry_resumes_exact_action_from_before_tool_snapshot(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = ResumeAwareAgent()
    path = tmp_path / "interrupted.jsonl"

    with InterruptAfterBeforeStore(path) as store:
        runtime = TaskRuntime(store)
        with pytest.raises(RuntimeError, match="simulated worker loss"):
            await runtime.run(
                agent,
                environment,
                execution_id="resume-pending",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                max_iterations=2,
            )

        pending = await store.load_head(
            str(uuid5(NAMESPACE_URL, "corral:execution:resume-pending"))
        )
        assert pending is not None
        assert pending.pending_action is not None
        pending_action_id = pending.pending_action.id

        final = await runtime.run(
            agent,
            environment,
            execution_id="resume-pending",
            started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            max_iterations=2,
        )

    assert final.submission == "42"
    assert counter["calls"] == 1
    assert counter["action_ids"] == [pending_action_id]
    assert agent.calls == 2


@pytest.mark.anyio()
async def test_failed_session_outcome_is_not_submitted(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = OutcomeSessionAgent(
        AgentOutcome(status="harness_failure", error="SDK process crashed")
    )

    with JSONLStateStore(tmp_path / "failed.jsonl") as store:
        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id="failed-session",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.runtime.status == "failed"
    assert final.submission is None
    assert final.runtime.metadata["agent_status"] == "harness_failure"
    assert final.runtime.metadata["error"] == "SDK process crashed"
    assert final.tool_statistics == {}


@pytest.mark.anyio()
async def test_completed_outcome_without_submit_tool_is_rejected(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)

    class PlainTextAgent:
        model = "test-model"

        async def run_session(self, session):
            del session
            return AgentOutcome(status="completed", answer="42")

    with JSONLStateStore(tmp_path / "plain-text.jsonl") as store:
        final = await TaskRuntime(store).run(
            PlainTextAgent(),
            environment,
            execution_id="plain-text-session",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.runtime.status == "failed"
    assert final.submission is None
    assert final.runtime.metadata["agent_status"] == "protocol_failure"
    assert "without calling submit_answer" in final.runtime.metadata["error"]


@pytest.mark.anyio()
async def test_surrendered_session_uses_runtime_owned_sentinel(tmp_path):
    counter = {"calls": 0, "action_ids": []}
    environment = _environment(counter)
    agent = OutcomeSessionAgent(AgentOutcome(status="surrendered"))

    with JSONLStateStore(tmp_path / "surrendered.jsonl") as store:
        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id="surrendered-session",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.runtime.status == "surrendered"
    assert final.submission == "SURRENDER"
    assert final.tool_statistics == {"submit_answer": 1}
