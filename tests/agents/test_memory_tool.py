"""Tests for Corral's agent-callable durable memory tools."""

import json

import pytest
from tests.agents.commit_session import start_session

from corral.agents.schema import AgentOutcome
from corral.agents.session import MEMORY_READ_TOOL_NAME, MEMORY_WRITE_TOOL_NAME
from corral.core import Action
from corral.core.environment import Environment
from corral.core.task import TaskDefinition


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_environment() -> Environment:
    return Environment(
        "memory-task",
        TaskDefinition(
            name="memory-task",
            description="Remember a lesson.",
            tools=[],
            scoring_fn=lambda _answer: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
    )


@pytest.mark.anyio
async def test_memory_tools_are_exposed_and_persist_between_attempts(tmp_path):
    environment = make_environment()
    first = await start_session(
        environment, actor_id="learner", store_path=tmp_path / "first.sqlite3"
    )

    names = {
        tool["function"]["name"] for tool in first.tools if tool["type"] == "function"
    }
    assert {MEMORY_READ_TOOL_NAME, MEMORY_WRITE_TOOL_NAME} <= names

    written = await first.execute(
        Action(
            name=MEMORY_WRITE_TOOL_NAME,
            arguments={"content": "Check the evidence before committing."},
        )
    )
    assert written.success

    second = await start_session(
        environment,
        actor_id="learner",
        previous_state=first.state,
        store_path=tmp_path / "second.sqlite3",
    )
    read = await second.execute(Action(name=MEMORY_READ_TOOL_NAME))

    assert read.success
    payload = json.loads(read.result or "{}")
    assert payload["entries"][0]["content"] == ("Check the evidence before committing.")

    third = await start_session(
        environment,
        actor_id="learner",
        previous_state=second.state,
        store_path=tmp_path / "third.sqlite3",
    )
    read_again = await third.execute(Action(name=MEMORY_READ_TOOL_NAME))
    assert json.loads(read_again.result or "{}") == payload


@pytest.mark.anyio
async def test_subagents_receive_the_previous_attempt_state(tmp_path):
    environment = make_environment()
    first = await start_session(
        environment, actor_id="parent", store_path=tmp_path / "first.sqlite3"
    )
    second = await start_session(
        environment,
        actor_id="parent",
        previous_state=first.state,
        store_path=tmp_path / "second.sqlite3",
    )

    class Child:
        observed_previous_state = None

        async def run_session(self, session):
            self.observed_previous_state = session.previous_state
            return AgentOutcome(status="iteration_limit", error="test")

    child = Child()
    child_run_id = await second.spawn_subagent(child, handoff="test")
    await second.wait_for_subagent(child_run_id)

    assert child.observed_previous_state is first.state


@pytest.mark.anyio
async def test_memory_is_scoped_to_the_agent_identity(tmp_path):
    environment = make_environment()
    first = await start_session(
        environment, actor_id="learner-a", store_path=tmp_path / "first.sqlite3"
    )
    await first.execute(
        Action(
            name=MEMORY_WRITE_TOOL_NAME,
            arguments={"content": "Private lesson."},
        )
    )

    other = await start_session(
        environment,
        actor_id="learner-b",
        previous_state=first.state,
        store_path=tmp_path / "second.sqlite3",
    )
    read = await other.execute(Action(name=MEMORY_READ_TOOL_NAME))

    assert read.success
    assert json.loads(read.result or "{}") == {"entries": []}


@pytest.mark.anyio
async def test_memory_write_rejects_empty_or_oversized_content(tmp_path):
    session = await start_session(
        make_environment(), store_path=tmp_path / "memory.sqlite3"
    )

    empty = await session.execute(
        Action(name=MEMORY_WRITE_TOOL_NAME, arguments={"content": "  "})
    )
    oversized = await session.execute(
        Action(name=MEMORY_WRITE_TOOL_NAME, arguments={"content": "x" * 4001})
    )

    assert not empty.success
    assert not oversized.success


@pytest.mark.anyio
async def test_memory_write_is_idempotent_for_a_retried_invocation(tmp_path):
    session = await start_session(
        make_environment(), store_path=tmp_path / "memory.sqlite3"
    )
    action = Action(
        name=MEMORY_WRITE_TOOL_NAME,
        arguments={"content": "Retry this safely."},
    )
    invocation_id = "stable-invocation"
    first = await session._memory_effects(
        action,
        based_on_hash=session.state.through_commit_hash,
        invocation_id=invocation_id,
    )
    second = await session._memory_effects(
        action,
        based_on_hash=session.state.through_commit_hash,
        invocation_id=invocation_id,
    )

    assert first.observation == second.observation
    entries = session.get_agent_state("memory")["entries"]
    assert len(entries) == 1
