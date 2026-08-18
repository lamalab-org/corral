"""Tests for the in-memory agent task session."""

from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from corral.agents.hooks import AgentHooks, HookContext, HookPoint
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.session import AgentSession
from corral.core.action import Action, submit_answer_tool, with_submit_answer_tool
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.core.tool_catalog import ToolCatalogMismatchError, ToolCatalogSnapshot
from corral.observability import NoOpObserver, ObservationContext


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def test_submit_answer_schema_is_canonical_and_present_exactly_once():
    conflicting = {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "non-canonical",
            "parameters": {"type": "object"},
        },
    }

    assert with_submit_answer_tool([conflicting, conflicting]) == [submit_answer_tool()]


def test_session_reads_only_the_complete_state_catalog_snapshot():
    def ping(value: str) -> str:
        """Return a pong value."""
        return f"pong:{value}"

    task = TaskDefinition(
        name="task",
        description="call ping",
        tools=["ping"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(
            pool={"ping": tool(ping)},
            workspace_factory=None,
        ),
    )
    state = environment.initial_state()
    persisted = ToolCatalogSnapshot.model_validate(
        state.metadata.environment["tool_catalog"]
    )
    interface = AgentSession(environment, state)

    assert interface.tools == persisted.detached_tools()
    assert [item["function"]["name"] for item in interface.tools] == [
        "ping",
        "submit_answer",
    ]

    # A bound session remains a view of State, never a live schema lookup.
    environment.tools["ping"].description = "A different executable schema."
    assert interface.tools == persisted.detached_tools()

    # A new binding detects that same drift before the agent can run.
    with pytest.raises(ToolCatalogMismatchError, match="Refusing to bind"):
        AgentSession(environment, state)


@pytest.mark.anyio()
async def test_session_mcp_transport_uses_state_backed_tool_execution():
    calls: list[str] = []

    def ping(value: str) -> str:
        """Return a pong value."""
        calls.append(value)
        return f"pong:{value}"

    task = TaskDefinition(
        name="task",
        description="call ping",
        tools=["ping"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(
            pool={"ping": tool(ping)},
            workspace_factory=None,
        ),
    )
    state = environment.initial_state(started_at=datetime.now(timezone.utc))
    previous_interface = AgentSession(environment, state)
    previous_interface.set_agent_state(
        "reflective",
        {"lessons": ["verify the measurement"]},
    )
    interface = AgentSession(
        environment,
        state,
        last_score={"trial_id": "prior", "score": 0.5},
        previous_state=previous_interface.state,
        max_iterations=3,
    )
    interface.set_agent_state("current", {"phase": "reasoning"})

    await interface.record_message(
        {"role": "assistant", "content": "continue from state"}
    )
    assert interface.messages == interface.state.messages
    assert interface.messages[-1]["content"] == "continue from state"
    assert interface.final_messages() == interface.messages

    try:
        async with (
            interface.open_mcp() as mcp,
            streamablehttp_client(mcp.url) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            endpoint = urlsplit(mcp.url)
            path_parts = endpoint.path.strip("/").split("/")
            assert path_parts[-1] == "mcp"
            assert len(path_parts[0]) >= 32

            # A parallel localhost task that discovers the port still cannot
            # reach this session without its unguessable path capability.
            unscoped_url = urlunsplit(
                (endpoint.scheme, endpoint.netloc, "/mcp", "", "")
            )

            def unscoped_status() -> int:
                request = Request(
                    unscoped_url,
                    data=b"{}",
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=2) as response:
                        return response.status
                except HTTPError as exc:
                    return exc.code

            assert await anyio.to_thread.run_sync(unscoped_status) == 404

            await session.initialize()
            listed = await session.list_tools()
            assert [item.name for item in listed.tools] == ["ping", "submit_answer"]

            result = await session.call_tool("ping", {"value": "x"})
            assert result.isError is False
            assert result.content[0].text == "pong:x"
    finally:
        interface.close()

    assert calls == ["x"]
    assert interface.state.tool_statistics == {"ping": 1}
    assert interface.previous_evaluation == {
        "trial_id": "prior",
        "score": 0.5,
    }
    assert interface.iteration_limit == 3
    assert interface.state.runtime.metadata["previous_evaluation"] == {
        "trial_id": "prior",
        "score": 0.5,
    }
    assert interface.state.runtime.metadata["max_iterations"] == 3
    assert interface.state.runtime.metadata["previous_state_hash"] == (
        previous_interface.state.state_hash
    )
    assert interface.get_agent_state("current") == {"phase": "reasoning"}
    assert interface.get_agent_state("reflective", previous=True) == {
        "lessons": ["verify the measurement"]
    }
    restored = type(interface.state).from_json(interface.state.to_json())
    restored_namespaces = restored.runtime.metadata["agent_state"]
    assert restored_namespaces["current"] == {"phase": "reasoning"}


@pytest.mark.anyio()
async def test_successful_submission_closes_the_session_action_boundary():
    calls: list[str] = []

    def ping(value: str) -> str:
        """Return a pong value."""
        calls.append(value)
        return f"pong:{value}"

    task = TaskDefinition(
        name="task",
        description="answer",
        tools=["ping"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={"ping": tool(ping)}, workspace_factory=None),
    )
    state = await anyio.to_thread.run_sync(
        lambda: environment.initial_state(started_at=datetime.now(timezone.utc))
    )
    interface = AgentSession(
        environment,
        state,
        max_iterations=1,
    )

    submitted = await interface.execute(
        Action(name="submit_answer", arguments={"answer": "42"})
    )
    terminal_hash = interface.state.state_hash
    rejected = await interface.execute(Action(name="ping", arguments={"value": "x"}))

    assert submitted.success is True
    assert rejected.success is False
    assert "no further actions" in str(rejected.error)
    assert interface.state.state_hash == terminal_hash
    assert interface.state.submission == "42"
    assert interface.state.tool_statistics == {"submit_answer": 1}
    assert calls == []


@pytest.mark.anyio()
async def test_delegate_uses_shared_hooks_and_submission_protocol():
    task = TaskDefinition(
        name="task",
        description="answer",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={}, workspace_factory=None),
    )
    interface = AgentSession(
        environment,
        environment.initial_state(started_at=datetime.now(timezone.utc)),
        max_iterations=3,
    )
    hooks = AgentHooks()
    events: list[str] = []

    async def before(context: HookContext) -> None:
        events.append(context.hook_point.value)
        await context.session.record_message(
            {"role": "assistant", "content": "delegate context"}
        )

    def after(context: HookContext) -> None:
        events.append(context.hook_point.value)
        assert context.data["status"] == "completed"

    hooks.register(HookPoint.BEFORE_TASK, before)
    hooks.register(HookPoint.AFTER_TASK, after)

    class Delegate:
        def __init__(self) -> None:
            self.hooks = hooks

        async def run_session(self, session: AgentSession) -> AgentOutcome:
            assert session is interface
            assert session.iteration_limit == 2
            assert session.messages[-1]["content"] == "delegate context"
            submitted = await session.execute(
                Action(name="submit_answer", arguments={"answer": "42"})
            )
            assert submitted.success
            return AgentOutcome(
                status="completed",
                answer="42",
                usage=AgentUsage(input_tokens=5, output_tokens=2, llm_calls=1),
            )

    outcome = await interface.run_delegate(Delegate(), max_iterations=2)

    assert outcome == AgentOutcome(
        status="completed",
        answer="42",
        usage=AgentUsage(input_tokens=5, output_tokens=2, llm_calls=1),
    )
    assert events == ["before_task", "after_task"]
    assert interface.submission == "42"
    # The outer composite folds the delegate's returned usage exactly once.
    assert interface.state.usage.llm_calls == 0
    assert interface.state.usage.tool_calls == 1
    assert interface.iteration_limit == 3


@pytest.mark.anyio()
async def test_delegate_cannot_complete_without_submit_answer():
    task = TaskDefinition(
        name="task",
        description="answer",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={}, workspace_factory=None),
    )
    interface = AgentSession(
        environment,
        environment.initial_state(started_at=datetime.now(timezone.utc)),
        max_iterations=1,
    )

    class PlainTextDelegate:
        async def run_session(self, session: AgentSession) -> AgentOutcome:
            assert session is interface
            return AgentOutcome(status="completed", answer="42")

    outcome = await interface.run_delegate(PlainTextDelegate())

    assert outcome.status == "protocol_failure"
    assert "without calling submit_answer" in str(outcome.error)
    assert interface.submission is None


@pytest.mark.anyio()
async def test_session_branch_inherits_runtime_services_and_can_be_adopted(tmp_path):
    calls: list[int] = []

    def measure(value: int) -> str:
        """Record a measurement."""
        calls.append(value)
        return str(value)

    task = TaskDefinition(
        name="task",
        description="measure and answer",
        tools=["measure"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="parent",
        toolset=Toolset(
            pool={"measure": tool(measure)},
            workspace_factory=None,
        ),
    )
    observer = NoOpObserver()
    observation_context = ObservationContext(execution_id="parent")
    state = await anyio.to_thread.run_sync(
        lambda: environment.initial_state(started_at=datetime.now(timezone.utc))
    )
    interface = AgentSession(
        environment,
        state,
        max_iterations=4,
        observer=observer,
        observation_context=observation_context,
    )
    durable_before = interface.durable_state
    branch = interface.fork_branch(execution_id="candidate")

    try:
        assert branch.advance_head is False
        assert branch.observer is observer
        assert branch.observation_context is observation_context
        assert branch.iteration_limit == interface.iteration_limit
        assert branch.environment is not interface.environment
        assert branch.workspace != interface.workspace

        measured = await branch.execute(Action(name="measure", arguments={"value": 7}))
        assert measured.success
        assert interface.state.tool_statistics == {}

        interface.adopt_branch(branch)
        assert interface.state.tool_statistics == {"measure": 1}
        assert interface.durable_state == durable_before

        submitted = await interface.execute(
            Action(name="submit_answer", arguments={"answer": "7"})
        )
        assert submitted.success
    finally:
        branch.close()
        branch.environment.shutdown_jobs()

    assert calls == [7]
    assert interface.submission == "7"
    assert interface.state.tool_statistics == {"measure": 1, "submit_answer": 1}
