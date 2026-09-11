"""Runtime transport selection, authored tool execution and invocation scope."""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tests.agents.commit_session import start_session

from corral.agents.hooks import AgentHooks, HookPoint
from corral.agents.schema import AgentOutcome
from corral.agents.session import (
    AgentSession,
    AgentSessionCapabilities,
    run_agent_session,
)
from corral.backend.mcp import open_mcp_host
from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.events import AgentTurnRecorded
from corral.core.task import TaskDefinition
from corral.core.tool import tool
from corral.observability import NoOpObserver
from corral.persistence import SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@tool
def echo(text: str) -> str:
    """Return the supplied text."""
    return text


@pytest.fixture()
async def runtime(tmp_path):
    async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
        yield TaskRuntime(store, NoOpObserver())


def make_environment(tool_pool=None):
    tool_pool = tool_pool or {"echo": echo}
    task = TaskDefinition(
        name="transport-task",
        description="Use tools and submit an answer.",
        tools=list(tool_pool),
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    return Environment(
        "transport-task",
        task,
        toolset=Toolset(pool=tool_pool, workspace_factory=None),
    )


async def run(runtime, agent, *, execution_id="transport-execution", tool_pool=None):
    return await runtime.run(
        agent,
        make_environment(tool_pool),
        execution_id=execution_id,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_iterations=20,
    )


async def call_tool(session, name, arguments):
    if session.tool_connection.transport == "python":
        response = await session.execute(Action(name=name, arguments=arguments))
        assert response.success, response.error
        return response.result
    async with (
        streamablehttp_client(session.tool_connection.mcp_url) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        await client.initialize()
        result = await client.call_tool(name, arguments)
        assert result.isError is False, result.content
        return result.content[0].text


async def assert_closed(url):
    if url is not None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ConnectError):
                await client.post(url, json={})


def track_hosts(monkeypatch):
    hosts = []

    @asynccontextmanager
    async def open_host():
        async with open_mcp_host() as host:
            hosts.append(host)
            yield host

    monkeypatch.setattr("corral.runtime.task_runner.open_mcp_host", open_host)
    return hosts


@pytest.mark.anyio()
@pytest.mark.parametrize("standalone", [False, True])
async def test_python_agent_runs_without_a_listener(runtime, monkeypatch, standalone):
    hosts = track_hosts(monkeypatch)

    def unexpected_socket(*args, **kwargs):
        pytest.fail("a Python-only execution must not create a socket")

    monkeypatch.setattr("corral.backend.mcp.socket.socket", unexpected_socket)

    class LocalAgent:
        async def run_session(self, session):
            if standalone:
                assert session.mcp_host is None
            assert session.tool_connection.transport == "python"
            assert session.tool_connection.url is None
            assert await call_tool(session, "echo", {"text": "local"}) == "local"
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    if standalone:
        session = await start_session(make_environment())
        try:
            result = await run_agent_session(
                LocalAgent(),
                session.environment,
                session.state,
                actor=session.actor,
                state_store=session.state_store,
                max_iterations=10,
            )
            state = result.state
        finally:
            await session.state_store.aclose()
    else:
        state = await run(runtime, LocalAgent())
    assert state.submission == "done", state.runtime.metadata
    assert state.tool_statistics == {"echo": 1, "submit_answer": 1}
    assert len(hosts) == (0 if standalone else 1)
    for host in hosts:
        assert host.port is None
        assert host._closed
        assert not host._bindings


@pytest.mark.anyio()
@pytest.mark.parametrize("parent_transport", ["python", "mcp"])
@pytest.mark.parametrize("delegate_transport", ["python", "mcp"])
async def test_delegate_transport_is_scoped_and_restores_the_caller(
    runtime, monkeypatch, parent_transport, delegate_transport
):
    hosts = track_hosts(monkeypatch)

    class Delegate:
        tool_transport = delegate_transport

        async def run_session(self, session):
            self.connection = session.tool_connection
            assert self.connection.transport == delegate_transport
            assert await call_tool(session, "echo", {"text": "delegate"}) == "delegate"
            return AgentOutcome(status="iteration_limit", error="delegate budget ended")

    delegate = Delegate()

    class Parent:
        tool_transport = parent_transport

        async def run_session(self, session):
            self.session = session
            self.connection = session.tool_connection
            assert self.connection.transport == parent_transport
            assert await call_tool(session, "echo", {"text": "before"}) == "before"
            outcome = await session.run_delegate(delegate)
            assert outcome.status == "iteration_limit"
            assert session.tool_connection is self.connection
            if delegate.connection.url is not None:
                async with httpx.AsyncClient() as client:
                    assert (
                        await client.post(delegate.connection.url, json={})
                    ).status_code == 404
            if self.connection.url and delegate.connection.url:
                assert (
                    httpx.URL(self.connection.url).port
                    == httpx.URL(delegate.connection.url).port
                )
            assert await call_tool(session, "echo", {"text": "after"}) == "after"
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    parent = Parent()
    state = await run(runtime, parent)
    assert state.submission == "done", state.runtime.metadata
    assert state.tool_statistics == {"echo": 3, "submit_answer": 1}
    assert len(hosts) == 1
    assert parent.session.tool_connection.transport == "python"
    await assert_closed(parent.connection.url)


@pytest.mark.anyio()
@pytest.mark.parametrize("parent_transport", ["python", "mcp"])
async def test_mcp_subagents_get_their_own_catalogs_and_record_their_own_calls(
    runtime, monkeypatch, parent_transport
):
    hosts = track_hosts(monkeypatch)

    class Child:
        tool_transport = "mcp"

        def __init__(self, inspect):
            self.session_capabilities = AgentSessionCapabilities(
                inspect_subagents=inspect
            )

        async def run_session(self, session):
            self.session = session
            self.connection = session.tool_connection
            self.run_id = session.actor.run_id
            async with (
                streamablehttp_client(self.connection.mcp_url) as streams,
                ClientSession(streams[0], streams[1]) as client,
            ):
                await client.initialize()
                catalog = await client.list_tools()
                names = {entry.name for entry in catalog.tools}
                assert (
                    "inspect_subagent" in names
                ) == self.session_capabilities.inspect_subagents
                response = await client.call_tool("echo", {"text": self.run_id})
                assert response.isError is False
                assert response.content[0].text == self.run_id
            return AgentOutcome(status="completed", answer=self.run_id)

    children = [Child(inspect=True), Child(inspect=False)]

    class Parent:
        tool_transport = parent_transport

        async def run_session(self, session):
            self.connection = session.tool_connection
            child_ids = [
                await session.spawn_subagent(child, handoff="use echo")
                for child in children
            ]
            outcomes = await asyncio.gather(
                *(session.wait_for_subagent(run_id) for run_id in child_ids)
            )
            assert {outcome.answer for outcome in outcomes} == set(child_ids)
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    parent = Parent()
    state = await run(runtime, parent)
    assert state.submission == "done", state.runtime.metadata
    assert len(hosts) == 1
    if parent.connection.url is not None:
        assert httpx.URL(parent.connection.url).port == hosts[0].port
    assert len({child.connection.url for child in children}) == 2
    assert len({httpx.URL(child.connection.url).port for child in children}) == 1
    echo_invocations = [
        invocation
        for invocation in state.tool_invocations.values()
        if invocation.tool_name == "echo"
    ]
    assert {invocation.requested_by_run_id for invocation in echo_invocations} == {
        child.run_id for child in children
    }
    for child in children:
        assert child.session.tool_connection.transport == "python"
        await assert_closed(child.connection.url)


@pytest.mark.anyio()
async def test_runtime_releases_mcp_when_an_agent_raises(runtime):
    class FailingAgent:
        tool_transport = "mcp"

        async def run_session(self, session):
            self.session = session
            self.url = session.tool_connection.mcp_url
            raise RuntimeError("harness failed")

    agent = FailingAgent()
    state = await run(runtime, agent)
    assert state.runtime.status == "failed"
    assert "harness failed" in state.runtime.metadata["error"]
    assert agent.session.tool_connection.transport == "python"
    await assert_closed(agent.url)


@pytest.mark.anyio()
async def test_runtime_releases_mcp_when_an_agent_is_cancelled(runtime):
    class CancelledAgent:
        tool_transport = "mcp"

        async def run_session(self, session):
            self.session = session
            self.url = session.tool_connection.mcp_url
            scope.cancel()
            await anyio.lowlevel.checkpoint()

    agent = CancelledAgent()
    with anyio.CancelScope() as scope:
        await run(runtime, agent)
    assert agent.session.tool_connection.transport == "python"
    await assert_closed(agent.url)


@pytest.mark.anyio()
async def test_before_task_cancellation_starts_no_listener(runtime, monkeypatch):
    hosts = track_hosts(monkeypatch)

    def unexpected_socket(*args, **kwargs):
        pytest.fail("a rejected invocation must not create a socket")

    monkeypatch.setattr("corral.backend.mcp.socket.socket", unexpected_socket)
    hooks = AgentHooks()
    hooks.register(
        HookPoint.BEFORE_TASK,
        lambda context: setattr(context, "should_continue", False),
    )

    class Agent:
        tool_transport = "mcp"

        async def run_session(self, session):
            pytest.fail("the before-task hook cancelled this invocation")

    agent = Agent()
    agent.hooks = hooks
    state = await run(runtime, agent)
    assert state.runtime.metadata["agent_status"] == "cancelled"
    assert len(hosts) == 1
    assert hosts[0].port is None
    assert hosts[0]._closed
    assert not hosts[0]._bindings


@pytest.mark.anyio()
async def test_invalid_transport_is_rejected_before_calling_the_agent(runtime):
    class InvalidAgent:
        tool_transport = "unsupported"

        async def run_session(self, session):
            pytest.fail("an invalid transport must not reach the agent")

    state = await run(runtime, InvalidAgent())
    assert state.runtime.status == "failed"
    assert "Unsupported agent tool_transport" in state.runtime.metadata["error"]


@pytest.mark.anyio()
async def test_concurrent_executions_on_one_runtime_have_independent_hosts(
    runtime, monkeypatch
):
    hosts = track_hosts(monkeypatch)
    ready = [asyncio.Event(), asyncio.Event()]

    class Agent:
        tool_transport = "mcp"

        def __init__(self, index):
            self.index = index

        async def run_session(self, session):
            self.connection = session.tool_connection
            self.host = session.mcp_host
            ready[self.index].set()
            await ready[1 - self.index].wait()
            answer = session.execution_id
            assert await call_tool(session, "echo", {"text": answer}) == answer
            await call_tool(session, "submit_answer", {"answer": answer})
            return AgentOutcome(status="completed", answer=answer)

    agents = [Agent(0), Agent(1)]
    with anyio.fail_after(10):
        states = await asyncio.gather(
            *(
                run(runtime, agent, execution_id=f"execution-{index}")
                for index, agent in enumerate(agents)
            )
        )
    assert len(hosts) == 2
    assert agents[0].host is not agents[1].host
    assert hosts[0].port != hosts[1].port
    assert [state.submission for state in states] == ["execution-0", "execution-1"]
    for agent in agents:
        await assert_closed(agent.connection.url)


@pytest.mark.anyio()
async def test_submit_response_and_sdk_cleanup_precede_server_shutdown(
    runtime, monkeypatch
):
    hosts = track_hosts(monkeypatch)
    commits = []

    class Observer(NoOpObserver):
        def record_commit(self, commit, *, context=None):
            commits.append((commit.event.type, hosts[0]._closed if hosts else None))

    runtime.observer = Observer()

    class Agent:
        tool_transport = "mcp"

        async def run_session(self, session):
            url = session.tool_connection.mcp_url
            async with (
                streamablehttp_client(url) as streams,
                ClientSession(streams[0], streams[1]) as client,
            ):
                await client.initialize()
                response = await client.call_tool("submit_answer", {"answer": "done"})
                assert not response.isError
                assert session.state.is_terminal
                assert not hosts[0]._closed
                await session.record_message(
                    {"role": "assistant", "content": "SDK finished"}
                )
                catalog = await client.list_tools()
                assert "submit_answer" in {entry.name for entry in catalog.tools}
            return AgentOutcome(status="completed", answer="done")

    state = await run(runtime, Agent())
    assert state.submission == "done", state.runtime.metadata
    assert ("submission.accepted", False) in commits
    assert ("agent.completed", False) in commits
    assert commits[-1] == ("execution.completed", True)


@pytest.mark.anyio()
async def test_recovery_uses_fresh_binding_and_completed_state_starts_no_host(
    runtime, monkeypatch
):
    hosts = track_hosts(monkeypatch)
    urls = []

    class Agent:
        tool_transport = "mcp"

        async def run_session(self, session):
            urls.append(session.tool_connection.mcp_url)
            await call_tool(session, "echo", {"text": f"attempt-{len(urls)}"})
            if len(urls) == 1:
                scope.cancel()
                await anyio.lowlevel.checkpoint()
            await call_tool(session, "submit_answer", {"answer": "recovered"})
            return AgentOutcome(status="completed", answer="recovered")

    agent = Agent()
    with anyio.CancelScope() as scope:
        await run(runtime, agent)
    before = [
        commit.hash
        async for commit in runtime.state_store.for_execution(
            "transport-execution"
        ).iter_commits("main")
    ]
    state = await run(runtime, agent)
    assert state.submission == "recovered", state.runtime.metadata
    assert state.tool_statistics == {"echo": 2, "submit_answer": 1}
    assert len(hosts) == 2
    assert urls[0] != urls[1]
    after = [
        commit.hash
        async for commit in runtime.state_store.for_execution(
            "transport-execution"
        ).iter_commits("main")
    ]
    assert after[: len(before)] == before
    assert all(url not in state.model_dump_json() for url in urls)
    inspected = await run(runtime, agent)
    assert inspected.through_commit_hash == state.through_commit_hash
    assert len(hosts) == 2
    for url in urls:
        await assert_closed(url)


@pytest.mark.anyio()
@pytest.mark.parametrize("invocation", ["direct", "delegate", "subagent"])
async def test_standalone_mcp_requires_a_caller_owned_host(monkeypatch, invocation):
    def unexpected_socket(*args, **kwargs):
        pytest.fail("a session without a host must not create a listener")

    monkeypatch.setattr("corral.backend.mcp.socket.socket", unexpected_socket)

    class MCPAgent:
        tool_transport = "mcp"

        async def run_session(self, session):
            pytest.fail("an MCP agent must not run without a host")

    class Parent:
        async def run_session(self, session):
            if invocation == "delegate":
                return await session.run_delegate(MCPAgent())
            child_id = await session.spawn_subagent(MCPAgent(), handoff="use MCP")
            return await session.wait_for_subagent(child_id)

    session = await start_session(make_environment())
    try:
        with pytest.raises(RuntimeError, match="MCP agents require.*pass mcp_host"):
            await run_agent_session(
                MCPAgent() if invocation == "direct" else Parent(),
                session.environment,
                session.state,
                actor=session.actor,
                state_store=session.state_store,
                max_iterations=10,
            )
    finally:
        await session.state_store.aclose()


@pytest.mark.anyio()
@pytest.mark.parametrize("parent_transport", ["python", "mcp"])
async def test_standalone_session_borrows_host_shared_with_delegates(parent_transport):
    class Delegate:
        tool_transport = "mcp"

        async def run_session(self, session):
            assert session.mcp_host is host
            await call_tool(session, "echo", {"text": "standalone delegate"})
            return AgentOutcome(status="iteration_limit", error="budget ended")

    class Parent:
        tool_transport = parent_transport

        async def run_session(self, session):
            assert session.mcp_host is host
            await session.run_delegate(Delegate())
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    session = await start_session(make_environment())
    try:
        async with open_mcp_host() as host:
            result = await run_agent_session(
                Parent(),
                session.environment,
                session.state,
                actor=session.actor,
                state_store=session.state_store,
                max_iterations=10,
                mcp_host=host,
            )
            assert result.state.submission == "done"
            assert host.port is not None
            assert not host._closed
            assert not host._bindings
        assert host._closed
    finally:
        await session.state_store.aclose()


@pytest.mark.anyio()
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("child_transport", ["python", "mcp"])
async def test_handled_child_failure_preserves_parent_completion(
    nested, child_transport
):
    class FailedChild:
        tool_transport = child_transport

        async def run_session(self, session):
            raise RuntimeError("recoverable child failure")

    class RecoveringAgent:
        async def run_session(self, session):
            child_id = await session.spawn_subagent(
                FailedChild(), handoff="try a subtask"
            )
            with pytest.raises(RuntimeError, match="recoverable child failure"):
                await session.wait_for_subagent(child_id)
            if not nested:
                await call_tool(session, "submit_answer", {"answer": "fallback"})
            return AgentOutcome(
                status="completed", answer="fallback", metadata={"recovered": True}
            )

    class Parent:
        async def run_session(self, session):
            child_id = await session.spawn_subagent(
                RecoveringAgent(), handoff="recover"
            )
            outcome = await session.wait_for_subagent(child_id)
            assert outcome.status == "completed"
            await call_tool(session, "submit_answer", {"answer": outcome.answer})
            return outcome

    session = await start_session(make_environment())
    try:
        async with open_mcp_host() as host:
            result = await run_agent_session(
                Parent() if nested else RecoveringAgent(),
                session.environment,
                session.state,
                actor=session.actor,
                state_store=session.state_store,
                max_iterations=10,
                mcp_host=host,
            )
        assert result.state.submission == "fallback"
        assert result.outcome.status == "completed"
        assert result.final_commit.event.metadata == {"recovered": True}
        assert result.final_commit.event.result_summary["answer"] == "fallback"
    finally:
        await session.state_store.aclose()


@pytest.mark.anyio()
async def test_unobserved_child_failure_still_propagates_at_completion():
    class FailedChild:
        async def run_session(self, session):
            raise RuntimeError("unobserved child failure")

    class Parent:
        async def run_session(self, session):
            await session.spawn_subagent(FailedChild(), handoff="try a subtask")
            return AgentOutcome(status="iteration_limit", error="budget ended")

    session = await start_session(make_environment())
    try:
        with pytest.raises(RuntimeError, match="unobserved child failure"):
            await run_agent_session(
                Parent(),
                session.environment,
                session.state,
                actor=session.actor,
                state_store=session.state_store,
                max_iterations=10,
            )
    finally:
        await session.state_store.aclose()


@pytest.mark.anyio()
async def test_fork_borrows_host_and_dispatches_to_its_branch(runtime, monkeypatch):
    hosts = track_hosts(monkeypatch)

    class Delegate:
        tool_transport = "mcp"

        async def run_session(self, session):
            self.url = session.tool_connection.mcp_url
            await call_tool(session, "echo", {"text": session.branch_id})
            return AgentOutcome(status="iteration_limit", error="budget ended")

    delegate = Delegate()

    class Parent:
        tool_transport = "mcp"

        async def run_session(self, session):
            fork = await session.fork_branch(branch_id="experiment")
            assert fork.mcp_host is session.mcp_host
            await fork.run_delegate(delegate)
            assert (
                httpx.URL(delegate.url).port
                == httpx.URL(session.tool_connection.url).port
            )
            assert fork.state.tool_statistics == {"echo": 1}
            assert not session.state.tool_statistics
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    state = await run(runtime, Parent())
    assert state.submission == "done", state.runtime.metadata
    assert state.tool_statistics == {"submit_answer": 1}
    assert len(hosts) == 1


@pytest.mark.anyio()
async def test_mcp_delegate_dispatch_preserves_its_iteration_budget(
    runtime, monkeypatch
):
    budgets = []
    original = AgentSession.execute

    async def execute(session, action, **kwargs):
        assert session.tool_connection.transport == "mcp"
        budgets.append(session.iteration_limit)
        return await original(session, action, **kwargs)

    monkeypatch.setattr(AgentSession, "execute", execute)

    class Delegate:
        tool_transport = "mcp"

        async def run_session(self, session):
            await call_tool(session, "echo", {"text": "delegate"})
            return AgentOutcome(status="iteration_limit", error="budget ended")

    class Parent:
        tool_transport = "mcp"

        async def run_session(self, session):
            await session.run_delegate(Delegate(), max_iterations=3)
            await call_tool(session, "submit_answer", {"answer": "done"})
            return AgentOutcome(status="completed", answer="done")

    state = await run(runtime, Parent())
    assert state.submission == "done", state.runtime.metadata
    assert budgets == [3, 20]


@pytest.mark.anyio()
@pytest.mark.parametrize("transport", ["python", "mcp"])
async def test_runtime_cancellation_waits_for_synchronous_tool(
    runtime, transport, monkeypatch
):
    hosts = track_hosts(monkeypatch)
    started, stopped = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    @tool
    def blocking() -> str:
        """Wait until the test allows the synchronous tool to finish."""
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        loop.call_soon_threadsafe(stopped.set)
        return "finished"

    class Agent:
        tool_transport = transport

        async def run_session(self, session):
            await call_tool(session, "blocking", {})
            return AgentOutcome(status="iteration_limit", error="budget ended")

    owner = asyncio.create_task(run(runtime, Agent(), tool_pool={"blocking": blocking}))
    try:
        with anyio.fail_after(10):
            await started.wait()
            owner.cancel()
            await asyncio.sleep(0.05)
            assert not owner.done()
            assert not hosts[0]._closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert stopped.is_set()
            assert hosts[0]._closed
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.anyio()
async def test_parent_cancellation_cleans_up_descendant_bindings(runtime, monkeypatch):
    hosts = track_hosts(monkeypatch)
    ready = asyncio.Event()
    urls, cleaned = [], []

    class Child:
        tool_transport = "mcp"

        def __init__(self, depth):
            self.depth = depth

        async def run_session(self, session):
            urls.append(session.tool_connection.mcp_url)
            try:
                if self.depth:
                    await session.spawn_subagent(
                        Child(self.depth - 1), handoff="continue"
                    )
                else:
                    ready.set()
                await anyio.sleep_forever()
            finally:
                cleaned.append(self.depth)

    class Parent:
        async def run_session(self, session):
            await session.spawn_subagent(Child(1), handoff="continue")
            await anyio.sleep_forever()

    owner = asyncio.create_task(run(runtime, Parent()))
    with anyio.fail_after(10):
        await ready.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
    assert sorted(cleaned) == [0, 1]
    assert len(hosts) == 1
    assert hosts[0]._closed
    for url in urls:
        await assert_closed(url)


@pytest.mark.anyio()
async def test_recovered_submission_stops_host_before_final_commit(
    runtime, monkeypatch
):
    hosts = track_hosts(monkeypatch)
    completions = []

    class Observer(NoOpObserver):
        def record_commit(self, commit, *, context=None):
            if commit.event.type == "execution.completed":
                completions.append(hosts[-1]._closed)

    runtime.observer = Observer()

    class Agent:
        tool_transport = "mcp"

        async def run_session(self, session):
            assert len(hosts) == 1, "recovery should resume the pending submission"
            await session._append_agent(
                AgentTurnRecorded(
                    actions=(
                        Action(
                            id="pending-submission",
                            name="submit_answer",
                            arguments={"answer": "recovered"},
                        ),
                    )
                ),
                "pending-submission",
            )
            scope.cancel()
            await anyio.lowlevel.checkpoint()

    with anyio.CancelScope() as scope:
        await run(runtime, Agent())
    state = await run(runtime, Agent())
    assert state.submission == "recovered", state.runtime.metadata
    assert len(hosts) == 2
    assert completions == [True]
    assert state.tool_statistics == {"submit_answer": 1}


@pytest.mark.anyio()
async def test_agent_completion_waits_for_unawaited_mcp_request(runtime):
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    @tool
    def blocking() -> str:
        """Finish only once agent teardown is waiting for this request."""
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        return "finished"

    class Agent:
        tool_transport = "mcp"

        async def run_session(self, session):
            self.url = session.tool_connection.mcp_url

            async def request():
                async with httpx.AsyncClient() as client:
                    return await client.post(
                        self.url,
                        headers={"Accept": "application/json, text/event-stream"},
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": "blocking", "arguments": {}},
                        },
                    )

            self.request = asyncio.create_task(request())
            await started.wait()
            return AgentOutcome(status="iteration_limit", error="budget ended")

    agent = Agent()
    owner = asyncio.create_task(run(runtime, agent, tool_pool={"blocking": blocking}))
    try:
        with anyio.fail_after(10):
            await started.wait()
            await asyncio.sleep(0)
            async with httpx.AsyncClient() as client:
                assert (await client.post(agent.url, json={})).status_code == 404
            assert not owner.done()
            store = runtime.state_store.for_execution("transport-execution")
            events = [commit.event.type async for commit in store.iter_commits("main")]
            assert "agent.completed" not in events
            release.set()
            response = await agent.request
            assert response.json()["result"]["content"][0]["text"] == "finished"
            state = await owner
            assert state.tool_statistics == {"blocking": 1}
            events = [commit.event.type async for commit in store.iter_commits("main")]
            assert events.index("tool.completed") < events.index("agent.completed")
            assert events[-1] == "execution.failed"
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        if hasattr(agent, "request"):
            await asyncio.gather(agent.request, return_exceptions=True)
