"""Tests for AgentSession-native lifecycle hooks."""

from datetime import datetime, timezone

import pytest

from corral.agents.hooks import (
    AgentHooks,
    CriticalHookError,
    HookContext,
    HookPoint,
)
from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSession, run_agent_session
from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class SessionAgent:
    model = "test-model"

    def __init__(self, hooks: AgentHooks | None = None) -> None:
        self.hooks = hooks or AgentHooks()

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        assert any(
            message.get("content") == "injected context" for message in session.messages
        )
        response = await session.execute(
            Action(name="submit_answer", arguments={"answer": "42"})
        )
        assert response.success
        return AgentOutcome(status="completed", answer="42")


def make_session(*, agent: SessionAgent | None = None) -> AgentSession:
    task = TaskDefinition(
        name="test_task",
        description="answer 42",
        tools=[],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "test_task",
        task,
        toolset=Toolset(pool={}, workspace_factory=None),
    )
    state = environment.initial_state(started_at=datetime.now(timezone.utc))
    return AgentSession(
        environment,
        state,
        hooks=agent.hooks if agent is not None else None,
        agent=agent,
    )


def context(
    session: AgentSession,
    agent: SessionAgent,
    point: HookPoint = HookPoint.BEFORE_TASK,
) -> HookContext:
    return HookContext(session=session, agent=agent, hook_point=point)


@pytest.mark.anyio()
async def test_sync_and_async_hooks_run_in_priority_order():
    hooks = AgentHooks()
    order: list[str] = []

    def low(_context: HookContext) -> None:
        order.append("low")

    async def high(_context: HookContext) -> None:
        order.append("high")

    hooks.register(HookPoint.BEFORE_TASK, low, priority=1)
    hooks.register(HookPoint.BEFORE_TASK, high, priority=10)
    agent = SessionAgent(hooks)

    await hooks.run(HookPoint.BEFORE_TASK, context(make_session(), agent))

    assert order == ["high", "low"]


@pytest.mark.anyio()
async def test_hook_context_exposes_only_the_current_session_surface():
    agent = SessionAgent()
    session = make_session()
    hook_context = context(session, agent)

    assert hook_context.session is session
    assert hook_context.state is session.state
    assert hook_context.messages == session.messages
    assert isinstance(hook_context.messages, tuple)
    assert not hasattr(hook_context, "interface")
    assert not hasattr(hook_context, "iteration_data")
    assert not hasattr(agent.hooks, "execute")


@pytest.mark.anyio()
async def test_hook_stop_flag_prevents_lower_priority_callbacks():
    hooks = AgentHooks()
    order: list[str] = []

    def stop(hook_context: HookContext) -> None:
        order.append("stop")
        hook_context.should_continue = False

    def never(_context: HookContext) -> None:
        order.append("never")

    hooks.register(HookPoint.BEFORE_TASK, stop, priority=10)
    hooks.register(HookPoint.BEFORE_TASK, never, priority=1)
    agent = SessionAgent(hooks)
    result = await hooks.run(HookPoint.BEFORE_TASK, context(make_session(), agent))

    assert order == ["stop"]
    assert result.should_continue is False


@pytest.mark.anyio()
async def test_critical_errors_propagate_and_noncritical_errors_do_not():
    hooks = AgentHooks()
    continued: list[bool] = []

    def noncritical(_context: HookContext) -> None:
        raise ValueError("ignored")

    def normal(_context: HookContext) -> None:
        continued.append(True)

    hooks.register(HookPoint.BEFORE_TASK, noncritical, priority=10)
    hooks.register(HookPoint.BEFORE_TASK, normal, priority=1)
    agent = SessionAgent(hooks)
    await hooks.run(HookPoint.BEFORE_TASK, context(make_session(), agent))
    assert continued == [True]

    def critical(_context: HookContext) -> None:
        raise CriticalHookError("stop")

    hooks.register(HookPoint.AFTER_TASK, critical)
    with pytest.raises(CriticalHookError, match="stop"):
        await hooks.run(
            HookPoint.AFTER_TASK,
            context(make_session(), agent, HookPoint.AFTER_TASK),
        )


def test_registration_remove_and_clear():
    hooks = AgentHooks()

    def callback(_context: HookContext) -> None:
        pass

    hooks.register(HookPoint.BEFORE_TASK, callback)
    hooks.register(HookPoint.AFTER_TASK, callback)
    assert hooks.has_hooks(HookPoint.BEFORE_TASK)
    assert hooks.has_hooks(HookPoint.AFTER_TASK)

    hooks.remove(HookPoint.BEFORE_TASK, callback)
    assert not hooks.has_hooks(HookPoint.BEFORE_TASK)
    hooks.clear()
    assert not hooks.has_hooks(HookPoint.AFTER_TASK)


@pytest.mark.anyio()
async def test_shared_runner_invokes_hooks_and_persists_hook_state():
    hooks = AgentHooks()
    events: list[str] = []

    async def before(hook_context: HookContext) -> None:
        events.append(hook_context.hook_point.value)
        hook_context.metadata["source"] = "test"
        await hook_context.session.record_message(
            {"role": "assistant", "content": "injected context"}
        )

    def after(hook_context: HookContext) -> None:
        events.append(hook_context.hook_point.value)
        assert hook_context.data["status"] == "completed"

    hooks.register(HookPoint.BEFORE_TASK, before)
    hooks.register(HookPoint.AFTER_TASK, after)
    agent = SessionAgent(hooks)
    session = make_session(agent=agent)

    result = await run_agent_session(
        agent,
        session.environment,
        session.state,
        max_iterations=10,
    )

    assert result.outcome == AgentOutcome(status="completed", answer="42")
    assert events == ["before_task", "after_task"]
    hook_state = result.state.runtime.metadata["agent_state"]["hooks"]
    assert hook_state["metadata"] == {"source": "test"}


@pytest.mark.anyio()
async def test_before_task_hook_can_cancel_the_agent():
    hooks = AgentHooks()

    def cancel(hook_context: HookContext) -> None:
        hook_context.should_continue = False

    hooks.register(HookPoint.BEFORE_TASK, cancel)
    agent = SessionAgent(hooks)
    session = make_session(agent=agent)

    result = await run_agent_session(
        agent, session.environment, session.state, max_iterations=10
    )

    assert result.outcome.status == "cancelled"
    assert result.state.submission is None


@pytest.mark.anyio()
async def test_after_task_hook_runs_when_the_agent_raises():
    hooks = AgentHooks()
    seen: list[str] = []

    def after(hook_context: HookContext) -> None:
        seen.append(str(hook_context.data["status"]))

    hooks.register(HookPoint.AFTER_TASK, after)

    class RaisingAgent(SessionAgent):
        async def run_session(self, session: AgentSession) -> AgentOutcome:
            raise RuntimeError("agent broke")

    agent = RaisingAgent(hooks)
    session = make_session(agent=agent)

    with pytest.raises(RuntimeError, match="agent broke"):
        await run_agent_session(
            agent, session.environment, session.state, max_iterations=10
        )

    assert seen == ["raised"]
