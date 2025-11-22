"""Tests for the core hook system."""

import pytest

from corral.agents.hooks.core import (
    AgentHooks,
    CriticalHookError,
    HookContext,
    HookPoint,
)


class MockAgent:
    """Mock agent for testing."""


class MockInterface:
    """Mock interface for testing."""


def test_hook_registration_and_execution():
    """Test basic hook registration and execution."""
    hooks = AgentHooks()
    executed = []

    def test_hook(context: HookContext) -> None:
        executed.append(context.task_id)

    hooks.register(HookPoint.BEFORE_TASK, test_hook)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.BEFORE_TASK, context)

    assert executed == ["test_task"]


def test_hook_priority_order():
    """Test that hooks execute in priority order (highest first)."""
    hooks = AgentHooks()
    execution_order = []

    def hook_low(context: HookContext) -> None:
        execution_order.append("low")

    def hook_high(context: HookContext) -> None:
        execution_order.append("high")

    def hook_medium(context: HookContext) -> None:
        execution_order.append("medium")

    hooks.register(HookPoint.BEFORE_TASK, hook_low, priority=1)
    hooks.register(HookPoint.BEFORE_TASK, hook_high, priority=10)
    hooks.register(HookPoint.BEFORE_TASK, hook_medium, priority=5)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.BEFORE_TASK, context)

    assert execution_order == ["high", "medium", "low"]


def test_hook_context_modification():
    """Test that hooks can modify context."""
    hooks = AgentHooks()

    def modify_hook(context: HookContext) -> None:
        context.metadata["modified"] = True
        context.messages.append({"role": "user", "content": "Modified"})

    hooks.register(HookPoint.BEFORE_TASK, modify_hook)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.BEFORE_TASK, context)

    assert context.metadata["modified"] is True
    assert len(context.messages) == 1
    assert context.messages[0]["content"] == "Modified"


def test_hook_should_continue_flag():
    """Test that should_continue flag stops execution."""
    hooks = AgentHooks()
    execution_order = []

    def stop_hook(context: HookContext) -> None:
        execution_order.append("stop")
        context.should_continue = False

    def never_executed(context: HookContext) -> None:
        execution_order.append("never")

    hooks.register(HookPoint.BEFORE_TASK, stop_hook, priority=10)
    hooks.register(HookPoint.BEFORE_TASK, never_executed, priority=5)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.BEFORE_TASK, context)

    assert execution_order == ["stop"]
    assert context.should_continue is False


def test_critical_hook_error_propagates():
    """Test that CriticalHookError is re-raised."""
    hooks = AgentHooks()

    def critical_error_hook(context: HookContext) -> None:
        raise CriticalHookError("Critical failure")

    hooks.register(HookPoint.BEFORE_TASK, critical_error_hook)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    with pytest.raises(CriticalHookError) as exc_info:
        hooks.execute(HookPoint.BEFORE_TASK, context)

    assert "Critical failure" in str(exc_info.value)


def test_non_critical_error_is_caught(caplog):
    """Test that non-critical errors are logged but don't stop execution."""
    hooks = AgentHooks()
    executed = []

    def error_hook(context: HookContext) -> None:
        raise ValueError("Non-critical error")

    def normal_hook(context: HookContext) -> None:
        executed.append("normal")

    hooks.register(HookPoint.BEFORE_TASK, error_hook, priority=10)
    hooks.register(HookPoint.BEFORE_TASK, normal_hook, priority=5)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    # Should not raise
    hooks.execute(HookPoint.BEFORE_TASK, context)

    # Normal hook should still execute
    assert executed == ["normal"]


def test_no_hooks_registered():
    """Test execution with no hooks registered."""
    hooks = AgentHooks()

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    # Should return immediately without error
    result = hooks.execute(HookPoint.BEFORE_TASK, context)
    assert result is context


def test_hook_remove():
    """Test removing a specific hook."""
    hooks = AgentHooks()
    executed = []

    def hook1(context: HookContext) -> None:
        executed.append("hook1")

    def hook2(context: HookContext) -> None:
        executed.append("hook2")

    hooks.register(HookPoint.BEFORE_TASK, hook1)
    hooks.register(HookPoint.BEFORE_TASK, hook2)

    # Remove hook1
    hooks.remove(HookPoint.BEFORE_TASK, hook1)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.BEFORE_TASK, context)

    assert executed == ["hook2"]


def test_hook_clear_all():
    """Test clearing all hooks."""
    hooks = AgentHooks()

    def hook1(context: HookContext) -> None:
        pass

    def hook2(context: HookContext) -> None:
        pass

    hooks.register(HookPoint.BEFORE_TASK, hook1)
    hooks.register(HookPoint.AFTER_TASK, hook2)

    assert hooks.has_hooks(HookPoint.BEFORE_TASK)
    assert hooks.has_hooks(HookPoint.AFTER_TASK)

    hooks.clear()

    assert not hooks.has_hooks(HookPoint.BEFORE_TASK)
    assert not hooks.has_hooks(HookPoint.AFTER_TASK)


def test_hook_clear_specific_point():
    """Test clearing hooks for a specific point."""
    hooks = AgentHooks()

    def hook1(context: HookContext) -> None:
        pass

    def hook2(context: HookContext) -> None:
        pass

    hooks.register(HookPoint.BEFORE_TASK, hook1)
    hooks.register(HookPoint.AFTER_TASK, hook2)

    hooks.clear(HookPoint.BEFORE_TASK)

    assert not hooks.has_hooks(HookPoint.BEFORE_TASK)
    assert hooks.has_hooks(HookPoint.AFTER_TASK)


def test_iteration_data_in_context():
    """Test that iteration_data can be populated and accessed."""
    hooks = AgentHooks()

    def check_iteration_data(context: HookContext) -> None:
        context.iteration_data["llm_response"] = "test response"
        context.iteration_data["parsed_actions"] = []

    hooks.register(HookPoint.AFTER_ITERATION, check_iteration_data)

    context = HookContext(
        task_id="test_task",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hooks.execute(HookPoint.AFTER_ITERATION, context)

    assert context.iteration_data["llm_response"] == "test response"
    assert context.iteration_data["parsed_actions"] == []
