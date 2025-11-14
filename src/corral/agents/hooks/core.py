"""Core hook system for agent lifecycle events.
This module provides a flexible callback-based hook system that allows updating context at various points in an agent's execution lifecycle.

Example:
        >>> hooks = AgentHooks()
        >>> def my_hook(context: HookContext) -> None:
        ...     print(f"Task: {context.task_id}")
        >>> hooks.register(HookPoint.BEFORE_TASK, my_hook, priority=10)
        >>> # Later we can in agent code:
        >>> context = HookContext(task_id="task1", ...)
        >>> hooks.execute(HookPoint.BEFORE_TASK, context)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from loguru import logger


class HookPoint(str, Enum):
    """Available hook points in agent lifecycle.

    Hook points represent specific moments in the agent execution where
    custom callbacks can be injected to modify behavior or observe state.

    Minimal hook points design:
    - BEFORE_TASK: For intervention injection after task guide is created
    - BEFORE_ITERATION: Before each LLM call
    - AFTER_ITERATION: After tools execute with full iteration context
    - AFTER_TASK: Cleanup after task completion
    """

    # Task lifecycle
    BEFORE_TASK = "before_task"
    """Called once at the start of a task, after prompt/task guide initialization.
    Perfect for intervention injection."""

    AFTER_TASK = "after_task"
    """Called once at the end of a task, regardless of success/failure"""

    # Iteration lifecycle
    BEFORE_ITERATION = "before_iteration"
    """Called at the start of each agent iteration, before LLM call"""

    AFTER_ITERATION = "after_iteration"
    """Called at the end of each agent iteration, after all tools execute.
    Context includes: llm_response, parsed_actions, tool_results in iteration_data"""

    ## WE CAN ADD MORE LATER IF NEEDED for example tool execution hooks, LLM parsing hooks, etc.


@dataclass
class HookContext:
    """Context passed to hooks - mutable state that hooks can inspect/modify.

    The context object provides access to the agent's current state and allows
    hooks to modify behavior by setting control flags or updating metadata.

    Attributes:
        task_id: The current task identifier
        agent: Reference to the agent instance (BaseAgent)
        interface: Router interface for tool execution (CorralRouter)
        messages: The agent's conversation history (list[LiteLLMMessage])
        iteration: Current iteration number (0-indexed)
        should_continue: Flag to control execution flow (set to False to stop)
        skip_current_step: Flag to skip the current step without stopping
        metadata: Dictionary for storing arbitrary hook-specific data
        iteration_data: Rich data from current iteration (populated in AFTER_ITERATION)
            Contains: llm_response, parsed_actions, tool_results, final_answer, etc.
    """

    task_id: str
    agent: Any  # BaseAgent
    interface: (
        Any  # CorralRouter, good for taking other information from environment state
    )
    messages: list[Any]  # list[LiteLLMMessage]
    iteration: int

    # Control flags
    should_continue: bool = True
    skip_current_step: bool = False

    # Optional data
    metadata: dict[str, Any] = field(default_factory=dict)

    # Rich iteration data (primarily for AFTER_ITERATION hook)
    # Example contents:
    # {
    #   "llm_response": "...",
    #   "parsed_actions": [Action(...)],
    #   "tool_results": [{"tool_name": "...", "arguments": {...}, "result": ...}],
    #   "final_answer": "..." (if detected),
    #   "surrender_reason": "..." (if surrendered)
    # }
    iteration_data: dict[str, Any] = field(default_factory=dict)


class HookCallback(Protocol):
    """Protocol for hook callbacks.

    Hook callbacks receive a HookContext and can modify it to change
    agent behavior. They should not return values( all communication
    happens through the context object)
    """

    def __call__(self, context: HookContext) -> None:
        """Execute hook with given context.

        Args:
            context: The hook context containing agent state and control flags
        """
        ...


class AgentHooks:
    """Manager for agent lifecycle hooks.

    This class manages registration and execution of hooks at various points
    in the agent lifecycle. Hooks are executed in priority order (higher first).
    """

    def __init__(self):
        self._hooks: dict[str, list[tuple[HookCallback, int]]] = {}

    def register(
        self, hook_point: HookPoint | str, callback: HookCallback, priority: int = 0
    ) -> None:
        """Register a hook callback for a specific hook point.

        Multiple hooks can be registered for the same hook point.
        They will be executed in priority order (higher priority first).

        Args:
            hook_point: The lifecycle point to hook into
            callback: Function to call at this hook point
            priority: Execution priority (higher = earlier). Default 0.
        """
        point = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if point not in self._hooks:
            self._hooks[point] = []

        self._hooks[point].append((callback, priority))
        # Sort by priority (descending)
        self._hooks[point].sort(key=lambda x: x[1], reverse=True)

    def execute(self, hook_point: HookPoint | str, context: HookContext) -> HookContext:
        """Execute all hooks registered for a hook point.

        Hooks are executed in priority order. If a hook sets
        context.should_continue = False, execution stops early.

        Errors in individual hooks are logged but don't stop execution
        of subsequent hooks.

        Args:
            hook_point: The hook point to execute
            context: The context to pass to hooks

        Returns:
            The context (potentially modified)
        """
        point = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if point not in self._hooks:
            return context

        for callback, _ in self._hooks[point]:
            try:
                callback(context)

                # Check if hook requested to stop
                if not context.should_continue:
                    break

            except Exception as e:
                logger.error(f"Error executing hook at {point}: {e}")

        return context

    def remove(self, hook_point: HookPoint | str, callback: HookCallback) -> None:
        """Remove a specific hook callback.

        Args:
            hook_point: The hook point to remove the callback from
            callback: The callback to remove
        """
        point = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if point in self._hooks:
            self._hooks[point] = [
                (cb, pri) for cb, pri in self._hooks[point] if cb != callback
            ]

    def clear(self, hook_point: HookPoint | str | None = None) -> None:
        """Clear all hooks, or hooks for a specific point.

        Args:
            hook_point: The hook point to clear, or None to clear all hooks
        """
        if hook_point is None:
            self._hooks.clear()
        else:
            point = (
                hook_point.value if isinstance(hook_point, HookPoint) else hook_point
            )
            if point in self._hooks:
                del self._hooks[point]

    def has_hooks(self, hook_point: HookPoint | str) -> bool:
        """Check if any hooks are registered for a hook point.

        Args:
            hook_point: The hook point to check

        Returns:
            True if at least one hook is registered for this point
        """
        point = hook_point.value if isinstance(hook_point, HookPoint) else hook_point
        return point in self._hooks and len(self._hooks[point]) > 0

    # TODO: Add method to list all registered hooks before agent execution
