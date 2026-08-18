"""Async lifecycle hooks for first-class :class:`AgentSession` agents."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol

from corral.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from pydantic import JsonValue

    from corral.agents.session import Agent, AgentSession
    from corral.core.state import State


class CriticalHookError(RuntimeError):
    """A hook failure that must abort the current agent session."""


class HookPoint(str, Enum):
    """Lifecycle points exposed by the current session-agent API."""

    BEFORE_TASK = "before_task"
    AFTER_TASK = "after_task"


@dataclass(slots=True)
class HookContext:
    """One hook invocation backed exclusively by an :class:`AgentSession`.

    The canonical transcript is deliberately read-only through `messages`.
    Hooks that need to add context must call `await session.record_message(...)`;
    hooks that need to act must call `await session.execute(Action(...))`. This
    prevents an intervention from maintaining a second mutable conversation or
    bypassing State transitions.
    """

    session: AgentSession
    agent: Agent
    hook_point: HookPoint
    data: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    should_continue: bool = True

    @property
    def task_id(self) -> str:
        """Return the task ID recorded in State, falling back to execution ID."""
        value = self.session.initial_state.metadata.task.get("id")
        return str(value or self.session.execution_id)

    @property
    def state(self) -> State:
        """Return the session's current immutable State."""
        return self.session.state

    @property
    def messages(self) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the canonical, immutable transcript view."""
        return self.session.messages


class HookCallback(Protocol):
    """A synchronous or asynchronous callback over one session context."""

    def __call__(self, context: HookContext) -> Awaitable[None] | None:
        """Inspect or modify the session using its public async API."""
        ...


class AgentHooks:
    """Thread-safe hook registration with asynchronous ordered execution."""

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[HookCallback, int]]] = {}
        self._lock = RLock()

    @staticmethod
    def _point(value: HookPoint | str) -> HookPoint:
        try:
            return value if isinstance(value, HookPoint) else HookPoint(value)
        except ValueError as exc:
            raise ValueError(f"unknown agent hook point: {value!r}") from exc

    def register(
        self,
        hook_point: HookPoint | str,
        callback: HookCallback,
        priority: int = 0,
    ) -> None:
        """Register `callback` in descending priority order."""
        point = self._point(hook_point)
        with self._lock:
            callbacks = self._hooks.setdefault(point, [])
            callbacks.append((callback, priority))
            callbacks.sort(key=lambda item: item[1], reverse=True)

    async def run(
        self,
        hook_point: HookPoint | str,
        context: HookContext,
    ) -> HookContext:
        """Run registered callbacks, awaiting asynchronous hooks when needed."""
        point = self._point(hook_point)
        if context.hook_point is not point:
            raise ValueError(
                f"hook context is for {context.hook_point.value!r}, not {point.value!r}"
            )
        with self._lock:
            callbacks = tuple(self._hooks.get(point, ()))

        for callback, _priority in callbacks:
            try:
                result = callback(context)
                if inspect.isawaitable(result):
                    await result
                if not context.should_continue:
                    break
            except CriticalHookError:
                raise
            except Exception as exc:
                logger.warning(f"Hook at {point.value} failed and was skipped: {exc}")
        return context

    def remove(self, hook_point: HookPoint | str, callback: HookCallback) -> None:
        """Remove one callback from one lifecycle point."""
        point = self._point(hook_point)
        with self._lock:
            callbacks = self._hooks.get(point)
            if callbacks is not None:
                self._hooks[point] = [
                    (registered, priority)
                    for registered, priority in callbacks
                    if registered != callback
                ]

    def clear(self, hook_point: HookPoint | str | None = None) -> None:
        """Clear callbacks for one point, or all callbacks when omitted."""
        with self._lock:
            if hook_point is None:
                self._hooks.clear()
            else:
                self._hooks.pop(self._point(hook_point), None)

    def has_hooks(self, hook_point: HookPoint | str) -> bool:
        """Return whether at least one callback is registered for `hook_point`."""
        point = self._point(hook_point)
        with self._lock:
            return bool(self._hooks.get(point))


__all__ = [
    "AgentHooks",
    "CriticalHookError",
    "HookCallback",
    "HookContext",
    "HookPoint",
]
