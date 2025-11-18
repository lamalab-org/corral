"""Adapter registry for agent-specific hook behavior."""

from collections.abc import Callable
from typing import Any

from corral.agents.hooks.core import HookContext


class HookAdapterRegistry:
    """Registry for agent-specific hook adapters.

    Allows hooks to define custom behavior per agent type without
    modifying the Agent base class. This is used by individual hooks only if it is required
    to adapt their behavior based on the agent in context.

    The registry accepts callables with flexible signatures that start with HookContext,
    followed by any additional arguments required by the specific adapter.
    """

    def __init__(self, default_adapter: Callable[..., None]):
        """Initialize the registry with a default adapter.

        Args:
            default_adapter: The fallback adapter to use when no agent-specific
                adapter is registered. Must accept HookContext as first argument.
        """
        self._adapters: dict[str, Callable[..., None]] = {}
        self._default = default_adapter

    def register(
        self, agent_class_name: str, adapter: Callable[..., None]
    ) -> Callable[..., None]:
        """Register an adapter for a specific agent type.

        Args:
            agent_class_name: The name of the agent class (e.g., "ReActAgent")
            adapter: The adapter function to use for this agent type.
                Must accept HookContext as first argument.

        Returns:
            The adapter function (allows use as decorator)
        """
        self._adapters[agent_class_name] = adapter
        return adapter  # Allow use as decorator

    def execute(self, context: HookContext, *args: Any, **kwargs: Any) -> None:
        """Execute the appropriate adapter for the agent in context.

        Args:
            context: The hook context containing agent information
            *args: Additional positional arguments to pass to the adapter
            **kwargs: Additional keyword arguments to pass to the adapter
        """
        agent_type = context.agent.__class__.__name__
        adapter = self._adapters.get(agent_type, self._default)
        return adapter(context, *args, **kwargs)
