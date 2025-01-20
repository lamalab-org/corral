from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from corral.base import Tool


class ToolRegistry:
    """A registry for managing and discovering tools."""

    def __init__(self):
        self._tools: dict[str, dict[str, Tool]] = {}
        self._environments: dict[str, list[str]] = {}

    def register(self, tool: Tool, environment: str = "default") -> None:
        """Register a tool in the specified environment.

        Args:
            tool: Tool instance to register
            environment: Environment to register the tool in
        """
        if environment not in self._tools:
            self._tools[environment] = {}
            self._environments[environment] = []

        self._tools[environment][tool.name] = tool
        if tool.name not in self._environments[environment]:
            self._environments[environment].append(tool.name)

    def deregister(self, tool_name: str, environment: str = "default") -> bool:
        """
        Deregister a tool from the specified environment.

        Args:
            tool_name: Name of the tool to deregister
            environment: Environment to deregister from

        Returns:
            bool: True if tool was deregistered, False if not found
        """
        if environment in self._tools and tool_name in self._tools[environment]:
            del self._tools[environment][tool_name]
            self._environments[environment].remove(tool_name)
            return True
        return False

    def get_tool(self, name: str, environment: str = "default") -> Tool | None:
        """Get a tool by name from the specified environment."""
        return self._tools.get(environment, {}).get(name)

    def get_all_tools(self, environment: str = "default") -> Sequence[Tool]:
        """Get all tool instances from the specified environment."""
        return list(self._tools.get(environment, {}).values())

    def list_tools(self, environment: str = "default") -> Sequence[str]:
        """List all tool names in the specified environment."""
        return self._environments.get(environment, [])

    def list_environments(self) -> list[str]:
        """List all available environments."""
        return list(self._tools.keys())

    def search_tools(self, query: str, environment: str = "default") -> Sequence[Tool]:
        """Search for tools by name or description."""
        env_tools = self._tools.get(environment, {})

        return [
            tool
            for tool in env_tools.values()
            if query.lower() in tool.name.lower()
            or query.lower() in tool.description.lower()
        ]

    def check_health(self, environment: str = "default") -> dict[str, Sequence[str]]:
        """
        Run health checks for all tools in the specified environment.

        Returns:
            Dict[str, List[str]]: Dictionary mapping tool names to lists of error messages.
                                 Empty list means tool is healthy.
        """
        health_status = {}
        for tool_name, tool in self._tools.get(environment, {}).items():
            if hasattr(tool, "check_health"):
                is_healthy, errors = tool.check_health()
                health_status[tool_name] = errors if not is_healthy else []
            else:
                health_status[tool_name] = ["Tool does not implement health checks"]

        return health_status
