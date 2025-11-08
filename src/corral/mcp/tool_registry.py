"""
Tool registry for auto-discovering Corral tools from domain modules.

This module provides a registry that can automatically discover and register
tools from domain-specific modules by importing them and extracting Tool instances.
"""

import importlib
import inspect
from typing import Any

from loguru import logger

from corral.backend.tool import Tool


class ToolRegistry:
    """
    Registry for auto-discovering and managing Corral tools.

    The registry loads tools from domain modules by auto-discovering
    all Tool instances (created by @tool decorator) from module attributes.
    """

    def __init__(self, domain_module: str | None = None):
        """
        Initialize the tool registry.

        Args:
            domain_module: Optional domain module path (e.g., "retrosynthesis.tools")
                         If provided, tools will be loaded immediately.
        """
        self.tools: dict[str, Tool] = {}

        if domain_module:
            self.load_tools_from_module(domain_module)

    def load_tools_from_module(self, module_path: str) -> None:
        """
        Load tools from a domain module by discovering @tool decorated functions.

        This method imports the module and inspects all attributes to find
        Tool instances (created by the @tool decorator).

        Args:
            module_path: Python module path (e.g., "retrosynthesis.tools")

        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            module = importlib.import_module(module_path)
            logger.info(f"Loaded module: {module_path}")
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            raise

        # Auto-discover Tool instances from module attributes
        discovered_tools = self._discover_tools_from_module(module)
        if discovered_tools:
            self.register_tools(discovered_tools)
            logger.info(
                f"Auto-discovered {len(discovered_tools)} tools from {module_path}"
            )
        else:
            logger.warning(f"No tools found in module {module_path}")

    def _discover_tools_from_module(self, module: Any) -> dict[str, Tool]:
        """
        Auto-discover Tool instances from a module's attributes.

        Finds all module attributes that are Tool instances (created by @tool decorator).

        Args:
            module: Imported Python module

        Returns:
            Dictionary mapping tool names to Tool instances
        """
        tools = {}
        for _name, obj in inspect.getmembers(module):
            if isinstance(obj, Tool):
                tools[obj.name] = obj
                logger.debug(f"Discovered tool: {obj.name}")

        return tools

    def register_tools(self, tools: dict[str, Tool]) -> None:
        """
        Register multiple tools at once.

        Args:
            tools: Dictionary mapping tool names to Tool instances
        """
        for name, tool in tools.items():
            self.register_tool(name, tool)

    def register_tool(self, name: str, tool: Tool) -> None:
        """
        Register a single tool.

        Args:
            name: Tool name (used as identifier)
            tool: Tool instance to register
        """
        if name in self.tools:
            logger.warning(f"Tool {name} already registered, overwriting")

        self.tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Tool:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            KeyError: If tool is not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self.tools[name]

    def get_all_tools(self) -> dict[str, Tool]:
        """
        Get all registered tools.

        Returns:
            Dictionary mapping tool names to Tool instances
        """
        return self.tools.copy()

    def list_tool_names(self) -> list[str]:
        """
        Get a list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools
