"""
MCP Server implementation for Corral tools.

This module provides an MCP server that wraps Corral tools, allowing them to be
used by any MCP-compatible client (Claude Desktop, VS Code, custom agents, etc.).
"""

import asyncio
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from corral.mcp.schema_converter import tool_to_json_schema
from corral.mcp.tool_registry import ToolRegistry
from corral.router.verbosity import ToolVerbosity, VerbosityConfig


class MCPServer:
    """
    MCP server that wraps Corral tools.

    This server automatically discovers tools from a domain module and exposes
    them via the Model Context Protocol.

    Args:
        domain_module (str): Python module path containing tools (e.g., "retrosynthesis.tools")
        server_name (str | None): Optional server name (defaults to "corral-{domain}")
        work_dir (str | None): Optional working directory to pass to tools with hidden_args
    """

    def __init__(
        self,
        domain_module: str,
        server_name: str | None = None,
        work_dir: str | None = None,
        verbosity: ToolVerbosity = ToolVerbosity.COMPREHENSIVE,
    ):
        """
        Initialize the MCP server.

        Args:
            domain_module: Python module path containing tools (e.g., "retrosynthesis.tools")
            server_name: Optional server name (defaults to "corral-{domain}")
            work_dir: Optional working directory to pass to tools with hidden_args
            verbosity: Tool description verbosity level (defaults to COMPREHENSIVE)
        """
        # Extract domain name from module path
        domain = domain_module.split(".")[-2] if "." in domain_module else domain_module
        self.domain = domain
        self.work_dir = work_dir
        self.verbosity = verbosity

        # Initialize tool registry
        self.registry = ToolRegistry(domain_module)
        logger.info(f"Initialized MCP server for domain: {domain}")
        logger.info(f"Loaded {len(self.registry)} tools")

        # Create MCP server
        self.server = Server(server_name or f"corral-{domain}")

        # Register MCP handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[MCPTool]:
            """List all available tools."""
            tools = []
            for tool_name, tool in self.registry.get_all_tools().items():
                # Filter description based on verbosity level
                filtered_description = VerbosityConfig.filter_tool_description(
                    tool.description, self.verbosity
                )

                mcp_tool = MCPTool(
                    name=tool.name,
                    description=filtered_description,
                    inputSchema=tool_to_json_schema(tool, self.verbosity),
                )
                tools.append(mcp_tool)
                logger.debug(f"Listing tool: {tool_name}")

            logger.info(
                f"Listed {len(tools)} tools with {self.verbosity.value} verbosity"
            )
            return tools

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any] | None
        ) -> list[TextContent]:
            """Execute a tool with the given arguments."""
            logger.info(f"Calling tool: {name} with arguments: {arguments}")

            try:
                # Get the tool from registry
                tool = self.registry.get_tool(name)

                # Prepare arguments
                call_args = arguments or {}

                # Add hidden arguments if needed
                if tool.hidden_args:
                    # If work_dir is in hidden_args and we have one, use it
                    if "work_dir" in tool.hidden_args and self.work_dir:
                        call_args["work_dir"] = self.work_dir
                    else:
                        # Use all hidden args from tool definition
                        call_args.update(tool.hidden_args)

                # Execute the tool in a thread pool to avoid blocking
                result = await asyncio.to_thread(tool.execute, **call_args)

                # Convert result to string if needed
                if not isinstance(result, str):
                    result = str(result)

                logger.info(f"Tool {name} executed successfully")
                return [TextContent(type="text", text=result)]

            except KeyError:
                error_msg = f"Tool '{name}' not found"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            except Exception as e:
                error_msg = f"Error executing tool '{name}': {e!s}"
                logger.error(error_msg)
                logger.exception(e)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

    async def run_stdio(self) -> None:
        """
        Run the server using stdio transport.

        This is the standard transport for MCP servers used with Claude Desktop
        and other local clients.
        """
        logger.info(f"Starting MCP server for {self.domain} on stdio")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    def run(self) -> None:
        """
        Run the server (convenience method).

        This method starts the server using stdio transport.
        """
        asyncio.run(self.run_stdio())


def create_server(
    domain_module: str,
    server_name: str | None = None,
    work_dir: str | None = None,
    verbosity: ToolVerbosity = ToolVerbosity.COMPREHENSIVE,
) -> MCPServer:
    """
    Factory function to create an MCP server.

    Args:
        domain_module: Python module path containing tools (e.g., "retrosynthesis.tools")
        server_name: Optional server name (defaults to "corral-{domain}")
        work_dir: Optional working directory to pass to tools
        verbosity: Tool description verbosity level (defaults to COMPREHENSIVE)

    Returns:
        Configured MCPServer instance
    """
    return MCPServer(domain_module, server_name, work_dir, verbosity)
