"""
MCP Integration for Corral

This package provides Model Context Protocol (MCP) server support for Corral tools.
It enables Corral's scientific tools to be exposed as MCP servers that can be used
by any MCP-compatible client (Claude Desktop, VS Code, custom agents, etc.).

Key Components:
- ToolRegistry: Auto-discovers and registers tools from domain modules
- MCPServer: Wraps Corral tools as an MCP server using FastMCP
- schema_converter: Converts Corral tool schemas to MCP JSON Schema format
- main: CLI entry point for running MCP servers
"""

from corral.mcp.cli import main
from corral.mcp.mcp_server import MCPServer, create_server
from corral.mcp.tool_registry import ToolRegistry

__all__ = ["MCPServer", "ToolRegistry", "create_server", "main"]
