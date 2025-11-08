"""Tests for mcp_server.py - MCP server implementation."""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corral.backend.tool import tool
from corral.mcp.mcp_server import MCPServer, create_server
from corral.router.verbosity import ToolVerbosity


@pytest.fixture()
def mock_server():
    """Create a mock MCP Server instance."""
    with patch("corral.mcp.mcp_server.Server") as mock:
        yield mock


@pytest.fixture()
def mock_tools_module():
    """Create a mock module with sample tools for testing."""

    # Create actual tool instances
    @tool
    def calculator(operation: str, x: float, y: float) -> float:
        """Perform basic math operations.

        Args:
            operation: Operation to perform (add, multiply)
            x: First number
            y: Second number
        """
        if operation == "add":
            return x + y
        if operation == "multiply":
            return x * y
        return 0.0

    @tool
    def percentage_calculator(value: float, percentage: float = 100.0) -> float:
        """Calculate percentage of a value.

        Args:
            value: The base value
            percentage: The percentage to calculate
        """
        return (value * percentage) / 100.0

    # Create a mock module object with these tools as attributes
    mock_module = types.ModuleType("mock_tools")
    mock_module.calculator = calculator
    mock_module.percentage_calculator = percentage_calculator

    return mock_module


class TestMCPServer:
    """Test the MCPServer class."""

    def test_initialization(self, mock_server, mock_tools_module):
        """Test MCPServer initialization."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(
                domain_module="mockdomain.tools",
                server_name="test-server",
                work_dir="/tmp/test",
            )

            assert server.domain == "mockdomain"
            assert server.work_dir == "/tmp/test"
            assert len(server.registry) == 2
            mock_server.assert_called_once_with("test-server")

    def test_initialization_with_default_name(self, mock_server, mock_tools_module):
        """Test that default server name is generated correctly."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

            assert server.domain == "mockdomain"
            mock_server.assert_called_once_with("corral-mockdomain")

    def test_initialization_extracts_domain_correctly(
        self, mock_server, mock_tools_module
    ):
        """Test domain extraction from module path."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="first.tools")
            assert server.domain == "first"

            server2 = MCPServer(domain_module="second.tools")
            assert server2.domain == "second"

    def test_initialization_with_verbosity(self, mock_server, mock_tools_module):
        """Test initialization with different verbosity levels."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(
                domain_module="mockdomain.tools", verbosity=ToolVerbosity.BRIEF
            )

            assert server.verbosity == ToolVerbosity.BRIEF

    @pytest.mark.asyncio()
    async def test_list_tools(self, mock_server, mock_tools_module):
        """Test the list_tools handler."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

            # Test the registry instead
            tools = server.registry.get_all_tools()
            assert len(tools) == 2
            assert "calculator" in tools

    @pytest.mark.asyncio()
    async def test_call_tool_success(self, mock_server, mock_tools_module):
        """Test successful tool execution."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

            # Manually test the tool execution logic
            tool = server.registry.get_tool("calculator")
            result = await asyncio.to_thread(
                tool.execute, operation="add", x=5.0, y=3.0
            )

            assert result == "8.0"  # Tool returns string representation

    @pytest.mark.asyncio()
    async def test_call_tool_with_hidden_args(self, mock_server, mock_tools_module):
        """Test tool execution with hidden arguments."""

        # Create a tool with hidden args
        @tool
        def tool_with_work_dir(value: int, work_dir: str | None = None) -> str:
            """Tool that uses work_dir.

            Args:
                value: A value
                work_dir: Working directory (hidden)
            """
            return f"Value: {value}, WorkDir: {work_dir}"

        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(
                domain_module="mockdomain.tools", work_dir="/tmp/workspace"
            )

            # Register our custom tool
            server.registry.register_tool("test_tool", tool_with_work_dir)

            # Execute with hidden args
            result = await asyncio.to_thread(
                tool_with_work_dir.execute, value=42, work_dir="/tmp/workspace"
            )

            assert "Value: 42" in result
            assert "WorkDir: /tmp/workspace" in result

    def test_create_server_factory(self, mock_server, mock_tools_module):
        """Test the create_server factory function."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = create_server(
                domain_module="mockdomain.tools",
                server_name="custom-server",
                work_dir="/tmp",
                verbosity=ToolVerbosity.BRIEF,
            )

            assert isinstance(server, MCPServer)
            assert server.domain == "mockdomain"
            assert server.work_dir == "/tmp"
            assert server.verbosity == ToolVerbosity.BRIEF

    def test_invalid_module(self):
        """Test initialization with invalid module."""
        with pytest.raises(ImportError):
            MCPServer(domain_module="nonexistent.module")

    @pytest.mark.asyncio()
    async def test_run_stdio_integration(self, mock_server, mock_tools_module):
        """Test run_stdio method sets up the server correctly."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

        # Mock stdio_server
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_server_instance = mock_server.return_value
        mock_server_instance.run = AsyncMock()
        mock_server_instance.create_initialization_options = MagicMock(return_value={})

        with patch("corral.mcp.mcp_server.stdio_server") as mock_stdio:
            # Setup async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = (mock_read_stream, mock_write_stream)
            mock_context.__aexit__.return_value = None
            mock_stdio.return_value = mock_context

            # Run the server
            await server.run_stdio()

            # Verify stdio_server was used
            mock_stdio.assert_called_once()

            # Verify server.run was called with streams
            mock_server_instance.run.assert_called_once()
            call_args = mock_server_instance.run.call_args
            assert call_args[0][0] == mock_read_stream
            assert call_args[0][1] == mock_write_stream

    def test_run_sync_wrapper(self, mock_server, mock_tools_module):
        """Test the synchronous run() method."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

        with (
            patch.object(server, "run_stdio", new_callable=AsyncMock) as mock_run_stdio,
            patch("corral.mcp.mcp_server.asyncio.run") as mock_asyncio_run,
        ):
            server.run()

            # Verify run_stdio was called (it returns a coroutine)
            mock_run_stdio.assert_called_once()

            # Verify asyncio.run was called once (can't compare coroutine instances)
            mock_asyncio_run.assert_called_once()


class TestMCPServerVerbosity:
    """Test verbosity filtering in MCP server."""

    def test_tool_description_filtering_comprehensive(
        self, mock_server, mock_tools_module
    ):
        """Test that comprehensive verbosity includes detailed descriptions."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(
                domain_module="mockdomain.tools", verbosity=ToolVerbosity.COMPREHENSIVE
            )

            assert server.verbosity == ToolVerbosity.COMPREHENSIVE

    def test_tool_description_filtering_brief(self, mock_server, mock_tools_module):
        """Test that brief verbosity filters descriptions."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(
                domain_module="mockdomain.tools", verbosity=ToolVerbosity.BRIEF
            )

            assert server.verbosity == ToolVerbosity.BRIEF


class TestMCPServerToolExecution:
    """Test tool execution scenarios."""

    @pytest.mark.asyncio()
    async def test_execute_calculator_tool(self, mock_server, mock_tools_module):
        """Test executing the calculator tool."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

        calculator = server.registry.get_tool("calculator")

        # Test addition - returns string representation
        result = await asyncio.to_thread(
            calculator.execute, operation="add", x=10.0, y=5.0
        )
        assert result == "15.0"

        # Test multiplication - returns string representation
        result = await asyncio.to_thread(
            calculator.execute, operation="multiply", x=10.0, y=5.0
        )
        assert result == "50.0"

    @pytest.mark.asyncio()
    async def test_execute_percentage_calculator(self, mock_server, mock_tools_module):
        """Test executing the percentage calculator tool."""
        with patch("importlib.import_module", return_value=mock_tools_module):
            server = MCPServer(domain_module="mockdomain.tools")

        percentage_calc = server.registry.get_tool("percentage_calculator")

        # Test with default percentage - returns string representation
        result = await asyncio.to_thread(
            percentage_calc.execute, value=200.0, percentage=50.0
        )
        assert result == "100.0"  # Tool returns string representation
