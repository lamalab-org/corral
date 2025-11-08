"""Tests for tool_registry.py - tool discovery and registration."""

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from corral.backend.tool import Tool, tool
from corral.mcp.tool_registry import ToolRegistry


@pytest.fixture()
def sample_tools_module():
    """Create a temporary module with sample tools."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        module_dir = Path(tmpdir)
        tools_file = module_dir / "sample_tools.py"

        # Write sample tools to the file
        tools_file.write_text('''
from corral.backend.tool import tool

@tool
def add_numbers(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x: First number
        y: Second number
    """
    return x + y

@tool
def multiply_numbers(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First number
        y: Second number
    """
    return x * y
''')

        # Load the module
        spec = importlib.util.spec_from_file_location("sample_tools", tools_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["sample_tools"] = module
        spec.loader.exec_module(module)

        yield module

        # Clean up
        del sys.modules["sample_tools"]


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_initialization_without_module(self):
        """Test that registry can be initialized without a module."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.list_tool_names() == []

    def test_initialization_with_valid_module(self, sample_tools_module):
        """Test that registry loads tools from a module on initialization."""
        # Mock the module import
        with patch("importlib.import_module", return_value=sample_tools_module):
            registry = ToolRegistry("mock_tools.tools")

            assert len(registry) == 2
            tool_names = registry.list_tool_names()
            assert "add_numbers" in tool_names
            assert "multiply_numbers" in tool_names

    def test_initialization_with_invalid_module(self):
        """Test that initialization with invalid module raises ImportError."""
        with pytest.raises(ImportError):
            ToolRegistry("nonexistent.module.path")

    def test_load_tools_from_module(self, sample_tools_module):
        """Test loading tools from a valid module."""
        registry = ToolRegistry()
        assert len(registry) == 0

        with patch("importlib.import_module", return_value=sample_tools_module):
            registry.load_tools_from_module("mock_tools.tools")

            assert len(registry) == 2
            assert "add_numbers" in registry

    def test_load_tools_from_invalid_module(self):
        """Test loading tools from invalid module raises ImportError."""
        registry = ToolRegistry()

        with pytest.raises(ImportError):
            registry.load_tools_from_module("invalid.module")

    def test_register_tool(self):
        """Test registering a single tool."""
        registry = ToolRegistry()

        @tool
        def test_tool(x: int) -> int:
            """Test tool.

            Args:
                x: A number
            """
            return x * 2

        registry.register_tool("test_tool", test_tool)

        assert len(registry) == 1
        assert "test_tool" in registry
        assert registry.get_tool("test_tool") == test_tool

    def test_register_tool_overwrites_existing(self):
        """Test that registering a tool with the same name overwrites the existing one."""
        registry = ToolRegistry()

        @tool
        def tool1(x: int) -> int:
            """First tool.

            Args:
                x: A number
            """
            return x

        @tool
        def tool2(x: int) -> int:
            """Second tool.

            Args:
                x: A number
            """
            return x * 2

        registry.register_tool("my_tool", tool1)
        assert registry.get_tool("my_tool") == tool1

        registry.register_tool("my_tool", tool2)
        assert registry.get_tool("my_tool") == tool2

    def test_register_tools(self):
        """Test registering multiple tools at once."""
        registry = ToolRegistry()

        @tool
        def tool_a(x: int) -> int:
            """Tool A.

            Args:
                x: A number
            """
            return x

        @tool
        def tool_b(x: int) -> int:
            """Tool B.

            Args:
                x: A number
            """
            return x * 2

        tools = {"tool_a": tool_a, "tool_b": tool_b}
        registry.register_tools(tools)

        assert len(registry) == 2
        assert "tool_a" in registry
        assert "tool_b" in registry

    def test_get_tool(self, sample_tools_module):
        """Test getting a tool by name."""
        with patch("importlib.import_module", return_value=sample_tools_module):
            registry = ToolRegistry("mock_tools.tools")

            add_tool = registry.get_tool("add_numbers")
            assert isinstance(add_tool, Tool)
            assert add_tool.name == "add_numbers"

    def test_get_tool_not_found(self):
        """Test getting a non-existent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="Tool 'nonexistent' not found"):
            registry.get_tool("nonexistent")

    def test_get_all_tools(self, sample_tools_module):
        """Test getting all tools returns a copy."""
        with patch("importlib.import_module", return_value=sample_tools_module):
            registry = ToolRegistry("mock_tools.tools")

            all_tools = registry.get_all_tools()
            assert isinstance(all_tools, dict)
            assert len(all_tools) == 2

            # Verify it's a copy by modifying it
            original_length = len(registry)
            all_tools.clear()
            assert len(registry) == original_length

    def test_list_tool_names(self, sample_tools_module):
        """Test listing all tool names."""
        with patch("importlib.import_module", return_value=sample_tools_module):
            registry = ToolRegistry("mock_tools.tools")

            tool_names = registry.list_tool_names()
            assert isinstance(tool_names, list)
            assert "add_numbers" in tool_names

    def test_len(self, sample_tools_module):
        """Test __len__ returns the correct number of tools."""
        registry = ToolRegistry()
        assert len(registry) == 0

        with patch("importlib.import_module", return_value=sample_tools_module):
            registry.load_tools_from_module("mock_tools.tools")
            assert len(registry) == 2

    def test_contains(self, sample_tools_module):
        """Test __contains__ checks for tool existence."""
        with patch("importlib.import_module", return_value=sample_tools_module):
            registry = ToolRegistry("mock_tools.tools")

            assert "add_numbers" in registry
            assert "nonexistent_tool" not in registry

    def test_discover_tools_with_decorator(self, sample_tools_module):
        """Test discovering tools created with @tool decorator."""
        registry = ToolRegistry()
        discovered = registry._discover_tools_from_module(sample_tools_module)

        assert len(discovered) == 2
        assert "add_numbers" in discovered
        assert "multiply_numbers" in discovered

    def test_discover_tools_with_class_based_tools(self):
        """Test discovering class-based Tool instances."""
        # Create a temporary module with class-based tools
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir)
            tools_file = module_dir / "class_tools.py"

            tools_file.write_text("""
from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool

class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="A custom tool",
            arguments=[
                ToolArgument("value", "int", "A value"),
            ],
        )

    def execute(self, value: int) -> int:
        return value * 2

# Create an instance
my_tool_instance = MyTool()
""")

            spec = importlib.util.spec_from_file_location("class_tools", tools_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["class_tools"] = module
            spec.loader.exec_module(module)

            try:
                registry = ToolRegistry()
                discovered = registry._discover_tools_from_module(module)

                assert len(discovered) == 1
                assert "my_tool" in discovered
                assert isinstance(discovered["my_tool"], Tool)
            finally:
                del sys.modules["class_tools"]

    def test_empty_module(self):
        """Test loading a module with no tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir)
            tools_file = module_dir / "empty_module.py"
            tools_file.write_text("# Empty module with no tools\n")

            spec = importlib.util.spec_from_file_location("empty_module", tools_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["empty_module"] = module
            spec.loader.exec_module(module)

            try:
                registry = ToolRegistry()
                discovered = registry._discover_tools_from_module(module)

                assert len(discovered) == 0
            finally:
                del sys.modules["empty_module"]
