"""Integration tests for the complete MCP workflow."""

import asyncio
import types
from unittest.mock import patch

import pytest

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool, tool
from corral.mcp.schema_converter import tool_to_json_schema
from corral.mcp.tool_registry import ToolRegistry
from corral.router.verbosity import ToolVerbosity


class TestEndToEndWorkflow:
    """Test the complete workflow from tool definition to MCP server."""

    def test_complete_workflow_with_decorator_tool(self):
        """Test complete workflow using @tool decorator."""

        # Step 1: Create a tool using decorator
        @tool
        def calculate_area(length: float, width: float) -> float:
            """Calculate the area of a rectangle.

            Args:
                length: Length of the rectangle
                width: Width of the rectangle
            """
            return length * width

        # Step 2: Create registry and register tool
        registry = ToolRegistry()
        registry.register_tool("calculate_area", calculate_area)

        # Step 3: Verify tool is registered
        assert "calculate_area" in registry
        assert len(registry) == 1

        # Step 4: Get the tool and verify it works
        tool_instance = registry.get_tool("calculate_area")
        result = tool_instance.execute(length=5.0, width=3.0)
        assert result == "15.0"  # Tool returns string representation

        # Step 5: Convert to JSON schema
        schema = tool_to_json_schema(tool_instance)
        assert schema["type"] == "object"
        assert "length" in schema["properties"]
        assert "width" in schema["properties"]
        assert schema["properties"]["length"]["type"] == "number"

    def test_complete_workflow_with_class_tool(self):
        """Test complete workflow using class-based Tool."""

        # Step 1: Create a class-based tool
        class TemperatureConverter(Tool):
            def __init__(self):
                super().__init__(
                    name="temperature_converter",
                    description="Convert temperature between Celsius and Fahrenheit",
                    arguments=[
                        ToolArgument(
                            "value", "float", "Temperature value", required=True
                        ),
                        ToolArgument(
                            "from_unit",
                            "str",
                            "Source unit",
                            required=True,
                            choices=["C", "F"],
                        ),
                        ToolArgument(
                            "to_unit",
                            "str",
                            "Target unit",
                            required=True,
                            choices=["C", "F"],
                        ),
                    ],
                )

            def execute(self, value: float, from_unit: str, to_unit: str) -> float:
                if from_unit == to_unit:
                    return value
                if from_unit == "C" and to_unit == "F":
                    return (value * 9 / 5) + 32
                if from_unit == "F" and to_unit == "C":
                    return (value - 32) * 5 / 9
                msg = f"Invalid conversion: {from_unit} to {to_unit}"
                raise ValueError(msg)

        # Step 2: Create instance and register
        converter = TemperatureConverter()
        registry = ToolRegistry()
        registry.register_tool("temperature_converter", converter)

        # Step 3: Verify and test
        assert "temperature_converter" in registry
        tool_instance = registry.get_tool("temperature_converter")

        # Test conversion
        result = tool_instance.execute(value=0.0, from_unit="C", to_unit="F")
        assert result == 32.0

        # Step 4: Convert to JSON schema
        schema = tool_to_json_schema(tool_instance)
        assert schema["type"] == "object"
        assert len(schema["properties"]) == 3
        assert schema["properties"]["from_unit"]["enum"] == ["C", "F"]
        assert schema["properties"]["to_unit"]["enum"] == ["C", "F"]

    def test_mcp_server_with_mock_tools(self):
        """Test MCP server with mock tools."""
        # Create mock module with tools
        mock_module = types.ModuleType("mockmath")

        @tool
        def calculator(operation: str, x: float, y: float) -> float:
            """Perform basic calculations.

            Args:
                operation: Operation to perform
                x: First number
                y: Second number
            """
            operations = {
                "add": lambda a, b: a + b,
                "multiply": lambda a, b: a * b,
            }
            return operations[operation](x, y)

        @tool
        def percentage_calculator(value: float, percentage: float) -> float:
            """Calculate percentage of a value.

            Args:
                value: The base value
                percentage: Percentage to calculate
            """
            return (value * percentage) / 100

        mock_module.calculator = calculator
        mock_module.percentage_calculator = percentage_calculator

        # Test with registry
        with patch("importlib.import_module", return_value=mock_module):
            registry = ToolRegistry("mockmath.tools")

            # Verify tools are loaded
            assert len(registry) == 2
            assert "calculator" in registry

            # Test calculator
            calc = registry.get_tool("calculator")
            result = calc.execute(operation="add", x=10.0, y=5.0)
            assert result == "15.0"  # Tool returns string representation

            # Test percentage calculator
            perc_calc = registry.get_tool("percentage_calculator")
            result = perc_calc.execute(value=200.0, percentage=25.0)
            assert result == "50.0"  # Tool returns string representation

    @pytest.mark.asyncio()
    async def test_async_tool_execution(self):
        """Test asynchronous tool execution."""

        @tool
        def slow_computation(x: int) -> int:
            """A slow computation.

            Args:
                x: Input value
            """
            return x * 2

        registry = ToolRegistry()
        registry.register_tool("slow_computation", slow_computation)

        tool_instance = registry.get_tool("slow_computation")

        # Execute in thread pool (as MCP server does) - returns string representation
        result = await asyncio.to_thread(tool_instance.execute, x=42)
        assert result == "84"


class TestVerbosityIntegration:
    """Test verbosity filtering across the stack."""

    def test_verbosity_comprehensive(self):
        """Test comprehensive verbosity keeps all details."""

        @tool
        def verbose_tool(param: str) -> str:
            """[brief] Short [comprehensive] A detailed description of the tool.

            Args:
                param: [brief] Param [comprehensive] A detailed parameter description
            """
            return param

        registry = ToolRegistry()
        registry.register_tool("verbose_tool", verbose_tool)

        tool_instance = registry.get_tool("verbose_tool")
        schema = tool_to_json_schema(
            tool_instance, verbosity=ToolVerbosity.COMPREHENSIVE
        )

        # Should include comprehensive content
        assert (
            "detailed parameter description"
            in schema["properties"]["param"]["description"]
        )

    def test_verbosity_brief(self):
        """Test brief verbosity filters to essentials."""

        @tool
        def verbose_tool(param: str) -> str:
            """[brief]Short[/brief] [comprehensive]A detailed description of the tool[/comprehensive].

            Args:
                param: [brief]Param[/brief] [comprehensive]A detailed parameter description[/comprehensive]
            """
            return param

        registry = ToolRegistry()
        registry.register_tool("verbose_tool", verbose_tool)

        tool_instance = registry.get_tool("verbose_tool")
        schema = tool_to_json_schema(tool_instance, verbosity=ToolVerbosity.BRIEF)

        # Should only have brief content
        assert schema["properties"]["param"]["description"] == "Param"


class TestMultipleToolsIntegration:
    """Test working with multiple tools."""

    def test_multiple_tools_registration_and_execution(self):
        """Test registering and using multiple tools together."""

        # Create multiple tools
        @tool
        def add(x: float, y: float) -> float:
            """Add two numbers.

            Args:
                x: First number
                y: Second number
            """
            return x + y

        @tool
        def multiply(x: float, y: float) -> float:
            """Multiply two numbers.

            Args:
                x: First number
                y: Second number
            """
            return x * y

        @tool
        def power(base: float, exponent: float) -> float:
            """Raise base to exponent.

            Args:
                base: Base number
                exponent: Exponent
            """
            return base**exponent

        # Register all tools
        registry = ToolRegistry()
        registry.register_tools(
            {
                "add": add,
                "multiply": multiply,
                "power": power,
            }
        )

        # Verify all are registered
        assert len(registry) == 3
        assert "add" in registry
        assert "multiply" in registry
        assert "power" in registry

        # Execute each tool - all return string representations
        assert registry.get_tool("add").execute(x=5.0, y=3.0) == "8.0"
        assert registry.get_tool("multiply").execute(x=5.0, y=3.0) == "15.0"
        assert registry.get_tool("power").execute(base=2.0, exponent=3.0) == "8.0"

        # Convert all to schemas
        for tool_name in ["add", "multiply", "power"]:
            tool_instance = registry.get_tool(tool_name)
            schema = tool_to_json_schema(tool_instance)
            assert schema["type"] == "object"
            assert "x" in schema["properties"] or "base" in schema["properties"]


class TestErrorHandling:
    """Test error handling across the MCP stack."""

    def test_registry_missing_tool(self):
        """Test error when accessing non-existent tool."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="Tool 'missing' not found"):
            registry.get_tool("missing")

    def test_tool_execution_error(self):
        """Test error handling during tool execution."""

        @tool
        def failing_tool(x: int) -> int:
            """A tool that fails.

            Args:
                x: Input value
            """
            if x < 0:
                msg = "x must be positive"
                raise ValueError(msg)
            return x

        registry = ToolRegistry()
        registry.register_tool("failing_tool", failing_tool)

        tool_instance = registry.get_tool("failing_tool")

        # Should work with valid input - returns string representation
        assert tool_instance.execute(x=5) == "5"

        # Should fail with invalid input
        with pytest.raises(ValueError, match="x must be positive"):
            tool_instance.execute(x=-1)

    def test_invalid_module_import(self):
        """Test error when loading invalid module."""
        with pytest.raises(ImportError):
            ToolRegistry("this.module.does.not.exist")
