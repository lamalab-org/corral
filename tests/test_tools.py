from typing import Literal

import pytest
from jsonschema.validators import validator_for
from pydantic import Field

from corral.core.tool import Tool, tool
from corral.workspace import WorkspaceFilesystem, build_workspace_tools


# Sample functions for testing (new Field-based style)
def sample_valid_tool(
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Operation to perform"
    ),
    x: float = Field(description="First number to operate on"),
    y: float = Field(description="Second number to operate on"),
) -> str:
    """Perform basic math operations."""
    operations = {
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y if y != 0 else "Error: Division by zero",
    }
    return str(operations[operation]())


def sample_tool_with_defaults(
    x: float = Field(description="First number to add"),
    y: float = Field(default=1.0, description="Second number to add"),
) -> str:
    """Add two numbers, with optional second number."""
    return str(x + y)


# Fixtures
@pytest.fixture()
def calculator_tool():
    """Fixture providing a basic calculator tool"""
    return tool(sample_valid_tool)


@pytest.fixture()
def calculator_with_defaults():
    """Fixture providing a calculator with default arguments"""
    return tool(sample_tool_with_defaults)


# Test classes
class TestToolCreation:
    def test_valid_tool_creation(self, calculator_tool):
        """Test that a valid function can be converted to a tool"""
        assert isinstance(calculator_tool, Tool)
        assert calculator_tool.name == "sample_valid_tool"

        # Test argument parsing
        arg_names = {arg.name for arg in calculator_tool.arguments}
        assert arg_names == {"operation", "x", "y"}

        # Test choices parsing
        operation_arg = next(
            arg for arg in calculator_tool.arguments if arg.name == "operation"
        )
        assert operation_arg.choices == ["add", "subtract", "multiply", "divide"]

    @pytest.mark.parametrize(
        ("operation", "x", "y", "expected"),
        [
            ("add", 5, 3, "8"),
            ("subtract", 5, 3, "2"),
            ("multiply", 4, 2, "8"),
            ("divide", 5, 0, "Error: Division by zero"),
        ],
    )
    def test_tool_operations(self, calculator_tool, operation, x, y, expected):
        """Test various calculator operations"""
        result = calculator_tool.execute(operation=operation, x=x, y=y)
        assert result == expected

    def test_default_arguments(self, calculator_with_defaults):
        """Test handling of default arguments"""
        args = calculator_with_defaults.arguments
        x_arg = next(arg for arg in args if arg.name == "x")
        y_arg = next(arg for arg in args if arg.name == "y")

        assert x_arg.required
        assert not y_arg.required
        assert y_arg.default == 1.0

        # Test execution with default
        result = calculator_with_defaults.execute(x=5)
        assert result == "6.0"


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("args", "expected_valid", "error_message"),
        [
            ({"x": 1.0}, False, "Missing required argument"),
            ({"operation": "invalid_op", "x": 1.0, "y": 2.0}, False, "Must be one of"),
        ],
    )
    def test_argument_validation(
        self, calculator_tool, args, expected_valid, error_message
    ):
        """Test argument validation with various inputs"""
        is_valid, error = calculator_tool.validate_arguments(args)
        assert is_valid == expected_valid
        if error_message:
            assert error_message in (error or "")


class TestDocstringValidation:
    def test_missing_docstring(self):
        """Test rejection of functions without docstrings"""

        def no_docstring(x: int = Field(description="X")) -> str:
            return str(x)

        with pytest.raises(ValueError) as exc_info:
            tool(no_docstring)
        assert "docstring" in str(exc_info.value)


def test_usage_guide():
    """Test that get_usage_guide renders argument metadata."""

    @tool
    def documented_tool(
        name: str = Field(description="a name"),
    ) -> str:
        """A documented tool."""
        return "ok"

    guide = documented_tool.get_usage_guide()
    assert "- name (string, required): a name" in guide


def test_integration_with_field_annotations():
    """Test with Field annotations and union types.

    This test verifies that a function with complex types like `list[float] | None`
    is properly handled and the tool is correctly created.
    """

    @tool
    def test_function(
        slab_cif: str = Field(description="CIF string of the slab."),
        adsorbate_cif: str = Field(description="CIF string of the adsorbate."),
        height: float = Field(
            default=2.0,
            description="Height above the slab surface where the adsorbate should be placed.",
        ),
        site: list[float] | None = Field(
            default=None,
            description="Optional fractional coordinate [x, y, z] for placement.",
        ),
    ) -> str:
        """Place an adsorbate on a slab at a specified adsorption site."""
        return "Test result"

    # Verify all arguments were correctly parsed
    args = {arg.name: arg for arg in test_function.arguments}

    assert len(args) == 4
    assert "slab_cif" in args
    assert "adsorbate_cif" in args
    assert "height" in args
    assert "site" in args

    # Verify the docstring description was properly captured
    assert "place an adsorbate on a slab" in test_function.description.lower()

    # Test that the tool can be executed
    result = test_function.execute(
        slab_cif="sample_slab", adsorbate_cif="sample_adsorbate", height=2.5
    )
    assert result == "Test result"


# Tests for provider tool schemas
def test_tool_openai_format_basic():
    """Test basic conversion of a tool to OpenAI function format."""

    @tool
    def sample_tool(param1: str, param2: int = 5) -> str:
        """Sample tool description

        Args:
            param1: First parameter
            param2: Second parameter (default: 5)

        Returns:
            Result string
        """
        return f"{param1}-{param2}"

    tool_def = sample_tool.get_openai_tool_format()
    function = tool_def["function"]

    # Check structure
    assert tool_def["type"] == "function"
    assert "name" in function
    assert "description" in function
    assert "parameters" in function

    # Check values
    assert function["name"] == "sample_tool"
    assert "Sample tool description" in function["description"]

    # Check schema
    schema = function["parameters"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema

    # Check properties
    assert "param1" in schema["properties"]
    assert "param2" in schema["properties"]

    # Check param1 (required)
    assert schema["properties"]["param1"]["type"] == "string"
    assert "param1" in schema["required"]

    # Check param2 (optional with default) - optional params get ['type', 'null']
    param2_type = schema["properties"]["param2"]["type"]
    if isinstance(param2_type, list):
        assert "integer" in param2_type
        assert "null" in param2_type
    else:
        assert param2_type == "integer"
    assert schema["properties"]["param2"]["default"] == 5


def test_tool_openai_format_with_choices():
    """Test provider schema conversion with parameter choices."""

    @tool
    def tool_with_choices(mode: Literal["fast", "accurate", "balanced"]) -> str:
        """Tool with restricted parameter values

        Args:
            mode: Operation mode

        Returns:
            Result
        """
        return f"Mode: {mode}"

    schema = tool_with_choices.get_openai_tool_format()["function"]["parameters"]

    # Check that choices are converted to enum
    assert "enum" in schema["properties"]["mode"]
    assert schema["properties"]["mode"]["enum"] == [
        "fast",
        "accurate",
        "balanced",
    ]


def test_tool_openai_format_complex_types():
    """Test provider schema conversion with complex parameter types."""

    @tool
    def complex_tool(
        name: str, count: int, value: float, active: bool, tags: str = "[]"
    ) -> str:
        """Tool with various parameter types

        Args:
            name: Name as string
            count: Count as integer
            value: Value as float
            active: Active flag as boolean
            tags: JSON array of tags (default: [])

        Returns:
            Result
        """
        return "result"

    schema = complex_tool.get_openai_tool_format()["function"]["parameters"]
    props = schema["properties"]

    # Check type mappings (required params have simple types)
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["value"]["type"] == "number"
    assert props["active"]["type"] == "boolean"

    # Optional param with default gets ['type', 'null']
    tags_type = props["tags"]["type"]
    if isinstance(tags_type, list):
        assert "string" in tags_type
    else:
        assert tags_type == "string"

    # Check required fields
    required = schema["required"]
    assert "name" in required
    assert "count" in required
    assert "value" in required
    assert "active" in required
    # Strict provider schemas keep nullable/defaulted fields in `required`.
    assert "tags" in required


def test_tool_openai_format_preserves_descriptions():
    """Test that parameter descriptions are preserved in provider format."""

    @tool
    def documented_tool(param1: str, param2: int) -> str:
        """Tool with detailed documentation

        Args:
            param1: This is a detailed description of param1
            param2: This is a detailed description of param2

        Returns:
            Result string
        """
        return "result"

    schema = documented_tool.get_openai_tool_format()["function"]["parameters"]
    props = schema["properties"]

    assert "detailed description of param1" in props["param1"]["description"]
    assert "detailed description of param2" in props["param2"]["description"]


def test_tool_openai_format_with_hidden_args():
    """Test that hidden args are not exposed in provider format."""

    @tool(hidden_args=["api_key"])
    def api_tool(endpoint: str, api_key: str = "secret") -> str:
        """Call an API endpoint

        Args:
            endpoint: The API endpoint to call

        Returns:
            API response
        """
        return f"Calling {endpoint} with {api_key}"

    schema = api_tool.get_openai_tool_format()["function"]["parameters"]
    props = schema["properties"]

    # Only endpoint should be in the schema
    assert "endpoint" in props
    assert "api_key" not in props

    # endpoint should be required
    assert "endpoint" in schema["required"]


def test_file_tool_schemas_pass_metaschema_validation(tmp_path):
    """Every filesystem tool schema must satisfy its JSON metaschema."""

    tools = build_workspace_tools(WorkspaceFilesystem(tmp_path))
    assert "write_file" in tools
    for file_tool in tools.values():
        schema = file_tool.params_json_schema
        validator_for(schema).check_schema(schema)
