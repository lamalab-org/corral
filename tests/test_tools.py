from typing import Union

import pytest

from corral.backend.tool import Tool, tool
from corral.backend.tool_utils import format_type_annotation


# Sample functions for testing
def sample_valid_tool(operation: str, x: float, y: float) -> float:
    """Perform basic math operations.

    Args:
        operation: Operation to perform (choices: ["add", "subtract", "multiply", "divide"])
        x: First number to operate on
        y: Second number to operate on

    Returns:
        float: Result of the mathematical operation
    """
    operations = {
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y if y != 0 else "Error: Division by zero",
    }
    return operations[operation]()


def sample_tool_with_defaults(x: float, y: float = 1.0) -> float:
    """Add two numbers, with optional second number.

    Args:
        x: First number to add
        y: Second number to add (defaults to 1.0)

    Returns:
        float: Sum of the numbers
    """
    return x + y


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
            (
                {"operation": "add", "x": "not a number", "y": 1.0},
                False,
                "Invalid type",
            ),
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

        def no_docstring(x: int) -> int:
            return x

        with pytest.raises(ValueError) as exc_info:
            tool(no_docstring)
        assert "docstring" in str(exc_info.value)

    def test_missing_args_section(self):
        """Test rejection of docstrings without Args section"""

        def bad_docs(x: int) -> int:
            """Just does something."""
            return x

        with pytest.raises(ValueError) as exc_info:
            tool(bad_docs)
        assert "Args" in str(exc_info.value)

    def test_missing_type_hints(self):
        """Test rejection of functions without type hints"""

        def no_types(x, y):
            """Do something.

            Args:
                x: First number
                y: Second number
            """
            return x + y

        with pytest.raises(TypeError) as exc_info:
            tool(no_types)
        assert "type hints" in str(exc_info.value)

    @pytest.mark.parametrize("missing_param", ["x", "y", "operation"])
    def test_missing_parameter_docs(self, missing_param):
        """Test rejection of missing parameter documentation"""
        params = {
            "operation": "Math operation",
            "x": "First number",
            "y": "Second number",
        }

        # Remove the specified parameter's documentation
        del params[missing_param]

        # Create docstring with missing parameter
        arg_docs = []
        for param, desc in params.items():
            arg_docs.append(f"{param}: {desc}")

        docstring = """Do math.

        Args:
            {}

        Returns:
            float: The result
        """.format("\n        ".join(arg_docs))

        def test_func(operation: str, x: float, y: float) -> float:
            pass

        # Assign the docstring
        test_func.__doc__ = docstring

        with pytest.raises(ValueError) as exc_info:
            tool(test_func)
        assert "Missing documentation for parameters" in str(exc_info.value)


def test_format_type_annotation():
    """Test the format_type_annotation function with various types including unions."""
    # Basic types
    assert format_type_annotation(str) == "str"
    assert format_type_annotation(int) == "int"
    assert format_type_annotation(float) == "float"

    # Union types using | operator
    union_type = str | int
    assert format_type_annotation(union_type) == "str | int"

    # Union types using typing.Union

    union_type_old = Union[str, int]  # noqa: UP007
    assert format_type_annotation(union_type_old) == "str | int"

    # Optional type (which is Union[T, None])
    optional_type = str | None
    assert format_type_annotation(optional_type) == "str | None"

    # Nested unions and complex types
    complex_union = list[str | int] | None
    formatted = format_type_annotation(complex_union)
    assert "list" in formatted
    assert "str | int" in formatted
    assert "None" in formatted

    # Tuple with mixed types
    assert format_type_annotation(tuple[str, int]) == "tuple[str, int]"


def test_integration_with_actual_docstring():
    """Test with a docstring similar to the original function.

    This test verifies that a function with union types like `list[float] | None`
    is properly parsed and the tool is correctly created with appropriate
    type information.
    """

    @tool
    def test_function(
        slab_cif: str,
        adsorbate_cif: str,
        height: float = 2.0,
        site: list[float] | None = None,
    ) -> str:
        """
        Place an adsorbate on a slab at a specified adsorption site.
        If no site is specified, choose one from the top sites automatically.

        Args:
            slab_cif: CIF string of the slab.
            adsorbate_cif: CIF string of the adsorbate.
            height: Height (Å) above the slab surface where the adsorbate should be placed.
            site: Optional fractional coordinate [x, y, z] for placement.
                If None, the first top site will be used.
        Returns:
            str: CIF string of the combined structure.
        """
        return "Test result"

    # Verify all arguments were correctly parsed
    args = {arg.name: arg for arg in test_function.arguments}

    assert len(args) == 4
    assert "slab_cif" in args
    assert "adsorbate_cif" in args
    assert "height" in args
    assert "site" in args

    # Check specific properties of the site parameter
    site_param = args["site"]
    expected_type = format_type_annotation(list[float] | None)

    # Then make the assertion strict
    assert (
        site_param.type == expected_type
    ), f"Expected type '{expected_type}', got '{site_param.type}'"

    # Verify the docstring description was properly captured
    assert "place an adsorbate on a slab" in test_function.description.lower()

    # Test that the tool can be executed
    result = test_function.execute(
        slab_cif="sample_slab", adsorbate_cif="sample_adsorbate", height=2.5
    )
    assert result == "Test result"


# Tests for MCP integration methods
def test_tool_for_mcp_basic():
    """Test basic conversion of tool to MCP format"""

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

    mcp_def = sample_tool.for_mcp()

    # Check structure
    assert "name" in mcp_def
    assert "description" in mcp_def
    assert "inputSchema" in mcp_def

    # Check values
    assert mcp_def["name"] == "sample_tool"
    assert "Sample tool description" in mcp_def["description"]

    # Check schema
    schema = mcp_def["inputSchema"]
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


def test_tool_for_mcp_with_choices():
    """Test MCP conversion with parameter choices"""

    @tool
    def tool_with_choices(mode: str) -> str:
        """Tool with restricted parameter values

        Args:
            mode: Operation mode (choices: ["fast", "accurate", "balanced"])

        Returns:
            Result
        """
        return f"Mode: {mode}"

    mcp_def = tool_with_choices.for_mcp()

    # Check that choices are converted to enum
    assert "enum" in mcp_def["inputSchema"]["properties"]["mode"]
    assert mcp_def["inputSchema"]["properties"]["mode"]["enum"] == [
        "fast",
        "accurate",
        "balanced",
    ]


def test_tool_for_mcp_with_verbosity():
    """Test MCP conversion with different verbosity levels"""
    from corral.router.verbosity import ToolVerbosity

    @tool
    def verbose_tool(param: str) -> str:
        """Brief description of the tool

        Long detailed description that should be filtered
        based on verbosity level.

        Args:
            param: Parameter description

        Returns:
            Result
        """
        return param

    # Test COMPREHENSIVE (default)
    comprehensive = verbose_tool.for_mcp()
    assert "Brief description" in comprehensive["description"]

    # Test BRIEF
    brief = verbose_tool.for_mcp(verbosity=ToolVerbosity.BRIEF)
    assert "description" in brief

    # Test WORKFLOW
    workflow = verbose_tool.for_mcp(verbosity=ToolVerbosity.WORKFLOW)
    assert "description" in workflow


def test_tool_for_mcp_complex_types():
    """Test MCP conversion with complex parameter types"""

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

    mcp_def = complex_tool.for_mcp()
    props = mcp_def["inputSchema"]["properties"]

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
    required = mcp_def["inputSchema"]["required"]
    assert "name" in required
    assert "count" in required
    assert "value" in required
    assert "active" in required
    assert "tags" not in required  # Has default


def test_tool_from_mcp_raises_not_implemented():
    """Test that from_mcp raises NotImplementedError"""

    mcp_definition = {
        "name": "test_tool",
        "description": "Test tool",
        "inputSchema": {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        },
    }

    with pytest.raises(NotImplementedError) as exc_info:
        Tool.from_mcp(mcp_definition)

    assert "not currently supported" in str(exc_info.value)
    assert "Corral tools are designed to be converted TO MCP format" in str(
        exc_info.value
    )


def test_tool_for_mcp_preserves_descriptions():
    """Test that parameter descriptions are preserved in MCP format"""

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

    mcp_def = documented_tool.for_mcp()
    props = mcp_def["inputSchema"]["properties"]

    assert "detailed description of param1" in props["param1"]["description"]
    assert "detailed description of param2" in props["param2"]["description"]


def test_tool_for_mcp_with_hidden_args():
    """Test that hidden args are not exposed in MCP format"""

    @tool(hidden_args=["api_key"])
    def api_tool(endpoint: str, api_key: str = "secret") -> str:
        """Call an API endpoint

        Args:
            endpoint: The API endpoint to call

        Returns:
            API response
        """
        return f"Calling {endpoint} with {api_key}"

    mcp_def = api_tool.for_mcp()
    props = mcp_def["inputSchema"]["properties"]

    # Only endpoint should be in the schema
    assert "endpoint" in props
    assert "api_key" not in props

    # endpoint should be required
    assert "endpoint" in mcp_def["inputSchema"]["required"]
