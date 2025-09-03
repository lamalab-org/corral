from typing import Union

import pytest

from corral.base import Tool
from corral.utils import format_type_annotation, tool


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
