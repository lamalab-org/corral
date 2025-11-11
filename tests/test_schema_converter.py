"""Tests for schema_converter.py - conversion of Corral tools to MCP JSON Schema."""

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool, tool
from corral.mcp.schema_converter import (
    _extract_inner_type,
    _map_simple_type,
    _python_type_to_json_schema,
    tool_argument_to_json_schema,
    tool_to_json_schema,
)
from corral.router.verbosity import ToolVerbosity


class TestExtractInnerType:
    """Test _extract_inner_type function."""

    def test_simple_list(self):
        assert _extract_inner_type("list[str]", "list") == "str"

    def test_nested_list(self):
        assert _extract_inner_type("list[list[str]]", "list") == "list[str]"

    def test_deeply_nested_list(self):
        assert _extract_inner_type("list[list[list[int]]]", "list") == "list[list[int]]"

    def test_dict_with_types(self):
        assert _extract_inner_type("dict[str, int]", "dict") == "str, int"

    def test_dict_with_complex_value(self):
        assert _extract_inner_type("dict[str, list[int]]", "dict") == "str, list[int]"

    def test_list_of_dict(self):
        assert _extract_inner_type("list[dict[str, int]]", "list") == "dict[str, int]"

    def test_non_matching_type(self):
        assert _extract_inner_type("str", "list") == ""

    def test_empty_brackets(self):
        # Edge case - though not valid Python, should handle gracefully
        result = _extract_inner_type("list[]", "list")
        assert result == ""


class TestMapSimpleType:
    """Test _map_simple_type function."""

    def test_str_type(self):
        assert _map_simple_type("str") == "string"

    def test_int_type(self):
        assert _map_simple_type("int") == "integer"

    def test_float_type(self):
        assert _map_simple_type("float") == "number"

    def test_bool_type(self):
        assert _map_simple_type("bool") == "boolean"

    def test_dict_type(self):
        assert _map_simple_type("dict") == "object"

    def test_any_type(self):
        assert _map_simple_type("Any") == "object"

    def test_unknown_type_defaults_to_string(self):
        assert _map_simple_type("CustomType") == "string"


class TestPythonTypeToJsonSchema:
    """Test _python_type_to_json_schema function."""

    def test_simple_string_type(self):
        result = _python_type_to_json_schema("str", required=True)
        assert result == {"type": "string"}

    def test_simple_int_type(self):
        result = _python_type_to_json_schema("int", required=True)
        assert result == {"type": "integer"}

    def test_optional_type_with_pipe(self):
        result = _python_type_to_json_schema("str | None", required=False)
        assert result == {"type": ["string", "null"]}

    def test_optional_type_without_space(self):
        result = _python_type_to_json_schema("str|None", required=False)
        assert result == {"type": ["string", "null"]}

    def test_list_of_strings(self):
        result = _python_type_to_json_schema("list[str]", required=True)
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_list_of_ints(self):
        result = _python_type_to_json_schema("list[int]", required=True)
        assert result == {"type": "array", "items": {"type": "integer"}}

    def test_list_of_union_types(self):
        result = _python_type_to_json_schema("list[str | int]", required=True)
        assert result == {"type": "array", "items": {"type": ["string", "integer"]}}

    def test_dict_type(self):
        result = _python_type_to_json_schema("dict", required=True)
        assert result == {"type": "object"}

    def test_dict_with_brackets(self):
        """Test dict with type parameters adds additionalProperties."""
        result = _python_type_to_json_schema("dict[str, int]", required=True)
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_not_required_simple_type(self):
        result = _python_type_to_json_schema("str", required=False)
        assert result == {"type": ["string", "null"]}

    def test_nested_list_of_strings(self):
        """Test nested list type (list of lists)."""
        result = _python_type_to_json_schema("list[list[str]]", required=True)
        assert result == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        }

    def test_nested_list_of_integers(self):
        """Test nested list with integer type."""
        result = _python_type_to_json_schema("list[list[int]]", required=True)
        assert result == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }

    def test_deeply_nested_list(self):
        """Test deeply nested list (3 levels)."""
        result = _python_type_to_json_schema("list[list[list[str]]]", required=True)
        assert result == {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

    def test_list_of_dicts(self):
        """Test list containing dict types."""
        result = _python_type_to_json_schema("list[dict[str, int]]", required=True)
        assert result == {
            "type": "array",
            "items": {"type": "object", "additionalProperties": {"type": "integer"}},
        }

    def test_dict_with_type_params(self):
        """Test dict with key and value types."""
        result = _python_type_to_json_schema("dict[str, int]", required=True)
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_dict_with_list_value(self):
        """Test dict with list as value type."""
        result = _python_type_to_json_schema("dict[str, list[int]]", required=True)
        assert result == {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "integer"}},
        }

    def test_dict_with_dict_value(self):
        """Test nested dict types."""
        result = _python_type_to_json_schema("dict[str, dict[str, int]]", required=True)
        assert result == {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
        }

    def test_optional_nested_list(self):
        """Test optional nested list type."""
        result = _python_type_to_json_schema("list[list[str]] | None", required=False)
        expected_items = {"type": "array", "items": {"type": "string"}}
        # The outer array type should be nullable
        assert result["type"] == ["array", "null"]
        assert result["items"] == expected_items

    def test_list_of_union_with_nested_types(self):
        """Test that union detection doesn't trigger for nested types."""
        # This should NOT be treated as a union because the | is inside nested brackets
        result = _python_type_to_json_schema("list[dict[str, int]]", required=True)
        assert result["type"] == "array"
        assert "items" in result
        assert result["items"]["type"] == "object"

    def test_optional_simple_array(self):
        """Test optional simple array (required=False)."""
        result = _python_type_to_json_schema("list[str]", required=False)
        assert result["type"] == ["array", "null"]
        assert result["items"] == {"type": "string"}

    def test_optional_simple_dict(self):
        """Test optional simple dict (required=False)."""
        result = _python_type_to_json_schema("dict[str, int]", required=False)
        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"] == {"type": "integer"}

    def test_optional_dict_no_params(self):
        """Test optional dict without type parameters (required=False)."""
        result = _python_type_to_json_schema("dict", required=False)
        assert result["type"] == ["object", "null"]
        assert "additionalProperties" not in result

    def test_optional_deeply_nested_array(self):
        """Test optional deeply nested array (3 levels, required=False)."""
        result = _python_type_to_json_schema("list[list[list[int]]]", required=False)
        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "array"
        assert result["items"]["items"]["items"]["type"] == "integer"

    def test_optional_dict_with_complex_value(self):
        """Test optional dict with nested list value (required=False)."""
        result = _python_type_to_json_schema("dict[str, list[int]]", required=False)
        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"]["type"] == "array"
        assert result["additionalProperties"]["items"]["type"] == "integer"

    def test_optional_dict_with_nested_dict(self):
        """Test optional dict with nested dict value (required=False)."""
        result = _python_type_to_json_schema(
            "dict[str, dict[str, int]]", required=False
        )
        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"]["type"] == "object"
        assert (
            result["additionalProperties"]["additionalProperties"]["type"] == "integer"
        )

    def test_optional_list_of_dicts(self):
        """Test optional list containing dicts (required=False)."""
        result = _python_type_to_json_schema("list[dict[str, int]]", required=False)
        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == "object"
        assert result["items"]["additionalProperties"]["type"] == "integer"

    def test_optional_array_with_union_items(self):
        """Test optional array with union types in items (required=False)."""
        result = _python_type_to_json_schema("list[str | int]", required=False)
        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == ["string", "integer"]

    def test_optional_type_with_pipe_none_array(self):
        """Test optional array with explicit | None syntax."""
        result = _python_type_to_json_schema("list[int] | None", required=False)
        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == "integer"

    def test_optional_type_with_pipe_none_dict(self):
        """Test optional dict with explicit | None syntax."""
        result = _python_type_to_json_schema("dict[str, str] | None", required=False)
        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"]["type"] == "string"


class TestToolArgumentToJsonSchema:
    """Test tool_argument_to_json_schema function."""

    def test_simple_required_argument(self):
        arg = ToolArgument("name", "str", "User name", required=True)
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "string"
        assert result["description"] == "User name"
        assert "default" not in result

    def test_optional_argument_with_default(self):
        arg = ToolArgument(
            "limit", "int", "Maximum results", required=False, default=10
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == ["integer", "null"]
        assert result["description"] == "Maximum results"
        assert result["default"] == 10

    def test_argument_with_choices(self):
        arg = ToolArgument(
            "operation",
            "str",
            "Math operation",
            required=True,
            choices=["add", "subtract", "multiply", "divide"],
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "string"
        assert result["enum"] == ["add", "subtract", "multiply", "divide"]

    def test_list_argument(self):
        arg = ToolArgument("tags", "list[str]", "List of tags", required=True)
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_verbosity_filtering_comprehensive(self):
        arg = ToolArgument(
            "param",
            "str",
            "[brief] Short desc [comprehensive] Long detailed description",
            required=True,
        )
        result = tool_argument_to_json_schema(
            arg, verbosity=ToolVerbosity.COMPREHENSIVE
        )

        assert "Long detailed description" in result["description"]

    def test_verbosity_filtering_brief(self):
        arg = ToolArgument(
            "param",
            "str",
            "[brief]Short desc[/brief] [comprehensive]Long detailed description[/comprehensive]",
            required=True,
        )
        result = tool_argument_to_json_schema(arg, verbosity=ToolVerbosity.BRIEF)

        assert result["description"] == "Short desc"

    def test_nested_list_argument(self):
        """Test argument with nested list type."""
        arg = ToolArgument(
            "matrix", "list[list[int]]", "2D matrix of integers", required=True
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "array"
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "integer"
        assert result["description"] == "2D matrix of integers"

    def test_dict_with_complex_value_argument(self):
        """Test argument with dict containing list values."""
        arg = ToolArgument(
            "data",
            "dict[str, list[int]]",
            "Mapping of names to integer lists",
            required=True,
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "object"
        assert "additionalProperties" in result
        assert result["additionalProperties"]["type"] == "array"
        assert result["additionalProperties"]["items"]["type"] == "integer"

    def test_deeply_nested_list_argument(self):
        """Test argument with deeply nested list (3 levels)."""
        arg = ToolArgument(
            "tensor", "list[list[list[float]]]", "3D tensor", required=True
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == "array"
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "array"
        assert result["items"]["items"]["items"]["type"] == "number"

    def test_optional_list_argument(self):
        """Test optional list argument (required=False)."""
        arg = ToolArgument("tags", "list[str]", "Optional list of tags", required=False)
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == "string"

    def test_optional_dict_argument(self):
        """Test optional dict argument (required=False)."""
        arg = ToolArgument(
            "metadata", "dict[str, str]", "Optional metadata", required=False
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"]["type"] == "string"

    def test_optional_nested_list_argument(self):
        """Test optional nested list argument (required=False)."""
        arg = ToolArgument(
            "matrix", "list[list[int]]", "Optional 2D matrix", required=False
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == ["array", "null"]
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "integer"

    def test_optional_dict_with_list_value_argument(self):
        """Test optional dict with complex value (required=False)."""
        arg = ToolArgument(
            "lookup", "dict[str, list[int]]", "Optional lookup table", required=False
        )
        result = tool_argument_to_json_schema(arg)

        assert result["type"] == ["object", "null"]
        assert result["additionalProperties"]["type"] == "array"
        assert result["additionalProperties"]["items"]["type"] == "integer"


class TestToolToJsonSchema:
    """Test tool_to_json_schema function."""

    def test_tool_with_no_arguments(self):
        """Test a tool with no arguments."""

        # Use Tool class directly since @tool decorator requires Args section
        class NoArgsTool(Tool):
            def __init__(self):
                super().__init__(
                    name="no_args_tool",
                    description="A tool with no arguments.",
                    arguments=[],
                )

            def execute(self, **kwargs) -> str:
                return "result"

        tool_instance = NoArgsTool()
        schema = tool_to_json_schema(tool_instance)

        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_tool_with_required_arguments(self):
        """Test a tool with only required arguments."""

        @tool
        def calculator(operation: str, x: float, y: float) -> float:
            """Perform basic math operations.

            Args:
                operation: Operation to perform
                x: First number
                y: Second number
            """
            return x + y

        schema = tool_to_json_schema(calculator)

        assert schema["type"] == "object"
        assert "operation" in schema["properties"]
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["operation"]["type"] == "string"
        assert schema["properties"]["x"]["type"] == "number"
        assert schema["properties"]["y"]["type"] == "number"
        assert set(schema["required"]) == {"operation", "x", "y"}

    def test_tool_with_optional_arguments(self):
        """Test a tool with optional arguments."""

        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person.

            Args:
                name: Person's name
                greeting: Greeting to use (defaults to Hello)
            """
            return f"{greeting}, {name}!"

        schema = tool_to_json_schema(greet)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "greeting" in schema["properties"]
        assert schema["required"] == ["name"]
        assert schema["properties"]["greeting"]["default"] == "Hello"

    def test_tool_with_mixed_argument_types(self):
        """Test a tool with various argument types."""

        class MixedTool(Tool):
            def __init__(self):
                super().__init__(
                    name="mixed_tool",
                    description="Tool with mixed argument types",
                    arguments=[
                        ToolArgument("text", "str", "Text input", required=True),
                        ToolArgument(
                            "count", "int", "Count", required=False, default=1
                        ),
                        ToolArgument("tags", "list[str]", "Tags", required=True),
                        ToolArgument("metadata", "dict", "Metadata", required=False),
                    ],
                )

            def execute(
                self, text: str, count: int, tags: list, metadata: dict | None = None
            ) -> str:
                return "result"

        tool_instance = MixedTool()
        schema = tool_to_json_schema(tool_instance)

        assert schema["type"] == "object"
        assert len(schema["properties"]) == 4
        assert set(schema["required"]) == {"text", "tags"}
        assert schema["properties"]["text"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == ["integer", "null"]
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["metadata"]["type"] == ["object", "null"]

    def test_tool_with_choices(self):
        """Test a tool with argument choices."""

        class ToolWithChoices(Tool):
            def __init__(self):
                super().__init__(
                    name="tool_with_choices",
                    description="Tool with choices",
                    arguments=[
                        ToolArgument(
                            "mode",
                            "str",
                            "Operating mode",
                            required=True,
                            choices=["fast", "accurate", "balanced"],
                        ),
                    ],
                )

            def execute(self, mode: str) -> str:
                return mode

        tool_instance = ToolWithChoices()
        schema = tool_to_json_schema(tool_instance)

        assert schema["properties"]["mode"]["enum"] == ["fast", "accurate", "balanced"]

    def test_verbosity_comprehensive(self):
        """Test schema generation with comprehensive verbosity."""

        @tool
        def verbose_tool(param: str) -> str:
            """[brief] A tool [comprehensive] with detailed description.

            Args:
                param: [brief] Parameter [comprehensive] with detailed explanation
            """
            return param

        schema = tool_to_json_schema(
            verbose_tool, verbosity=ToolVerbosity.COMPREHENSIVE
        )

        assert "detailed" in schema["properties"]["param"]["description"]

    def test_verbosity_brief(self):
        """Test schema generation with brief verbosity."""

        @tool
        def verbose_tool(param: str) -> str:
            """[brief]A tool[/brief] [comprehensive]with detailed description[/comprehensive].

            Args:
                param: [brief]Parameter[/brief] [comprehensive]with detailed explanation[/comprehensive]
            """
            return param

        schema = tool_to_json_schema(verbose_tool, verbosity=ToolVerbosity.BRIEF)

        assert schema["properties"]["param"]["description"] == "Parameter"

    def test_tool_with_nested_list_arguments(self):
        """Test tool with nested list (2D matrix) argument."""

        class MatrixTool(Tool):
            def __init__(self):
                super().__init__(
                    name="matrix_tool",
                    description="Process a 2D matrix",
                    arguments=[
                        ToolArgument(
                            "matrix",
                            "list[list[int]]",
                            "2D matrix of integers",
                            required=True,
                        ),
                    ],
                )

            def execute(self, matrix: list[list[int]]) -> str:
                return "processed"

        tool_instance = MatrixTool()
        schema = tool_to_json_schema(tool_instance)

        assert schema["type"] == "object"
        assert "matrix" in schema["properties"]
        assert schema["properties"]["matrix"]["type"] == "array"
        assert schema["properties"]["matrix"]["items"]["type"] == "array"
        assert schema["properties"]["matrix"]["items"]["items"]["type"] == "integer"
        assert schema["required"] == ["matrix"]

    def test_tool_with_dict_complex_values(self):
        """Test tool with dict containing complex value types."""

        class DataTool(Tool):
            def __init__(self):
                super().__init__(
                    name="data_tool",
                    description="Process structured data",
                    arguments=[
                        ToolArgument(
                            "data",
                            "dict[str, list[int]]",
                            "Mapping of keys to integer lists",
                            required=True,
                        ),
                    ],
                )

            def execute(self, data: dict) -> str:
                return "processed"

        tool_instance = DataTool()
        schema = tool_to_json_schema(tool_instance)

        assert schema["type"] == "object"
        assert "data" in schema["properties"]
        assert schema["properties"]["data"]["type"] == "object"
        assert "additionalProperties" in schema["properties"]["data"]
        assert schema["properties"]["data"]["additionalProperties"]["type"] == "array"

    def test_tool_with_multiple_complex_types(self):
        """Test tool with various complex type combinations."""

        class ComplexTool(Tool):
            def __init__(self):
                super().__init__(
                    name="complex_tool",
                    description="Tool with complex type arguments",
                    arguments=[
                        ToolArgument(
                            "matrix",
                            "list[list[float]]",
                            "2D float matrix",
                            required=True,
                        ),
                        ToolArgument(
                            "lookup",
                            "dict[str, list[str]]",
                            "String lookup table",
                            required=True,
                        ),
                        ToolArgument(
                            "tensor",
                            "list[list[list[int]]]",
                            "3D tensor",
                            required=False,
                        ),
                        ToolArgument(
                            "nested_dict",
                            "dict[str, dict[str, int]]",
                            "Nested dictionary",
                            required=False,
                        ),
                    ],
                )

            def execute(self, **kwargs) -> str:
                return "processed"

        tool_instance = ComplexTool()
        schema = tool_to_json_schema(tool_instance)

        # Check matrix (list[list[float]])
        assert schema["properties"]["matrix"]["type"] == "array"
        assert schema["properties"]["matrix"]["items"]["type"] == "array"
        assert schema["properties"]["matrix"]["items"]["items"]["type"] == "number"

        # Check lookup (dict[str, list[str]])
        assert schema["properties"]["lookup"]["type"] == "object"
        assert schema["properties"]["lookup"]["additionalProperties"]["type"] == "array"
        assert (
            schema["properties"]["lookup"]["additionalProperties"]["items"]["type"]
            == "string"
        )

        # Check tensor (list[list[list[int]]])
        assert schema["properties"]["tensor"]["type"] == ["array", "null"]
        assert schema["properties"]["tensor"]["items"]["type"] == "array"
        assert schema["properties"]["tensor"]["items"]["items"]["type"] == "array"
        assert (
            schema["properties"]["tensor"]["items"]["items"]["items"]["type"]
            == "integer"
        )

        # Check nested_dict (dict[str, dict[str, int]])
        assert schema["properties"]["nested_dict"]["type"] == ["object", "null"]
        assert (
            schema["properties"]["nested_dict"]["additionalProperties"]["type"]
            == "object"
        )
        assert (
            schema["properties"]["nested_dict"]["additionalProperties"][
                "additionalProperties"
            ]["type"]
            == "integer"
        )

        # Check required fields
        assert set(schema["required"]) == {"matrix", "lookup"}

    def test_tool_with_list_of_dicts(self):
        """Test tool with list containing dict elements."""

        class ListDictTool(Tool):
            def __init__(self):
                super().__init__(
                    name="list_dict_tool",
                    description="Process list of dictionaries",
                    arguments=[
                        ToolArgument(
                            "records",
                            "list[dict[str, int]]",
                            "List of record dictionaries",
                            required=True,
                        ),
                    ],
                )

            def execute(self, records: list) -> str:
                return "processed"

        tool_instance = ListDictTool()
        schema = tool_to_json_schema(tool_instance)

        assert schema["properties"]["records"]["type"] == "array"
        assert schema["properties"]["records"]["items"]["type"] == "object"
        assert (
            schema["properties"]["records"]["items"]["additionalProperties"]["type"]
            == "integer"
        )

    def test_tool_with_optional_complex_types(self):
        """Test tool with various optional complex types to verify null handling."""

        class OptionalComplexTool(Tool):
            def __init__(self):
                super().__init__(
                    name="optional_complex_tool",
                    description="Tool with optional complex type arguments",
                    arguments=[
                        ToolArgument(
                            "tags", "list[str]", "Optional list of tags", required=False
                        ),
                        ToolArgument(
                            "metadata",
                            "dict[str, str]",
                            "Optional metadata dictionary",
                            required=False,
                        ),
                        ToolArgument(
                            "matrix",
                            "list[list[int]]",
                            "Optional 2D matrix",
                            required=False,
                        ),
                        ToolArgument(
                            "lookup_table",
                            "dict[str, list[float]]",
                            "Optional lookup table with float arrays",
                            required=False,
                        ),
                        ToolArgument("name", "str", "Required name", required=True),
                    ],
                )

            def execute(self, **kwargs) -> str:
                return "processed"

        tool_instance = OptionalComplexTool()
        schema = tool_to_json_schema(tool_instance)

        # Check optional simple list
        assert schema["properties"]["tags"]["type"] == ["array", "null"]
        assert schema["properties"]["tags"]["items"]["type"] == "string"

        # Check optional simple dict
        assert schema["properties"]["metadata"]["type"] == ["object", "null"]
        assert (
            schema["properties"]["metadata"]["additionalProperties"]["type"] == "string"
        )

        # Check optional nested list
        assert schema["properties"]["matrix"]["type"] == ["array", "null"]
        assert schema["properties"]["matrix"]["items"]["type"] == "array"
        assert schema["properties"]["matrix"]["items"]["items"]["type"] == "integer"

        # Check optional dict with complex value
        assert schema["properties"]["lookup_table"]["type"] == ["object", "null"]
        assert (
            schema["properties"]["lookup_table"]["additionalProperties"]["type"]
            == "array"
        )
        assert (
            schema["properties"]["lookup_table"]["additionalProperties"]["items"][
                "type"
            ]
            == "number"
        )

        # Check required simple type (should not have null)
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["name"]["type"] != ["string", "null"]

        # Check required fields
        assert schema["required"] == ["name"]
