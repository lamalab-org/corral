"""Tests for schema_converter.py - conversion of Corral tools to MCP JSON Schema."""

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool, tool
from corral.mcp.schema_converter import (
    _map_simple_type,
    _python_type_to_json_schema,
    tool_argument_to_json_schema,
    tool_to_json_schema,
)
from corral.router.verbosity import ToolVerbosity


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
        result = _python_type_to_json_schema("dict[str, int]", required=True)
        assert result == {"type": "object"}

    def test_not_required_simple_type(self):
        result = _python_type_to_json_schema("str", required=False)
        assert result == {"type": ["string", "null"]}


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
