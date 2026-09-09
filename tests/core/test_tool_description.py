from corral.core.tool import tool
from corral.core.tool_description import (
    default_argument_description,
    default_tool_description,
)


def test_default_description_prefers_brief_section():
    description = """Fallback text.

    [BRIEF]Short runtime description.[/BRIEF]
    [DETAILED]Internal detail that must not reach the agent.[/DETAILED]
    """

    assert default_tool_description(description) == "Short runtime description."


def test_default_description_preserves_untagged_docstring():
    assert default_tool_description("Add two values.") == "Add two values."


def test_default_argument_prefers_args_brief_section():
    description = (
        "[ARGS_BRIEF]Input path.[/ARGS_BRIEF]"
        "[ARGS_DETAILED]Absolute path on the worker.[/ARGS_DETAILED]"
    )

    assert default_argument_description(description) == "Input path."


def test_provider_schema_always_uses_fixed_default():
    @tool
    def inspect_value(value: str) -> str:
        """[BRIEF]Inspect a value.[/BRIEF]
        [DETAILED]A longer internal explanation.[/DETAILED]

        Args:
            value: [ARGS_BRIEF]Value to inspect.[/ARGS_BRIEF]
                [ARGS_DETAILED]Extra argument detail.[/ARGS_DETAILED]
        """
        return value

    function = inspect_value.get_openai_tool_format()["function"]

    assert function["description"] == "Inspect a value."
    assert function["parameters"]["properties"]["value"]["description"] == (
        "Value to inspect."
    )
