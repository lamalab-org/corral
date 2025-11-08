"""
Schema converter for Corral tools to MCP JSON Schema format.

This module provides utilities to convert Corral's ToolArgument specifications
into MCP-compatible JSON Schema format.
"""

from typing import Any

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool
from corral.router.verbosity import ToolVerbosity, VerbosityConfig


def _python_type_to_json_schema(
    python_type: str, required: bool = True
) -> dict[str, Any]:
    """
    Convert a Python type string to JSON Schema type specification.

    Args:
        python_type: String representation of the Python type (e.g., "str", "int", "list[str]")
        required: Whether the field is required

    Returns:
        JSON Schema type specification
    """
    # Handle optional types (e.g., "str | None")
    is_optional = "| None" in python_type or "|None" in python_type
    if is_optional:
        python_type = python_type.replace("| None", "").replace("|None", "").strip()

    # Handle list types
    if python_type.startswith("list["):
        # Extract inner type (e.g., "list[str]" -> "str")
        inner_type = python_type[5:-1].strip()

        # Handle list of unions (e.g., "list[str | int]")
        if "|" in inner_type:
            inner_types = [t.strip() for t in inner_type.split("|")]
            items_schema = {"type": [_map_simple_type(t) for t in inner_types]}
        else:
            items_schema = {"type": _map_simple_type(inner_type)}

        schema = {"type": "array", "items": items_schema}
    elif python_type.startswith("dict"):
        # Handle dict types
        schema = {"type": "object"}
    else:
        # Simple types
        json_type = _map_simple_type(python_type)
        schema = {"type": json_type}

    # Add null option if optional and required=False
    if (is_optional or not required) and isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]

    return schema


def _map_simple_type(python_type: str) -> str:
    """Map Python type strings to JSON Schema types."""
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object",
        "Any": "object",
    }
    return type_mapping.get(python_type, "string")


def tool_argument_to_json_schema(
    arg: ToolArgument, verbosity: ToolVerbosity = ToolVerbosity.COMPREHENSIVE
) -> dict[str, Any]:
    """
    Convert a Corral ToolArgument to JSON Schema property definition.

    Args:
        arg: ToolArgument instance to convert
        verbosity: Tool description verbosity level for filtering argument descriptions

    Returns:
        JSON Schema property definition for the argument
    """
    schema = _python_type_to_json_schema(arg.type, arg.required)

    # Filter argument description based on verbosity
    filtered_description = VerbosityConfig.filter_argument_description(
        arg.description, verbosity
    )
    schema["description"] = filtered_description

    # Add enum constraint if choices are specified
    if arg.choices is not None:
        schema["enum"] = arg.choices

    # Add default value if specified
    if arg.default is not None:
        schema["default"] = arg.default

    return schema


def tool_to_json_schema(
    tool: Tool, verbosity: ToolVerbosity = ToolVerbosity.COMPREHENSIVE
) -> dict[str, Any]:
    """
    Convert a Corral Tool to MCP-compatible JSON Schema format.

    Args:
        tool: Corral Tool instance to convert
        verbosity: Tool description verbosity level for filtering argument descriptions

    Returns:
        JSON Schema representing the tool's input parameters

    Example:
        {
            "type": "object",
            "properties": {
                "molecule_smiles": {
                    "type": "string",
                    "description": "SMILES string of the target molecule"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10
                }
            },
            "required": ["molecule_smiles"]
        }
    """
    properties = {}
    required_fields = []

    for arg in tool.arguments:
        properties[arg.name] = tool_argument_to_json_schema(arg, verbosity)
        if arg.required:
            required_fields.append(arg.name)

    schema = {
        "type": "object",
        "properties": properties,
    }

    if required_fields:
        schema["required"] = required_fields

    return schema
