"""
Schema converter for Corral tools to MCP JSON Schema format.

This module provides utilities to convert Corral's ToolArgument specifications
into MCP-compatible JSON Schema format.
"""

from typing import Any

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool
from corral.router.verbosity import ToolVerbosity, VerbosityConfig


def _extract_inner_type(type_str: str, outer_type: str) -> str:
    """
    Extract the inner type from a generic type annotation.

    Args:
        type_str: The full type string (e.g., "list[str]", "dict[str, int]")
        outer_type: The outer type to match (e.g., "list", "dict")

    Returns:
        The inner type string (e.g., "str", "str, int")
    """
    if not type_str.startswith(f"{outer_type}["):
        return ""

    # Find matching closing bracket
    bracket_count = 0
    start_idx = len(outer_type) + 1

    for i in range(start_idx, len(type_str)):
        if type_str[i] == "[":
            bracket_count += 1
        elif type_str[i] == "]":
            if bracket_count == 0:
                return type_str[start_idx:i]
            bracket_count -= 1

    return type_str[start_idx:-1] if type_str.endswith("]") else ""


def _python_type_to_json_schema(
    python_type: str, required: bool = True
) -> dict[str, Any]:
    """
    Convert a Python type string to JSON Schema type specification.

    Args:
        python_type: String representation of the Python type (e.g., "str", "int", "list[str]", "list[list[int]]")
        required: Whether the field is required

    Returns:
        JSON Schema type specification
    """
    # Handle optional types (e.g., "str | None")
    is_optional = "| None" in python_type or "|None" in python_type
    if is_optional:
        python_type = python_type.replace("| None", "").replace("|None", "").strip()

    # Handle list types (including nested lists)
    if python_type.startswith("list["):
        inner_type = _extract_inner_type(python_type, "list").strip()

        # Handle list of unions (e.g., "list[str | int]")
        if "|" in inner_type and not inner_type.startswith(("list[", "dict[")):
            inner_types = [t.strip() for t in inner_type.split("|")]
            items_schema = {"type": [_map_simple_type(t) for t in inner_types]}
        # Handle nested lists or dicts (e.g., "list[list[str]]", "list[dict[str, int]]")
        elif inner_type.startswith(("list[", "dict[")):
            items_schema = _python_type_to_json_schema(inner_type, required=True)
        else:
            items_schema = {"type": _map_simple_type(inner_type)}

        schema = {"type": "array", "items": items_schema}
    elif python_type.startswith("dict["):
        # Extract key and value types from dict[K, V]
        inner_type = _extract_inner_type(python_type, "dict").strip()

        # For dict with type parameters, we still use object type in JSON Schema
        # but we could add additionalProperties if needed
        schema = {"type": "object"}

        # If there are type parameters, add additionalProperties
        if inner_type:
            # Split by comma, handling nested types
            parts = []
            current = ""
            bracket_count = 0
            for char in inner_type:
                if char == "," and bracket_count == 0:
                    parts.append(current.strip())
                    current = ""
                else:
                    if char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1
                    current += char
            if current:
                parts.append(current.strip())

            # If we have a value type (second parameter), add additionalProperties
            if len(parts) >= 2:
                value_type = parts[1]
                if value_type.startswith(("list[", "dict[")):
                    schema["additionalProperties"] = _python_type_to_json_schema(
                        value_type, required=True
                    )
                else:
                    schema["additionalProperties"] = {
                        "type": _map_simple_type(value_type)
                    }
    elif python_type.startswith("dict"):
        # Handle simple dict without type parameters
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
