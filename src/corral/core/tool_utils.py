"""Formatting helpers for core tool schemas and annotations."""

from types import UnionType
from typing import Any, Union, get_args, get_origin

from loguru import logger


def format_json_schema_type(prop: dict[str, Any]) -> str:
    """Convert a JSON Schema property dict into a human-readable type string.

    Recursively resolves nested schemas (arrays, tuples via prefixItems,
    anyOf unions, enum literals, and scalar types) into a compact
    representation suitable for text-based tool guides.

    Args:
        prop: A single property dict from a JSON Schema.

    Returns:
        A formatted type string, e.g. "list[tuple[string, number]]".
    """
    if "anyOf" in prop:
        parts = [format_json_schema_type(variant) for variant in prop["anyOf"]]
        return " | ".join(parts)

    if "enum" in prop:
        choices = ", ".join(repr(c) for c in prop["enum"])
        return f"Literal[{choices}]"

    raw_type = prop.get("type")

    if raw_type == "array":
        if "prefixItems" in prop:
            inner = ", ".join(
                format_json_schema_type(item) for item in prop["prefixItems"]
            )
            return f"tuple[{inner}]"
        if "items" in prop:
            inner = format_json_schema_type(prop["items"])
            return f"list[{inner}]"
        return "list"

    if isinstance(raw_type, str):
        return raw_type

    if isinstance(raw_type, list):
        return " | ".join(raw_type)

    return "any"


def format_type_annotation(annotation) -> str:
    """Formats type annotations to readable strings with proper error handling."""
    try:
        # Handle None type
        if annotation is type(None):
            return "None"

        # Handle basic types
        if isinstance(annotation, type):
            return annotation.__name__

        # Handle typing constructs
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types (including Optional which is Union[T, None])
        if origin is Union:
            # Handle Optional[T] which is Union[T, None]
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return f"{format_type_annotation(non_none_type)} | None"
            else:
                # Handle regular Union[T, U, ...]
                formatted_args = [format_type_annotation(arg) for arg in args]
                return " | ".join(formatted_args)

        # Handle Python 3.10+ union syntax (str | int) using types.UnionType
        try:
            is_union_type = isinstance(annotation, UnionType)
        except (ImportError, AttributeError):
            is_union_type = False

        if is_union_type:
            # Recursively format all types in the union
            formatted_args = [
                format_type_annotation(arg) for arg in annotation.__args__
            ]
            # Handle Optional[T] (T | None)
            if len(formatted_args) == 2 and "None" in formatted_args:
                non_none_type = next(a for a in formatted_args if a != "None")
                return f"{non_none_type} | None"
            return " | ".join(formatted_args)

        # Handle generic types like List[str], Dict[str, int], etc.
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            if args:
                formatted_args = [format_type_annotation(arg) for arg in args]
                return f"{origin_name}[{', '.join(formatted_args)}]"
            return origin_name

        # Fallback to string representation for unknown types
        return str(annotation).replace("typing.", "")

    except Exception as e:
        logger.warning(f"Could not format type annotation {annotation}: {e}")
        return str(annotation)
