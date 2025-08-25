import inspect
import re
from collections.abc import Callable
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from loguru import logger

from corral.backend.schema import ToolArgument


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


def extract_tagged_content(text: str, tag: str) -> str | None:
    """Extract content from a specific tagged section with error handling"""
    try:
        pattern = rf"\[{re.escape(tag)}\](.*?)\[/{re.escape(tag)}\]"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    except Exception as e:
        logger.warning(f"Error extracting tagged content for tag '{tag}': {e}")
        return None


def extract_main_description_from_complex_docstring(docstring: str) -> str:
    """Extract main description from complex docstring with tagged sections"""
    try:
        # First try to get BRIEF section
        brief = extract_tagged_content(docstring, "BRIEF")
        if brief:
            return brief

        # If no BRIEF, get content before first tagged section or Args section
        lines = docstring.strip().split("\n")
        description_lines = []

        for line in lines:
            line_text = line.strip()
            # Stop at first tagged section or Args section
            if line_text.startswith(("[", "Args:")):
                break
            if line_text:
                description_lines.append(line_text)

        return (
            " ".join(description_lines)
            if description_lines
            else "No description available"
        )

    except Exception as e:
        logger.warning(f"Error extracting main description: {e}")
        return "No description available"


def parse_argument_from_lines(arg_lines: list[str]) -> dict[str, Any]:
    """Parse a single argument from its lines with improved error handling"""
    if not arg_lines:
        return {}

    try:
        # First line should contain "arg_name: description"
        first_line = arg_lines[0]
        if ":" not in first_line:
            return {}

        # Extract argument name and start of description
        colon_pos = first_line.find(":")
        arg_name_part = first_line[:colon_pos].strip()
        first_desc_part = first_line[colon_pos + 1 :].strip()

        # Extract bare argument name (remove type annotation if present)
        if "(" in arg_name_part and ")" in arg_name_part:
            arg_name = arg_name_part.split("(")[0].strip()
        else:
            arg_name = arg_name_part.strip()

        # Validate argument name
        if not arg_name or not arg_name.isidentifier():
            logger.warning(f"Invalid argument name: {arg_name}")
            return {}

        # Combine all description lines
        all_desc_parts = [first_desc_part] + [line.strip() for line in arg_lines[1:]]
        full_description = " ".join(part for part in all_desc_parts if part)

        # Parse choices if specified
        choices = None
        if "(choices:" in full_description:
            try:
                desc_parts = full_description.split("(choices:", 1)
                full_description = desc_parts[0].strip()
                choices_str = desc_parts[1].split(")", 1)[0].strip()
                # Safely evaluate choices
                choices = eval(
                    choices_str
                )  # This should be replaced with ast.literal_eval for safety
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Invalid choices format for argument {arg_name}: {e}")

        # Extract tagged sections from argument description
        raises_info = extract_tagged_content(full_description, "RAISES")
        limitations_info = extract_tagged_content(full_description, "LIMITATIONS")

        return {
            "name": arg_name,
            "description": full_description,
            "choices": choices,
            "raises": raises_info,
            "limitations": limitations_info,
        }

    except Exception as e:
        logger.error(f"Error parsing argument from lines: {e}")
        return {}


def parse_args_section(args_section: str) -> list[dict[str, Any]]:
    """Parse the Args section to extract individual arguments with their descriptions"""
    if not args_section:
        return []

    try:
        # Split into lines and remove "Args:" header
        lines = args_section.splitlines()
        if lines and lines[0].strip().startswith("Args:"):
            lines = lines[1:]

        arguments = []
        current_arg_lines = []

        # Pattern to match argument start: "arg_name:" or "arg_name (type):"
        arg_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:")

        for line in lines:
            if arg_pattern.match(line):
                # New argument found, process previous one
                if current_arg_lines:
                    arg_data = parse_argument_from_lines(current_arg_lines)
                    if arg_data.get("name"):
                        arguments.append(arg_data)
                current_arg_lines = [line]
            elif current_arg_lines:
                # Continuation of current argument
                current_arg_lines.append(line)

        # Process the last argument
        if current_arg_lines:
            arg_data = parse_argument_from_lines(current_arg_lines)
            if arg_data.get("name"):
                arguments.append(arg_data)

        return arguments

    except Exception as e:
        logger.error(f"Error parsing args section: {e}")
        return []


def is_complex_docstring(docstring: str) -> bool:
    """Check if docstring has tagged sections like [BRIEF], [DETAILED], etc."""
    if not docstring:
        return False

    tagged_keywords = [
        "BRIEF",
        "DETAILED",
        "PROCEDURAL",
        "CONTEXTUAL",
        "WORKFLOW_INTEGRATION",
        "SYNTACTICAL",
        "RAISES",
        "LIMITATIONS",
    ]
    return any(f"[{keyword}]" in docstring for keyword in tagged_keywords)


def parse_docstring(func: Callable) -> tuple[str, list[ToolArgument]]:
    """Parse function docstring to get description and arguments with comprehensive error handling."""

    # Validate function
    if not callable(func):
        raise ValueError(f"Expected callable function, got {type(func)}")

    # Get docstring
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    try:
        # Determine if this is a complex docstring with tagged sections
        if is_complex_docstring(doc):
            # Extract main description from complex docstring
            # description = extract_main_description_from_complex_docstring(doc)
            doc_without_args_returns = re.sub(
                r"Args:.*?(?=Returns:|$)", "", doc, flags=re.DOTALL
            )
            description = doc_without_args_returns.strip()
        else:
            # Handle simple docstring - use first section as description
            sections = doc.split("\n\n")
            description = sections[0].strip()

        # Validate description
        if not description or description == "No description available":
            logger.warning(f"Function {func.__name__} has empty or invalid description")

        # Find Args section (works for both simple and complex docstrings)
        args_section = None

        # Look for Args section in the docstring
        args_match = re.search(
            r"Args:(.*?)(?=Returns:|$)", doc, re.DOTALL | re.IGNORECASE
        )
        if args_match:
            args_section = "Args:" + args_match.group(1)
        else:
            # Fallback: look for Args section in split sections (for simple docstrings)
            sections = doc.split("\n\n")
            for section in sections:
                if section.strip().startswith("Args:"):
                    args_section = section.strip()
                    break

        if not args_section:
            raise ValueError(
                f"Function {func.__name__} docstring must have an 'Args:' section"
            )

        # Parse arguments from the Args section
        parsed_args = parse_args_section(args_section)

        # Get type hints and signature from function
        try:
            type_hints = get_type_hints(func)
            signature = inspect.signature(func)
        except Exception as e:
            raise ValueError(
                f"Error getting type hints for {func.__name__}: {e}"
            ) from e

        arguments = []

        # Convert parsed arguments to ToolArgument objects
        for arg_data in parsed_args:
            try:
                arg_name = arg_data["name"]

                # Skip if not a real parameter
                if arg_name not in type_hints:
                    logger.warning(
                        f"Argument {arg_name} not found in type hints for {func.__name__}"
                    )
                    continue

                # Get type and default from function signature
                arg_annotation = type_hints[arg_name]
                arg_type = format_type_annotation(arg_annotation)

                param = signature.parameters.get(arg_name)
                has_default = param and param.default != inspect.Parameter.empty
                default_value = param.default if has_default else None

                arguments.append(
                    ToolArgument(
                        name=arg_name,
                        type=arg_type,
                        description=arg_data["description"],
                        required=not has_default,
                        default=default_value,
                        choices=arg_data["choices"],
                    )
                )

            except Exception as e:
                logger.error(
                    f"Error processing argument {arg_data.get('name', 'unknown')} for {func.__name__}: {e}"
                )
                continue

        return description, arguments

    except Exception as e:
        logger.error(f"Error parsing docstring for {func.__name__}: {e}")
        raise ValueError(
            f"Error parsing docstring for function {func.__name__}: {e}"
        ) from e
