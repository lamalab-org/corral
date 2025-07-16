import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, get_type_hints

from pydantic import BaseModel

from corral.backend.tool_utils import parse_docstring


class ToolCallStatus(Enum):
    SUCCESS = "success"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGS = "invalid_args"
    EXECUTION_ERROR = "execution_error"


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolArgument:
    """Specification for a tool argument"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    choices: list[Any] | None = None  # from transformers


class Tool:
    """Base class for tools
    Inherit from this class to create new tools.
    Should have an execute method that performs the tool's functionality.
    TODO: might need to take state
    TODO: add descriptions of the arguments of the class, i.e., name, description, arguments
    """

    def __init__(self, name: str, description: str, arguments: list[ToolArgument]):
        self.name = name
        self.description = description
        self.arguments = arguments

    def validate_arguments(
        self, provided_args: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate that all required arguments are provided with correct types"""
        for arg in self.arguments:
            if arg.required and arg.name not in provided_args:
                return False, f"Missing required argument: {arg.name}"

            if arg.name in provided_args:
                value = provided_args[arg.name]

                # Check choices if specified
                if arg.choices is not None and value not in arg.choices:
                    return (
                        False,
                        f"Invalid value for {arg.name}. Must be one of: {arg.choices}",
                    )

                try:
                    # Basic type checking
                    if arg.type == "int":
                        int(value)
                    elif arg.type == "float":
                        float(value)
                    elif arg.type == "bool":
                        isinstance(value, bool)
                except ValueError:
                    return (
                        False,
                        f"Invalid type for argument {arg.name}. Expected {arg.type}",
                    )

        return True, None

    def execute(self, **kwargs) -> str:
        """Execute the tool functionality"""
        raise NotImplementedError

    def get_usage_guide(self) -> str:
        """Generate a usage guide for the tool"""
        args_desc = []
        for arg in self.arguments:
            required = (
                "required" if arg.required else f"optional, default: {arg.default}"
            )
            args_desc.append(
                f"- {arg.name} ({arg.type}, {required}): {arg.description}"
            )

        return f"""Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


def tool(func: Callable) -> Tool:
    """
    Decorator to convert a function into a Tool. The decorated function must have:
    1. A complete docstring with description and Args section.
    2. Type hints for all parameters.
    3. A return type hint.

    The docstring must follow this format:
    ```
                Brief description of what the tool does.

                Args:
                    param1: Description of first parameter (choices: ["optional", "list", "of", "choices"])
                    param2: Description of second parameter
                    ...

                Returns:
                    Description of what the function returns
    ```

    Args:
        func (Callable): The function to convert into a tool

    Returns:
        Tool: A Tool instance wrapping the function

    Raises:
        ValueError: If the function lacks proper docstring, type hints, or has invalid format
        TypeError: If the function signature is incompatible with Tool requirements
    """
    # Validate function has a docstring
    if not func.__doc__:
        raise ValueError(
            f"Function {func.__name__} must have a docstring describing its purpose and arguments. "
            "See the decorator documentation for the required format."
        )

    # Validate function has type hints
    type_hints = get_type_hints(func)
    if not type_hints:
        raise TypeError(
            f"Function {func.__name__} must have type hints for all parameters and return type. "
            "Example: def func(param1: str, param2: int) -> str"
        )

    # Validate return type is specified
    if "return" not in type_hints:
        raise TypeError(
            f"Function {func.__name__} must specify a return type hint. "
            "Example: def func(param: str) -> str"
        )

    # Validate docstring format
    doc = inspect.getdoc(func)
    if doc is None or "Args:" not in doc:
        raise ValueError(
            f"Function {func.__name__}'s docstring must have an 'Args:' section. "
            "See the decorator documentation for the required format."
        )

    try:
        description, arguments = parse_docstring(func)
    except Exception as e:
        raise ValueError(
            f"Error parsing docstring for function {func.__name__}: {e!s}. "
            "Please ensure it follows the required format shown in the decorator documentation."
        ) from e

    # Validate all parameters have documentation
    signature_params = set(inspect.signature(func).parameters.keys())
    documented_params = {arg.name for arg in arguments}
    if missing_docs := signature_params - documented_params:
        raise ValueError(
            f"Missing documentation for parameters: {', '.join(missing_docs)}. "
            "All parameters must be documented in the Args section of the docstring."
        )

    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(
                name=func.__name__, description=description, arguments=arguments
            )

        def execute(self, **kwargs):
            return str(func(**kwargs))

    return FunctionTool()
