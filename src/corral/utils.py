import inspect
from typing import Callable, List, Optional, Sequence, Union, get_type_hints

from modal import App, Image, Mount, Secret, Volume

from corral.base import ModalTool, Tool, ToolArgument

MODAL_TOOL_REGISTRY = {}


def parse_docstring(func: Callable) -> tuple[str, List[ToolArgument]]:
    """Parse function docstring to get description and arguments"""
    doc = inspect.getdoc(func)
    if not doc:
        raise ValueError(f"Function {func.__name__} must have a docstring")

    # Split docstring into sections
    sections = doc.split("\n\n")
    description = sections[0].strip()

    # Find Args section
    args_section = None
    for section in sections:
        if section.strip().startswith("Args:"):
            args_section = section.strip()
            break

    if not args_section:
        raise ValueError(f"Docstring must have an 'Args:' section")

    # Parse arguments section, skip the "Args:" line
    args_lines = [
        line.strip() for line in args_section.splitlines()[1:] if line.strip()
    ]
    arguments = []

    # Get type hints from function
    type_hints = get_type_hints(func)

    # Parse each argument line
    for line in args_lines:
        if ":" not in line:
            continue
        arg_name, arg_desc = line.split(":", 1)
        arg_name = arg_name.strip()
        arg_desc = arg_desc.strip()

        # Parse choices if specified in format (choices: [val1, val2, ...])
        choices = None
        if "(choices:" in arg_desc:
            desc_parts = arg_desc.split("(choices:", 1)
            arg_desc = desc_parts[0].strip()
            choices_str = desc_parts[1].split(")", 1)[0].strip()
            try:
                choices = eval(choices_str)  # Convert string representation to list
            except:
                raise ValueError(f"Invalid choices format for argument {arg_name}")

        # Get type from type hints
        if arg_name not in type_hints:
            continue  # Skip non-argument sections like Returns

        arg_type = type_hints[arg_name].__name__

        # Check if argument has default value
        signature = inspect.signature(func)
        param = signature.parameters.get(arg_name)
        has_default = param.default != inspect.Parameter.empty if param else False
        default_value = param.default if has_default else None

        arguments.append(
            ToolArgument(
                name=arg_name,
                type=arg_type,
                description=arg_desc,
                required=not has_default,
                default=default_value,
                choices=choices,
            )
        )

    return description, arguments


def tool(func: Callable) -> Tool:
    """
    Decorator to convert a function into a Tool. The decorated function must have:
    1. A complete docstring with description and Args section
    2. Type hints for all parameters
    3. A return type hint

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

    Example:
    ```python
    @tool
    def calculator(operation: str, x: float, y: float) -> float:
        '''Perform basic math operations.

        Args:
            operation: Operation to perform (choices: ["add", "subtract", "multiply", "divide"])
            x: First number to operate on
            y: Second number to operate on

        Returns:
            float: Result of the mathematical operation
        '''
        operations = {
            "add": lambda: x + y,
            "subtract": lambda: x - y,
            "multiply": lambda: x * y,
            "divide": lambda: x / y if y != 0 else "Error: Division by zero",
        }
        return operations[operation]()
    ```

    Args:
        func: The function to convert into a tool

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
    if "Args:" not in doc:
        raise ValueError(
            f"Function {func.__name__}'s docstring must have an 'Args:' section. "
            "See the decorator documentation for the required format."
        )

    try:
        description, arguments = parse_docstring(func)
    except Exception as e:
        raise ValueError(
            f"Error parsing docstring for function {func.__name__}: {str(e)}. "
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


def modal_tool(
    app: App,
    image: Optional[Image] = None,
    secrets: Optional[Sequence[Secret]] = None,
    mounts: Optional[Sequence[Mount]] = None,
    volumes: Optional[dict[Union[str, str], Volume]] = None,
    memory: Optional[int] = None,
    timeout: Optional[int] = None,
    cpu: Optional[float] = None,
    retries: Optional[int] = None,
    gpu: Optional[str] = None,
    keep_warm: Optional[int] = None,
    block_network: bool = False,
    **kwargs,
):
    """
    Decorator that returns a ModalTool instance (which inherits from Tool)
    that can be used both as a Tool and as a Modal function.
    """
    if app is None:
        raise ValueError(
            "The 'app' argument is required. This is the Modal App instance."
        )

    def decorator(func: Callable):
        # Convert the function to a Modal function with explicit arguments
        modal_kwargs = {
            "image": image,
            "secrets": secrets,
            "mounts": mounts,
            "volumes": volumes,
            "memory": memory,
            "timeout": timeout,
            "cpu": cpu,
            "retries": retries,
            "gpu": gpu,
            "keep_warm": keep_warm,
            "block_network": block_network,
            **kwargs,
        }
        name = func.__name__
        # Remove None values from modal_kwargs
        modal_kwargs = {k: v for k, v in modal_kwargs.items() if v is not None}

        modal_func = app.function(**modal_kwargs)(func)

        description, arguments = parse_docstring(func)
        # Convert arguments to ToolArgument instances
        # Create and return a ModalTool instance
        tool_instance = ModalTool(
            modal_func=modal_func,
            name=name,
            description=description,
            arguments=arguments,
        )

        # Register the tool for later use
        MODAL_TOOL_REGISTRY[name] = tool_instance
        func.tool = tool_instance

        # Return the modal function
        return modal_func

    return decorator
