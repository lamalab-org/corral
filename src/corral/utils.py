import inspect
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import get_type_hints

from litellm import embedding
from modal import App, Image, Mount, Secret, Volume

from corral.agents.utils import LiteLLMMessage
from corral.base import ModalTool, Tool, ToolArgument

MODAL_TOOL_REGISTRY = {}


def parse_docstring(func: Callable) -> tuple[str, list[ToolArgument]]:
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
        raise ValueError("Docstring must have an 'Args:' section")

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
            except ValueError:
                raise ValueError(
                    f"Invalid choices format for argument {arg_name}"
                ) from None

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


def create_modal_function(func: Callable, app: App, **modal_kwargs) -> Callable:
    """Create a Modal function with the given configuration."""
    modal_kwargs = {k: v for k, v in modal_kwargs.items() if v is not None}
    return app.function(**modal_kwargs)(func)


def modal_tool(
    app: App,
    image: Image | None = None,
    secrets: Sequence[Secret] | None = None,
    mounts: Sequence[Mount] | None = None,
    volumes: dict[str, Volume] | None = None,
    memory: int | None = None,
    timeout: int | None = None,
    cpu: float | None = None,
    retries: int | None = None,
    gpu: str | None = None,
    keep_warm: int | None = None,
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

    def decorator(func: Callable):
        modal_func = create_modal_function(func, app, **modal_kwargs)
        description, arguments = parse_docstring(func)

        tool_instance = ModalTool(
            modal_func=modal_func,
            name=func.__name__,
            description=description,
            arguments=arguments,
        )

        # Register the tool for later use
        MODAL_TOOL_REGISTRY[func.__name__] = tool_instance

        # Return the modal function
        return modal_func

    return decorator


def serialize_messages(messages: list[LiteLLMMessage]) -> list[dict]:
    """
    Serialize LiteLLMMessage objects to a format that can be saved to a JSON file.

    Args:
        messages (List[LiteLLMMessage]): The messages to serialize.

    Returns:
        List[Dict]: The serialized messages.
    """
    serializable_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            message_dict = msg.copy()
        else:
            message_dict = {"role": msg.role, "content": msg.content}

            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                message_dict["name"] = msg.name
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

        serializable_messages.append(message_dict)

    return serializable_messages


def save_agent_messages(
    messages: list[LiteLLMMessage],
    task_id: str,
    agent_name: str,
    output_dir: str = "agent_logs",
) -> str:
    """Save agent conversation to a JSON file for logging and analysis purposes.

    This function handles both regular dictionaries and LiteLLMMessage objects,
    properly serializing them for storage.

    Args:
        messages: List of message objects (LiteLLMMessages or dictionaries)
        task_id: The ID of the task being solved
        agent_name: The name of the agent that generated the messages
        output_dir: Directory to save the logs (will be created if it doesn't exist)

    Returns:
        str: Path to the saved file
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    file_path = Path(output_dir) / filename

    # Convert messages to serializable format
    serializable_messages = serialize_messages(messages)

    # Write to file with metadata and pretty formatting
    with Path(file_path).open("w") as f:
        json.dump(
            {
                "task_id": task_id,
                "agent": agent_name,
                "timestamp": timestamp,
                "messages": serializable_messages,
            },
            f,
            indent=2,
        )

    return file_path


def embed_text(
    chunks: list, model: str = "openai/text-embedding-3-small"
) -> list[list[float]]:
    """
    Embed a list of text chunks using the specified model.
    Args:
        chunks: List of text chunks to embed
        model: Model to use for embeddings. Default: "text-embedding-3-small"

    Returns:
        List of embeddings, each corresponding to a chunk

    Raises:
        ValueError: If chunks is not a non-empty list of strings
    """
    if (
        not chunks
        or not isinstance(chunks, list)
        or not all(isinstance(chunk, str) for chunk in chunks)
    ):
        raise ValueError("Input must be a non-empty list of strings")

    result_embeddings = embedding(
        model=model,
        input=chunks,
    )
    return [item["embedding"] for item in result_embeddings["data"]]


def chunk_text(text: str) -> list[str]:
    """
    Split a long text into smaller chunks based on the number of lines.
    Args:
        text: The text to be split into chunks

    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    return [chunk.strip() for chunk in text.split("\n")]
