from collections.abc import Callable, Sequence

import modal
from modal import App, Image, Secret, Volume

from corral.backend.tool import Tool, ToolArgument, arguments_to_schema
from corral.backend.tool_utils import parse_docstring

MODAL_TOOL_REGISTRY = {}


class ModalTool(Tool):
    def __init__(
        self,
        modal_func: Callable,
        name: str,
        description: str,
        arguments: list[ToolArgument],
    ):
        super().__init__(
            name=name,
            description=description,
            params_json_schema=arguments_to_schema(arguments),
        )
        self._modal_func = modal_func

    def execute(self, **kwargs):
        return self._modal_func.remote(**kwargs)


def remote_call(function_name: str, env_name: str = "chemenv"):
    """
    Decorator to call a function in a remote environment.
    This decorator is used to call a function in a remote environment
    using the Modal library.

    Args:
        function_name (str): The name of the function to call
        env_name (str): The name of the environment to use

    Returns:
        Callable: A wrapper function that calls the remote function
    """

    def wrapper(**kwargs) -> str:
        remote = modal.Function.from_name(env_name, function_name)
        return remote.remote(**kwargs)

    return wrapper


def create_modal_function(func: Callable, app: App, **modal_kwargs) -> Callable:
    """Create a Modal function with the given configuration."""
    modal_kwargs = {k: v for k, v in modal_kwargs.items() if v is not None}
    return app.function(**modal_kwargs)(func)


def modal_tool(
    app: App,
    image: Image | None = None,
    secrets: Sequence[Secret] | None = None,
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
