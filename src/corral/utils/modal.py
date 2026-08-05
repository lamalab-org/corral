import modal


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
