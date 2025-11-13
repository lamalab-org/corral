import re
from typing import Any

from promptstore import PromptStore

from corral.agents.utils import LiteLLMMessage


class StringPrompt:
    """A simple string-based prompt that mimics PromptStore prompt interface with Jinja2 support."""

    def __init__(self, content):
        self.content = content

    def fill(self, replacements):
        # Extract all placeholders from the template (Jinja2 double-brace format)
        # Match {{variable}} patterns
        template_placeholders = set(re.findall(r"\{\{([^}]*)\}\}", self.content))

        # Separate framework keys (with underscore prefix) from user keys
        framework_keys = {k for k in replacements if k.startswith("_")}
        user_keys = set(replacements.keys()) - framework_keys

        # Check for extra user keys that don't match any placeholders
        extra_keys = user_keys - template_placeholders
        if extra_keys:
            raise KeyError(
                f"Extra keys provided that don't match any placeholders: {sorted(extra_keys)}"
            )

        result = self.content
        for key, value in replacements.items():
            # Replace {{key}} with the value
            result = result.replace(f"{{{{{key}}}}}", str(value))

        # Check for any remaining unfilled placeholders
        remaining_placeholders = re.findall(r"\{\{([^}]*)\}\}", result)
        if remaining_placeholders:
            raise KeyError(f"Missing values for placeholders: {remaining_placeholders}")

        return result


def ensure_jinja_compatible(prompt: str | Any) -> Any:
    """
    Convert a prompt to a Jinja2-compatible format with a .fill() method.

    This function takes various prompt formats and ensures they have a .fill() method
    that can be used for Jinja-style variable substitution. It also converts old-style
    {variable} format to Jinja2 {{variable}} format.

    Args:
        prompt (str | Any): The prompt to convert. Can be:
            - A string (will be converted to Jinja2 format if needed, then wrapped in StringPrompt)
            - An object that already has a .fill() method (returned as-is)
            - Any other object (will raise TypeError)

    Returns:
        Any: A prompt object with a .fill() method that accepts a dict of variables
            and returns the filled prompt string

    Raises:
        TypeError: If the prompt is not a string or doesn't have a .fill() method

    Examples:
        >>> # String prompt with old format
        >>> prompt = ensure_jinja_compatible("Hello {name}")
        >>> prompt.fill({"name": "World"})
        'Hello World'

        >>> # String prompt with Jinja2 format (unchanged)
        >>> prompt = ensure_jinja_compatible("Hello {{name}}")
        >>> prompt.fill({"name": "World"})
        'Hello World'

        >>> # Already compatible prompt object
        >>> from promptstore import Prompt
        >>> prompt = ensure_jinja_compatible(Prompt("Task: {{task}}"))
        >>> prompt.fill({"task": "Solve this"})
        'Task: Solve this'
    """
    if isinstance(prompt, str):
        # Convert old-style {variable} to Jinja2 {{variable}} format
        converted_prompt = _convert_to_jinja2_format(prompt)
        return StringPrompt(converted_prompt)
    elif hasattr(prompt, "fill") and callable(prompt.fill):
        return prompt
    else:
        raise TypeError(
            f"Prompt must be a string or an object with a .fill() method, "
            f"got {type(prompt).__name__}"
        )


def _convert_to_jinja2_format(text: str) -> str:
    """
    Convert old-style {variable} format to Jinja2 {{variable}} format.

    This function is careful to:
    1. Only convert single braces {variable} to double braces {{variable}}
    2. Leave already-doubled braces {{variable}} unchanged
    3. Handle nested structures and edge cases

    Args:
        text: The text potentially containing {variable} placeholders

    Returns:
        Text with all single-brace variables converted to double-brace Jinja2 format

    Examples:
        >>> _convert_to_jinja2_format("Task: {task_guide}")
        'Task: {{task_guide}}'
        >>> _convert_to_jinja2_format("Task: {{task_guide}}")
        'Task: {{task_guide}}'
        >>> _convert_to_jinja2_format("{a} and {b}")
        '{{a}} and {{b}}'
    """
    pattern = r"(?<!\{)\{([^{}]+)\}(?!\})"

    # Replace {variable} with {{variable}}
    return re.sub(pattern, r"{{\1}}", text)


def get_prompt(
    store: PromptStore,
    prompt_input: str | None | Any,
    default_uuid: str | None,
) -> Any:
    """Get prompt from store or create a string prompt.

    Args:
        store: The PromptStore instance
        prompt_input: A string, None, or existing prompt object
        default_uuid: UUID to use from store if prompt_input is None

    Returns:
        A prompt object with a fill method
    """
    if prompt_input is None:
        if default_uuid is None:
            raise ValueError("default_uuid cannot be None when prompt_input is None")
        return store.get(default_uuid)
    elif isinstance(prompt_input, str):
        return StringPrompt(prompt_input)
    else:
        return prompt_input


def create_prompt(
    system_prompt: Any,
    user_prompt: Any,
    task_guide: str | list,
    history: list[LiteLLMMessage] | None = None,
    surrender_prompt: Any | None = None,
    enable_surrender: bool = False,
    **kwargs,
) -> list[LiteLLMMessage]:
    """Create prompt for LLM including context and history

    Args:
        system_prompt: The system prompt object
        user_prompt: The user prompt object
        task_guide (Union[str, list]): The task guide or prompt to use
        history (list[LiteLLMMessage], optional): Message history to include. Defaults to None.
        surrender_prompt: The surrender prompt object. Instructions for how the agent can surrender from unsolvable tasks. Defaults to None.
        enable_surrender (bool, optional): Whether to enable the surrender option, which allows the agent to give up solving a task. Defaults to False.
        **kwargs: Additional keyword arguments for building user content

    Returns:
        List[LiteLLMMessage]: The prepared messages for the LLM
    """
    messages: list[LiteLLMMessage] = []

    if history is None:
        history = []

    if history:
        messages.extend(history)

    if system_prompt:
        messages.append(LiteLLMMessage(role="system", content=system_prompt))

    # Add surrender instructions if enabled
    if enable_surrender and surrender_prompt:
        surrender_instructions = surrender_prompt.fill({})
        kwargs["surrender_instructions"] = surrender_instructions

    user_content = build_user_content(user_prompt, task_guide=task_guide, **kwargs)

    messages.append(LiteLLMMessage(role="user", content=user_content))

    return messages


def build_user_content(
    user_prompt: Any, task_guide: str | list, **kwargs
) -> str | list:
    """
    Fill the user prompt with all provided parameters.

    This function passes all kwargs to the prompt's fill() method, allowing the prompt
    template itself to determine which variables it requires. If the prompt template
    contains placeholders that are not provided in kwargs, an error will be raised.

    Args:
        user_prompt: The user prompt object with a fill method
        task_guide (Union[str, list]): Task guide used for describing the environment task
        **kwargs: All keyword arguments to be passed to the prompt template

    Returns:
        Union[str, list]: The filled user prompt

    Raises:
        KeyError: If the prompt template contains placeholders not provided in kwargs
        ValueError: If task_guide is not str or list
    """
    fill_kwargs = kwargs.copy()

    fill_kwargs["task_guide"] = task_guide

    # Provide default empty string for surrender_instructions if not specified
    if "surrender_instructions" not in fill_kwargs:
        fill_kwargs["surrender_instructions"] = ""

    if isinstance(task_guide, list):
        LIST_PROMPT = "The task is to correctly answer the question with an image specified below."
        fill_kwargs["task_guide"] = LIST_PROMPT

        try:
            user_prompt_text = user_prompt.fill(fill_kwargs)
            user_content = [{"type": "text", "text": user_prompt_text}]
            user_content.extend(task_guide)
            return user_content
        except Exception as e:
            raise KeyError(
                f"Prompt template contains undefined placeholders. Error: {e}. "
                f"Available variables: {list(fill_kwargs.keys())}"
            ) from e
    elif isinstance(task_guide, str):
        try:
            return user_prompt.fill(fill_kwargs)
        except Exception as e:
            raise KeyError(
                f"Prompt template contains undefined placeholders. Error: {e}. "
                f"Available variables: {list(fill_kwargs.keys())}"
            ) from e
    else:
        raise ValueError(f"task_guide should be str or list, got {type(task_guide)}")
