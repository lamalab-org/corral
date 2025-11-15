from typing import Any

from promptstore import Prompt, PromptStore

from corral.agents.utils import LiteLLMMessage


class ValidatedPrompt:
    """
    A wrapper around PromptStore's Prompt that adds validation for extra keys.

    This class provides the same Jinja2 templating capabilities as PromptStore's Prompt,
    but adds validation to catch typos in variable names by raising errors when extra
    keys are provided that don't match any template placeholders.

    Framework keys (those starting with underscore) are allowed as extras and won't
    trigger validation errors.
    """

    def __init__(self, prompt: Prompt):
        """Initialize with a PromptStore Prompt object."""
        self._prompt = prompt

    def fill(self, replacements: dict) -> str:
        """
        Fill the prompt template with provided variables, validating for extra keys.

        Args:
            replacements: Dictionary of variables to fill the template

        Returns:
            The filled template string

        Raises:
            KeyError: If extra user keys are provided that don't match any placeholders
        """
        # Separate framework keys (with underscore prefix) from user keys
        framework_keys = {k for k in replacements if k.startswith("_")}
        user_keys = set(replacements.keys()) - framework_keys

        # Get template variables from the underlying Prompt
        template_variables = set(self._prompt.variables)

        # Check for extra user keys that don't match any placeholders
        extra_keys = user_keys - template_variables
        if extra_keys:
            raise KeyError(
                f"Extra keys provided that don't match any placeholders: {sorted(extra_keys)}"
            )

        # Use PromptStore's Prompt.fill() for actual Jinja2 rendering
        return self._prompt.fill(replacements)


def ensure_jinja_compatible(prompt: str | Any) -> Any:
    """
    Convert a prompt to a Jinja2-compatible format with a .fill() method.

    This function takes various prompt formats and ensures they have a .fill() method
    that can be used for Jinja-style variable substitution using Jinja2 templating.
    String prompts are wrapped in ValidatedPrompt for typo detection.

    Args:
        prompt (str | Any): The prompt to convert. Can be:
            - A string (will be wrapped in ValidatedPrompt with PromptStore's Prompt)
            - An object that already has a .fill() method (returned as-is)
            - Any other object (will raise TypeError)

    Returns:
        Any: A prompt object with a .fill() method that accepts a dict of variables
            and returns the filled prompt string

    Raises:
        TypeError: If the prompt is not a string or doesn't have a .fill() method

    Examples:
        >>> # String prompt with Jinja2 format
        >>> prompt = ensure_jinja_compatible("Hello {{name}}")
        >>> prompt.fill({"name": "World"})
        'Hello World'

        >>> # Already compatible prompt object
        >>> from promptstore import Prompt
        >>> prompt = ensure_jinja_compatible(Prompt(content="Task: {{task}}", version=1))
        >>> prompt.fill({"task": "Solve this"})
        'Task: Solve this'
    """
    if isinstance(prompt, str):
        # Wrap PromptStore's Prompt in ValidatedPrompt for typo detection
        base_prompt = Prompt(content=prompt, version=1)
        return ValidatedPrompt(base_prompt)
    elif hasattr(prompt, "fill") and callable(prompt.fill):
        return prompt
    else:
        raise TypeError(
            f"Prompt must be a string or an object with a .fill() method, "
            f"got {type(prompt).__name__}"
        )


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
        base_prompt = Prompt(content=prompt_input, version=1)
        return ValidatedPrompt(base_prompt)
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
