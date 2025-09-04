from typing import Any

from promptstore import PromptStore

from corral.agents.utils import LiteLLMMessage


class StringPrompt:
    """A simple string-based prompt that mimics PromptStore prompt interface."""

    def __init__(self, content):
        self.content = content

    def fill(self, replacements):
        result = self.content
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


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
    **kwargs,
) -> list[LiteLLMMessage]:
    """Create prompt for LLM including context and history

    Args:
        system_prompt: The system prompt object
        user_prompt: The user prompt object
        task_guide (Union[str, list]): The task guide or prompt to use
        history (list[LiteLLMMessage], optional): Message history to include. Defaults to None.
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
