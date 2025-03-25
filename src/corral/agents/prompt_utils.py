from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from promptstore import PromptStore


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
    default_uuid: str,
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
        return store.get(default_uuid)
    elif isinstance(prompt_input, str):
        return StringPrompt(prompt_input)
    else:
        return prompt_input
