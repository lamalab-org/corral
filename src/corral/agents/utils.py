from __future__ import annotations

from typing import Any, TypedDict

import litellm
import openai
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RETRY_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIError,
    openai.APIStatusError,
    openai.InternalServerError,
)


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str
    tool_call_id: str | None
    name: str | None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, log_level=20),
    reraise=True,
)
def llm_call(
    model: str,
    messages: list[LiteLLMMessage],
    temperature: float,
    tools: dict[str, Any] | None = None,
    api_endpoint: str | None = None,
    **kwargs,
) -> list[LiteLLMMessage] | dict:
    """
    Call LiteLLM API with or without tools based on parameters

    Args:
        model (str): The model to use.
        messages (List[LiteLLMMessage]): The messages to send to the model.
        temperature (float): The temperature to use.
        tools (Dict[str, Any], optional): The tools to use. If provided, will use tool calling.
        api_endpoint (str, optional): The API endpoint to use. When using VLLM.
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.

    Returns:
        List[LiteLLMMessage] | Dict: The response from the model. Returns a dictionary for tool calling,
        otherwise returns a list with a single message.
    """
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "api_endpoint": api_endpoint,
            **kwargs,
        }

        if tools:
            params.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                }
            )
            response = litellm.completion(**params)
            return response.choices[0].message

        else:
            response = litellm.completion(**params)
            content = response.choices[0].message.content
            return [LiteLLMMessage(role="assistant", content=content)]

    except Exception as e:
        raise ValueError(f"Error in LiteLLM API call: {e}") from e
