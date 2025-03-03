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
    model: str, messages: list[LiteLLMMessage], temperature: float, **kwargs
) -> list[LiteLLMMessage]:
    """
    Call LiteLLM API

    Args:
        model (str): The model to use.
        messages (List[LiteLLMMessage]): The messages to send to the model.
        temperature (float): The temperature to use.
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.

    Returns:
        List[LiteLLMMessage]: The messages returned by the model.
    """
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        return [LiteLLMMessage(role="assistant", content=content)]

    except Exception as e:
        raise ValueError(f"Error in LiteLLM API call: {e}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, log_level=20),
    reraise=True,
)
def llm_tool_call(
    model: str,
    messages: list[LiteLLMMessage],
    tools: dict[str, Any],
    temperature: float,
    api_endpoint: str | None = None,
    **kwargs,
) -> dict:
    """
    Call LiteLLM API with tools

    Args:
        model (str): The model to use.
        messages (List[LiteLLMMessage]): The messages to send to the model.
        tools (Dict[str, Any]): The tools to use.
        temperature (float): The temperature to use.
        api_endpoint (str): The API endpoint to use. When using VLLM.
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.

    Returns:
        Dict: The response from the model.
    """
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            api_endpoint=api_endpoint,
            tool_choice="auto",
            **kwargs,
        )
        return response.choices[0].message

    except Exception as e:
        raise ValueError(f"Error in LiteLLM API call: {e}") from e
