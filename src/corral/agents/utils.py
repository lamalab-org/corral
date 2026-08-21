import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import litellm
import openai
from litellm.exceptions import BudgetExceededError, RateLimitError
from litellm.types.utils import Message
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)

from corral.agents.schema import BudgetExhaustedError
from corral.report.logging import logger

RETRY_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIError,
    openai.APIStatusError,
    openai.InternalServerError,
)

# 4xx responses are client errors: a malformed request (bad output schema,
# invalid params) fails identically on every attempt, so retrying just burns the
# whole 30/60/90s wait chain before the same error escapes. These few 4xx codes
# are the exception — they are genuinely transient — so they stay retryable
# alongside all 5xx, timeouts, and connection errors.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def _should_retry(exc: BaseException) -> bool:
    """Retry transient failures only; fail fast on non-retryable 4xx errors."""
    if not isinstance(exc, RETRY_EXCEPTIONS):
        return False
    status = getattr(exc, "status_code", None)
    if status is not None and 400 <= status < 500:
        return status in RETRYABLE_STATUS_CODES
    # 5xx, client-side timeouts, and connection errors (no status) are transient.
    return True


# Exceptions that should stop the benchmark immediately (not retry or continue)
STOP_BENCHMARK_EXCEPTIONS = (
    BudgetExceededError,  # litellm budget exceeded
    openai.AuthenticationError,  # Invalid API key
)

# Keywords in error messages that indicate quota/credits exhausted (not a temporary rate limit)
QUOTA_EXHAUSTED_KEYWORDS = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing",
    "spending limit",
    "budget",
)


def before_sleep_loguru(retry_state):
    # Surface *what* is being retried, not just the attempt count: without the
    # triggering exception a transient failure is invisible until (and unless)
    # the final one escapes, leaving only a bare LiteLLM error in the logs.
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    cause = f"{type(exc).__name__}: {exc}" if exc is not None else "unknown error"
    next_action = retry_state.next_action
    wait = f"{next_action.sleep:.0f}s" if next_action is not None else "unknown"
    retry_object = getattr(retry_state, "retry_object", None)
    stop = getattr(retry_object, "stop", None)
    maximum_attempts = getattr(stop, "max_attempt_number", 3)
    attempt_label = (
        f"{retry_state.attempt_number}/{maximum_attempts}"
        if maximum_attempts is not None
        else str(retry_state.attempt_number)
    )
    logger.warning(f"LLM call retry {attempt_label} after {cause}; waiting {wait}")


class LLMResponseMetadata(TypedDict, total=False):
    """Metadata for LLM responses"""

    id: str | None
    logprobs: Any | None
    usage: dict[str, int] | None


class LLMResponse:
    """
    Wrapper for LLM responses with optional metadata.

    This class wraps the raw LLM message response and includes optional metadata
    like logprobs, message ID, and token usage.

    Args:
        message (Message): The raw LLM message response
        metadata (LLMResponseMetadata | None): Optional metadata including logprobs, id, and usage

    Properties:
        content: Access message content
        role: Access message role
        tool_calls: Access tool calls (if any)
        logprobs: Access logprobs from metadata
        id: Access message ID from metadata
        usage: Access token usage from metadata
    """

    def __init__(
        self, message: Message, metadata: LLMResponseMetadata | None = None
    ) -> None:
        self.message = message
        self.metadata = metadata or LLMResponseMetadata()

    @property
    def content(self) -> str | None:
        """Get the message content"""
        return self.message.content

    @property
    def role(self) -> str:
        """Get the message role"""
        return self.message.role

    @property
    def tool_calls(self) -> Any:
        """Get tool calls from the message"""
        return getattr(self.message, "tool_calls", None)

    # Access to metadata
    @property
    def logprobs(self) -> Any:
        """Get logprobs from metadata"""
        return self.metadata.get("logprobs")

    @property
    def id(self) -> str | None:
        """Get message ID from metadata"""
        return self.metadata.get("id")

    @property
    def usage(self) -> dict[str, int] | None:
        """Get token usage from metadata"""
        return self.metadata.get("usage")


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str | list
    tool_call_id: str | None
    name: str | None
    id: str | None


def _usage_member(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _reasoning_tokens(usage: Any) -> int | None:
    """Read reasoning-token details from Chat Completions or Responses usage."""
    direct = _usage_member(usage, "reasoning_tokens")
    if direct is not None:
        return int(direct or 0)
    for details_key in (
        "completion_tokens_details",
        "output_tokens_details",
    ):
        details = _usage_member(usage, details_key)
        reported = _usage_member(details, "reasoning_tokens")
        if reported is not None:
            return int(reported or 0)
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_chain(wait_fixed(30), wait_fixed(60), wait_fixed(90)),
    retry=retry_if_exception(_should_retry),
    before_sleep=before_sleep_loguru,
    reraise=True,
)
async def llm_call(
    model: str,
    messages: list[LiteLLMMessage],
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    api_endpoint: str | None = None,
    return_usage: bool = True,
    **kwargs,
) -> LLMResponse:
    """
    Call LiteLLM with a streaming transport and reconstruct its final response.

    Args:
        model (str): The model to use.
        messages (list[LiteLLMMessage]): The messages to send to the model.
        temperature (float): The temperature to use.
        tools (dict[str, Any], optional): The tools to use. If provided, will use tool calling.
        api_endpoint (str, optional): The API endpoint to use. When using VLLM.
        return_usage (bool, optional): If True, includes token usage in metadata. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.
            Streaming is always enabled, even if `stream=False` is supplied.
            If 'logprobs' is True in kwargs, logprobs will be included in response metadata.

    Returns:
        LLMResponse: Wrapper containing the message and optional metadata (id, logprobs if requested, usage).
    """
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "api_base": api_endpoint,
            **kwargs,
        }

        # GPT-5.6 rejects chat-completions requests that combine function tools
        # with reasoning controls. The explicit LiteLLM `responses/` route
        # keeps the configured provider model unchanged while selecting the
        # endpoint that supports both features. This is also stable across the
        # range of LiteLLM versions used by the task-specific environments.
        use_gpt_5_6_responses = (
            tools is not None
            and kwargs.get("reasoning_effort") is not None
            and model.startswith("openai/gpt-5.6")
        )
        if use_gpt_5_6_responses:
            params["model"] = model.replace("openai/", "openai/responses/", 1)

        if "anthropic" in model:
            params["max_tokens"] = 8192

        # When extended thinking is on (LiteLLM turns `reasoning_effort` into an
        # Anthropic `thinking` block), Anthropic rejects any `temperature` other
        # than 1 with a 400. Force it so a reasoning run is not aborted; this also
        # covers reasoning calls made through this shared LiteLLM helper.
        if kwargs.get("reasoning_effort") or kwargs.get("thinking"):
            params["temperature"] = 1

        if tools is not None:
            params["tools"] = tools
            # Responses defaults to automatic tool selection. Older LiteLLM
            # bridges reject the otherwise redundant chat-completions value.
            if not use_gpt_5_6_responses:
                params["tool_choice"] = "auto"
        # Always consume the provider response as a stream. Besides making long
        # generations observable at the transport layer, this keeps an active
        # response from looking idle to gateways with read/idle timeouts. Build
        # the chunks back into LiteLLM's ordinary ModelResponse so callers keep
        # the same message, tool-call, logprob, and usage interface.
        params["stream"] = True
        response_stream = await litellm.acompletion(**params)
        chunks = [chunk async for chunk in response_stream]
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        if response is None:
            raise ValueError("LLM stream returned no response chunks")

        message = response.choices[0].message

        # If message content is None, try to get reasoning_content
        if message.content is None:
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content is not None:
                # Ensure reasoning_content is a string
                message.content = str(reasoning_content) if reasoning_content else None

        metadata = LLMResponseMetadata()
        metadata["id"] = response.id

        # Include logprobs if requested via kwargs
        if kwargs.get("logprobs"):
            metadata["logprobs"] = response.choices[0].logprobs

        # Include usage info if requested
        if return_usage:
            response_usage = response.usage
            usage = {
                "prompt_tokens": int(
                    (
                        _usage_member(response_usage, "prompt_tokens")
                        or _usage_member(response_usage, "input_tokens")
                        or 0
                    )
                    if response_usage
                    else 0
                ),
                "completion_tokens": int(
                    (
                        _usage_member(response_usage, "completion_tokens")
                        or _usage_member(response_usage, "output_tokens")
                        or 0
                    )
                    if response_usage
                    else 0
                ),
                "total_tokens": int(
                    (_usage_member(response_usage, "total_tokens") or 0)
                    if response_usage
                    else 0
                ),
            }
            reasoning_tokens = _reasoning_tokens(response_usage)
            if reasoning_tokens is not None:
                usage["reasoning_tokens"] = reasoning_tokens
            metadata["usage"] = usage

        return LLMResponse(message, metadata)

    except STOP_BENCHMARK_EXCEPTIONS as e:
        # Re-raise as BudgetExhaustedError to stop benchmark immediately
        raise BudgetExhaustedError(
            f"Benchmark stopped: {type(e).__name__} - {e}"
        ) from e

    except RateLimitError as e:
        # Check if this is a quota exhaustion (not a temporary rate limit)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in QUOTA_EXHAUSTED_KEYWORDS):
            raise BudgetExhaustedError(
                f"Benchmark stopped - quota exhausted: {e}"
            ) from e
        # Otherwise, it's a temporary rate limit - re-raise to let retry handle it
        raise

    except Exception as e:
        raise e


def format_examples(examples: list[str] | None) -> str:
    """Format few-shot part of the prompt from a list of shots

    Args:
        examples (List[str], optional): The examples to format. Defaults to None.

    Returns:
        str: The formatted examples
    """
    if examples is None:
        return ""
    else:
        example_prompt = f"""To help you in understanding this task, the next {
            len(examples)
        } examples are provided:\n\n"""
        return example_prompt + "\n\n".join(examples)


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

            if hasattr(msg, "id") and msg.id:
                message_dict["id"] = msg.id
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                message_dict["name"] = msg.name
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                message_dict["tool_calls"] = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        tool_call = {
                            "id": tc.get("id"),
                            "function": {
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments"),
                            },
                        }
                    else:
                        tool_call = {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    message_dict["tool_calls"].append(tool_call)

        serializable_messages.append(message_dict)

    return serializable_messages


def save_agent_messages(
    messages: list[LiteLLMMessage],
    task_id: str,
    agent_name: str,
    model: str,
    output_dir: str | None = None,
    tools: list[dict] | None = None,
    tool_verbosity: str = "brief",
    trace_metadata: dict[str, Any] | None = None,
) -> str:
    """Save agent conversation to a JSON file for logging and analysis purposes.

    This function handles both regular dictionaries and LiteLLMMessage objects,
    properly serializing them for storage.

    Args:
        messages (list[LiteLLMMessage]): List of message objects (LiteLLMMessages or dictionaries)
        task_id (str): The ID of the task being solved
        agent_name (str): The name of the agent that generated the messages
        output_dir (str, optional): Directory to save the logs (will be created if it doesn't exist). Default is "agent_logs".
        tools (list[dict], optional): List of available tools used by the agent. Defaults to None.
        tool_verbosity (str, optional): Verbosity level for tool descriptions. Defaults to "brief".
        trace_metadata (dict, optional): Agent-specific trace information saved
            beside `messages`. It is deliberately not merged into individual
            messages, which keeps replayed/API-bound messages schema-compatible.

    Returns:
        str: Path to the saved file
    """
    if output_dir is None:
        output_dir = f"agent_logs-{agent_name}-{model}-{tool_verbosity}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    file_path = Path(output_dir) / filename

    # Convert messages to serializable format
    serializable_messages = serialize_messages(messages)

    # Prepare log data with metadata
    log_data = {
        "task_id": task_id,
        "model": model,
        "agent": agent_name,
        "tool_verbosity": tool_verbosity,
        "tools": tools,
        "timestamp": timestamp,
        "messages": serializable_messages,
    }
    if trace_metadata is not None:
        log_data["trace_metadata"] = trace_metadata

    # Write to file with metadata and pretty formatting
    with Path(file_path).open("w") as f:
        json.dump(
            log_data,
            f,
            indent=2,
        )

    return file_path


def get_context_window(model: str) -> int:
    """
    Return the total context window (max tokens) for a given LiteLLM model name.

    Tries litellm.get_max_tokens(model) first, then falls back to the
    model cost/context map. Returns None if the model isn't known.

    Args:
        model (str): The model name, e.g., "gpt-4o", "claude-haiku-4-5".

    Returns:
        int: The max input tokens for the model, or None if unknown.
    """
    if (
        model
        == "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
    ):
        return 131072
    return litellm.model_cost.get(model, {}).get("max_input_tokens", None)


def remove_old_budget_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove any existing CONTEXT_BUDGET messages from the first message in the list.

    Args:
        messages (List[dict[str, Any]]): The messages to filter.

    Returns:
        List[dict[str, Any]]: The filtered messages.
    """
    if messages and "[CONTEXT_BUDGET]" in str(messages[0].get("content", "")):
        # Replace content between CONTEXT_BUDGET tokens with nothing in first message
        content = str(messages[0]["content"])
        start_tag = "[CONTEXT_BUDGET]"
        end_tag = "[/CONTEXT_BUDGET]"

        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            # Keep everything after end_tag (remove budget from beginning)
            messages[0]["content"] = content[end_idx + len(end_tag) :].lstrip()
    return messages


def count_tokens_and_add(
    messages: list[dict[str, Any]], model: str, count: int
) -> list[dict[str, Any]]:
    """
    Count tokens in messages and add token count as metadata if supported.

    Args:
        messages (list[dict[str, Any]]): The messages to count tokens for.
        model (str): The model to use for token counting.
        count (int): The token count to add as metadata.

    Returns:
        list[dict[str, Any]]: The messages with token count metadata added.
    """
    window = get_context_window(model=model)
    if window is None:
        window = get_context_window(model=model.rsplit("/", maxsplit=1)[-1])

    if window is None:
        window = 8192  # Default to 8k if unknown

    messages = remove_old_budget_message(messages)
    budget_message = f"""[CONTEXT_BUDGET]
Note that the context budget is not exact since it is based on previous iterations.
model: {model}
max_context_tokens: {window}
prompt_tokens_now: {count}
reserve_for_output: 100
remaining_budget: {window - count - 100}
actions_if_low_budget:
  - avoid reading entire files
  - prefer short answers (<= 150 tokens)
  - summarize or drop thoughts if needed
[/CONTEXT_BUDGET]
"""
    # Ensure content is a string before concatenation
    messages[0]["content"] = budget_message + str(messages[0]["content"])
    return messages


def convert_outermost_triple_quotes(text: str) -> str:
    """Convert only the outermost triple-quoted strings to JSON format."""
    result = []
    i = 0

    while i < len(text):
        if text[i : i + 3] == '"""':
            # Found opening triple quote - find its closing match
            # Look backwards to see if this is a JSON value start
            preceding = text[:i].rstrip()
            if preceding and (preceding[-1] in ":," or preceding.endswith("{")):
                # This is a JSON value, find the matching closing quote
                start = i + 3
                j = start

                # Scan for the closing triple quote
                # Skip to end or find """ followed by JSON delimiter
                while j <= len(text) - 3:
                    if text[j : j + 3] == '"""':
                        # Check what comes after
                        after = text[j + 3 :].lstrip()
                        if not after or after[0] in ",}":
                            # This is the closing quote
                            content = text[start:j]
                            result.append(json.dumps(content))
                            i = j + 3
                            break
                    j += 1
                else:
                    # Didn't find closing, keep original
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1

    return "".join(result)
