import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import litellm
import openai
from litellm.exceptions import BudgetExceededError
from litellm.types.utils import Message
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)

from corral.types import BudgetExhaustedError

RETRY_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIError,
    openai.APIStatusError,
    openai.InternalServerError,
)

# Exceptions that should stop the benchmark immediately (not retry or continue)
STOP_BENCHMARK_EXCEPTIONS = (
    BudgetExceededError,  # litellm budget exceeded
    openai.AuthenticationError,  # Invalid API key
)


TYPE_MAPPING = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "list[str]": "array",
    "none": "null",
    "dict": "object",
}


def before_sleep_loguru(retry_state):
    logger.info(
        f"Retrying: {retry_state.attempt_number}, wait: {retry_state.next_action.sleep} seconds"
    )


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str | list
    tool_call_id: str | None
    name: str | None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_chain(wait_fixed(30), wait_fixed(60), wait_fixed(90)),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=before_sleep_loguru,
    reraise=True,
)
def llm_call(
    model: str,
    messages: list[LiteLLMMessage],
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    api_endpoint: str | None = None,
    return_usage: bool = False,
    **kwargs,
) -> Message | tuple[Message, dict[str, Any]]:
    """
    Call LiteLLM API with or without tools based on parameters

    Args:
        model (str): The model to use.
        messages (list[LiteLLMMessage]): The messages to send to the model.
        temperature (float): The temperature to use.
        tools (dict[str, Any], optional): The tools to use. If provided, will use tool calling.
        api_endpoint (str, optional): The API endpoint to use. When using VLLM.
        return_usage (bool, optional): If True, returns tuple of (message, usage_info). Defaults to False.
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.

    Returns:
        Message | tuple[Message, dict]: The response from the LiteLLM API, optionally with usage info.
    """
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "api_base": api_endpoint,
            **kwargs,
        }

        if "anthropic" in model:
            params["max_tokens"] = 8192

        if tools is not None:
            params.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                }
            )
            response = litellm.completion(**params)

        else:
            response = litellm.completion(**params)

        message = response.choices[0].message

        if return_usage:
            # Extract usage information from the response
            usage_info = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
                if response.usage
                else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0)
                if response.usage
                else 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0)
                if response.usage
                else 0,
            }
            return message, usage_info

        return message

    except STOP_BENCHMARK_EXCEPTIONS as e:
        # Re-raise as BudgetExhaustedError to stop benchmark immediately
        logger.error(f"Budget/credits exhausted or authentication failed: {e}")
        raise BudgetExhaustedError(
            f"Benchmark stopped: {type(e).__name__} - {e}"
        ) from e

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
        example_prompt = f"To help you in understanding this task, the next {
            len(examples)} examples are provided:\n\n"
        return example_prompt + "\n\n".join(examples)


def convert_dict_arg(arg: dict) -> dict:
    """
    Convert a single argument dictionary (from our ToolArgument format)
    to an OpenAI-compatible JSON schema property.
    """
    if not isinstance(arg, dict):
        raise TypeError(
            f"Expected argument specification to be a dictionary, but got {type(arg)}: {arg}"
        )

    arg_type = arg.get("type")
    if not arg_type:
        raise ValueError(
            f"Argument 'type' is missing for argument: {arg.get('name', 'unknown')}"
        )

    prop = {"description": arg.get("description", "")}

    if arg_type == "list[str]":
        prop["type"] = "array"
        prop["items"] = {"type": "string"}
        prop["description"] += (
            ' - Provide as an array of strings, e.g., ["item1", "item2"]'
        )
    else:
        json_type = TYPE_MAPPING.get(arg_type)
        if not json_type:
            logger.warning(
                f"Unknown argument type '{arg_type}'. Defaulting to 'string'."
            )
            prop["type"] = "string"
        else:
            prop["type"] = json_type

    if arg.get("choices"):
        prop["enum"] = arg["choices"]

    if "default" in arg and arg["default"] is not None:
        prop["default"] = arg["default"]

    return prop


def _parse_argument_string_to_dict(arg_string: str) -> dict | None:
    """
    Parses a human-readable argument string back into a structured dictionary.
    Handles formats like: "name (type, required): description"
    or "name (type, optional, default: value): description"
    """
    # Regex to capture the different parts of the argument string
    pattern = re.compile(
        r"^(?P<name>\w+)\s+\((?P<type>[^,]+),\s*(?P<req_opt>required|optional(?:,\s*default:\s*(?P<default>.*?))?)\):\s*(?P<desc>.*)$",
        re.DOTALL,
    )
    match = pattern.match(arg_string)

    if not match:
        logger.warning(f"Could not parse argument string: {arg_string}")
        return None

    data = match.groupdict()

    arg_dict = {
        "name": data["name"],
        "type": data["type"],
        "required": data["req_opt"] == "required",
        "description": data["desc"],
    }

    if data["default"] is not None:
        # Here we are just storing the default as a string. A more robust
        # implementation might try to cast it to the correct type.
        arg_dict["default"] = data["default"]

    return arg_dict


def convert_to_openai_tool_format(tools_dict: dict) -> list[dict]:
    """
    Convert a dictionary of tools into the OpenAI tool calling format.
    This is now robust and can handle arguments as a list of dicts OR a list of strings.
    """
    if "tools" not in tools_dict or not isinstance(tools_dict["tools"], list):
        logger.warning(
            "No 'tools' list found in the provided dictionary. Returning empty list."
        )
        return []

    openai_tools = []
    for tool in tools_dict["tools"]:
        if not all(k in tool for k in ["name", "description", "arguments"]):
            logger.warning(f"Skipping malformed tool, missing required keys: {tool}")
            continue

        function_spec = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        if not isinstance(tool["arguments"], list):
            logger.warning(
                f"Skipping tool '{tool['name']}' because its arguments are not a list. Got: {type(tool['arguments'])}"
            )
            continue

        for arg_spec in tool["arguments"]:
            arg_dict = None
            # UPDATED LOGIC: Handle both string and dict formats
            if isinstance(arg_spec, str):
                arg_dict = _parse_argument_string_to_dict(arg_spec)
            elif isinstance(arg_spec, dict):
                arg_dict = arg_spec
            else:
                logger.error(
                    f"Argument spec for tool '{tool['name']}' is neither a string nor a dictionary: {arg_spec}"
                )
                continue

            if not arg_dict:
                continue  # Skip if parsing failed or spec was invalid

            try:
                property_entry = convert_dict_arg(arg_dict)
                arg_name = arg_dict["name"]
                function_spec["parameters"]["properties"][arg_name] = property_entry

                if arg_dict.get("required", True):
                    function_spec["parameters"]["required"].append(arg_name)
            except (TypeError, ValueError, KeyError) as e:
                logger.error(
                    f"Skipping invalid argument in tool '{tool['name']}': {arg_dict}. Error: {e}"
                )
                continue

        if function_spec["name"] and function_spec["description"]:
            openai_tools.append({"type": "function", "function": function_spec})

    return openai_tools


def parse_string_argument(arg_string: str) -> dict | None:
    """
    Parse a string argument format like "path (str, required): Path to the directory"
    This is a fallback for malformed API responses.
    """
    # Pattern to match "name (type, required/optional): description"
    pattern = r"^(\w+)\s*\(([^,]+)(?:,\s*(required|optional))?\):\s*(.+)$"
    match = re.match(pattern, arg_string.strip())

    if match:
        name = match.group(1)
        arg_type = match.group(2).strip()
        required_str = match.group(3)
        description = match.group(4).strip()

        return {
            "name": name,
            "type": arg_type,
            "description": description,
            "required": required_str != "optional" if required_str else True,
        }

    return None


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
        window = get_context_window(model=model.split("/")[-1])

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
