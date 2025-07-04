import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import litellm
import openai
from litellm.types.utils import Message
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RETRY_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIError,
    openai.APIStatusError,
    openai.InternalServerError,
)

TYPE_MAPPING = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "list[str]": "array",
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
    wait=wait_exponential(multiplier=2, min=1),
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
        messages (List[LiteLLMMessage]): The messages to send to the model.
        temperature (float): The temperature to use.
        tools (Dict[str, Any], optional): The tools to use. If provided, will use tool calling.
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
        example_prompt = f"To help you in understanding this task, the next {len(examples)} examples are provided:\n\n"
        return example_prompt + "\n\n".join(examples)


def convert_dict_arg(arg: dict) -> dict:
    """Convert a single argument dictionary to OpenAI tool format."""
    if not isinstance(arg, dict):
        raise ValueError(f"Expected arg to be a dictionary but got: {arg}")

    arg_type = arg.get("type")
    if not arg_type:
        raise ValueError(
            f"Argument type is missing for argument: {arg.get('name', 'unknown')}"
        )

    if arg_type == "str":
        prop = {"type": "string", "description": arg.get("description", "")}
    elif arg_type == "bool":
        prop = {"type": "boolean", "description": arg.get("description", "")}
    elif arg_type == "int":
        prop = {
            "type": "integer",  # More specific than "number"
            "description": arg.get("description", ""),
        }
    elif arg_type == "float":
        prop = {"type": "number", "description": arg.get("description", "")}
    elif arg_type == "list[str]":
        # Very explicit array schema to prevent character-by-character parsing
        prop = {
            "type": "array",
            "items": {"type": "string"},
            "description": f"{arg.get('description', '')} - Provide as array of complete strings, e.g., [\"Li2O3\", \"CaCO3\"]",
            "minItems": 1,
        }
    else:
        # Fallback for unknown types
        prop = {
            "type": "string",
            "description": f"{arg.get('description', '')} (type: {arg_type})",
        }

    # Add choices/enum if specified
    if arg.get("choices"):
        prop["enum"] = arg["choices"]

    # Add default if specified
    if arg.get("default") is not None:
        prop["default"] = arg["default"]

    return prop


def convert_to_openai_tool_format(tools_dict: dict) -> list:
    """
    Convert a dictionary of tools into the OpenAI tool calling format.

    Args:
        tools_dict (dict): Dictionary with a 'tools' list containing tool specifications

    Returns:
        list: List of tools in OpenAI tool calling format
    """
    openai_tools = []

    for tool in tools_dict["tools"]:
        function = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        if isinstance(tool["arguments"], str):
            arg_names = [arg_name.strip() for arg_name in tool["arguments"].split(",")]
            for arg_name in arg_names:
                if arg_name:
                    function["parameters"]["properties"][arg_name] = {
                        "type": "string",
                        "description": f"Argument: {arg_name}",
                    }
                    function["parameters"]["required"].append(arg_name)

        else:
            for arg in tool["arguments"]:
                property_entry = convert_dict_arg(arg)
                function["parameters"]["properties"][arg["name"]] = property_entry

                if arg.get("required", True):
                    function["parameters"]["required"].append(arg["name"])

        openai_tools.append({"type": "function", "function": function})

    return openai_tools


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
    output_dir: str = "agent_logs",
) -> str:
    """Save agent conversation to a JSON file for logging and analysis purposes.

    This function handles both regular dictionaries and LiteLLMMessage objects,
    properly serializing them for storage.

    Args:
        messages (list[LiteLLMMessage]): List of message objects (LiteLLMMessages or dictionaries)
        task_id (str): The ID of the task being solved
        agent_name (str): The name of the agent that generated the messages
        output_dir (str, optional): Directory to save the logs (will be created if it doesn't exist). Default is "agent_logs".

    Returns:
        str: Path to the saved file
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    file_path = Path(output_dir) / filename

    # Convert messages to serializable format
    serializable_messages = serialize_messages(messages)

    # Write to file with metadata and pretty formatting
    with Path(file_path).open("w") as f:
        json.dump(
            {
                "task_id": task_id,
                "agent": agent_name,
                "timestamp": timestamp,
                "messages": serializable_messages,
            },
            f,
            indent=2,
        )

    return file_path
