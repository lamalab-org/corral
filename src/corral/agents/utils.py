import json
from typing import Any, TypedDict

import litellm
import openai
from litellm.types.utils import Message
from loguru import logger
from promptstore import Prompt
from tenacity import (
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


LIST_PROMPT = (
    "The task is to correctly answer the question with an image specified below."
)


TYPE_MAPPING = {
    "str": "string",
    "bool": "boolean",
    "int": "number",
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
    tools: dict[str, Any] | None = None,
    api_endpoint: str | None = None,
    **kwargs,
) -> Message:
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
        Message: The response from the LiteLLM API.
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

        return response.choices[0].message

    except Exception as e:
        raise ValueError(f"Error in LiteLLM API call: {e}") from e


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


def _build_user_content(
    agent: str,
    user_prompt: Prompt,
    task_guide: str | list,
    tools: str = "",
    examples: list[str] | None = None,
    history: list[LiteLLMMessage] | None = None,
    iterations: int = 0,
) -> list | str:
    """
    Fill the user prompt with the required parameters, managing the different types of agents.
    Additionally, it manages the case when the task_guide is a list of messages.

    Args:
        agent (str): The type of agent being prompted. Important to know the variables to fill.
        user_prompt (Prompt): The user prompt to use.
        task_guide (str | list): Task guide used for describing the environment task.
        tools (str, optional): The tools to use. Defaults to an empty string.
        examples (List[str], optional): The examples to use. Defaults to None.
        history (List[LiteLLMMessage], optional): The history items to include. Defaults to None.
        iterations (int, optional): The number of iterations. Defaults to 0.

    Returns:
        list | str: The filled user prompt.
    """
    if history is None:
        history = []

    base_kwargs = {
        "task_guide": LIST_PROMPT,
        "examples": format_examples(examples),
    }

    if agent == "react":
        base_kwargs["task_guide"] += (
            f" To solve the task you have available the next tools:\n\n{tools}"
        )
        base_kwargs["history"] = json.dumps(history)
    elif agent == "tool_calling":
        pass
    elif agent == "llm_planner":
        base_kwargs["tools"] = json.dumps(tools)
        base_kwargs["iterations"] = str(iterations)
    else:
        raise ValueError(f"Unknown agent type: {agent}")

    if isinstance(task_guide, list):
        user_prompt_text = user_prompt.fill(base_kwargs)
        user_content = [{"type": "text", "text": user_prompt_text}]
        user_content.extend(task_guide)
        return user_content
    elif isinstance(task_guide, str):
        base_kwargs["task_guide"] = task_guide
        return user_prompt.fill(base_kwargs)
    else:
        raise ValueError(f"task_guide should be str or list, got {type(task_guide)}")


def convert_dict_arg(arg: dict) -> dict:
    """Convert a single argument dictionary to OpenAI tool format."""
    if not isinstance(arg, dict):
        raise ValueError(f"Expected arg to be a dictionary but got: {arg}")

    arg_type = arg.get("type")
    if not arg_type:
        raise ValueError(
            f"Argument type is missing for argument: {arg.get('name', 'unknown')}"
        )

    mapped_type = TYPE_MAPPING.get(arg_type)
    if not mapped_type:
        raise ValueError(
            f"Unsupported argument type: {arg_type} for argument: {arg.get('name')}"
        )

    prop = {"type": mapped_type, "description": arg.get("description", "")}

    if mapped_type == "array":
        prop["items"] = {"type": "string"}

    if arg.get("choices"):
        prop["enum"] = arg["choices"]

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

                if arg.get("required", False):
                    function["parameters"]["required"].append(arg["name"])

        openai_tools.append({"type": "function", "function": function})

    return openai_tools
