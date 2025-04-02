from __future__ import annotations

import importlib.resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corral.evaluate import BenchmarkInterface

import json
from dataclasses import dataclass

from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import LiteLLMMessage, llm_call


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


class ToolCallingAgent:
    """
    Agent that uses native function calling from the providers to solve the task

    Args:
        model (str): The model to use for running the agent
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider (e.g., OpenAI, VLLM, or self-hosted models) to handle tool/function calling requests. Defaults to None.
        system_prompt (str, optional): The system prompt to use.
            Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str, optional): The user prompt to use. Defaults to a simple prompt with `task_guide` and `examples` as variables.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        kwargs: Additional keyword arguments to pass to the LiteLLM API
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        **kwargs,
    ):
        """Initialize the agent"""
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        if prompt_store:
            self.store = prompt_store
        else:
            with importlib.resources.path("corral.agents", "") as style_path:
                self.store = PromptStore(f"{style_path}/prompts")
        self.kwargs = kwargs

        self.system_prompt = (
            get_prompt(
                self.store, system_prompt, "400fcecf-f5f2-464b-aff5-8a4377c9685c"
            ).fill({})
            if system_prompt is None
            else system_prompt
        )

        self.user_prompt = get_prompt(
            self.store, user_prompt, "fe04453b-5469-4611-bba6-6d81487df787"
        )

    def create_prompt(
        self, task_guide: str | list, history: list[LiteLLMMessage], examples: list[str]
    ) -> list[LiteLLMMessage]:
        """Create the initial prompt messages for the agent

        Args:
            task_guide (str): The task guide to use.
            history (list[LiteLLMMessage]): The history items to include.
            examples (list[str]): The few-shot examples to include.

        Returns:
            List[LiteLLMMessage]: The prompt messages.
        """
        messages: list[LiteLLMMessage] = []
        if self.system_prompt:
            messages.append(LiteLLMMessage(role="system", content=self.system_prompt))
        if isinstance(task_guide, list):
            user_prompt = self.user_prompt.fill(
                {
                    "task_guide": "The task is to correctly answer the question with an image specified below. If you want the tools to use use the images, first save the image, and provide the path to the tools.",
                    "examples": examples,
                }
            )
            user_content = [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ]
            user_content.extend(task_guide)
        elif isinstance(task_guide, str):
            user_content = self.user_prompt.fill(
                {"task_guide": task_guide, "examples": examples}
            )
        else:
            raise ValueError("task_guide must be a string or a list")

        messages.append(LiteLLMMessage(role="user", content=user_content))

        # History is meant to be the conversation history, so we add it to the messages
        if history:
            messages.extend(history)

        return messages

    def convert_to_openai_tool_format(self, tools_dict: dict) -> list:
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

            for arg in tool["arguments"]:
                arg_type = arg["type"]
                if arg_type == "str":
                    arg_type = "string"
                elif arg_type == "bool":
                    arg_type = "boolean"
                elif arg_type == "int" or arg_type == "float":
                    arg_type = "number"
                elif arg_type == "list[str]":
                    arg_type = "array"
                    property_entry = {
                        "type": arg_type,
                        "description": arg["description"],
                        "items": {"type": "string"},
                    }
                    function["parameters"]["properties"][arg["name"]] = property_entry

                    if arg["required"]:
                        function["parameters"]["required"].append(arg["name"])

                    continue
                else:
                    raise ValueError(f"Invalid argument type: {arg_type}")

                property_entry = {"type": arg_type, "description": arg["description"]}

                if arg["choices"]:
                    property_entry["enum"] = arg["choices"]

                function["parameters"]["properties"][arg["name"]] = property_entry

                if arg["required"]:
                    function["parameters"]["required"].append(arg["name"])

            openai_tools.append({"type": "function", "function": function})

        return openai_tools

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
        """Run the agent to solve the task

        Args:
            interface (BenchmarkInterface): The interface to use.
            task_id (str): The task ID to solve.
            history (list[LiteLLMMessage], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. `task_prompt` is intended to be a plan or description about the task, that should always be provided when this agent is called as a subagent of a main orchestrator. Defaults to None.
            examples (list[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            tuple[str, List[LiteLLMMessage]]: The final answer and messages.
        """
        if history is None:
            history = []

        tools = self.convert_to_openai_tool_format(
            interface.get_available_tools_for_task(task_id)
        )
        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt

        messages = self.create_prompt(
            task_guide=task_guide, history=history, examples=examples
        )

        for _i in range(self.max_iterations):
            llm_response = llm_call(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                api_endpoint=self.api_endpoint,
                **self.kwargs,
            )

            content = llm_response.content
            if content:
                messages.append(LiteLLMMessage(role="assistant", content=content))
                if "Final Answer:" in content:
                    return content, messages

            tool_calls = llm_response.tool_calls
            if tool_calls:
                messages.append(llm_response)
                for called_tool in tool_calls:
                    action = Action(
                        tool_name=called_tool.function.name,
                        arguments=json.loads(called_tool.function.arguments),
                    )
                    function_call = interface.execute_tool(
                        task_id, action.tool_name, action.arguments
                    )
                    result = str(function_call.result)
                    if result is None:
                        result = function_call.error

                    function_name = str(called_tool.function.name)

                    messages.append(
                        LiteLLMMessage(
                            tool_call_id=called_tool.id,
                            role="tool",
                            content=result,
                            name=function_name,
                        )
                    )
            else:
                messages.append(
                    LiteLLMMessage(role="assistant", content=llm_response.content)
                )

        return "Error solving the task. Maximum iterations reached.", messages
