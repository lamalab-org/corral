from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

import litellm
from loguru import logger

from corral.evaluate import BenchmarkInterface


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str
    tool_call_id: Optional[str]
    name: Optional[str]


# This should go into the prompt management system
# ToDo: Implement a prompt management system
# ToDo: Move this to the prompt management system
REACT_PROMPT_NO_TOOL = """Task Guide:

{task_guide}

You must think what to do next. You can use some of the tools available in the system.

When you think that the task is completed, you can submit the answer.
For that, answer with: "Final Answer: <your answer>". It is very important to follow this format.
"""


class ToolCallingAgent:
    """Agent that uses tool calling
    This agent uses the LiteLLM API to solve tasks by calling tools.

    Args:
        model (str): The LiteLLM model to use. Defaults to "gpt-4".
        max_iterations (int): The maximum number of iterations to run. Defaults to 10.
        system_prompt (str): The system prompt to use. Defaults to "You are a helpful AI assistant that solves tasks step by step."
        **kwargs: Additional keyword arguments to pass to the LiteLLM API.
    """

    system_prompt = "You are a helpful AI assistant that solves tasks step by step."

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 10,
        api_endpoint: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the agent"""
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        if system_prompt:
            self.system_prompt = system_prompt
        self.kwargs = kwargs

    def format_call(
        self, messages: List[LiteLLMMessage], tools: List[Dict[str, Any]]
    ) -> Any:
        """Call the LiteLLM API with the given messages and tools
        Args:
            messages (List[LiteLLMMessage]): The messages to send to LiteLLM.
            tools (List[Dict[str, Any]]): The tools to send to LiteLLM.
        Returns:
            Any: The response from LiteLLM.
        """
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            api_base=self.api_endpoint,
            tool_choice="auto",
        )
        logger.info(f"Response: {response}")
        return response.choices[0].message

    def create_history_messages(
        self, history: List[Dict[str, Any]]
    ) -> List[LiteLLMMessage]:
        """Create LiteLLMMessage objects from history items ensuring they have required fields
        Args:
            history (List[Dict[str, Any]]): The history items to convert.
        Returns:
            List[LiteLLMMessage]: The converted history items.
        """
        history_messages: List[LiteLLMMessage] = []
        required_fields = {"role", "content"}

        for item in history:
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid history item (not a dict): {item}")
                continue

            if not all(field in item for field in required_fields):
                logger.warning(f"Skipping history item missing required fields: {item}")
                continue

            valid_fields = {"role", "content", "tool_call_id", "name"}
            filtered_item = {k: v for k, v in item.items() if k in valid_fields}

            try:
                message: LiteLLMMessage = {
                    "role": filtered_item["role"],
                    "content": filtered_item["content"],
                }
                if "tool_call_id" in filtered_item:
                    message["tool_call_id"] = filtered_item["tool_call_id"]
                if "name" in filtered_item:
                    message["name"] = filtered_item["name"]

                history_messages.append(message)
            except Exception as e:
                logger.warning(
                    f"Failed to create LiteLLMMessage from item: {item}. Error: {e}"
                )

    def create_prompt(
        self, task_guide: str, history: List[Dict[str, Any]]
    ) -> List[LiteLLMMessage]:
        """Create the initial prompt messages for the agent
        Args:
            task_guide (str): The task guide to use.
            history (List[Dict[str, Any]]): The history items to include.
        Returns:
            List[LiteLLMMessage]: The prompt messages.
        """
        messages: List[LiteLLMMessage] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": REACT_PROMPT_NO_TOOL.format(task_guide=task_guide),
            },
        ]

        if history:
            messages.extend(self.create_history_messages(history))

        return messages

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: List[Dict[str, Any]] = None,
    ) -> str:
        """Run the agent to solve the task
        Args:
            interface (BenchmarkInterface): The interface to use.
            task_id (str): The task ID to solve.
            history (List[Dict[str, Any]]): The history items to include. Defaults to None.

        Returns:
            str: The final answer from the agent.
        """
        if history is None:
            history = []

        tools = interface.get_available_tools(task_id)
        # I think this task prompt is without the tools descriptions
        task_guide = interface.get_task_prompt(task_id)
        messages = self.create_prompt(task_guide=task_guide, history=history)

        for i in range(self.max_iterations):
            llm_response = self.format_call(messages, tools)

            content = llm_response.content
            if content:
                messages.append({"role": "assistant", "content": content})
                if "Final Answer:" in content:
                    return content

            tool_calls = llm_response.tool_calls
            if tool_calls:
                called_tool = tool_calls[0]
                function_name = called_tool.function.name
                function_args = json.loads(called_tool.function.arguments)
                try:
                    function_call = interface.execute_tool(
                        task_id, function_name, json.dumps(function_args)
                    )
                except Exception as e:
                    function_call = f"Error: {e}"

                messages.append(
                    {
                        "tool_call_id": called_tool.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_call,
                    }
                )

        return "Error solving the task. Maximum iterations reached."
