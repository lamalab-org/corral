import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.utils import (
    LiteLLMMessage,
    convert_to_openai_tool_format,
)
from corral.evaluate import BenchmarkInterface


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


class ToolCallingAgent(BaseAgent):
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
        model: str = "openai/gpt-4",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        **kwargs,
    ):
        """Initialize the agent"""
        user_prompt_id = "fe04453b-5469-4611-bba6-6d81487df787"
        super().__init__(
            model=model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            prompt_store=prompt_store,
            user_prompt_id=user_prompt_id,
            **kwargs,
        )

    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
        """Run the agent to solve the task

        Args:
            interface (BenchmarkInterface): The interface to use
            task_id (str): The task ID to solve
            history (list[LiteLLMMessage]], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. Defaults to None.
            examples (list[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            Tuple[str, list[LiteLLMMessage]]: The final answer and messages
        """
        if history is None:
            history = []

        tools = convert_to_openai_tool_format(
            interface.get_available_tools_for_task(task_id)
        )
        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt

        messages = self.create_prompt(
            task_guide=task_guide,
            history=None,
            examples=examples,
            agent_type="tool_calling",
        )
        messages.extend(history)

        for _i in range(self.max_iterations):
            try:
                llm_response = self.get_llm_response(messages, tools)

                content = llm_response.content
                if content:
                    final_answer_match = re.search(
                        r"Final Answer:\s*(.*)", content, re.IGNORECASE
                    )
                    if final_answer_match:
                        messages.append(
                            LiteLLMMessage(role="assistant", content=content)
                        )
                        return final_answer_match.group(1).strip(), messages

                tool_calls = llm_response.tool_calls
                if tool_calls:
                    messages.append(llm_response)

                    for called_tool in tool_calls:
                        action = Action(
                            tool_name=called_tool.function.name,
                            arguments=json.loads(called_tool.function.arguments),
                        )
                        try:
                            function_call = interface.execute_tool(
                                task_id, action.tool_name, action.arguments
                            )
                            result = str(function_call.result)
                            if result is None:
                                result = str(function_call.error)
                        except Exception as e:
                            result = str(e)

                        function_name = str(called_tool.function.name)

                        messages.append(
                            LiteLLMMessage(
                                role="tool",
                                tool_call_id=called_tool.id,
                                content=result,
                                name=function_name,
                            )
                        )
                else:
                    messages.append(
                        LiteLLMMessage(role="assistant", content=llm_response.content)
                    )
            except Exception as e:
                # Append error message but continue with the next iteration
                logger.error(f"Error during agent iteration: {e}")
                messages.append(
                    LiteLLMMessage(
                        role="system",
                        content=f"Error during tool execution: {e!s}",
                    )
                )

        return "Error solving the task. Maximum iterations reached.", messages
