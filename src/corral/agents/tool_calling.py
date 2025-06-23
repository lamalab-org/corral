import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.prompt_utils import create_prompt
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
    Agent that uses native function calling from LLM providers to solve tasks.

    This agent leverages the built-in tool/function calling capabilities of modern LLMs
    (like OpenAI's function calling) rather than parsing text-based tool invocations.
    Tools are automatically made available to the LLM and called directly through the API.

    ## Required Prompt Fields

    The user prompt for ToolCallingAgent must contain the following Jinja template fields:
    - **{{task_guide}}**: The main task instructions and description
    - **{{examples}}**: Few-shot examples formatted as a string (optional, can be empty)

    ### Default User Prompt Template:
    The default prompt (ID: "fe04453b-5469-4611-bba6-6d81487df787") expects:

    ### Custom Prompt Requirements:
    If providing a custom user_prompt, it must:
    1. Include {{task_guide}} placeholder for task instructions
    2. Include {{examples}} placeholder for few-shot examples
    3. Instruct the agent to indicate completion with "Final Answer:" format
    4. Support Jinja templating with .fill() method
    5. NOT include tool descriptions (tools are provided via function calling API)

    Args:
        model (str): The model to use for running the agent. Defaults to "openai/gpt-4o".
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider (e.g., OpenAI, VLLM, or self-hosted models) to handle tool/function calling requests. Defaults to None.
        system_prompt (str | Any, optional): The system prompt to use. Can be a string, PromptStore ID, or prompt object
            that implements .fill() method. Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str | Any, optional): The user prompt template. Must contain {{task_guide}} and {{examples}} fields.
            Can be a string, PromptStore ID, or prompt object that implements .fill() method.
            Should NOT include tool descriptions as tools are provided via function calling API.
        extractor_prompt (str | Any, optional): The prompt to use for extracting final answers. Can be a string,
            PromptStore ID, or prompt object that implements .fill() method. Defaults to None.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        system_prompt_id (str, optional): The ID of the system prompt to use. Defaults to "400fcecf-f5f2-464b-aff5-8a4377c9685c".
        user_prompt_id (str, optional): The ID of the user prompt to use. Defaults to "fe04453b-5469-4611-bba6-6d81487df787".
        extractor_prompt_id (str, optional): The ID of the extractor prompt to use. Defaults to "9d37e4a0-26c5-438a-ba1b-a273388fcded".
        **kwargs: Additional keyword arguments to pass to the LiteLLM API
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | Any | None = None,
        user_prompt: str | Any | None = None,
        extractor_prompt: str | Any | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        system_prompt_id: str = "400fcecf-f5f2-464b-aff5-8a4377c9685c",
        user_prompt_id: str | None = "fe04453b-5469-4611-bba6-6d81487df787",
        extractor_prompt_id: str | None = "9d37e4a0-26c5-438a-ba1b-a273388fcded",
        **kwargs,
    ):
        """Initialize the agent"""
        super().__init__(
            model=model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
            temperature=temperature,
            prompt_store=prompt_store,
            system_prompt_id=system_prompt_id,
            user_prompt_id=user_prompt_id,
            extractor_prompt_id=extractor_prompt_id,
            **kwargs,
        )

    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Run the agent to solve the task

        Args:
            interface (BenchmarkInterface): The interface to use
            task_id (str): The task ID to solve
            history (list[LiteLLMMessage]], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. Defaults to None.
            examples (list[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            str: The final answer to the task
        """
        tools = convert_to_openai_tool_format(
            interface.get_available_tools_for_task(task_id)
        )
        if task_prompt is None:
            task_guide = interface.get_task_guide(task_id)
        else:
            task_guide = task_prompt

        self.messages = create_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            task_guide=task_guide,
            history=history,
            examples=examples,
        )

        for _i in range(self.max_iterations):
            try:
                llm_response = self.get_llm_response(tools)

                content = llm_response.content
                if content:
                    final_answer_match = re.search(
                        r"Final Answer:\s*(.*)", content, re.IGNORECASE
                    )
                    if final_answer_match:
                        self.messages.append(
                            LiteLLMMessage(role="assistant", content=content)
                        )
                        return final_answer_match.group(1).strip()

                tool_calls = llm_response.tool_calls
                if tool_calls:
                    self.messages.append(llm_response)

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

                        self.messages.append(
                            LiteLLMMessage(
                                role="tool",
                                tool_call_id=called_tool.id,
                                content=result,
                                name=function_name,
                            )
                        )
                else:
                    self.messages.append(
                        LiteLLMMessage(role="assistant", content=llm_response.content)
                    )
            except Exception as e:
                # Append error message but continue with the next iteration
                logger.error(f"Error during agent iteration: {e}")
                self.messages.append(
                    LiteLLMMessage(
                        role="system",
                        content=f"Error during tool execution: {e!s}",
                    )
                )

        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content="Error: Maximum iterations reached without finding a final answer.",
                name="tool-calling-error",
            )
        )

        return "Error solving the task. Maximum iterations reached."
