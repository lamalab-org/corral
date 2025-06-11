import importlib.resources
from abc import ABC, abstractmethod
from typing import Any

import openai
from litellm.types.utils import Message
from loguru import logger
from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import (
    LiteLLMMessage,
    llm_call,
    save_agent_messages,
)
from corral.evaluate import BenchmarkInterface


class BaseAgent(ABC):
    """
    Base Agent class that captures common functionality for different agent types.

    This abstract base class provides common initialization parameters and methods
    for agent implementations.

    ## Prompt System Overview

    The BaseAgent uses a flexible prompt system that supports both string-based prompts
    and Jinja-templated prompts from a PromptStore. **All prompts must be Jinja-compatible
    objects that support a `.fill()` method for parameter substitution.**

    ### Prompt Types:
    - **system_prompt**: Sets the agent's behavior and role (e.g., "You are a helpful AI assistant")
    - **user_prompt**: Contains the main task instructions with placeholders for dynamic content
    - **extractor_prompt**: Used to extract and clean final answers from agent responses.
      This prompt takes the raw agent output and extracts just the answer portion, removing
      explanatory text, formatting artifacts, or multiple choice options.

    ### How to Provide Custom Prompts:

    Users can provide custom prompts in three ways:

    1. **String Prompts**: Pass a string directly, which will be wrapped in a StringPrompt object
       that supports Jinja templating with `{{variable}}` syntax.
       ```python
       user_prompt = (
           "Task: {{task_guide}}\n\nExamples: {{examples}}\n\nSolve this step by step."
       )
       ```

    2. **PromptStore IDs**: Pass a UUID string to reference pre-stored prompts in the PromptStore.
       If no ID is provided, default prompts are used for each agent type.
       ```python
       user_prompt_id = "d880c4d3-fe60-4cf4-813b-2008076cd595"  # ReAct prompt
       ```

    3. **Prompt Objects**: Pass any object that implements a `.fill(dict)` method for Jinja-style
       variable substitution. This includes custom Prompt classes or objects from external libraries.
       ```python
       from promptstore import Prompt

       user_prompt = Prompt("Custom template: {{task_guide}}")
       ```

    ### Prompt Variable Filling:

    Prompts are filled with variables using Jinja syntax (`{{variable_name}}`). The specific
    variables available depend on the agent type and are automatically provided by the
    `_build_user_content()` method:

    **Common variables for all agent types:**
    - **task_guide**: The main task description or instructions
    - **examples**: Formatted few-shot examples (when provided, otherwise empty string)

    ## Extractor
    The extractor is a specialized prompt that cleans the final answer from the agent's response.
    It is used to ensure the final output is concise and formatted correctly, removing any
    unnecessary text or artifacts from the agent's response.

    The extractor temperature is set to 0.0 to ensure deterministic output.

    **Extractor prompt variables:**
    - **message**: The full agent response message
    - **answer**: The initially parsed answer from the agent response

    The `_build_user_content()` method handles populating the different variables automatically based on
    the agent type specified in the `agent_type` parameter during prompt creation.

    Args:
        model (str): The model to use for running the agent. Defaults to "openai/gpt-4o".
        max_iterations (int, optional): The maximum number of iterations (number of LLM calls) to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider
            (e.g., OpenAI, VLLM, or self-hosted models) to handle requests. Defaults to None.
        system_prompt (str | Any, optional): The system prompt to use. Can be a string, PromptStore ID, or prompt object
            that implements .fill() method. If None, uses default system prompt. Must support Jinja templating.
        user_prompt (str | Any, optional): The user prompt to use. Can be a string, PromptStore ID, or prompt object
            that implements .fill() method. Different templates are used for each agent type. Must support Jinja templating.
        extractor_prompt (str | Any, optional): The extractor prompt for cleaning final answers. Can be a string,
            PromptStore ID, or prompt object that implements .fill() method. Must support Jinja templating.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use for managing templated prompts.
            If None, uses default store from package resources.
        system_prompt_id (str, optional): The ID of the system prompt to use from the prompt store.
            Defaults to "400fcecf-f5f2-464b-aff5-8a4377c9685c".
        user_prompt_id (str, optional): The ID of the user prompt to use from the prompt store.
            Defaults to None (each agent type has its own default).
        extractor_prompt_id (str, optional): The ID of the extractor prompt to use from the prompt store.
            Defaults to "9d37e4a0-26c5-438a-ba1b-a273388fcded".
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
        user_prompt_id: str | None = None,
        extractor_prompt_id: str | None = "9d37e4a0-26c5-438a-ba1b-a273388fcded",
        **kwargs,
    ):
        """Initialize the base agent with common parameters"""
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.messages = []

        if prompt_store:
            self.store = prompt_store
        else:
            with importlib.resources.path("corral.agents", "") as style_path:
                self.store = PromptStore(f"{style_path}/prompts")

        self.kwargs = kwargs

        self.system_prompt = (
            get_prompt(self.store, system_prompt, system_prompt_id).fill({})
            if system_prompt is None
            else system_prompt
        )

        if user_prompt_id and user_prompt is None:
            self.user_prompt = get_prompt(self.store, user_prompt, user_prompt_id)
        else:
            self.user_prompt = user_prompt

        if extractor_prompt_id and extractor_prompt is None:
            self.extractor_prompt = get_prompt(
                self.store, extractor_prompt, extractor_prompt_id
            )
        else:
            self.extractor_prompt = extractor_prompt

    def get_llm_response(self, tools: dict[str, Any] | None = None) -> Any:
        """Get response from the LLM using LiteLLM

        Args:
            messages (list[LiteLLMMessage]): The messages to send to the LLM
            tools (dict[str, Any], optional): Optional tools/functions for function calling

        Returns:
            Any: The response from the LLM
        """
        try:
            return llm_call(
                model=self.model,
                messages=self.messages,
                tools=tools,
                temperature=self.temperature,
                api_endpoint=self.api_endpoint,
                **self.kwargs,
            )

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")

            for message in reversed(self.messages):
                if message["role"] != "assistant":
                    if "content" in message:
                        content = str(message["content"])
                        if len(content) > 100:
                            message["content"] = content[:100] + "..."
                else:
                    break

            error_message = f"RateLimitError: {e!s}"

            return Message(role="user", content=error_message, tool_calls=[])

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise e

    @abstractmethod
    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """
        Run the agent to solve a task

        This method must be implemented by all subclasses

        Args:
            interface (BenchmarkInterface): The benchmark interface to use
            task_id (str): The task ID to solve
            history (list[LiteLLMMessage], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. Defaults to None.
            examples (list[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            str: The final answer from the agent
        """
        raise NotImplementedError("Subclasses must implement run_agent()")

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """Run the agent to solve a task

        This method is a wrapper around run to provide a consistent interface

        Args:
            interface (BenchmarkInterface): The benchmark interface to use
            task_id (str): The task ID to solve
            history (list[LiteLLMMessage], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. Defaults to None.
            examples (list[str], optional): List with the few-shot examples to use. Defaults to None.
            verbose (bool, optional): Whether to save agent messages. Defaults to False.

        Returns:
            str: The final answer from the agent
        """
        if history is None:
            history = []

        try:
            final_answer = self.run(interface, task_id, history, task_prompt, examples)

            if "Error" in final_answer:
                logger.error(f"Error in agent response: {final_answer}")
                return final_answer

        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e

        prompt = self.extractor_prompt.fill(
            {
                "answer": final_answer,
                "message": self.messages[-1]["content"],
            }
        )

        try:
            answer = llm_call(
                model=self.model,
                messages=[LiteLLMMessage(role="user", content=prompt)],
                temperature=0.0,
                api_endpoint=self.api_endpoint,
                **self.kwargs,
            )
            if verbose:
                save_agent_messages(self.messages, task_id, self.__class__.__name__)

            return answer.content

        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return final_answer
