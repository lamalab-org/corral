import importlib.resources
import json
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import (
    LiteLLMMessage,
    format_examples,
    llm_call,
    save_agent_messages,
)
from corral.evaluate import BenchmarkInterface


class BaseAgent(ABC):
    """
    Base Agent class that captures common functionality for different agent types.

    This abstract base class provides common initialization parameters and methods
    for agent implementations.

    Args:
        model (str): The model to use for running the agent
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider
            (e.g., OpenAI, VLLM, or self-hosted models) to handle requests. Defaults to None.
        system_prompt (str, optional): The system prompt to use.
            Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str, optional): The user prompt to use. Different for each agent type.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        kwargs: Additional keyword arguments to pass to the LiteLLM API
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
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

    def get_llm_response(
        self, messages: list[LiteLLMMessage], tools: dict[str, Any] | None = None
    ) -> Any:
        """Get response from the LLM using LiteLLM

        Args:
            messages (list[LiteLLMMessage]): The messages to send to the LLM
            tools (dict[str, Any], optional): Optional tools/functions for function calling

        Returns:
            Any: The response from the LLM
        """
        return llm_call(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            **self.kwargs,
        )

    def create_prompt(self, task_guide: str | list, **kwargs) -> list[LiteLLMMessage]:
        """Create prompt for LLM including context and history

        Args:
            task_guide (Union[str, list]): The task guide or prompt to use
            **kwargs: Additional keyword arguments that can include:
                - history (list[LiteLLMMessage]): The history of messages
                - examples (list[str]): Few-shot examples to include
                - agent_type (str): Type of agent for building user content. Defaults to "base"
                - tools (str): Available tools for the task

        Returns:
            List[LiteLLMMessage]: The prepared messages for the LLM
        """
        messages: list[LiteLLMMessage] = []
        if self.system_prompt:
            messages.append(LiteLLMMessage(role="system", content=self.system_prompt))

        user_content = self._build_user_content(task_guide=task_guide, **kwargs)

        messages.append(LiteLLMMessage(role="user", content=user_content))

        return messages

    def _build_user_content(self, task_guide: str | list, **kwargs) -> list | str:
        """
        Fill the user prompt with the required parameters, managing the different types of agents.
        Additionally, it manages the case when the task_guide is a list of messages.

        Args:
            task_guide (Union[str, List]): Task guide used for describing the environment task
            user_prompt (Prompt): The user prompt to use
            agent (str): The type of agent being prompted
            **kwargs: Additional keyword arguments that can include:
                - tools (str): The tools to use
                - examples (list[str]): The examples to use
                - history (list[LiteLLMMessage]): The history items to include
                - iterations (int): The number of iterations

        Returns:
            Union[List, str]: The filled user prompt
        """
        agent_type = kwargs.get("agent_type", None)
        if agent_type is None:
            raise ValueError("Agent type must be specified in kwargs")

        LIST_PROMPT = "The task is to correctly answer the question with an image specified below."

        tools = kwargs.get("tools", None)
        examples = kwargs.get("examples", None)

        base_kwargs = {
            "task_guide": LIST_PROMPT,
            "examples": format_examples(examples),
        }

        if agent_type == "react":
            history = kwargs.get("history", [])
            base_kwargs["task_guide"] += (
                f" To solve the task you have available the next tools:\n\n{tools}"
            )
            base_kwargs["history"] = json.dumps(history)
        elif agent_type == "tool_calling":
            pass
        elif agent_type == "llm_planner":
            base_kwargs["tools"] = json.dumps(tools)
            base_kwargs["iterations"] = str(self.max_iterations)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        if isinstance(task_guide, list):
            user_prompt_text = self.user_prompt.fill(base_kwargs)
            user_content = [{"type": "text", "text": user_prompt_text}]
            user_content.extend(task_guide)
            return user_content
        elif isinstance(task_guide, str):
            base_kwargs["task_guide"] = task_guide
            return self.user_prompt.fill(base_kwargs)
        else:
            raise ValueError(
                f"task_guide should be str or list, got {type(task_guide)}"
            )

    @abstractmethod
    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
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
            Tuple[str, list[LiteLLMMessage]]: The final answer and messages
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
        try:
            final_answer, messages = self.run(
                interface, task_id, history, task_prompt, examples
            )

            if "Error" in final_answer:
                logger.error(f"Error in agent response: {final_answer}")
                return final_answer, messages

        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e

        prompt = self.extractor_prompt.fill(
            {
                "answer": final_answer,
                "message": messages[-1]["content"],
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
                save_agent_messages(messages, task_id, self.__class__.__name__)

            return answer.content

        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return final_answer
