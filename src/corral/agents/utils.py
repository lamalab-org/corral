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

from corral.graph import GraphTrackerFactory

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


def tracked_llm_call(
    model: str,
    messages: list[LiteLLMMessage],
    temperature: float,
    tools: dict[str, Any] | None = None,
    api_endpoint: str | None = None,
    graph_factory: GraphTrackerFactory | None = None,
    task_id: str | None = None,
    trial_id: str | None = None,
    **kwargs,
) -> Any:
    """
    Call LLM with integrated graph tracking

    Args:
        model: The model to use
        messages: The messages to send
        temperature: The temperature to use
        tools: Optional tools to use
        api_endpoint: Optional API endpoint
        graph_factory: Optional graph tracker factory
        task_id: Optional task ID for tracking
        trial_id: Optional trial ID for tracking
        **kwargs: Additional arguments to pass to llm_call

    Returns:
        The LLM response
    """
    # Track the LLM call if we have tracking info
    tracker = None
    prompt_node_id = None

    if graph_factory and task_id:
        # Get or create tracker
        tracker = graph_factory.get_tracker(task_id, trial_id)

        # Track the prompt
        if tracker:
            prompt_node_id = tracker.track_llm_prompt(messages)

    # Call the LLM
    response = llm_call(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        api_endpoint=api_endpoint,
        **kwargs,
    )

    # Track the response if we're tracking
    if tracker and prompt_node_id:
        tracker.track_llm_response(
            content=response.content if response.content else str(response.tool_calls),
            source_node_id=prompt_node_id,
        )

    return response


class TrackedAgentMixin:
    """Mixin to add graph tracking to agents"""

    def __init__(self, graph_factory: GraphTrackerFactory | None = None, **kwargs):
        """Initialize the tracked agent mixin

        Args:
            graph_factory: Optional graph tracker factory
            **kwargs: Additional arguments to pass to parent constructor
        """
        # Initialize parent if possible
        super().__init__(**kwargs) if hasattr(super(), "__init__") else None

        # Set up graph tracking
        self.graph_factory = graph_factory or GraphTrackerFactory()
        self._active_tracker = None
        self._task_id = None
        self._trial_id = None

    def start_tracking(
        self,
        task_id: str,
        trial_id: str | None = None,
        agent_type: str | None = None,
    ) -> None:
        """Start tracking an agent run

        Args:
            task_id: ID of the task
            trial_id: Optional trial ID
            agent_type: Optional agent type, defaults to class name
        """
        if not agent_type:
            agent_type = self.__class__.__name__

        self._task_id = task_id
        self._trial_id = trial_id

        # Get or create tracker
        self._active_tracker = self.graph_factory.get_tracker(task_id, trial_id)
        if not self._active_tracker:
            self._active_tracker = self.graph_factory.create_tracker(
                task_id=task_id, agent_type=agent_type, trial_id=trial_id
            )

    def stop_tracking(self, visualize: bool = False) -> None:
        """Stop tracking and save results

        Args:
            visualize: Whether to generate visualization
        """
        if self._active_tracker and self._task_id:
            self.graph_factory.save_tracker(
                task_id=self._task_id, trial_id=self._trial_id, visualize=visualize
            )

        self._active_tracker = None
        self._task_id = None
        self._trial_id = None

    def get_tracker(self) -> Any | None:
        """Get the active tracker

        Returns:
            Active tracker or None
        """
        return self._active_tracker

    def tracked_llm_call(self, *args, **kwargs) -> Any:
        """Call LLM with tracking

        Args:
            *args: Arguments to pass to tracked_llm_call
            **kwargs: Keyword arguments to pass to tracked_llm_call

        Returns:
            LLM response
        """
        # Add tracking info if available
        if self._active_tracker and self._task_id:
            kwargs.update(
                {
                    "graph_factory": self.graph_factory,
                    "task_id": self._task_id,
                    "trial_id": self._trial_id,
                }
            )

        return tracked_llm_call(*args, **kwargs)


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
        base_kwargs["tools"] = json.dumps(tools)
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
