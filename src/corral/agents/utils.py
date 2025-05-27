import json
from typing import Any, TypedDict

import litellm
import openai
import requests
from litellm.types.utils import Message
from loguru import logger
from promptstore import Prompt
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from corral.graph import GraphTrackerFactory, NodeType

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


class AgentTrackingRequest(BaseModel):
    """Request model for agent tracking API endpoint"""

    node_type: str
    content: Any
    metadata: dict[str, Any] = {}


class TrackedAgentMixin:
    """Mixin for agents that want to track their interactions in a graph"""

    def __init__(
        self,
        graph_factory: GraphTrackerFactory | None = None,
        server_url: str = "http://localhost:8000",
    ):
        """Initialize the mixin

        Args:
            graph_factory: Optional graph factory to use (for backward compatibility)
            server_url: URL of the benchmark server for server-side tracking
        """
        self.graph_factory = graph_factory
        self.graph_tracker = None

        # Server tracking attributes
        self.server_url = server_url
        self.tracking_task_id = None
        self.tracking_agent_type = None
        self.tracking_trial_id = None
        self.use_server_tracking = True  # Default to server-side tracking

    def start_tracking(
        self, task_id: str, agent_type: str = "Agent", trial_id: str | None = None
    ) -> None:
        """Start tracking agent interactions

        Args:
            task_id: ID of the task
            agent_type: Type of agent (for metadata)
            trial_id: Optional trial ID
        """
        self.tracking_task_id = task_id
        self.tracking_agent_type = agent_type
        self.tracking_trial_id = trial_id

        logger.info(
            f"Agent {agent_type} started tracking for task {task_id}, trial {trial_id}"
        )

    def add_node(
        self, node_type: NodeType, content: Any, metadata: dict | None = None
    ) -> str | None:
        """Add a node to the graph (either server-side or local)

        Args:
            node_type: Type of node
            content: Content of the node
            metadata: Additional metadata

        Returns:
            Node ID if successful, None otherwise
        """
        if not self.tracking_task_id:
            return None

        metadata = metadata or {}
        metadata["component"] = "agent"
        metadata["agent_type"] = self.tracking_agent_type

        if self.use_server_tracking:
            # Send tracking data to server
            try:
                response = requests.post(
                    f"{self.server_url}/tasks/{self.tracking_task_id}/track",
                    json={
                        "node_type": node_type.value,
                        "content": content,
                        "metadata": metadata,
                    },
                )
                if response.status_code == 200:
                    return response.json().get("node_id")
                else:
                    logger.warning(
                        f"Failed to send tracking data: {response.status_code} {response.text}"
                    )
                    # Fall back to local tracking if available
                    if self.graph_tracker:
                        return self.graph_tracker.add_node(
                            node_type=node_type, content=content, metadata=metadata
                        )
            except Exception as e:
                logger.warning(f"Failed to send tracking data: {e}")
                # Fall back to local tracking if available
                if self.graph_tracker:
                    return self.graph_tracker.add_node(
                        node_type=node_type, content=content, metadata=metadata
                    )
        elif self.graph_tracker:
            # Use local graph tracker
            return self.graph_tracker.add_node(
                node_type=node_type, content=content, metadata=metadata
            )

        return None

    def track_llm_prompt(self, messages: list[dict]) -> str | None:
        """Track an LLM prompt

        Args:
            messages: List of messages to send to the LLM

        Returns:
            Node ID if successful, None otherwise
        """
        return self.add_node(
            node_type=NodeType.LLM_PROMPT,
            content=messages,
            metadata={"message_count": len(messages)},
        )

    def track_llm_response(
        self, content: str, prompt_node_id: str | None = None
    ) -> str | None:
        """Track an LLM response

        Args:
            content: Content of the LLM response
            prompt_node_id: Optional ID of the prompt node

        Returns:
            Node ID if successful, None otherwise
        """
        metadata = {"length": len(content)}
        if prompt_node_id:
            metadata["prompt_node_id"] = prompt_node_id

        return self.add_node(
            node_type=NodeType.LLM_RESPONSE, content=content, metadata=metadata
        )

    def track_thought(self, content: str) -> str | None:
        """Track an agent thought

        Args:
            content: Content of the thought

        Returns:
            Node ID if successful, None otherwise
        """
        return self.add_node(
            node_type=NodeType.LLM_RESPONSE, content=content, metadata={"thought": True}
        )

    def track_tool_call(self, tool_name: str, arguments: dict) -> str | None:
        """Track a tool call

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool

        Returns:
            Node ID if successful, None otherwise
        """
        return self.add_node(
            node_type=NodeType.TOOL_CALL,
            content={"tool_name": tool_name, "arguments": arguments},
            metadata={},
        )

    def track_tool_response(
        self, result: str, success: bool = True, tool_name: str | None = None
    ) -> str | None:
        """Track a tool response

        Args:
            result: Result of the tool call
            success: Whether the tool call was successful
            tool_name: Optional name of the tool

        Returns:
            Node ID if successful, None otherwise
        """
        metadata = {"success": success}
        if tool_name:
            metadata["tool_name"] = tool_name

        return self.add_node(
            node_type=NodeType.TOOL_RESPONSE, content=result, metadata=metadata
        )

    def track_final_submission(
        self, answer: str, score: float | None = None
    ) -> str | None:
        """Track the final submission

        Args:
            answer: The final answer
            score: Optional score

        Returns:
            Node ID if successful, None otherwise
        """
        metadata = {}
        if score is not None:
            metadata["score"] = score

        return self.add_node(
            node_type=NodeType.FINAL_SUBMISSION, content=answer, metadata=metadata
        )

    def get_tracker(self) -> Any:
        """Get the current tracker (for backward compatibility)"""
        return self.graph_tracker

    def stop_tracking(self, visualize: bool = False) -> None:
        """Stop tracking agent interactions

        Args:
            visualize: Whether to generate visualization (only for local tracking)
        """
        task_id = self.tracking_task_id
        trial_id = self.tracking_trial_id

        self.tracking_task_id = None
        self.tracking_trial_id = None

        # For backward compatibility with local tracking
        if self.graph_tracker and self.graph_factory:
            # Save the tracker
            self.graph_factory.save_tracker(
                task_id=task_id,
                trial_id=trial_id,
                visualize=visualize,
            )
            logger.info(
                f"Saved local tracking data for task {task_id}, trial {trial_id}"
            )


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
