import datetime
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Any

from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import LiteLLMMessage, _build_user_content, llm_call
from corral.evaluate import BenchmarkInterface


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


class ReActAgent:
    """
    Agent that uses the ReAct framework to solve tasks
    Based on https://arxiv.org/abs/2210.03629

    Args:
        model (str): The model to use for running the agent
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 10.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider (e.g., OpenAI, VLLM, or self-hosted models) to handle tool/function calling requests. Defaults to None.
        system_prompt (str, optional): The system prompt to use.
            Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str, optional): The user prompt to use. Defaults to a simple prompt with `task_guide`, `history` and `examples` as variables.
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.7.
        prompt_store (PromptStore, optional): The prompt store to use. Defaults to None.
        agent_id (str, optional): The ID of the agent. If not provided, a unique ID will be generated.
        kwargs: Additional keyword arguments to pass to the LiteLLM API
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        prompt_store: PromptStore | None = None,
        agent_id: str | None = None,
        **kwargs,
    ):
        """Initialize the agent"""
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.agent_id = (
            agent_id
            or f"llm_planner_agent-{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

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
            self.store,
            user_prompt,
            "d880c4d3-fe60-4cf4-813b-2008076cd595",
        )

    def get_llm_response(self, messages: list[LiteLLMMessage]) -> str:
        """Get response from the LLM using LiteLLM

        Args:
            messages(list[LiteLLMMessage]): The prompt to send to the LLM

        Returns:
            str: The response from the LLM
        """
        return llm_call(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            **self.kwargs,
        ).content

    def parse_llm_response(
        self, response: str
    ) -> tuple[Thought | None, list[Action] | None]:
        """Parse LLM response into Thought and Actions"""
        thought_match = re.search(
            r"Thought: (.*?)(?=\nAction:|Final Answer:|$)", response, re.DOTALL
        )
        action_matches = re.finditer(
            r"Action: (\w+)\nAction Input: ({.*?}(?=\nAction:|\nThought:|\nFinal Answer:|$))",
            response,
            re.DOTALL,
        )

        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        actions = []
        for action_match in action_matches:
            tool_name = action_match.group(1).strip()
            try:
                action_input = action_match.group(2).strip()
                arguments = json.loads(action_input)
                actions.append(Action(tool_name=tool_name, arguments=arguments))
            except json.JSONDecodeError:
                pass

        return thought, actions if actions else None

    def create_prompt(
        self,
        task_guide: str | list,
        history: list[LiteLLMMessage],
        examples: list[str] | None,
    ) -> list[LiteLLMMessage]:
        """Create prompt for LLM including context and history"""
        messages: list[LiteLLMMessage] = []
        messages.append(
            LiteLLMMessage(
                role="system",
                content=self.system_prompt,
                name=f"system_{self.agent_id}",
            )
        )

        user_content = _build_user_content(
            agent="react",
            user_prompt=self.user_prompt,
            task_guide=task_guide,
            history=history,
            examples=examples,
        )

        messages.append(
            LiteLLMMessage(
                role="user",
                content=user_content,
                name=f"user_{self.agent_id}",
            )
        )

        return messages

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
        """Main ReAct loop implementation

        Args:
            interface (BenchmarkInterface): The interface to use
            task_id (str): The task ID to solve
            history (List[Dict[str, Any]], optional): The history items to include. Defaults to None.
            task_prompt (str, optional): The task prompt to use. `task_prompt` is intended to be a plan or description about the task, that should always be provided when this agent is called as a subagent of a main orchestrator. Defaults to None.
            examples (List[str], optional): List with the few-shot examples to use. Defaults to None.

        Returns:
            tuple[str, list[LiteLLMMessage]]:: The final answer and messages history
        """
        if history is None:
            history = []
        if task_prompt is None:
            task_guide = interface.get_task_guide(task_id)
        else:
            task_guide = task_prompt

        messages = self.create_prompt(task_guide, history, examples)

        for _iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            llm_response = self.get_llm_response(messages)

            # Parse response
            thought, actions = self.parse_llm_response(llm_response)
            thought_prefix = f"Thought: {thought.content}\n" if thought else ""

            # Check for final answer
            final_answer_match = re.search(r"Final Answer: (.*)", llm_response)

            if final_answer_match:
                messages.append(
                    LiteLLMMessage(
                        role="assistant",
                        content=f"{thought_prefix}Final Answer: {final_answer_match.group(1)}",
                        name=f"assistant_{self.agent_id}",
                    )
                )
                return final_answer_match.group(1).strip(), messages

            # Execute tools if actions exist
            if actions:
                for action in actions:
                    action_content = f"{thought_prefix}Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}"
                    messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=action_content,
                            name=f"assistant_{self.agent_id}",
                        )
                    )

                    # Execute tool and get response
                    tool_response = interface.execute_tool(
                        task_id, action.tool_name, action.arguments, messages
                    )

                    observation = (
                        f"Observation: {tool_response.result}"
                        if tool_response.success
                        else f"Error: {tool_response.error}"
                    )

                    messages.append(
                        LiteLLMMessage(
                            role="user",
                            content=observation,
                            name=action.tool_name + f"_{self.agent_id}",
                        )
                    )

            # If no action or thought was parsed, break the loop
            if not thought and actions is None:
                break

        return (
            "Error solving the task: unable to complete it in the iteration limit",
            messages,
        )
