from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.evaluate import BenchmarkInterface

import json
import re
from dataclasses import dataclass
from typing import Any

from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import llm_call


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
        model (str): The model to use for planning
        max_iterations (int): The maximum number of iterations to plan
        api_endpoint (str, optional): The API endpoint to use for tool calls
        system_prompt (str, optional): The system prompt to use.
            Defaults to "You are a helpful AI assistant that solves tasks step by step."
        user_prompt (str, optional): The user prompt to use
        temperature (float): The temperature to use for sampling
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
        **kwargs,
    ):
        """Initialize the agent"""
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.store = PromptStore("./prompts")
        self.temperature = temperature
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

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM using LiteLLM

        Args:
            prompt (str): The prompt to send to LLM

        Returns:
            str: The response from LLM
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        return llm_call(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            **self.kwargs,
        ).content

    def parse_llm_response(self, response: str) -> tuple[Thought | None, Action | None]:
        """Parse LLM response into Thought and Action"""
        thought_match = re.search(
            r"Thought: (.*?)(?=\nAction:|Final Answer:|$)", response, re.DOTALL
        )
        action_match = re.search(
            r"Action: (\w+)\nAction Input: ({.*})", response, re.DOTALL
        )

        thought = Thought(thought_match.group(1).strip()) if thought_match else None

        action = None
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                arguments = json.loads(action_match.group(2).strip())
                action = Action(tool_name=tool_name, arguments=arguments)
            except json.JSONDecodeError:
                pass

        return thought, action

    def create_prompt(self, task_guide: str, history: list[str]) -> str:
        """Create prompt for LLM including context and history"""
        return self.user_prompt.fill(
            {"task_guide": task_guide, "history": chr(10).join(history)}
        )

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        task_prompt: str | None = None,
    ) -> tuple[str, list[str]]:
        """Main ReAct loop implementation

        Args:
            interface (BenchmarkInterface): The interface to use
            task_id (str): The task ID to solve
            task_prompt (str): The task prompt to use. Defaults to None.

        Returns:
            Tuple[str, List[str]]: The final answer and history
        """
        if task_prompt is None:
            task_guide = interface.get_task_guide(task_id)
        else:
            task_guide = task_prompt
        history: list[str] = []

        for _iteration in range(self.max_iterations):
            # Create prompt and get LLM response
            prompt = self.create_prompt(task_guide, history)
            llm_response = self.get_llm_response(prompt)

            # Parse response
            thought, action = self.parse_llm_response(llm_response)

            # Record thought
            if thought:
                history.append(f"Thought: {thought.content}")

            # Check for final answer
            final_answer_match = re.search(r"Final Answer: (.*)", llm_response)
            if final_answer_match:
                return final_answer_match.group(1).strip(), history

            # Execute tool if action exists
            if action:
                history.append(
                    f"Action: {action.tool_name}\nAction Input: {json.dumps(action.arguments)}"
                )

                # Execute tool and get response
                tool_response = interface.execute_tool(
                    task_id, action.tool_name, action.arguments
                )

                # Record observation
                observation = (
                    f"Observation: {tool_response.result}"
                    if tool_response.success
                    else f"Error: {tool_response.error}"
                )
                history.append(observation)

            # If no action or thought was parsed, break the loop
            if not thought and not action:
                break

        return (
            "Error solving the task: unable to complete it in the iteration limit",
            history,
        )
