from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.evaluate import BenchmarkInterface
from promptstore import PromptStore

from corral.agents.react import ReactAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.agents.utils import LiteLLMMessage, llm_call


class LLMPlanner:
    """Agent that uses the LLM planner to generate plans

    Args:
        model (str): The model to use for planning
        max_iterations (int): The maximum number of iterations to plan
        api_endpoint (str, optional): The API endpoint to use for tool calls
        system_prompt (str, optional): The system prompt to use
        temperature (float): The temperature to use for sampling
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.store = PromptStore("./prompts")
        self.kwargs = kwargs

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        user_prompt_uuid: str | None = "d77a15a2-4ded-4ceb-9b33-e84ab899c899",
        examples: str = "",
        tool_usage: bool = False,
    ) -> str:
        """Run the LLM planner agent

        Args:
            interface (BenchmarkInterface): The benchmark interface to use
            task_id (str): The task ID to solve
            user_prompt_uuid (str): The user prompt UUID to use
            examples (str): The examples to use for planning
            tool_usage (bool): Whether to use tool calling or not
        """
        if user_prompt_uuid is None:
            raise ValueError("User prompt UUID is required")
        else:
            user_prompt = self.store.get(user_prompt_uuid)

        tools = json.loads(interface.get_available_tools_for_task(task_id))["tools"]

        task_guide = interface.get_task_prompt(task_id)
        prompt = user_prompt.fill(
            {
                "examples": examples,
                "tools": json.dump(tools),
                "task_guide": task_guide,
                "iterations": self.max_iterations,
            },
        )

        messages = list[LiteLLMMessage] = []
        if self.system_prompt:
            messages.append(LiteLLMMessage(role="system", content=self.system_prompt))
        messages.append(LiteLLMMessage(role="user", content=prompt))

        plan = llm_call(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.kwargs,
        )
        messages.append(LiteLLMMessage(role="assistant", content=plan))

        if tool_usage:
            agent = ToolCallingAgent(
                model=self.model,
                max_iterations=self.max_iterations,
                api_endpoint=self.api_endpoint,
                temperature=self.temperature,
                **self.kwargs,
            )
        else:
            agent = ReactAgent(
                model=self.model,
                max_iterations=10,
            )

        for _i in range(self.max_iterations):
            final_answer, low_planner_messages = agent.run_agent(
                interface=interface,
                task_id=task_id,
                task_prompt=plan,
            )
            if final_answer.split(" ")[0] != "Error":
                messages.append(
                    LiteLLMMessage(
                        role="tool",
                        content=f"Final Answer: {final_answer}.\nIteration by the agent:\n{low_planner_messages}",
                    )
                )
                return final_answer, messages

            messages.append(
                LiteLLMMessage(
                    role="tool",
                    content=f"Error: {final_answer}.\nIteration by the agent:\n{low_planner_messages}\n\nPlease provide a new plan.",
                )
            )
            plan = llm_call(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **self.kwargs,
            )
            messages.append(LiteLLMMessage(role="assistant", content=plan))

        return "Error: Maximum iterations reached", messages
