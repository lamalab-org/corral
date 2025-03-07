from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.evaluate import BenchmarkInterface
from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
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
        user_prompt (str, optional): The user prompt to use
        temperature (float): The temperature to use for sampling
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.store = PromptStore("./prompts")
        self.kwargs = kwargs
        self.system_prompt = (
            get_prompt(
                self.store, system_prompt, "400fcecf-f5f2-464b-aff5-8a4377c9685c"
            ).fill({})
            if system_prompt is None
            else system_prompt
        )

        self.user_prompt = get_prompt(
            self.store, user_prompt, "d77a15a2-4ded-4ceb-9b33-e84ab899c899"
        )

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        examples: str = "",
        tool_usage: bool = False,
    ) -> str:
        """Run the LLM planner agent

        Args:
            interface (BenchmarkInterface): The benchmark interface to use
            task_id (str): The task ID to solve
            examples (str): The examples to use for planning
            tool_usage (bool): Whether to use tool calling or not
        """
        tools = json.loads(interface.get_available_tools_for_task(task_id))["tools"]

        task_guide = interface.get_task_prompt(task_id)
        prompt = self.user_prompt.fill(
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
        messages.append(
            LiteLLMMessage(role="assistant", content=plan), name="High-level planner"
        )

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
                **self.kwargs,
            )

        for _i in range(self.max_iterations):
            final_answer, low_level_planner_messages = agent.run_agent(
                interface=interface,
                task_id=task_id,
                task_prompt=plan,
            )
            if final_answer.split(" ")[0] != "Error":
                messages.append(
                    LiteLLMMessage(
                        role="assistant",
                        content=f"Final Answer: {final_answer}.\nIteration by the agent:\n{low_level_planner_messages}",
                        name="Low-level planner",
                    )
                )
                return final_answer, messages

            messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=f"Error: {final_answer}.\nIteration by the agent:\n{low_level_planner_messages}\n\nPlease provide a new plan.",
                    name="Low-level planner",
                )
            )
            plan = llm_call(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **self.kwargs,
            )
            messages.append(
                LiteLLMMessage(role="assistant", content=plan),
                name="High-level planner",
            )

        return "Error: Maximum iterations reached", messages
