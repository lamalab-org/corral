from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.evaluate import BenchmarkInterface

import json
from typing import Any

from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import LiteLLMMessage, llm_call


class ToolCallingAgent:
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
            self.store, user_prompt, "fe04453b-5469-4611-bba6-6d81487df787"
        )

    def create_prompt(
        self, prompt: str, task_guide: str, history: list[dict[str, Any]]
    ) -> list[LiteLLMMessage]:
        """Create the initial prompt messages for the agent

        Args:
            task_guide (str): The task guide to use.
            history (List[Dict[str, Any]]): The history items to include.

        Returns:
            List[LiteLLMMessage]: The prompt messages.
        """
        messages = list[LiteLLMMessage] = []
        if self.system_prompt:
            messages.append(LiteLLMMessage(role="system", content=self.system_prompt))
        messages.append(
            LiteLLMMessage(role="user", content=prompt.fill({"task_guide": task_guide}))
        )

        if history:
            messages.extend(self.create_history_messages(history))

        return messages

    def run_agent(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[dict[str, Any]] | None = None,
        task_prompt: str | None = None,
    ) -> tuple[str, list[LiteLLMMessage]]:
        """Run the agent to solve the task

        Args:
            interface (BenchmarkInterface): The interface to use.
            task_id (str): The task ID to solve.
            history (List[Dict[str, Any]]): The history items to include. Defaults to None.

        Returns:
            str: The final answer from the agent.
        """
        if history is None:
            history = []

        tools = json.loads(interface.get_available_tools_for_task(task_id))["tools"]
        # I think this task prompt is without the tools descriptions
        # We want this here since for this agent the tools go into the functions or tools
        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt
        user_prompt = self.user_prompt.fill({task_guide: task_guide})

        messages = self.create_prompt(
            user_prompt, task_guide=task_guide, history=history
        )

        for _i in range(self.max_iterations):
            llm_response = llm_call(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                api_endpoint=self.api_endpoint,
                **self.kwargs,
            )

            content = llm_response.content
            if content:
                messages.append(LiteLLMMessage(role="assistant", content=content))
                if "Final Answer:" in content:
                    return content, messages

            tool_calls = llm_response.tool_calls
            if tool_calls:
                for called_tool in tool_calls:
                    function_name = called_tool.function.name
                    function_args = json.loads(called_tool.function.arguments)
                    try:
                        function_call = interface.execute_tool(
                            task_id, function_name, json.dumps(function_args)
                        )
                    except Exception as e:
                        function_call = f"Error: {e}"

                    messages.append(
                        LiteLLMMessage(
                            role="tool",
                            content=function_call,
                            tool_call_id=called_tool.id,
                            name=str(function_name),
                        )
                    )

        return "Error solving the task. Maximum iterations reached.", messages
