from __future__ import annotations

import json

from promptstore import PromptStore

from corral.agents.utils import (
    LiteLLMMessage,
    llm_tool_call,
)


class MainAgent:
    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.prompt_store = PromptStore("./prompts")
        self.kwargs = kwargs

    def run(
        self,
        interface,
        task_id,
        max_iterations=10,
    ):
        task_guide = interface.get_task_prompt(task_id)

        env_tools = json.loads(interface.get_available_tools_for_task(task_id))["tools"]
        tools_names = [tool.name for tool in env_tools]

        prefix_uuid = "70545b35-005c-4aaf-ac3b-4979f8ab10cf"
        prefix_prompt = self.prompt_store.get(prefix_uuid)
        prefix = prefix_prompt.fill({"tools": json.dumps(env_tools)})

        format_uuid = "f4093177-d2bb-4b1b-8b77-d067b62a032e"
        format_prompt = self.prompt_store.get(format_uuid)
        format_filled = format_prompt.fill({"tools_names": tools_names})

        suffix_uuid = "07016776-37e1-4caf-abfe-fe3fc6872ebf"
        suffix_prompt = self.prompt_store.get(suffix_uuid)
        suffix = suffix_prompt.fill({"chat_id": task_guide})

        initial_prompt = f"{prefix}\n\n{format_filled}\n\n{suffix}"

        messages: list[LiteLLMMessage] = []
        messages.append(LiteLLMMessage(role="user", content=initial_prompt))

        for _i in range(max_iterations):
            try:
                response = json.loads(llm_tool_call(messages, self.kwargs))
            except Exception as e:
                response = f"Error: {e}"
                messages.append(LiteLLMMessage(role="assistant", content=response))
                continue

            messages.append(
                LiteLLMMessage(role="assistant", content=json.dumps(response))
            )

            action = response.get("action", "")
            if action:
                if action == "Final Answer":
                    return response.get("action_input"), messages
                else:
                    function_name = action
                    function_args = json.loads(response.get("action_input"))
                    try:
                        function_call = str(
                            interface.execute_tool(
                                task_id, function_name, json.dumps(function_args)
                            )
                        )
                    except Exception as e:
                        function_call = f"Error: {e}"

                    messages.append(
                        LiteLLMMessage(
                            role="tool",
                            content=function_call,
                            name=function_name,
                            tool_call_id=response.get("tool_call_id"),
                        )
                    )

            else:
                messages.append(
                    LiteLLMMessage(
                        role="user",
                        content="Incorrect output format. Please follow the correct format.",
                    )
                )

        return f"No final answer found after {max_iterations}", messages
