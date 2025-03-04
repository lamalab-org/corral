from __future__ import annotations

import json
import os
from abc import ABC
from typing import TypedDict

from promptstore import PromptStore

from corral.agents.llamp.tools import (
    MaterialsDielectric,
    MaterialsElasticity,
    MaterialsStructureText,
    MaterialsStructureVis,
    MaterialsSummary,
    MaterialsThermo,
)
from corral.agents.utils import llm_tool_call
from corral.base import Tool


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str
    name: str | None
    tool_call_id: str | None


class BaseAgent(ABC):
    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.mp_api_key = os.environ.get("MP_API_KEY")
        self.prompt_store = PromptStore("./prompts")
        self.tools = []
        self.kwargs = kwargs
        if hasattr(self, "_add_tools"):
            self._add_tools()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str | None:
        return self.__doc__

    def add_tools(self, tools):
        self.tools.extend(tools)

    def as_tool(self, **kwargs):
        return Tool()

    def run(
        self,
        interface,
        task_id,
        input_prompt,
        system_uuid="fb46ddea-eca3-458a-805f-aa6344780929",
        user_uuid="9f8a74c1-cd5e-4fc5-b50e-a2eebaffb409",
        max_iterations=5,
    ):
        tools_names = [tool.name for tool in self.tools]

        if system_uuid:
            system_prompt = self.prompt_store.get(system_uuid)
            system = system_prompt.fill(
                {"tools": self.tools, "tool_names": tools_names}
            )
        else:
            system = ""

        if user_uuid:
            user_prompt = self.prompt_store.get(user_uuid)
            user = user_prompt.fill({"input": input_prompt, "agent_scratchpad": ""})
        else:
            user = input_prompt

        messages = [LiteLLMMessage(role="system", content=system)]
        messages.append(LiteLLMMessage(role="user", content=user))

        for _i in range(max_iterations):
            try:
                response = json.loads(llm_tool_call(messages))
            except Exception as e:
                response = f"Error: {e}"
                messages.append(LiteLLMMessage(role="assistant", content=response))
                continue

            messages.append(
                LiteLLMMessage(role="assistant", content=json.dumps(response))
            )

            action = response.get("action")
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
                        )
                    )

            else:
                messages.append(
                    LiteLLMMessage(
                        role="user",
                        content="Incorrect output format. Please follow the correct format.",
                    )
                )


class MainAgent(BaseAgent):
    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 5,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model, max_iterations, api_endpoint, temperature, **kwargs)
        agent_tools = [
            MPDielectricExpert,
            MPElasticityExpert,
            MPElectronicExpert,
            MPMagnetismExpert,
            MPPiezoelectricExpert,
            MPStructureRetriever,
            MPStructureVisualizer,
            MPSummaryExpert,
            MPSynthesisExpert,
            MPThermoExpert,
        ]
        self.add_tools(agent_tools)

    def add_tools_to_environment(self, environment):
        for tool in self.tools:
            environment.add_tool(tool)

    def run_with_additional_tools(
        self, interface, task_id, input, additional_tools=None, max_iterations=5
    ):
        task_guide = interface.get_task_guide(task_id)
        for tool in self.tools:
            interface.add_tool(tool)

        prefix_uuid = "70545b35-005c-4aaf-ac3b-4979f8ab10cf"
        format_uuid = "f4093177-d2bb-4b1b-8b77-d067b62a032e"
        suffix_uuid = "07016776-37e1-4caf-abfe-fe3fc6872ebf"


class MPDielectricExpert(BaseAgent):
    def _add_tools(self):
        return super().add_tools([MaterialsDielectric()])


class MPSummaryExpert(BaseAgent):
    """Summary expert that has access to Materials Project summary endpoint"""

    def _add_tools(self):
        return [MaterialsSummary()]


class MPStructureRetriever(BaseAgent):
    """Structure expert who will retrieve structures from Materials Project."""

    def _add_tools(self):
        return [MaterialsStructureText()]


class MPStructureVisualizer(BaseAgent):
    """Structure expert who will retrieve structures from Materials Project and save to local storage for frontend visualization"""

    def _add_tools(self):
        return [MaterialsStructureVis()]


class MPThermoExpert(BaseAgent):
    """Theromodynamics expert that has access to Materials Project thermo endpoint"""

    def _add_tools(self):
        return [MaterialsThermo()]


class MPElasticityExpert(BaseAgent):
    """Elasticity expert that has access to Materials Project elasticity endpoint, including bulk, shear, and young's modulus, poisson ratio, and universal anisotropy index"""

    def _add_tools(self):
        return [MaterialsElasticity()]
