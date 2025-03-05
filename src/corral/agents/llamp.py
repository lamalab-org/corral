from __future__ import annotations

import json
from abc import ABC
from typing import TypedDict

from langchain.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from promptstore import PromptStore

from corral.agents.llamp.tools import (
    MaterialsDielectric,
    MaterialsElasticity,
    MaterialsElectronic,
    MaterialsMagnetism,
    MaterialsPiezoelectric,
    MaterialsStructureText,
    MaterialsSummary,
    MaterialsSynthesis,
    MaterialsThermo,
)
from corral.agents.utils import llm_tool_call
from corral.base import Tool, ToolArgument
from corral.evaluate import BenchmarkInterface

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


class LiteLLMMessage(TypedDict, total=False):
    role: str
    content: str
    name: str | None
    tool_call_id: str | None


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

    def add_tools(self, interface, task_id):
        tools = [
            MPThermoExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPElasticityExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPDielectricExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPMagnetismExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPElectronicExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPPiezoelectricExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPSummaryExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPSynthesisExpert(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            MPStructureRetriever(
                model=self.model,
                interface=interface,
                task_id=task_id,
                kwargs=self.kwargs,
            ).as_tool(),
            arxiv,
            wikipedia,
            PythonREPLTool(),
        ]

        for tool in tools:
            interface.add_tool_to_environment(task_id, tool)

        return

    def run(
        self,
        interface,
        task_id,
        max_iterations=10,
    ):
        self.add_tools(interface, task_id)

        task_guide = interface.get_task_prompt(task_id)

        env_tools = interface.get_available_tools_for_task(task_id)
        tools_names = [tool.name for tool in env_tools]

        prefix_uuid = "70545b35-005c-4aaf-ac3b-4979f8ab10cf"
        prefix_prompt = self.prompt_store.get(prefix_uuid)
        prefix = prefix_prompt.fill({"tools": env_tools})

        format_uuid = "f4093177-d2bb-4b1b-8b77-d067b62a032e"
        format_prompt = self.prompt_store.get(format_uuid)
        format = format_prompt.fill({"tools_names": tools_names})

        suffix_uuid = "07016776-37e1-4caf-abfe-fe3fc6872ebf"
        suffix_prompt = self.prompt_store.get(suffix_uuid)
        suffix = suffix_prompt.fill({"chat_id": task_guide})

        initial_prompt = f"{prefix}\n\n{format}\n\n{suffix}"

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


class MPAgent(ABC):
    def __init__(
        self,
        model: str,
        interface: BenchmarkInterface,
        task_id: str,
        max_iterations: int = 3,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.interface = interface
        self.task_id = task_id
        self.max_iterations = max_iterations
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.prompt_store = PromptStore("./prompts")
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str | None:
        return self.__doc__

    @property
    def tools(self):
        return []

    def as_tool(
        self,
    ) -> Tool:
        def execute(input: str):
            try:
                result, _ = self.run_agent(input)
                return result, _
            except Exception as e:
                error_response = (
                    f"Error on {self.__class__.__name__}: {e}. "
                    "Please decompose the request into multiple smaller requests "
                    "or specify 'limit' in request."
                )
                return error_response

        return Tool(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "input",
                    "str",
                    "Complete question to ask the assistant agent. Should include all the context and details needed to answer the question holistically.",
                ),
            ],
        )

    def run_agent(self, input: str) -> str:
        for tool in self.tools:
            self.interface.add_tool_to_environment(self.task_id, tool)

        system_prompt = self.prompt_store.get("fb46ddea-eca3-458a-805f-aa6344780929")
        user_prompt = self.prompt_store.get("9f8a74c1-cd5e-4fc5-b50e-a2eebaffb409")
        tool_names = [tool.name for tool in self.tools]
        system = system_prompt.fill({"tools": self.tools, "tool_names": tool_names})
        user = user_prompt.fill({"input": input, "agent_scratchpad": ""})
        messages: list[LiteLLMMessage] = []
        messages.append(LiteLLMMessage(role="system", content=system))
        messages.append(LiteLLMMessage(role="user", content=user))

        for _i in range(self.max_iterations):
            try:
                response = json.loads(llm_tool_call(messages))
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
                    function_name = self.tools[action]
                    function_args = json.loads(response.get("action_input"))
                    try:
                        function_call = str(
                            self.interface.execute_tool(
                                self.task_id, function_name, json.dumps(function_args)
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

        return f"No final answer found after {self.max_iterations}", messages


class MPSummaryExpert(MPAgent):
    """Summary expert that has access to Materials Project summary endpoint"""

    @property
    def tools(self):
        return [MaterialsSummary()]


class MPStructureRetriever(MPAgent):
    """Structure expert who will retrieve structures from Materials Project."""

    @property
    def tools(self):
        return [MaterialsStructureText()]


class MPThermoExpert(MPAgent):
    """Theromodynamics expert that has access to Materials Project thermo endpoint"""

    @property
    def tools(self):
        return [MaterialsThermo()]


class MPElasticityExpert(MPAgent):
    """Elasticity expert that has access to Materials Project elasticity endpoint, including bulk, shear, and young's modulus, poisson ratio, and universal anisotropy index"""

    @property
    def tools(self):
        return [MaterialsElasticity()]


class MPMagnetismExpert(MPAgent):
    """Magnetism expert that has access to Materials Project magnetism endpoint"""

    @property
    def tools(self):
        return [MaterialsMagnetism()]


class MPDielectricExpert(MPAgent):
    """Dielectric expert that has access to Materials Project dielectric endpoint"""

    @property
    def tools(self):
        return [MaterialsDielectric()]


class MPPiezoelectricExpert(MPAgent):
    """Piezoelectric expert that has access to Materials Project piezoelectric endpoint"""

    @property
    def tools(self):
        return [MaterialsPiezoelectric()]


class MPElectronicExpert(MPAgent):
    """Electronic expert that has access to Materials Project electronic endpoint"""

    @property
    def tools(self):
        return [MaterialsElectronic()]


class MPSynthesisExpert(MPAgent):
    """Materials synthesis expert that has access to Materials Project synthesis
    endpoint, where synthesis recipes are extracted  from scientific literature through
    text mining and natural language processing approaches"""

    @property
    def tools(self):
        return [MaterialsSynthesis()]
