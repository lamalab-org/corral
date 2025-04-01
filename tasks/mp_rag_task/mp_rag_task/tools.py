import json
import os
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import Path

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from llamp_tools import (
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
from loguru import logger
from promptstore import PromptStore

from corral.agents.utils import (
    LiteLLMMessage,
    llm_call,
)
from corral.base import Tool, ToolArgument
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


class MPAgent(ABC):
    def __init__(
        self,
        max_iterations: int = 3,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.model = "openai/gpt-4o"
        self.max_iterations = max_iterations
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
    @abstractmethod
    def tools(self):
        return []

    def as_tool(self) -> Tool:
        def execute(**args):
            try:
                return "0,0"
                input_question = args.get("input_question")
                logger.info(
                    f"Running {self.__class__.__name__} with input: {input_question}"
                )
                result, _ = self.run_agent(input_question)
                return result
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}: {e}")
                raise RuntimeError(
                    f"Error in {self.__class__.__name__}: {e}. "
                    "Please decompose the request into multiple smaller requests "
                    "or specify 'limit' in request."
                ) from e

        return Tool(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "input_question",
                    "str",
                    "Complete question to ask the assistant agent. Should include all the context and details needed to answer the question holistically.",
                ),
            ],
        )

    def run_agent(self, input_question: str) -> str:
        logger.info(f"Running {self.__class__.__name__} with input: {input_question}")
        env_tools = {"tools": []}

        for tool in self.tools:
            env_tools["tools"].append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "arguments": tool.arguments,
                }
            )

        system_prompt = self.prompt_store.get("fb46ddea-eca3-458a-805f-aa6344780929")
        user_prompt = self.prompt_store.get("9f8a74c1-cd5e-4fc5-b50e-a2eebaffb409")
        tool_names = [tool.name for tool in self.tools]
        system = system_prompt.fill(
            {"tools": json.dumps(env_tools), "tool_names": tool_names}
        )
        user = user_prompt.fill({"input": input_question, "agent_scratchpad": ""})
        messages: list[LiteLLMMessage] = []
        messages.append(LiteLLMMessage(role="system", content=system))
        messages.append(LiteLLMMessage(role="user", content=user))

        for _i in range(self.max_iterations):
            logger.info(f"Iteration {_i + 1} of {self.max_iterations}")
            try:
                response = json.loads(llm_call(messages))
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
                        tool = next(
                            (t for t in self.tools if t.name == function_name), None
                        )
                        if tool:
                            function_call = str(tool.execute(**function_args))
                        else:
                            function_call = f"Error: Tool '{function_name}' not found"
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


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
        "MP_thermo_expert": MPThermoExpert().as_tool(),
        "MP_elasticity_expert": MPElasticityExpert().as_tool(),
        "MP_dielectric_expert": MPDielectricExpert().as_tool(),
        "MP_magnetism_expert": MPMagnetismExpert().as_tool(),
        "MP_electronic_expert": MPElectronicExpert().as_tool(),
        "MP_piezoelectronic_expert": MPPiezoelectricExpert().as_tool(),
        "MP_summary_expert": MPSummaryExpert().as_tool(),
        "MP_synthesis_expert": MPSynthesisExpert().as_tool(),
        "MP_structure_retriever": MPStructureRetriever().as_tool(),
        "arxiv": arxiv,
        "wikipedia": wikipedia,
        "Python_REPL_tool": PythonREPLTool(),
    }
