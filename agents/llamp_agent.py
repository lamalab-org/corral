from __future__ import annotations

from .agent import (
    Agent,
    ConstrainedAgent,
    ModelName,
    Temperature,
)
from .prompts import (
    LLAMPPROMPT,
    LLAMPPROMPTSYSTEM,
    REACTFORMATPROMPT,
    REACTSYSTEMPROMPT,
)
from .tools import ToolStore


class BaseReActAgent:
    def __init__(
        self,
        model: str = ModelName.CLAUDE_3_SONNET.value,
        temperature: float = Temperature.ZERO.value,
        timeout: int = 60,
        caching: bool = True,
        max_calls: int = 10,
        toolstore: ToolStore | None = None,
        **model_kwargs,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.caching = caching
        self.max_calls = max_calls
        self.model_kwargs = model_kwargs
        self.sys_prompt = REACTSYSTEMPROMPT + LLAMPPROMPTSYSTEM
        self.user_prompt = REACTFORMATPROMPT + LLAMPPROMPT.format(chat_id="")
        self.toolstore = toolstore or ToolStore()
        self._scratchpad = ""

    def _format_prompt(self, task: str, calls_left: int):
        system = self.sys_prompt.format(tools=self.toolstore.get_tool_strs())
        user = self.user_prompt.format(
            tool_names=", ".join(self.toolstore.tool_names),
            input=task,
            scratchpad=self._scratchpad,
        )
        return system, user


class ReActAgent(BaseReActAgent, Agent):
    """ReAct agent with standard capabilities

    # Example usage:
    def create_react_agent(model_config: dict) -> ReActAgent:
        react_agent = ReActAgent(
            model=model_config.get('model', ModelName.CLAUDE_3_SONNET.value),
            temperature=model_config.get('temperature', Temperature.ZERO.value)
        )
        return react_agent
    """


class ConstReActAgent(BaseReActAgent, ConstrainedAgent):
    """ReAct agent with constrained capabilities

    def create_const_react_agent() -> ConstReActAgent:
    return ConstReActAgent()
    """
