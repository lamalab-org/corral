from importlib import import_module

from corral.agents.ai_scientist import (
    AIScientistAgent,
    AIScientistConfig,
    SakanaAIScientistConfig,
)
from corral.agents.base_agent import BaseAgent
from corral.agents.llm_planner import LLMPlanner
from corral.agents.react import ReActAgent
from corral.agents.reflexion_agent import ReflexionAgent
from corral.agents.schema import (
    SURRENDER_SENTINEL,
    AgentOutcome,
    AgentStatus,
    AgentUsage,
    Thought,
)
from corral.agents.session import (
    INSPECT_SUBAGENT_TOOL_NAME,
    Agent,
    AgentSessionCapabilities,
    inspect_subagent_tool,
)
from corral.agents.terminus import TerminusAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.action import Action

# ClaudeCodeAgent, CodexAgent and OpenHandsAgent drive black-box harnesses whose
# SDKs ship only as optional extras (`corral[claude]` / `corral[codex]` /
# `corral[openhands]`; the last is also gated to Python >= 3.12). Importing their
# modules eagerly would make a plain `import corral` fail wherever those extras
# are absent — e.g. task environments that only need the core runtime. They are
# therefore resolved lazily via the module-level `__getattr__` below: the
# module import, and any missing-dependency error, happens only when the
# attribute is actually accessed. They are intentionally not imported at module
# top-level (not even under `TYPE_CHECKING`) so no ruff type-checking rule is
# tripped; `__all__` still advertises them and `__getattr__` resolves them.
_LAZY_AGENTS = {
    "ClaudeCodeAgent": "corral.agents.claude_code",
    "CodexAgent": "corral.agents.codex",
    "OpenHandsAgent": "corral.agents.openhands",
}


def __getattr__(name: str):
    module_name = _LAZY_AGENTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "INSPECT_SUBAGENT_TOOL_NAME",
    "SURRENDER_SENTINEL",
    "AIScientistAgent",
    "AIScientistConfig",
    "Action",
    "Agent",
    "AgentOutcome",
    "AgentSessionCapabilities",
    "AgentStatus",
    "AgentUsage",
    "BaseAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "LLMPlanner",
    "OpenHandsAgent",
    "ReActAgent",
    "ReflexionAgent",
    "SakanaAIScientistConfig",
    "TerminusAgent",
    "Thought",
    "ToolCallingAgent",
    "inspect_subagent_tool",
]
