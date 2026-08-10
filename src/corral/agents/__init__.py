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
from corral.agents.schema import SURRENDER_SENTINEL, Action, AgentRunResult, Thought
from corral.agents.terminus import TerminusAgent
from corral.agents.tool_calling import ToolCallingAgent

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
    "SURRENDER_SENTINEL",
    "AIScientistAgent",
    "AIScientistConfig",
    "Action",
    "AgentRunResult",
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
]
