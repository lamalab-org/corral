from .base_agent import BaseAgent
from .llm_planner import LLMPlanner
from .react import ReActAgent
from .tool_calling import ToolCallingAgent

# Mapping of agent names to classes
_AGENT_CLASSES = {
    "BaseAgent": BaseAgent,
    "LLMPlanner": LLMPlanner,
    "ReActAgent": ReActAgent,
    "ToolCallingAgent": ToolCallingAgent,
}


def get_agent_class(name: str) -> type[BaseAgent]:
    """Get an agent class by name.

    Args:
        name: The name of the agent class (e.g., "ReActAgent", "ToolCallingAgent")

    Returns:
        type[BaseAgent]: The agent class
    """
    if name not in _AGENT_CLASSES:
        available = ", ".join(_AGENT_CLASSES.keys())
        raise ValueError(f"Unknown agent class '{name}'. Available: {available}")
    return _AGENT_CLASSES[name]


__all__ = [
    "BaseAgent",
    "LLMPlanner",
    "ReActAgent",
    "ToolCallingAgent",
    "get_agent_class",
]
