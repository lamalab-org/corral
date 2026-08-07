"""Safe adapters around Corral's allowed scientific tools."""

from corral.agents.ai_scientist.tools.corral_executor import (
    CorralExecutor,
    ToolCallBudgetExceeded,
)

__all__ = ["CorralExecutor", "ToolCallBudgetExceeded"]
