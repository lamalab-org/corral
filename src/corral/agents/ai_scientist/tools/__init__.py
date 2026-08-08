"""Safe adapters around Corral's allowed scientific tools."""

from corral.agents.ai_scientist.tools.corral_executor import (
    CorralExecutor,
    ResearchBudget,
    ToolCallBudgetExceeded,
)
from corral.agents.ai_scientist.tools.trial_pool import (
    BranchRuntime,
    ReplayDiverged,
    ReplayResult,
    TrialPool,
)

__all__ = [
    "BranchRuntime",
    "CorralExecutor",
    "ReplayDiverged",
    "ReplayResult",
    "ResearchBudget",
    "ToolCallBudgetExceeded",
    "TrialPool",
]
