"""Safe adapters around Corral's allowed scientific tools."""

from corral.agents.ai_scientist.tools.corral_executor import (
    CorralExecutor,
    ResearchBudget,
    ToolCallBudgetExceeded,
)
from corral.agents.ai_scientist.tools.trial_pool import (
    ArtifactPromotion,
    BranchRuntime,
    ReplayDiverged,
    ReplayEquivalence,
    ReplayResult,
    TrialPool,
)

__all__ = [
    "ArtifactPromotion",
    "BranchRuntime",
    "CorralExecutor",
    "ReplayDiverged",
    "ReplayEquivalence",
    "ReplayResult",
    "ResearchBudget",
    "ToolCallBudgetExceeded",
    "TrialPool",
]
