"""Built-in hooks for agent lifecycle events."""

from corral.agents.hooks.core import (
    AgentHooks,
    CriticalHookError,
    HookContext,
    HookPoint,
)
from corral.agents.hooks.intervention import (
    create_intervention_hook,
    create_trace_intervention_hook,
)
from corral.agents.hooks.sycophancy import create_sycophancy_hook

__all__ = [
    "AgentHooks",
    "CriticalHookError",
    "HookContext",
    "HookPoint",
    "create_intervention_hook",
    "create_sycophancy_hook",
    "create_trace_intervention_hook",
]
