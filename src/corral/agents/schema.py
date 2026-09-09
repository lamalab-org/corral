from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from corral.core.action import Action

AgentStatus = Literal[
    "completed",
    "surrendered",
    "iteration_limit",
    "timeout",
    "budget_exhausted",
    "cancelled",
    "tool_failure",
    "protocol_failure",
    "harness_failure",
    "agent_failure",
]


class BudgetExhaustedError(RuntimeError):
    """Raised when a model provider rejects the available budget."""


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Provider usage reported by one complete session agent run.

    `llm_calls` counts physical requests for direct model agents. For native
    harnesses, one SDK-reported turn is treated as one comparable LLM call.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "llm_calls",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"AgentUsage.{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """Typed terminal result returned by a first-class session agent."""

    status: AgentStatus
    answer: str | None = None
    error: str | None = None
    usage: AgentUsage = field(default_factory=AgentUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "completed":
            if self.answer is None:
                raise ValueError("a completed AgentOutcome requires an answer")
            if self.error is not None:
                raise ValueError("a completed AgentOutcome cannot contain an error")
            return
        if self.status == "surrendered":
            if self.answer is not None or self.error is not None:
                raise ValueError(
                    "a surrendered AgentOutcome has neither an answer nor an error"
                )
            return
        if self.answer is not None:
            raise ValueError("a failed AgentOutcome cannot contain an answer")
        if not self.error:
            raise ValueError("a failed AgentOutcome requires an error")

    @property
    def is_submit_worthy(self) -> bool:
        return self.status in {"completed", "surrendered"}


__all__ = [
    "SURRENDER_SENTINEL",
    "Action",
    "AgentOutcome",
    "AgentStatus",
    "AgentUsage",
    "BudgetExhaustedError",
    "Thought",
]

#: Exact value an agent submits through `submit_answer` when it gives up on a
#: task. The canonical tool transition derives the surrendered status from this
#: shared sentinel.
SURRENDER_SENTINEL = "SURRENDER"


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str
