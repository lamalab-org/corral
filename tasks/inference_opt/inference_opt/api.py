"""Public types used by submitted inference policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

__all__ = [
    "Answer",
    "AnswerType",
    "Answerable",
    "BudgetExhausted",
    "ComponentSpec",
    "LabeledExample",
    "Memory",
    "Message",
    "PolicyManifest",
    "PolicyApiMode",
    "Prompt",
    "Question",
    "SetupContext",
    "SolveContext",
    "StudentClient",
]

#: What a policy may return as its final answer. Anything else is coerced with ``str``.
Answerable = str | float | int

#: One chat message, e.g. ``{"role": "user", "content": "..."}``.
Message = Mapping[str, str]

#: What every client method accepts: a bare prompt or an explicit message list.
Prompt = str | Sequence[Message]

AnswerType = Literal["mcq", "numeric", "text"]
PolicyApiMode = Literal["primitive", "enhanced"]


class BudgetExhausted(RuntimeError):
    """Raised when the policy has no student-model calls remaining."""


@dataclass(frozen=True, slots=True)
class Question:
    """A benchmark question without its target answer."""

    id: str
    text: str
    benchmark: str
    answer_type: AnswerType
    choices: tuple[str, ...] | None = None
    choice_labels: tuple[str, ...] | None = None
    topic: str | None = None
    index: int = 0
    total: int = 1

    @property
    def is_multiple_choice(self) -> bool:
        return self.answer_type == "mcq" and bool(self.choices)

    def rendered_choices(self) -> str:
        """Return ``"A) first\\nB) second"``, or an empty string when free-form."""
        if not self.choices:
            return ""
        labels = self.choice_labels or tuple(
            chr(ord("A") + position) for position in range(len(self.choices))
        )
        return "\n".join(
            f"{label}) {choice}"
            for label, choice in zip(labels, self.choices, strict=False)
        )


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """A revealed training question, target, and baseline result."""

    question: Question
    answer: str
    baseline_completion: str = ""
    baseline_correct: bool = False


@dataclass(frozen=True, slots=True)
class Answer:
    """A structured policy reply; its final value becomes ``ANSWER: <final>``."""

    final: Answerable
    confidence: float | None = None
    rationale: str | None = None
    trace: tuple[Mapping[str, Any], ...] = ()


@runtime_checkable
class StudentClient(Protocol):
    """Budgeted interface to the student model."""

    def generate(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        component: str | None = None,
    ) -> str:
        """Complete one prompt. Charges one call. ``max_tokens=None`` uses the manifest's ``max_tokens_per_call``."""
        ...

    def sample(
        self,
        prompt: Prompt,
        *,
        n: int,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> list[str]:
        """Draw ``n`` completions of one prompt. Charges ``n`` calls."""
        ...

    def batch(
        self,
        prompts: Sequence[Prompt],
        **kwargs: Any,
    ) -> list[str]:
        """Complete several prompts. Charges ``len(prompts)`` calls."""
        ...

    @property
    def calls_used(self) -> int: ...

    @property
    def calls_remaining(self) -> int: ...


@runtime_checkable
class Memory(Protocol):
    """State shared across all questions of a run."""

    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def append(self, key: str, value: Any) -> None: ...
    def items(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Arguments passed to the optional policy setup hook."""

    student: StudentClient
    train_examples: tuple[LabeledExample, ...]
    memory: Memory | None
    artifacts: Path
    budget_remaining: int
    log: Callable[[str], None]
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolveContext:
    """Arguments passed to ``Policy.solve`` for one question."""

    student: StudentClient
    memory: Memory | None
    artifacts: Path
    budget_remaining: int
    question_budget: int
    log: Callable[[str], None]
    scratch: dict[str, Any] = field(default_factory=dict)
    #: Tags every call made inside it, so diagnostics can attribute spend and
    #: accuracy per component. ``None`` when the runtime supplies no tagging.
    component: Callable[[str], AbstractContextManager[None]] | None = None


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One named part of a multi-component policy, for per-component diagnostics."""

    name: str
    kind: str = "other"
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """How a policy wants to be run. Every field has a working default.

    Declared by the policy as a plain ``MANIFEST`` dict; unknown keys are rejected
    so a typo surfaces at ``dry_run_policy`` rather than silently doing nothing.
    """

    name: str = "policy"
    version: int = 1
    max_calls_per_question: int = 8
    setup_calls: int = 0
    max_tokens_per_call: int = 8192
    #: Whether questions may be solved concurrently. Set False when a question
    #: depends on earlier ones, e.g. through ``ctx.memory`` or state kept on the
    #: policy across calls.
    concurrent: bool = True
    components: tuple[ComponentSpec, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.concurrent, bool):
            raise ValueError("concurrent must be true or false")
        if self.max_calls_per_question < 1:
            raise ValueError("max_calls_per_question must be at least 1")
        if self.setup_calls < 0:
            raise ValueError("setup_calls cannot be negative")
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be at least 1")
