"""Public types used by submitted inference policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "Answer",
    "AnswerType",
    "Answerable",
    "BudgetExhausted",
    "LabeledExample",
    "Message",
    "PolicyApiMode",
    "PolicyManifest",
    "Prompt",
    "Question",
    "RunContext",
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
    """Raised when the run has no student-model calls remaining."""


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
    """A revealed training question, its gold answer, and the baseline's reply."""

    question: Question
    answer: str
    baseline_completion: str = ""
    baseline_correct: bool = False


@dataclass(frozen=True, slots=True)
class Answer:
    """A structured policy reply; its final value becomes ``ANSWER: <final>``."""

    final: Answerable


@runtime_checkable
class StudentClient(Protocol):
    """Budgeted interface to the student model.

    ``question_id`` attributes a call to one question in the run's diagnostics.
    """

    def generate(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
    ) -> str:
        """Complete one prompt. Charges one call."""
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

    def batch(self, prompts: Sequence[Prompt], **kwargs: Any) -> list[str]:
        """Complete several prompts side by side. Charges ``len(prompts)`` calls."""
        ...

    @property
    def calls_used(self) -> int: ...

    @property
    def calls_remaining(self) -> int: ...


@dataclass(frozen=True, slots=True)
class RunContext:
    """What a policy gets besides the questions: everything from the controller."""

    #: The metered student model. ``sample`` and ``batch`` exist only in
    #: enhanced tasks.
    student: StudentClient
    #: The revealed train questions with their gold answers.
    examples: tuple[LabeledExample, ...]
    #: ``submit(question_id, answer)`` records an answer straight away.
    submit: Callable[[str, Any], None]
    #: ``log(text, question_id=None)`` adds a line to the run's diagnostics.
    log: Callable[..., None]

    @property
    def budget(self) -> int:
        """Student calls left for this run."""
        return self.student.calls_remaining


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """How a policy wants to be run. Every field has a working default.

    Declared by the policy as a plain ``MANIFEST`` dict; unknown keys are rejected
    so a typo surfaces at ``dry_run_policy`` rather than silently doing nothing.
    """

    name: str = "policy"
    version: int = 1
    #: Default and ceiling for ``max_tokens`` on each student call.
    max_tokens_per_call: int = 16384

    def __post_init__(self) -> None:
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be at least 1")
