"""Public types used by submitted inference policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, runtime_checkable

__all__ = [
    "Answer",
    "AnswerType",
    "Answerable",
    "BudgetExhausted",
    "Completion",
    "LabeledExample",
    "Message",
    "PolicyManifest",
    "Prompt",
    "PromptItem",
    "Question",
    "RunContext",
    "StudentClient",
]

#: What a policy may return as its final answer. Anything else is coerced with ``str``.
Answerable = str | float | int

#: One chat message, e.g. ``{"role": "user", "content": "..."}``.
Message = Mapping[str, str]

#: One prompt: a bare string or an explicit message list.
Prompt = str | Sequence[Message]

#: One entry of a batch: a prompt, or ``{"prompt": ..., **settings}`` to give that call its own ``system``, ``temperature``, ``max_tokens``, ``stop``, ``seed``, ``question_id`` or ``component``.
PromptItem = Prompt | Mapping[str, Any]

AnswerType = Literal["mcq", "numeric", "text"]


class BudgetExhausted(RuntimeError):
    """Raised when the run has no student-model calls remaining."""


class Completion(str):  # noqa: SLOT000 - str subclasses cannot take non-empty __slots__
    """One student completion: the answer text, and how the student got there.

    It is a ``str`` holding the answer, so it works wherever text does.
    ``reasoning`` is the thinking the server returned separately from the answer,
    empty if there was none. ``finish_reason`` is ``"stop"``, or ``"length"`` when
    ``max_tokens`` ran out; a call cut off mid-reasoning has empty text but keeps
    its partial ``reasoning``, which a follow-up call can build on.
    """

    reasoning: str
    finish_reason: str

    def __new__(cls, text: str = "", reasoning: str = "", finish_reason: str = "") -> Self:
        completion = super().__new__(cls, text)
        completion.reasoning = reasoning
        completion.finish_reason = finish_reason
        return completion

    @property
    def truncated(self) -> bool:
        """Whether the call ended because it reached ``max_tokens``."""
        return self.finish_reason == "length"


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
    """A revealed example question, its gold answer, and the student's zero-shot reply."""

    question: Question
    answer: str
    #: The student's zero-shot reply; empty when it ran out of tokens, ``None``
    #: when it was never measured.
    baseline_completion: str | None = None
    #: The thinking behind that reply, if the server returned it separately.
    baseline_reasoning: str | None = None
    #: Whether that reply was right; ``None`` when it was never measured.
    baseline_correct: bool | None = None


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
        prompt: Prompt | Sequence[PromptItem],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
    ) -> Completion | list[Completion]:
        """Complete one prompt, or a list of prompts sent side by side.

        One prompt returns a ``Completion`` and charges one call; a failed request raises.
        A list returns one ``Completion`` per entry, in order, and charges one call per entry; a failed entry comes back empty and is logged.
        The keyword settings apply to every entry that does not set its own.
        """
        ...

    @property
    def calls_used(self) -> int: ...

    @property
    def calls_remaining(self) -> int: ...


@dataclass(frozen=True, slots=True)
class RunContext:
    """What a policy gets besides the questions: everything from the controller."""

    #: The metered student model.
    student: StudentClient
    #: The revealed example questions with their gold answers.
    examples: tuple[LabeledExample, ...]
    #: ``answer(question_id, text)``, or ``answer({question_id: text, ...})``,
    #: records answers straight away; a later answer replaces an earlier one.
    answer: Callable[..., None]
    #: ``log(text, question_id=None)`` adds a line to the run's diagnostics.
    log: Callable[..., None]

    @property
    def budget(self) -> int:
        """Student calls left for this run."""
        return self.student.calls_remaining


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """How a policy wants to be run. Every field has a working default.

    Declared by the policy as a plain ``MANIFEST`` dict; unknown keys are rejected so a typo surfaces at ``dry_run_policy`` rather than silently doing nothing.
    """

    name: str = "policy"
    version: int = 1
    #: ``max_tokens`` for a student call that does not set its own. No call
    #: generates more than ``runner.MAX_TOKENS_CAP`` tokens either way.
    max_tokens_per_call: int = 8192

    def __post_init__(self) -> None:
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be at least 1")
