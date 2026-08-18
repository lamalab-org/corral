"""Evaluation models and scorer implementations.

Evaluation deliberately lives outside the task-execution runtime. A scorer
receives an immutable, completed :class:`~corral.core.state.State`, computes
benchmark-only information, and returns it as a sibling result. It cannot add
correctness, ground-truth feedback, or benchmark coordinates to State.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, JsonValue

from corral.core._immutable import FrozenModel, validate_sha256_hex

if TYPE_CHECKING:
    from pathlib import Path

    from corral.core.state import State
    from corral.core.task import TaskDefinition


class EvaluationResult(FrozenModel):
    """Benchmark-only result associated with one immutable final State."""

    state_hash: str
    score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    feedback: str | None = None
    scorer_version: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def model_post_init(self, __context: object, /) -> None:
        validate_sha256_hex(self.state_hash, field_name="state_hash")


class Scorer(Protocol):
    """Evaluate a completed State without mutating execution data."""

    def evaluate(self, state: State) -> EvaluationResult: ...


def _callable_version(task: TaskDefinition) -> str:
    scorer = task.scoring_fn
    module = getattr(scorer, "__module__", type(scorer).__module__)
    name = getattr(scorer, "__qualname__", type(scorer).__qualname__)
    return f"{module}:{name}"


def _resolve_submission(
    task: TaskDefinition,
    submission: str,
    workspace: str | Path | None,
) -> str:
    """Resolve file-backed submissions only for the evaluation call."""
    if not task.resolve_answer:
        return submission
    if submission.startswith("{") and submission.endswith("}"):
        return submission
    if workspace is None:
        return submission
    from corral.utils.tool_helpers import smart_resolve_path

    return smart_resolve_path(submission, base_dir=str(workspace))


@dataclass(frozen=True, slots=True)
class TaskScorer:
    """Adapt a task's scoring callable to the pure :class:`Scorer` boundary."""

    task: TaskDefinition
    workspace: str | Path | None = None
    scorer_version: str | None = None

    def evaluate(self, state: State) -> EvaluationResult:
        """Evaluate a successful runtime output and leave State unchanged."""
        if state.submission is None or state.runtime.status == "surrendered":
            raise ValueError(
                "only a completed, non-surrendered submission can be evaluated"
            )

        state_hash = state.state_hash
        answer = _resolve_submission(self.task, state.submission, self.workspace)
        score = float(self.task.scoring_fn(answer))

        # The immutable model already prevents normal mutation. Checking the
        # content hash makes score purity an explicit runtime invariant too.
        if state.state_hash != state_hash:
            raise RuntimeError("scorer mutated State")

        return EvaluationResult(
            state_hash=state_hash,
            score=score,
            metrics={"score": score},
            scorer_version=self.scorer_version or _callable_version(self.task),
        )


__all__ = ["EvaluationResult", "Scorer", "TaskScorer"]
