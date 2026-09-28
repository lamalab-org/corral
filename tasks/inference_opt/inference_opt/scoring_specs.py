"""Select Inspect or Inspect Evals scorers without parsing completions."""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.scorer import (
    Scorer,
    choice,
    match,
)

__all__ = ["ANSWER_PREFIX", "BenchmarkSpec", "scorer_for", "spec_for"]

ANSWER_PREFIX = "ANSWER:"


def scorer_for(benchmark: str, answer_format: str) -> Scorer:
    """Select the Inspect scorer appropriate to one answer format."""
    if benchmark == "chembench":
        from inspect_evals.chembench.chembench import chembench_scorer

        return chembench_scorer()
    if answer_format in ("mcq_single", "mcq_multi"):
        return choice()
    if answer_format == "numeric":
        return match(location="end", numeric=True)
    return match(location="end", ignore_case=True)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """How one benchmark/answer-format group is graded by Inspect."""

    benchmark: str
    answer_format: str
    multiple_correct: bool = False

    @property
    def scorer_name(self) -> str:
        name = {
            "mcq_single": "choice",
            "mcq_multi": "choice",
            "numeric": "match_numeric",
        }.get(self.answer_format, "match")
        return "chembench" if self.benchmark == "chembench" else name

    def build_scorer(self) -> Scorer:
        return scorer_for(self.benchmark, self.answer_format)


def spec_for(benchmark: str, answer_format: str) -> BenchmarkSpec:
    """Resolve the Inspect scorer for one benchmark and answer format."""
    resolved = answer_format
    return BenchmarkSpec(
        benchmark=benchmark,
        answer_format=resolved,
        multiple_correct=resolved == "mcq_multi",
    )
