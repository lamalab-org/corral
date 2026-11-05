"""The Inspect scorer for each benchmark.

MATH uses inspect_evals' ``expression_exact_match_sympy`` unchanged; GSM-Symbolic
uses its repository's last-number rule (``last_number``); the v1 benchmarks use
Inspect's ``choice`` and ``match`` and inspect_evals' ChemBench scorer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    choice,
    match,
    scorer,
    stderr,
)

from inference_opt.answers import ANSWER_PREFIX

if TYPE_CHECKING:
    from inspect_ai.solver import TaskState

__all__ = ["ANSWER_PREFIX", "BenchmarkSpec", "scorer_for", "spec_for"]


@scorer(metrics=[accuracy(), stderr()])
def last_number() -> Scorer:
    """GSM-Symbolic's own grading: the last number in the reply, compared as a float.

    As in the repository's ``extract_final_answer``: commas are removed, every
    number is found with ``-?\\d+\\.?\\d*``, and the last one is the answer.
    """

    async def score(state: TaskState, target: Target) -> Score:
        numbers = re.findall(r"-?\d+\.?\d*", state.output.completion.replace(",", ""))
        if not numbers:
            return Score(value=INCORRECT, answer="", explanation="no number in the reply")
        answer = numbers[-1]
        correct = float(answer) == float(target.text)
        return Score(value=CORRECT if correct else INCORRECT, answer=answer)

    return score


def scorer_for(benchmark: str, answer_format: str) -> Scorer:
    """Select the Inspect scorer appropriate to one answer format."""
    if benchmark.startswith("gsm_"):
        return last_number()
    if benchmark.startswith("math_"):
        # inspect_evals' MATH scorer, unchanged: the answer is read from the last
        # "ANSWER: ..." line only, and checked for equality with sympy.
        from inspect_evals.math.math import expression_exact_match_sympy

        return expression_exact_match_sympy()
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
        if self.benchmark.startswith("math_"):
            return "expression_exact_match_sympy"
        if self.benchmark.startswith("gsm_"):
            return "last_number"
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
