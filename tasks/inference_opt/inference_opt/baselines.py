"""Measure the pinned zero-shot student baseline."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference_opt import datasets
from inference_opt.runner import run_policy

__all__ = [
    "BASELINE_POLICY_SOURCE",
    "BaselineResult",
    "headroom_verdict",
    "measure_baseline",
]

#: The pinned zero-shot policy. Written to a temporary directory and run through
#: the normal pipeline, so the baseline and every candidate share one code path.
BASELINE_POLICY_SOURCE = '''\
"""The pinned zero-shot baseline. One call, no system message, no demonstrations."""

MANIFEST = {
    "name": "baseline-zero-shot",
    "max_tokens_per_call": 8192,
}


class Policy:
    def solve(self, question, ctx):
        prompt = question.text
        if question.choices:
            prompt += "\\n\\n" + question.rendered_choices()
        if question.benchmark == "chembench":
            prompt += "\\n\\nRespond with your final answer wrapped as [ANSWER]<answer>[/ANSWER]."
        elif question.choices:
            prompt += (
                "\\n\\nAnswer with the letter of the correct option, in the "
                "format 'ANSWER: <letter>'."
            )
        return ctx.student.generate(prompt, temperature=0.0, max_tokens=8192, seed=0)
'''


@dataclass
class BaselineResult:
    """One student's zero-shot performance on one benchmark split."""

    benchmark: str
    model: str
    split: str
    accuracy: float = 0.0
    n: int = 0
    n_correct: int = 0
    n_unparseable: int = 0
    chance: float = 0.0
    per_item: dict[str, bool] = field(default_factory=dict)
    per_topic: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def headroom(self) -> float:
        """How much accuracy is left to win."""
        return max(0.0, 1.0 - self.accuracy)

    @property
    def floor_margin(self) -> float:
        """How far above guessing the student already is."""
        return self.accuracy - self.chance

    def verdict(self) -> str:
        return headroom_verdict(self.accuracy, self.chance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "n": self.n,
            "n_correct": self.n_correct,
            "n_unparseable": self.n_unparseable,
            "chance": round(self.chance, 4),
            "headroom": round(self.headroom, 4),
            "floor_margin": round(self.floor_margin, 4),
            "verdict": self.verdict(),
            "per_topic": self.per_topic,
            "error": self.error,
        }


def headroom_verdict(accuracy: float, chance: float) -> str:
    """Whether a (benchmark, model) pair can be moved at all.

    A pair at the ceiling has nothing left to win; one at the floor cannot be
    climbed off, and "improvement over baseline" does not rescue it — 0.17 to
    0.17 is noise, not a result. Shown in the baseline report so a human can
    decide whether to recalibrate item selection or pick a different student.
    """
    if 1.0 - accuracy < 0.15:
        return "ceiling"
    if accuracy - chance < 0.10:
        return "floor"
    if 1.0 - accuracy < 0.25:
        return "narrow"
    return "ok"


def _chance_level(items: list[datasets.FrozenItem]) -> float:
    """Expected accuracy from guessing, averaged over the split."""
    total = 0.0
    for item in items:
        if item.answer_format == "mcq_single" and item.options:
            total += 1.0 / len(item.options)
        elif item.answer_format == "mcq_multi" and item.options:
            total += 1.0 / (2 ** len(item.options))
    return total / len(items) if items else 0.0


def measure_baseline(
    benchmark: str,
    model: str,
    split: str,
    *,
    model_spec: str | None = None,
    base_url: str | None = None,
    timeout_s: int = 3600,
    max_connections: int = 8,
) -> BaselineResult:
    """Run the pinned zero-shot policy over one split and grade it."""
    items = datasets.load_items(benchmark, split)  # type: ignore[arg-type]
    targets = datasets.load_targets(benchmark, split)  # type: ignore[arg-type]
    unlabeled = [item.item_id for item in items if item.item_id not in targets]
    if unlabeled:
        raise datasets.DatasetError(
            f"{benchmark}/{split}: {len(unlabeled)} item(s) have no label, "
            f"e.g. {unlabeled[:3]}"
        )
    result = BaselineResult(
        benchmark=benchmark,
        model=model,
        split=split,
        n=len(items),
        chance=_chance_level(items),
    )

    with tempfile.TemporaryDirectory(prefix="inference-opt-baseline-") as scratch:
        workspace = Path(scratch) / "workspace"
        policy_dir = workspace / "policy"
        policy_dir.mkdir(parents=True)
        (policy_dir / "policy.py").write_text(BASELINE_POLICY_SOURCE, encoding="utf-8")
        config = {
            "benchmark": benchmark,
            "model_specs": {model: model_spec or model},
            "base_urls": {model: base_url},
            "policy_api": "primitive",
            "student_concurrency": max(1, max_connections),
        }
        run = run_policy(
            config=config,
            work_dir=workspace,
            run_id=f"baseline-{benchmark}-{model}-{split}",
            policy_dir=policy_dir,
            split=split,
            models=[model],
            budget_per_model=len(items) * 2,
            examples=[],
            write_copy=False,
            max_tokens_cap=8192,
            time_limit_s=timeout_s,
        )[model]

    failure = run.load_error or run.grading_error or (
        run.infrastructure_error if not run.answers else ""
    )
    if failure:
        result.error = failure[:600]
        return result
    result.n_correct = run.n_correct
    result.accuracy = run.accuracy
    result.n_unparseable = sum(
        1 for item_id in run.question_ids if not run.answers.get(item_id, "").strip()
    )
    result.per_item = {item_id: run.correct.get(item_id, False) for item_id in run.question_ids}
    by_topic: dict[str, dict[str, Any]] = {}
    for item in items:
        bucket = by_topic.setdefault(item.category or "(none)", {"n": 0, "n_correct": 0})
        bucket["n"] += 1
        bucket["n_correct"] += int(run.correct.get(item.item_id, False))
    result.per_topic = {
        name: {**bucket, "accuracy": bucket["n_correct"] / bucket["n"]}
        for name, bucket in sorted(by_topic.items())
    }
    return result
