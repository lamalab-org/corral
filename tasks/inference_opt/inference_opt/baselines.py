"""Measure the pinned zero-shot student baseline."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference_opt import datasets
from inference_opt.outcomes import read_outcomes
from inference_opt.eval_runner import PolicyEvaluator
from inference_opt.eval_runner.spec import RunSpec

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

MANIFEST = {"name": "baseline-zero-shot", "max_calls_per_question": 1}


class Policy:
    def solve(self, question, ctx):
        prompt = question.text
        if question.choices:
            prompt += "\\n\\n" + question.rendered_choices()
            prompt += "\\n\\nAnswer with the letter of the correct option."
        return ctx.student.generate(prompt, temperature=0.0, max_tokens=1024, seed=0)
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

    The go/no-go gate. A pair at the ceiling has nothing left to win; one at the
    floor cannot be climbed off, and "improvement over baseline" does not rescue
    it — 0.17 to 0.17 is noise, not a result. Either way the task measures nothing
    and should not ship.
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
    out_dir: Path | None = None,
    timeout_s: int = 3600,
) -> BaselineResult:
    """Run the pinned zero-shot policy over one split and grade it."""
    items = datasets.load_items(benchmark, split)  # type: ignore[arg-type]
    targets = datasets.load_targets(benchmark, split)  # type: ignore[arg-type]
    result = BaselineResult(
        benchmark=benchmark,
        model=model,
        split=split,
        n=len(items),
        chance=_chance_level(items),
    )

    with tempfile.TemporaryDirectory(prefix="inference-opt-baseline-") as scratch:
        scratch_root = Path(scratch)
        policy_dir = scratch_root / "policy"
        policy_dir.mkdir()
        (policy_dir / "policy.py").write_text(BASELINE_POLICY_SOURCE, encoding="utf-8")

        questions = scratch_root / "questions.jsonl"
        datasets.write_jsonl(
            questions,
            [
                {
                    **datasets.public_record(item),
                    "target": targets.get(item.item_id, ""),
                    "index": index,
                }
                for index, item in enumerate(items)
            ],
        )

        spec = RunSpec(
            run_id=f"baseline-{benchmark}-{model}-{split}",
            policy_dir=str(policy_dir),
            questions_path=str(questions),
            out_dir=str(out_dir or scratch_root / "out"),
            model_spec=model_spec or model,
            base_url=base_url,
            total_calls=len(items) * 2,
            max_calls_per_question=1,
            benchmark=benchmark,
            split=split,
        )
        summary = PolicyEvaluator().run(spec)
        if not summary.ok and summary.n_answered == 0:
            result.error = summary.error[:600]
            return result

        outcome = read_outcomes(spec.log_dir)
        result.n_correct = outcome.n_correct
        result.accuracy = outcome.n_correct / len(items) if items else 0.0
        result.n_unparseable = summary.n_unparseable
        result.per_item = {item.item_id: item.correct for item in outcome.items}
        result.per_topic = outcome.by_category()

    return result


def load_measured(path: Path) -> dict[str, Any]:
    """Read a baselines artifact written by ``scripts/measure_baselines.py``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
