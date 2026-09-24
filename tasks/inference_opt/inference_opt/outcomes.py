"""Read per-item outcomes from Inspect AI logs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ItemOutcome", "RunOutcome", "read_outcomes", "wilson_interval"]


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    """What happened on one question."""

    item_id: str
    correct: bool
    answer: str = ""
    target: str = ""
    calls: int = 0
    error: str = ""
    unparseable: bool = False
    benchmark: str = ""
    category: str | None = None


@dataclass
class RunOutcome:
    """Every item of one run, plus simple aggregates."""

    items: list[ItemOutcome] = field(default_factory=list)
    model_calls_logged: int = 0

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def n_correct(self) -> int:
        return sum(item.correct for item in self.items)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def stderr(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(max(0.0, self.accuracy * (1 - self.accuracy)) / self.n)

    def by_category(self) -> dict[str, dict[str, float | int]]:
        buckets: dict[str, list[ItemOutcome]] = {}
        for item in self.items:
            buckets.setdefault(item.category or "(none)", []).append(item)
        return {
            name: {
                "n": len(group),
                "n_correct": sum(i.correct for i in group),
                "accuracy": sum(i.correct for i in group) / len(group),
            }
            for name, group in sorted(buckets.items())
        }


def _score_is_correct(value: Any) -> bool:
    if isinstance(value, str):
        return value.upper() == "C"
    if isinstance(value, bool):
        return value
    return isinstance(value, (int, float)) and float(value) >= 1.0


def read_outcomes(
    log_dir: Path | str, predictions: list[dict[str, Any]] | None = None
) -> RunOutcome:
    """Collect correctness from Inspect logs and call metadata from predictions."""
    from inspect_ai.log import read_eval_log

    extra = {row["item_id"]: row for row in (predictions or [])}
    items: list[ItemOutcome] = []
    logged_calls = 0
    root = Path(log_dir)
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.eval")):
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            scores = sample.scores or {}
            correct = any(_score_is_correct(score.value) for score in scores.values())
            metadata = sample.metadata or {}
            policy_meta = metadata.get("policy", {}) or {}
            side = extra.get(str(sample.id), {})
            target = sample.target
            items.append(
                ItemOutcome(
                    item_id=str(sample.id),
                    correct=correct,
                    answer=str(policy_meta.get("answer", "")),
                    target=", ".join(target)
                    if isinstance(target, list)
                    else str(target),
                    calls=int(side.get("calls", policy_meta.get("calls", 0)) or 0),
                    error=str(side.get("error", policy_meta.get("error", "")) or ""),
                    unparseable=bool(policy_meta.get("unparseable", False)),
                    benchmark=str(metadata.get("benchmark", "")),
                    category=metadata.get("category"),
                )
            )
            logged_calls += sum(
                bool(usage) for usage in (sample.model_usage or {}).values()
            )
    return RunOutcome(items=items, model_calls_logged=logged_calls)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
