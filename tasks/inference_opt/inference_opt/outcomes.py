"""Read graded questions from an Inspect log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ItemOutcome", "read_outcomes"]


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    """One graded question."""

    item_id: str
    correct: bool
    answer: str = ""
    target: str = ""
    benchmark: str = ""
    category: str | None = None


def _score_is_correct(value: Any) -> bool:
    if isinstance(value, str):
        return value.upper() == "C"
    if isinstance(value, bool):
        return value
    return isinstance(value, (int, float)) and float(value) >= 1.0


def read_outcomes(log_dir: Path | str) -> list[ItemOutcome]:
    """Every sample in the logs under ``log_dir``."""
    from inspect_ai.log import read_eval_log

    items: list[ItemOutcome] = []
    root = Path(log_dir)
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.eval")):
        for sample in read_eval_log(str(path)).samples or []:
            target = sample.target
            metadata = sample.metadata or {}
            items.append(
                ItemOutcome(
                    item_id=str(sample.id),
                    correct=any(
                        _score_is_correct(score.value)
                        for score in (sample.scores or {}).values()
                    ),
                    answer=sample.output.completion if sample.output else "",
                    target=", ".join(target) if isinstance(target, list) else str(target),
                    benchmark=str(metadata.get("benchmark", "")),
                    category=metadata.get("category"),
                )
            )
    return items
