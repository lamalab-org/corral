"""The task ledger, kept in Corral state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from inference_opt.api import BudgetExhausted

__all__ = [
    "LEDGER_RESOURCE",
    "BudgetExhausted",
    "BudgetSpec",
    "RunRecord",
    "StateLedger",
]

#: The Corral resource that holds the ledger in committed state.
LEDGER_RESOURCE = "inference_state"


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Limits for one teacher task."""

    max_experiments: int = 20
    max_debug_runs: int = 10
    max_student_calls: int = 250
    max_reveals: int = 4
    max_probe_calls: int = 30
    reveal_batch: int = 5
    #: Test-split runs of a submitted policy; the score shown is the last one.
    max_submissions: int = 3

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> BudgetSpec:
        if not raw:
            return cls()
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        if "max_debug_questions" in raw and "max_debug_runs" not in known:
            known["max_debug_runs"] = raw["max_debug_questions"]
        return cls(**known)


@dataclass
class RunRecord:
    """One policy evaluation recorded in session state."""

    run_id: str
    kind: str
    policy_dir: str
    score: float | None = None
    baseline: float | None = None
    delta: float | None = None
    n_items: int = 0
    n_correct: int = 0
    calls_used: int = 0
    note: str = ""
    error: str = ""


class StateLedger:
    """JSON-shaped session state committed by Corral after trusted tool calls."""

    def __init__(self, state: dict[str, Any], spec: BudgetSpec) -> None:
        self.state = state
        self.spec = spec
        for key, default in {
            "experiments": 0,
            "debug_runs": 0,
            "student_calls": 0,
            "reveals": 0,
            "probe_calls": 0,
            "revealed_ids": [],
            "runs": [],
            "best_run_id": None,
            "submissions": 0,
            # The hidden test result of the last submission; never shown to the agent.
            "final": None,
        }.items():
            state.setdefault(key, default)

    @property
    def experiments(self) -> int:
        return int(self.state["experiments"])

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self.state["runs"]

    @property
    def revealed_ids(self) -> list[str]:
        return self.state["revealed_ids"]

    @property
    def best_run_id(self) -> str | None:
        return self.state.get("best_run_id")

    def reserve(
        self,
        *,
        experiments: int = 0,
        debug_runs: int = 0,
        calls: int = 0,
        reveals: int = 0,
        probe_calls: int = 0,
    ) -> None:
        proposed = {
            "experiments": self.experiments + experiments,
            "debug_runs": int(self.state["debug_runs"]) + debug_runs,
            "student_calls": int(self.state["student_calls"]) + calls,
            "reveals": int(self.state["reveals"]) + reveals,
            "probe_calls": int(self.state["probe_calls"]) + probe_calls,
        }
        limits = {
            "experiments": self.spec.max_experiments,
            "debug_runs": self.spec.max_debug_runs,
            "student_calls": self.spec.max_student_calls,
            "reveals": self.spec.max_reveals,
            "probe_calls": self.spec.max_probe_calls,
        }
        for name, value in proposed.items():
            if value > limits[name]:
                raise BudgetExhausted(
                    f"{name.replace('_', ' ')} budget exhausted ({limits[name]} allowed)"
                )
        self.state.update(proposed)

    def refund(self, *, calls: int = 0, experiments: int = 0, debug_runs: int = 0) -> None:
        """Give back what a run reserved but did not use."""
        self.state["student_calls"] = max(0, int(self.state["student_calls"]) - calls)
        self.state["experiments"] = max(0, self.experiments - experiments)
        self.state["debug_runs"] = max(0, int(self.state["debug_runs"]) - debug_runs)

    def record_run(self, record: RunRecord) -> None:
        self.runs.append(asdict(record))
        if record.kind == "experiment" and record.delta is not None:
            best = self.best_run()
            best_delta = None if best is None else best.get("delta")
            if best_delta is None or record.delta > best_delta:
                self.state["best_run_id"] = record.run_id

    def best_run(self) -> dict[str, Any] | None:
        return (
            next(
                (run for run in self.runs if run.get("run_id") == self.best_run_id),
                None,
            )
            if self.best_run_id
            else None
        )

    def mark_revealed(self, item_ids: list[str]) -> None:
        for item_id in item_ids:
            if item_id not in self.revealed_ids:
                self.revealed_ids.append(item_id)

    def remaining(self) -> dict[str, int]:
        return {
            "experiments": self.spec.max_experiments - self.experiments,
            "dry_runs": self.spec.max_debug_runs - int(self.state["debug_runs"]),
            "student_calls": self.spec.max_student_calls
            - int(self.state["student_calls"]),
            "reveals": self.spec.max_reveals - int(self.state["reveals"]),
            "probe_calls": self.spec.max_probe_calls - int(self.state["probe_calls"]),
            "submissions": self.spec.max_submissions - int(self.state["submissions"]),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": {
                "experiments": self.experiments,
                "dry_runs": int(self.state["debug_runs"]),
                "student_calls": int(self.state["student_calls"]),
                "reveals": int(self.state["reveals"]),
                "probe_calls": int(self.state["probe_calls"]),
                "submissions": int(self.state["submissions"]),
            },
            "limits": asdict(self.spec),
            "remaining": self.remaining(),
            "runs_recorded": len(self.runs),
            "best_run_id": self.best_run_id,
            "questions_revealed": len(self.revealed_ids),
        }

    def fraction_left(self) -> float:
        limits = (
            (self.spec.max_experiments - self.experiments, self.spec.max_experiments),
            (
                self.spec.max_student_calls - int(self.state["student_calls"]),
                self.spec.max_student_calls,
            ),
        )
        return min(left / total if total else 1.0 for left, total in limits)

    def advice(self) -> str:
        left = self.remaining()
        return f"Remaining: {left['experiments']} evaluation(s), {left['dry_runs']} dry run(s), {left['student_calls']} student call(s), {left['reveals']} reveal(s)."
