"""In-memory run meters and Corral-backed task state."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from inference_opt.api import BudgetExhausted

__all__ = [
    "Budget",
    "BudgetExhausted",
    "BudgetSpec",
    "QuestionAllocator",
    "RunRecord",
    "StateLedger",
]
RESERVED_FRACTION = 0.7


@dataclass
class Budget:
    """In-memory meter for one policy run."""

    max_experiments: int = 20
    max_debug_questions: int = 10
    max_student_calls: int = 250
    max_output_tokens: int = 0
    experiments: int = 0
    debug_questions: int = 0
    student_calls: int = 0
    output_tokens: int = 0

    def reserve_experiment(self, calls: int) -> None:
        if self.experiments >= self.max_experiments:
            raise BudgetExhausted(
                f"experiment budget exhausted ({self.max_experiments} used)"
            )
        self._reserve_calls(calls)
        self.experiments += 1

    def reserve_student_call(self, count: int = 1) -> None:
        self._reserve_calls(count)

    def reserve_debug(self, calls: int = 1) -> None:
        if self.debug_questions >= self.max_debug_questions:
            raise BudgetExhausted(
                f"debug-question budget exhausted ({self.max_debug_questions} used)"
            )
        self._reserve_calls(calls)
        self.debug_questions += 1

    def record_tokens(self, tokens: int) -> None:
        self.output_tokens += max(0, tokens)
        if 0 < self.max_output_tokens < self.output_tokens:
            raise BudgetExhausted(
                f"output-token budget exhausted ({self.output_tokens}/{self.max_output_tokens})"
            )

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_student_calls - self.student_calls)

    def _reserve_calls(self, calls: int) -> None:
        if calls < 0:
            raise ValueError("cannot reserve a negative number of calls")
        if self.student_calls + calls > self.max_student_calls:
            raise BudgetExhausted(
                f"student inference budget exhausted ({self.student_calls}/{self.max_student_calls} used, {calls} more requested)"
            )
        self.student_calls += calls

    def snapshot(self) -> dict[str, int]:
        return {
            "experiments": self.experiments,
            "debug_questions": self.debug_questions,
            "student_calls": self.student_calls,
            "output_tokens": self.output_tokens,
            "max_experiments": self.max_experiments,
            "max_debug_questions": self.max_debug_questions,
            "max_student_calls": self.max_student_calls,
        }


class QuestionAllocator:
    """Split a run's calls between per-question reserves and a shared pool."""

    def __init__(self, total_calls: int, questions: int, per_question_cap: int) -> None:
        if questions < 1:
            raise ValueError("a run needs at least one question")
        self.total_calls = max(0, total_calls)
        self.questions = questions
        self.per_question_cap = max(1, per_question_cap)
        self.reserve_each = int(self.total_calls * RESERVED_FRACTION) // questions
        self.pool = self.total_calls - self.reserve_each * questions
        self._used: dict[str, int] = {}
        self._lock = threading.RLock()

    def allowance(self, question_id: str) -> int:
        with self._lock:
            used = self._used.get(question_id, 0)
            return min(
                max(0, self.per_question_cap - used),
                max(0, self.reserve_each - used) + max(0, self.pool),
            )

    def charge(self, question_id: str, calls: int) -> None:
        with self._lock:
            used = self._used.get(question_id, 0)
            if used + calls > self.per_question_cap:
                raise BudgetExhausted(f"per-question cap reached for {question_id}")
            from_pool = max(0, calls - max(0, self.reserve_each - used))
            if from_pool > self.pool:
                raise BudgetExhausted(
                    f"run call budget exhausted at question {question_id}"
                )
            self.pool -= from_pool
            self._used[question_id] = used + calls

    @property
    def used_total(self) -> int:
        with self._lock:
            return sum(self._used.values())


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Limits for one teacher task."""

    max_experiments: int = 20
    max_debug_runs: int = 10
    max_student_calls: int = 250
    max_reveals: int = 4
    max_probe_calls: int = 30
    reveal_batch: int = 5

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "BudgetSpec":
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
    created_at: str = ""


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

    def refund(self, *, calls: int = 0) -> None:
        self.state["student_calls"] = max(0, int(self.state["student_calls"]) - calls)

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
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": {
                "experiments": self.experiments,
                "dry_runs": int(self.state["debug_runs"]),
                "student_calls": int(self.state["student_calls"]),
                "reveals": int(self.state["reveals"]),
                "probe_calls": int(self.state["probe_calls"]),
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
