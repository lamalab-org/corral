"""The task ledger, kept in Corral state."""

from __future__ import annotations

import copy
import fcntl
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_opt.api import BudgetExhausted

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "LEDGER_RESOURCE",
    "BudgetExhausted",
    "BudgetSpec",
    "RunRecord",
    "SharedLedger",
    "StateLedger",
]

#: The Corral resource that holds the ledger in committed state.
LEDGER_RESOURCE = "inference_state"

#: Usage counters that every limit in ``BudgetSpec`` is checked against.
_COUNTERS = ("experiments", "debug_runs", "student_calls", "reveals", "probe_calls")


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Limits for one teacher task."""

    max_experiments: int = 20
    max_debug_runs: int = 10
    max_student_calls: int = 250
    max_reveals: int = 4
    max_probe_calls: int = 30
    reveal_batch: int = 5
    #: Test-split runs of a submitted policy. One: submitting means "done", and a
    #: second chance taught agents to submit early and stop.
    max_submissions: int = 1

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> BudgetSpec:
        if not raw:
            return cls()
        known = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
        if "max_debug_questions" in raw and "max_debug_runs" not in known:
            known["max_debug_runs"] = raw["max_debug_questions"]
        return cls(**known)


def _target_gain(run: dict[str, Any]) -> float | None:
    """A run's change on the task's pass metric; runs recorded before it existed use ``delta``."""
    value = run.get("target_delta")
    return run.get("delta") if value is None else value


@dataclass
class RunRecord:
    """One policy evaluation recorded in session state."""

    run_id: str
    kind: str
    policy_dir: str
    score: float | None = None
    baseline: float | None = None
    delta: float | None = None
    #: Change on the task's pass metric (accuracy, or template consistency on
    #: level 2), the smaller one across students; the best run is chosen by it.
    target_delta: float | None = None
    n_items: int = 0
    n_correct: int = 0
    calls_used: int = 0
    note: str = ""
    error: str = ""
    #: ``blob_ref`` of the run's archived graded logs; empty outside Corral.
    private_artifact: str = ""


class SharedLedger:
    """Usage one trial's branches all draw on.

    Corral copies committed state into every branch an agent forks, so each
    branch's ledger starts from the same remaining budget. With shared budgets
    on, reservations also go through this file, kept beside the trial's
    workspace and out of the agent's reach, so the trial as a whole stays within
    the task's limits however many branches it runs. It also keeps every
    branch's runs, which the main ledger never sees.
    """

    def __init__(self, path: str | Path, branch: str = "") -> None:
        self.path = Path(path)
        #: Name of the branch using it; empty for the trial's main session.
        self.branch = branch

    @contextmanager
    def transaction(self, seed: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Lock, load, yield for changes, and write back.

        ``seed`` is the caller's own ledger state, used only to start the file
        when it does not exist yet: on the trial's first call, or after a
        resume that lost it.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = self._load(seed)
                yield data
                staged = self.path.with_suffix(".tmp")
                staged.write_text(json.dumps(data), encoding="utf-8")
                staged.replace(self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self, seed: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction(seed) as data:
            return copy.deepcopy(data)

    def _load(self, seed: Mapping[str, Any]) -> dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "used": {key: int(seed.get(key, 0)) for key in _COUNTERS},
            "revealed_ids": list(seed.get("revealed_ids") or []),
            "run_numbers": dict(seed.get("run_numbers") or {}),
            "runs": [],
        }


class StateLedger:
    """JSON-shaped session state committed by Corral after trusted tool calls.

    With a ``SharedLedger``, limits are checked against the trial's total
    usage across branches, and this branch's own state records its share.
    """

    def __init__(
        self,
        state: dict[str, Any],
        spec: BudgetSpec,
        shared: SharedLedger | None = None,
    ) -> None:
        self.state = state
        self.spec = spec
        self.shared = shared
        for key, default in {
            "experiments": 0,
            "debug_runs": 0,
            "student_calls": 0,
            "reveals": 0,
            "probe_calls": 0,
            "revealed_ids": [],
            "runs": [],
            # Run numbers per kind. Unlike the budget counters, never refunded,
            # so a refunded run cannot hand its id, and its folder, to the next.
            "run_numbers": {},
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
        """Questions revealed so far; with a shared ledger, by any branch."""
        own = self.state["revealed_ids"]
        if self.shared is None:
            return own
        pooled = self.shared.read(self.state)["revealed_ids"]
        return own + [item for item in pooled if item not in own]

    @property
    def best_run_id(self) -> str | None:
        return self.state.get("best_run_id")

    def _used(self) -> dict[str, int]:
        """Usage the limits apply to: the trial's total with a shared ledger."""
        if self.shared is None:
            return {key: int(self.state[key]) for key in _COUNTERS}
        return {
            key: int(value)
            for key, value in self.shared.read(self.state)["used"].items()
        }

    def reserve(
        self,
        *,
        experiments: int = 0,
        debug_runs: int = 0,
        calls: int = 0,
        reveals: int = 0,
        probe_calls: int = 0,
    ) -> None:
        wanted = {
            "experiments": experiments,
            "debug_runs": debug_runs,
            "student_calls": calls,
            "reveals": reveals,
            "probe_calls": probe_calls,
        }
        limits = {
            "experiments": self.spec.max_experiments,
            "debug_runs": self.spec.max_debug_runs,
            "student_calls": self.spec.max_student_calls,
            "reveals": self.spec.max_reveals,
            "probe_calls": self.spec.max_probe_calls,
        }

        def check(used: Mapping[str, Any], scope: str = "") -> dict[str, int]:
            proposed = {key: int(used[key]) + wanted[key] for key in _COUNTERS}
            for name, value in proposed.items():
                if value > limits[name]:
                    raise BudgetExhausted(
                        f"{name.replace('_', ' ')} budget exhausted "
                        f"({limits[name]} allowed{scope})"
                    )
            return proposed

        own = check(self.state)
        if self.shared is not None:
            with self.shared.transaction(self.state) as data:
                data["used"] = check(data["used"], " across all branches")
        self.state.update(own)

    def refund(self, *, calls: int = 0, experiments: int = 0, debug_runs: int = 0) -> None:
        """Give back what a run reserved but did not use."""
        returned = {"student_calls": calls, "experiments": experiments, "debug_runs": debug_runs}
        for key, amount in returned.items():
            self.state[key] = max(0, int(self.state[key]) - amount)
        if self.shared is not None:
            with self.shared.transaction(self.state) as data:
                for key, amount in returned.items():
                    data["used"][key] = max(0, int(data["used"][key]) - amount)

    def next_run_id(self, prefix: str) -> str:
        numbers = self.state["run_numbers"]
        number = int(numbers.get(prefix, 0)) + 1
        if self.shared is not None:
            # Numbered across branches, so a run id names one run in the trial.
            with self.shared.transaction(self.state) as data:
                pooled = data["run_numbers"]
                number = max(number, int(pooled.get(prefix, 0)) + 1)
                pooled[prefix] = number
        numbers[prefix] = number
        return f"{prefix}-{number}"

    def record_run(self, record: RunRecord) -> None:
        self.runs.append(asdict(record))
        if self.shared is not None:
            with self.shared.transaction(self.state) as data:
                data["runs"].append({**asdict(record), "branch": self.shared.branch})
        gain = _target_gain(asdict(record))
        if record.kind == "experiment" and gain is not None:
            best = self.best_run()
            best_gain = None if best is None else _target_gain(best)
            if best_gain is None or gain > best_gain:
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
        own = self.state["revealed_ids"]
        for item_id in item_ids:
            if item_id not in own:
                own.append(item_id)
        if self.shared is not None:
            with self.shared.transaction(self.state) as data:
                pooled = data["revealed_ids"]
                pooled.extend(item for item in item_ids if item not in pooled)

    def remaining(self) -> dict[str, int]:
        used = self._used()
        return {
            "experiments": self.spec.max_experiments - used["experiments"],
            "dry_runs": self.spec.max_debug_runs - used["debug_runs"],
            "student_calls": self.spec.max_student_calls - used["student_calls"],
            "reveals": self.spec.max_reveals - used["reveals"],
            "probe_calls": self.spec.max_probe_calls - used["probe_calls"],
            "submissions": self.spec.max_submissions - int(self.state["submissions"]),
        }

    def snapshot(self) -> dict[str, Any]:
        used = self._used()
        snapshot = {
            "used": {
                "experiments": used["experiments"],
                "dry_runs": used["debug_runs"],
                "student_calls": used["student_calls"],
                "reveals": used["reveals"],
                "probe_calls": used["probe_calls"],
                "submissions": int(self.state["submissions"]),
            },
            "limits": asdict(self.spec),
            "remaining": self.remaining(),
            "runs_recorded": len(self.runs),
            "best_run_id": self.best_run_id,
            "questions_revealed": len(self.revealed_ids),
        }
        if self.shared is not None:
            snapshot["shared_across_branches"] = True
        return snapshot

    def fraction_left(self) -> float:
        left = self.remaining()
        limits = (
            (left["experiments"], self.spec.max_experiments),
            (left["student_calls"], self.spec.max_student_calls),
        )
        return min(value / total if total else 1.0 for value, total in limits)

    def advice(self) -> str:
        left = self.remaining()
        return f"Remaining: {left['experiments']} evaluation(s), {left['dry_runs']} dry run(s), {left['student_calls']} student call(s), {left['reveals']} reveal(s)."
