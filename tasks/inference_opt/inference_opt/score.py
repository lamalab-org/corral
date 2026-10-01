"""The pass rule, and the scorer that reads the result ``submit_policy`` stored."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from inference_opt import datasets
from inference_opt.budget import LEDGER_RESOURCE

if TYPE_CHECKING:
    from corral.core.state import ExecutionState
    from inference_opt.runner import ModelRun

__all__ = [
    "HarnessError",
    "ScoreOutcome",
    "ScoreReport",
    "StateScorer",
    "final_result",
    "headroom_closed",
]

#: Fraction of crashed items at which a policy run is rejected.
CRASH_RATE_LIMIT = 0.5

#: A task passes (score 1) when every student closes at least half of the gap
#: between its zero-shot baseline and a perfect score; otherwise it scores 0.
DEFAULT_PASS_RULE: dict[str, Any] = {"kind": "headroom", "min_closed": 0.5}


def headroom_closed(n_correct: int, baseline: float, n_items: int) -> float | None:
    """Share of the baseline's wrong answers the policy got right; None at ceiling."""
    baseline_correct = round(baseline * n_items)
    if baseline_correct >= n_items:
        return None
    return (n_correct - baseline_correct) / (n_items - baseline_correct)


class ScoreOutcome(StrEnum):
    """Why a run scored what it did."""

    OK = "ok"
    POLICY_CRASHED = "policy_crashed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    NO_SUBMISSION = "no_submission"
    SUSPECTED_CHEATING = "suspected_cheating"


class HarnessError(RuntimeError):
    """An environment or infrastructure fault. Must propagate, never score 0."""


@dataclass
class ScoreReport:
    """Everything about one scoring pass that will not fit in a float."""

    outcome: str = ScoreOutcome.OK
    score: float = 0.0
    dataset_version: str = ""
    content_fingerprint: str = ""
    n_test_items: int = 0
    per_model: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def result(self) -> dict[str, Any]:
        """The score, a one-line verdict, and details for Corral's run report."""
        metrics: dict[str, float] = {}
        for model, entry in self.per_model.items():
            if model == "_aggregate":
                metrics["raw_delta"] = float(entry.get("raw_delta", 0.0))
                if "passed" in entry:
                    metrics["passed"] = float(entry["passed"])
                continue
            for key in ("accuracy", "baseline", "delta", "headroom_closed"):
                if entry.get(key) is not None:
                    metrics[f"{model}_{key}"] = float(entry[key])
        return {
            "score": self.score,
            "feedback": self._verdict(),
            "metadata": {
                "outcome": str(self.outcome),
                "metrics": metrics,
                "per_model": self.per_model,
                "n_test_items": self.n_test_items,
                "dataset_version": self.dataset_version,
                "content_fingerprint": self.content_fingerprint,
                "notes": self.notes,
            },
        }

    def _verdict(self) -> str:
        if self.outcome != ScoreOutcome.OK:
            return f"{self.outcome}: " + "; ".join(self.notes[-1:])
        closed = [
            f"{model} closed {entry['headroom_closed']:.0%} of headroom"
            if entry.get("headroom_closed") is not None
            else f"{model} had no headroom"
            for model, entry in self.per_model.items()
            if model != "_aggregate"
        ]
        return f"score {self.score:g}: " + ", ".join(closed)


def final_result(config: dict[str, Any], runs: dict[str, ModelRun]) -> dict[str, Any]:
    """Score one test-split run per model; raise HarnessError for environment faults."""
    report = ScoreReport()
    benchmark = str(config["benchmark"])
    models = list(config["models"])
    baselines = dict(config.get("baselines") or {})
    if not models:
        raise HarnessError(f"task config for {benchmark} lists no models")
    missing = [model for model in models if model not in baselines]
    if missing:
        raise HarnessError(
            f"no measured baseline for {missing} on {benchmark}; "
            "run scripts/measure_baselines.py --write-tasks first"
        )
    try:
        manifest = datasets.load_manifest()
    except datasets.DatasetError as exc:
        raise HarnessError(str(exc)) from exc
    report.dataset_version = str(manifest.get("dataset_version", ""))
    report.content_fingerprint = str(manifest.get("content_fingerprint", ""))

    deltas: dict[str, float] = {}
    for model in models:
        run = runs[model]
        n_items = run.n_items
        report.n_test_items = n_items
        if run.infrastructure_error and not run.answers:
            raise HarnessError(f"{model}: {run.infrastructure_error}")
        if run.grading_error:
            raise HarnessError(f"{model}: grading failed: {run.grading_error}")
        if run.error.startswith("time limit") and not run.answers:
            report.outcome = ScoreOutcome.TIMEOUT
            report.notes.append(f"{model}: {run.error}")
            return report.result()

        # The denominator is always the full test split: an unanswered question
        # is wrong, so the budget is a real constraint to plan against.
        accuracy = run.accuracy
        baseline_value = float(baselines[model])
        delta = accuracy - baseline_value
        deltas[model] = delta
        crash_rate = run.n_crashed / n_items if n_items else 0.0
        report.per_model[model] = {
            "accuracy": round(accuracy, 4),
            "baseline": round(baseline_value, 4),
            "delta": round(delta, 4),
            "headroom_closed": headroom_closed(run.n_correct, baseline_value, n_items),
            "n_correct": run.n_correct,
            "n_items": n_items,
            "n_answered": len(run.answers),
            "n_crashed": run.n_crashed,
            "crash_rate": round(crash_rate, 4),
            "calls_used": run.calls_used,
            "output_tokens": run.output_tokens,
            "budget_exhausted": run.budget_exhausted,
            "run_error": run.error[-500:],
        }
        if crash_rate > CRASH_RATE_LIMIT:
            report.outcome = ScoreOutcome.POLICY_CRASHED
            report.notes.append(f"{model}: {run.n_crashed}/{n_items} items crashed")
            return report.result()
        # A policy that beats the baseline without calling the student did not
        # perform inference-time optimisation.
        if run.calls_used == 0 and accuracy > baseline_value:
            report.outcome = ScoreOutcome.SUSPECTED_CHEATING
            report.notes.append(
                f"{model}: scored {accuracy:.3f} above a {baseline_value:.3f} "
                "baseline while making zero student calls"
            )
            return report.result()
        if run.budget_exhausted:
            report.outcome = ScoreOutcome.BUDGET_EXHAUSTED

    # Level 2 judges the weaker of the two students, so a policy has to help
    # both models rather than trading one off against the other.
    raw = min(deltas.values()) if config.get("joint") else deltas[models[0]]
    rule = dict(config.get("pass_rule") or DEFAULT_PASS_RULE)
    aggregate: dict[str, Any] = {
        "raw_delta": round(raw, 4),
        "joint": bool(config.get("joint")),
        "deltas": {model: round(value, 4) for model, value in deltas.items()},
        "pass_rule": rule,
    }
    if rule.get("kind") == "headroom":
        threshold = float(rule.get("min_closed", 0.5))
        closed = {model: report.per_model[model]["headroom_closed"] for model in models}
        for model, value in closed.items():
            if value is None:
                report.notes.append(
                    f"{model}: baseline is already perfect, so it cannot improve"
                )
        # A tiny tolerance keeps exact halves (e.g. 1 of 2) from failing on floats.
        passed = all(v is not None and v >= threshold - 1e-9 for v in closed.values())
        report.score = 1.0 if passed else 0.0
        aggregate["passed"] = passed
    elif rule.get("kind") == "continuous":
        scaled = float(config.get("scale", 1.0)) * raw + float(config.get("offset", 0.0))
        report.score = max(0.0, min(1.0, scaled))
        aggregate["scale"] = config.get("scale", 1.0)
    else:
        raise HarnessError(f"unknown pass_rule kind: {rule.get('kind')!r}")
    report.per_model["_aggregate"] = aggregate
    return report.result()


class StateScorer:
    """Corral's scorer: reads the test result ``submit_policy`` recorded."""

    def __call__(self, state: ExecutionState) -> float:
        return float(self.evaluate_state(state)["score"])

    def evaluate_state(self, state: ExecutionState) -> dict[str, Any]:
        resources = state.environment.values.get("resources") or {}
        ledger = resources.get(LEDGER_RESOURCE) or {}
        final = ledger.get("final")
        if not final:
            result = ScoreReport(
                outcome=ScoreOutcome.NO_SUBMISSION,
                notes=["no policy was submitted with submit_policy"],
            ).result()
        elif final.get("harness_error"):
            raise HarnessError(str(final["harness_error"]))
        else:
            result = dict(final["result"])
        metadata = dict(result.get("metadata") or {})
        if final:
            metadata["policy_hash"] = final.get("policy_hash", "")
            metadata["submission"] = final.get("submission", 0)
            metadata["private_artifact"] = final.get("private_artifact", "")
        # Every dry run and experiment, in order: how the policy improved on train.
        metadata["train_history"] = list(ledger.get("runs") or [])
        result["metadata"] = metadata
        return result
