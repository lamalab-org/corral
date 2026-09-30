"""Run the submitted policy and compute its test-set improvement."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_opt import datasets
from inference_opt.outcomes import read_outcomes
from inference_opt.errors import error_line, error_tail
from inference_opt.eval_runner import PolicyEvaluator
from inference_opt.eval_runner.spec import DEFAULT_STUDENT_CONCURRENCY, RunSpec

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "PolicyScorer",
    "ScoreOutcome",
    "ScoreReport",
    "policy_score",
    "resolve_submission",
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
    """Why a run scored what it did. Recorded in the diagnostics sidecar."""

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
    policy_dir: str = ""
    n_test_items: int = 0
    per_model: dict[str, Any] = field(default_factory=dict)
    per_item: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "outcome": str(self.outcome),
            "score": round(self.score, 4),
            "dataset_version": self.dataset_version,
            "content_fingerprint": self.content_fingerprint,
            "policy_dir": self.policy_dir,
            "n_test_items": self.n_test_items,
            "per_model": self.per_model,
            "per_item": self.per_item,
            "notes": self.notes,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")

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
                "policy_dir": self.policy_dir,
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


def resolve_submission(answer: str, work_dir: Path) -> tuple[Path | None, list[str]]:
    """Find the policy directory named by a submission."""
    notes: list[str] = []
    candidate = Path(str(answer).strip())

    # 1. The literal `submission.json` the agent was told to submit.
    root: Path | None = None
    if candidate.name == "submission.json" and candidate.is_file():
        root = candidate.parent
    elif candidate.is_dir():
        root = candidate
    elif candidate.is_file():
        root = candidate.parent
    if root is None:
        root = work_dir
        notes.append(
            f"submission path {answer!r} did not resolve; using workspace root"
        )

    # 2. The staged submission bundle.
    staged = root / "submission.json"
    if not staged.is_file() and (work_dir / "submission.json").is_file():
        staged = work_dir / "submission.json"
    if staged.is_file():
        try:
            bundle = json.loads(staged.read_text(encoding="utf-8"))
            policy_dir = (
                staged.parent / str(bundle.get("policy_dir", "policy"))
            ).resolve()
            if (policy_dir / "policy.py").is_file():
                notes.append(f"using staged submission from {staged.name}")
                return policy_dir, notes
            notes.append(f"staged policy_dir {policy_dir} has no policy.py")
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"could not read {staged.name}: {exc}")

    # 3. The path itself as a policy directory.
    for guess in (candidate, root, root / "policy", work_dir / "policy"):
        if (guess / "policy.py").is_file():
            notes.append(f"treating {guess} as the policy directory")
            return guess.resolve(), notes

    notes.append("no policy.py found anywhere reachable from the submission")
    return None, notes


def _write_test_questions(benchmark: str, path: Path) -> int:
    """Materialise public test questions for one run."""
    items = datasets.load_items(benchmark, "test")
    records = []
    for index, item in enumerate(items):
        record = datasets.public_record(item)
        record["index"] = index
        records.append(record)
    return datasets.write_jsonl(path, records)


class PolicyScorer:
    """Corral's scoring callable: a float when called, details via evaluate_submission."""

    def __init__(self, evaluate: Callable[[Any], dict[str, Any]]) -> None:
        self._evaluate = evaluate

    def __call__(self, answer: Any) -> float:
        return float(self._evaluate(answer)["score"])

    def evaluate_submission(self, answer: Any) -> dict[str, Any]:
        return self._evaluate(answer)


def policy_score(config: dict[str, Any], work_dir: str) -> PolicyScorer:
    """Build the scorer Corral calls with the agent's submitted answer."""

    def score_fn(answer: Any) -> dict[str, Any]:
        report = ScoreReport()
        workspace = Path(work_dir).expanduser()
        benchmark = str(config["benchmark"])
        models = list(config["models"])
        baselines = dict(config.get("baselines") or {})
        # `work_dir` is shared by every task, so each task reports into its own
        # folder; otherwise tasks overwrite one another's diagnostics.
        state = workspace / "state" / "-".join([benchmark, *models])

        # Configuration and dataset faults belong to the environment.
        if not models:
            raise HarnessError(f"task config for {benchmark} lists no models")
        missing_baselines = [model for model in models if model not in baselines]
        if missing_baselines:
            raise HarnessError(
                f"no measured baseline for {missing_baselines} on {benchmark}; "
                "run scripts/measure_baselines.py --write-tasks first"
            )

        try:
            manifest = datasets.load_manifest()
            report.dataset_version = str(manifest.get("dataset_version", ""))
            report.content_fingerprint = str(manifest.get("content_fingerprint", ""))
        except datasets.DatasetError as exc:
            raise HarnessError(str(exc)) from exc

        policy_dir, notes = resolve_submission(str(answer), workspace)
        report.notes.extend(notes)
        if policy_dir is None:
            report.outcome = ScoreOutcome.NO_SUBMISSION
            report.write(state / "scoring_failure.json")
            return report.result()
        report.policy_dir = str(policy_dir)

        with tempfile.TemporaryDirectory(prefix="inference-opt-score-") as scratch:
            scratch_root = Path(scratch)
            questions = scratch_root / "test.jsonl"
            targets = datasets.load_targets(benchmark, "test")
            items = datasets.load_items(benchmark, "test")
            missing = [item.item_id for item in items if item.item_id not in targets]
            if missing:
                raise HarnessError(
                    f"{len(missing)} test item(s) for {benchmark} have no label, "
                    f"e.g. {missing[:3]}"
                )
            n_items = _write_test_questions(benchmark, questions)
            report.n_test_items = n_items

            evaluator = PolicyEvaluator()
            deltas: dict[str, float] = {}

            specs = [
                RunSpec(
                    run_id=f"final-{benchmark}-{model}",
                    policy_dir=str(policy_dir),
                    questions_path=str(questions),
                    out_dir=str(scratch_root / model),
                    model_spec=str(config.get("model_specs", {}).get(model, model)),
                    base_url=config.get("base_urls", {}).get(model),
                    # The test allowance is per model, so a policy cannot starve one
                    # model to buy compute for the other and game the level-2 `min`.
                    total_calls=int(config["final_max_student_calls"]),
                    # The total budget is the only final allocation limit. The
                    # policy manifest may still define its own per-question cap.
                    max_calls_per_question=int(config["final_max_student_calls"]),
                    setup_calls=int(config.get("final_setup_calls", 0)),
                    benchmark=benchmark,
                    split="test",
                    policy_api=str(config.get("policy_api", "primitive")),
                    max_connections=int(
                        config.get("student_concurrency", DEFAULT_STUDENT_CONCURRENCY)
                    ),
                )
                for model in models
            ]
            # Each model has its own student server, so evaluate them side by side.
            summaries = evaluator.run_many([(spec, targets) for spec in specs])

            for model, spec, summary in zip(models, specs, summaries, strict=True):
                if summary.error and not summary.ok and summary.n_answered == 0:
                    lowered = summary.error.lower()
                    if "timeout" in lowered:
                        report.outcome = ScoreOutcome.TIMEOUT
                    else:
                        # A host that never produced a single answer is far more
                        # likely broken infrastructure than a broken policy.
                        raise HarnessError(
                            f"policy evaluator produced no answers for {model}: "
                            f"{error_line(summary.error)}\n{error_tail(summary.error, 3000)}"
                        )
                    report.notes.append(error_tail(summary.error, 3000))
                    report.write(state / "scoring_failure.json")
                    return report.result()

                outcome = read_outcomes(spec.log_dir, _read_predictions(spec))
                # The denominator is always the full test split: an unattempted
                # question is wrong, so budget is a real constraint to plan against.
                accuracy = outcome.n_correct / n_items if n_items else 0.0
                baseline_value = float(baselines[model])
                delta = accuracy - baseline_value
                deltas[model] = delta

                crash_rate = summary.n_crashed / n_items if n_items else 0.0
                entry = {
                    "accuracy": round(accuracy, 4),
                    "baseline": round(baseline_value, 4),
                    "delta": round(delta, 4),
                    "headroom_closed": headroom_closed(
                        outcome.n_correct, baseline_value, n_items
                    ),
                    "n_correct": outcome.n_correct,
                    "n_items": n_items,
                    "n_answered": summary.n_answered,
                    "n_crashed": summary.n_crashed,
                    "n_unparseable": summary.n_unparseable,
                    "crash_rate": round(crash_rate, 4),
                    "calls_used": summary.calls_used,
                    "output_tokens": summary.output_tokens,
                    "budget_exhausted_at": summary.budget_exhausted_at,
                    "execution": summary.execution,
                }
                report.per_model[model] = entry
                report.per_item.extend(
                    {
                        "item_id": item.item_id,
                        "model": model,
                        "policy_correct": item.correct,
                        "calls": item.calls,
                    }
                    for item in outcome.items
                )

                if crash_rate > CRASH_RATE_LIMIT:
                    report.outcome = ScoreOutcome.POLICY_CRASHED
                    report.notes.append(
                        f"{model}: {summary.n_crashed}/{n_items} items crashed"
                    )
                    report.write(state / "scoring_diagnostics.json")
                    return report.result()

                # A policy that beats the baseline without calling the student did
                # not perform inference-time optimisation.
                if summary.calls_used == 0 and accuracy > baseline_value:
                    report.outcome = ScoreOutcome.SUSPECTED_CHEATING
                    report.notes.append(
                        f"{model}: scored {accuracy:.3f} above a {baseline_value:.3f} "
                        "baseline while making zero student calls"
                    )
                    report.write(state / "scoring_diagnostics.json")
                    return report.result()

                if summary.budget_exhausted_at:
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
            closed = {m: report.per_model[m]["headroom_closed"] for m in models}
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
            scaled = float(config.get("scale", 1.0)) * raw + float(
                config.get("offset", 0.0)
            )
            report.score = max(0.0, min(1.0, scaled))
            aggregate["scale"] = config.get("scale", 1.0)
        else:
            raise HarnessError(f"unknown pass_rule kind: {rule.get('kind')!r}")
        report.per_model["_aggregate"] = aggregate
        report.write(state / "scoring_diagnostics.json")
        return report.result()

    return PolicyScorer(score_fn)


def _read_predictions(spec: RunSpec) -> list[dict[str, Any]]:
    path = Path(spec.predictions_path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
