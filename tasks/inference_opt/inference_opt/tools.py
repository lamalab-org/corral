"""Trusted teacher tools for building, testing and submitting an inference policy."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from corral.core.tool import Tool, tool
from inference_opt import datasets
from inference_opt.budget import BudgetSpec, RunRecord, StateLedger
from inference_opt.client import probe_student
from inference_opt.outcomes import read_outcomes
from inference_opt.policy import PolicyError, discover_policy
from inference_opt.eval_runner import PolicyEvaluator
from inference_opt.eval_runner.spec import RunSpec

__all__ = ["create_tools"]

#: Below this fraction of any budget, every tool result carries a submit reminder.
NAG_THRESHOLD = 0.2


def _compact(payload: dict[str, Any], summary: str, ledger: StateLedger) -> str:
    """Format a summary and compact JSON tool result."""
    payload = dict(payload)
    payload["budget"] = ledger.snapshot()["remaining"]
    lines = [summary, json.dumps(payload, sort_keys=True, default=str)]
    if ledger.fraction_left() <= NAG_THRESHOLD:
        lines.append(
            "NEXT: budget is nearly gone. Call submit_policy('policy'), then "
            "submit_answer('submission.json') NOW - an unsubmitted policy scores nothing."
        )
    return "\n".join(lines)


def _resolve(work_dir: str, policy_path: str) -> Path:
    root = Path(work_dir)
    candidate = (
        (root / policy_path).resolve()
        if not Path(policy_path).is_absolute()
        else Path(policy_path)
    )
    # Keep the agent inside its workspace.
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"policy path must be inside the workspace: {policy_path}")
    return candidate


def _run_dir(work_dir: str, run_id: str) -> Path:
    return Path(work_dir) / "runs" / run_id


def _stage_submission(
    work_dir: str,
    policy_dir: Path,
    manifest: dict[str, Any],
    train: dict[str, Any] | None,
    selected_by: str,
    rationale: str = "",
) -> Path:
    """Write the submission bundle the scorer looks for first."""
    root = Path(work_dir)
    bundle = {
        "schema": 1,
        "policy_dir": str(policy_dir.relative_to(root))
        if policy_dir.is_relative_to(root)
        else str(policy_dir),
        "entry": "policy.py",
        "manifest": manifest,
        "train": train or {},
        "selected_by": selected_by,
        "rationale": rationale[:2000],
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = root / "submission.json"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_questions(
    benchmark: str, split: str, path: Path, only: list[str] | None = None
) -> int:
    """Materialise questions with targets for one run (trusted side only)."""
    items = datasets.load_items(benchmark, split)  # type: ignore[arg-type]
    targets = datasets.load_targets(benchmark, split)  # type: ignore[arg-type]
    if only:
        wanted = set(only)
        items = [item for item in items if item.item_id in wanted]
    records = []
    for index, item in enumerate(items):
        record = datasets.public_record(item)
        record["target"] = targets.get(item.item_id, "")
        record["index"] = index
        records.append(record)
    return datasets.write_jsonl(path, records)


def create_tools(config: dict[str, Any], work_dir: str) -> dict[str, Tool]:
    """Build the tool set for one task.

    ``config`` is the task's immutable ``initial_input``. Mutable session state is
    supplied by Corral as a hidden JSON namespace and committed after each call.
    """
    del work_dir  # injected per call as a hidden workspace argument instead
    benchmark = str(config["benchmark"])
    models = list(config["models"])
    spec = BudgetSpec.from_mapping(config.get("budget"))
    model_specs = dict(config.get("model_specs") or {})
    base_urls = dict(config.get("base_urls") or {})
    max_calls_per_question = int(config.get("max_calls_per_question", 8))

    def _ledger(inference_state: Any) -> StateLedger:
        return StateLedger(inference_state or {}, spec)

    def _spec_for(
        work_dir: str,
        run_id: str,
        policy_dir: Path,
        questions: Path,
        model: str,
        total_calls: int,
        split: str,
    ) -> RunSpec:
        return RunSpec(
            run_id=run_id,
            policy_dir=str(policy_dir),
            questions_path=str(questions),
            out_dir=str(_run_dir(work_dir, run_id) / model),
            model_spec=model_specs.get(model, model),
            base_url=base_urls.get(model),
            total_calls=total_calls,
            max_calls_per_question=max_calls_per_question,
            setup_calls=int(config.get("setup_calls", 0)),
            revealed_path=str(Path(work_dir) / "revealed" / "train_revealed.jsonl"),
            benchmark=benchmark,
            split=split,
        )

    # -- trusted tools: they read gold answers, so they never run policy code ----

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def get_baseline(work_dir: str = "", inference_state: Any = None) -> str:
        """Return the measured zero-shot baseline.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the train-split baseline accuracy per model and per topic.
        """
        ledger = _ledger(inference_state)
        baselines = dict(config.get("baselines_train") or config.get("baselines") or {})
        payload = {
            "benchmark": benchmark,
            "models": models,
            "baseline_accuracy_train": baselines,
            "per_topic": config.get("baseline_topics", {}),
            "n_train": int(config.get("n_train", 30)),
            "n_test": int(config.get("n_test", 30)),
        }
        best = ", ".join(f"{m}={baselines.get(m, 0.0):.3f}" for m in models)
        return _compact(
            payload,
            f"Zero-shot baseline on {benchmark} (train split): {best}.",
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def reveal_train_questions(
        count: int = 5,
        strategy: str = "failures",
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Reveal a bounded set of labelled training questions.

        Args:
            count: How many new questions to reveal, at most 5 per call.
            strategy: Which to pick - "failures" (student got them wrong),
                "random", "hardest", or "topic:<name>".
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the newly revealed questions, their gold answers, and what
            the student answered zero-shot.
        """
        ledger = _ledger(inference_state)
        count = max(1, min(int(count), spec.reveal_batch))
        ledger.reserve(reveals=1)

        items = datasets.load_items(benchmark, "train")
        targets = datasets.load_targets(benchmark, "train")
        baseline_items = dict(config.get("baseline_items", {}).get(models[0], {}))
        already = set(ledger.revealed_ids)
        pool = [item for item in items if item.item_id not in already]

        if strategy.startswith("topic:"):
            wanted = strategy.split(":", 1)[1].strip().lower()
            pool = [i for i in pool if (i.category or "").lower() == wanted] or pool
        elif strategy == "failures":
            pool.sort(key=lambda i: bool(baseline_items.get(i.item_id, False)))
        elif strategy == "hardest":
            pool.sort(key=lambda i: i.reference_accuracy or 1.0)
        else:
            random.Random(len(already)).shuffle(pool)

        chosen = pool[:count]
        revealed_dir = Path(work_dir) / "revealed"
        existing = []
        store = revealed_dir / "train_revealed.jsonl"
        if store.is_file():
            existing = [
                json.loads(line)
                for line in store.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        new_records = []
        for item in chosen:
            record = datasets.public_record(item)
            record["target"] = targets.get(item.item_id, "")
            record["baseline_correct"] = bool(baseline_items.get(item.item_id, False))
            record["baseline_completion"] = str(
                config.get("baseline_completions", {}).get(item.item_id, "")
            )
            new_records.append(record)
        datasets.write_jsonl(store, existing + new_records)
        ledger.mark_revealed([item.item_id for item in chosen])

        return _compact(
            {
                "revealed": new_records,
                "total_revealed": len(existing) + len(new_records),
                "reveals_left": ledger.remaining()["reveals"],
                "stored_at": str(store.relative_to(Path(work_dir))),
            },
            f"Revealed {len(new_records)} train question(s) using strategy "
            f"{strategy!r}. Your policy's setup() will receive all of them.",
            ledger,
        )

    # -- trusted tools -----------------------------------------------------------

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def query_student(
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Query the student without recording an evaluation prediction.

        Args:
            prompt: The user message to send.
            system: Optional system message.
            temperature: Sampling temperature; 0.0 is deterministic.
            max_tokens: Maximum tokens to generate.
            n: How many samples to draw.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the completions the student produced.
        """
        ledger = _ledger(inference_state)
        count = max(1, min(int(n), 8))
        ledger.reserve(probe_calls=count, calls=count)
        completions = probe_student(
            models[0],
            prompt,
            system=system or None,
            temperature=temperature,
            max_tokens=max_tokens,
            n=count,
            served_name=model_specs.get(models[0], "").split("/")[-1] or None,
        )
        return _compact(
            {"model": models[0], "completions": completions},
            f"Student returned {len(completions)} completion(s).",
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def dry_run_policy(
        policy_path: str = "policy",
        question_ids: str = "",
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Run a policy through the evaluation pipeline on a small sample.

        Args:
            policy_path: Workspace-relative path to the policy directory.
            question_ids: Optional comma-separated item ids; defaults to two
                revealed train questions.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with per-question traces and any errors.
        """
        ledger = _ledger(inference_state)
        policy_dir = _resolve(work_dir, policy_path)

        try:
            # Loads the module, so an import-time failure is reported here with a
            # usable message rather than surfacing later as a failed run.
            discover_policy(policy_dir)
        except PolicyError as exc:
            return _compact(
                {"error": str(exc)},
                f"Policy could not be loaded: {exc}",
                ledger,
            )

        wanted = [item.strip() for item in question_ids.split(",") if item.strip()]
        if not wanted:
            train = datasets.load_items(benchmark, "train")
            revealed = set(ledger.revealed_ids)
            preferred = [i.item_id for i in train if i.item_id in revealed][:2]
            wanted = preferred or [item.item_id for item in train[:2]]

        run_id = f"dry-{len(ledger.runs) + 1}"
        out = _run_dir(work_dir, run_id)
        questions = out / "questions.jsonl"
        n_items = _write_questions(benchmark, "train", questions, only=wanted)
        budgeted = min(n_items * max_calls_per_question, 32)
        ledger.reserve(debug_runs=1, calls=budgeted)

        run_spec = _spec_for(
            work_dir, run_id, policy_dir, questions, models[0], budgeted, "train"
        )
        summary = PolicyEvaluator().run(run_spec)
        ledger.refund(calls=max(0, budgeted - summary.calls_used))

        predictions = []
        predictions_path = Path(run_spec.predictions_path)
        if predictions_path.is_file():
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        outcome = read_outcomes(run_spec.log_dir, predictions) if summary.ok else None

        traces = [
            {
                "item_id": row["item_id"],
                "answer": row["answer"],
                "calls": row["calls"],
                "error": row["error"],
                "log": row["log"],
                "correct": next(
                    (
                        i.correct
                        for i in (outcome.items if outcome else [])
                        if i.item_id == row["item_id"]
                    ),
                    None,
                ),
            }
            for row in predictions
        ]
        ledger.record_run(
            RunRecord(
                run_id=run_id,
                kind="dry_run",
                policy_dir=policy_path,
                n_items=n_items,
                calls_used=summary.calls_used,
                error=summary.error[:300],
            )
        )
        headline = (
            f"Dry run OK on {len(traces)} question(s); the policy runs end to end."
            if summary.ok and not summary.error
            else f"Dry run FAILED: {summary.error[:300]}"
        )
        return _compact(
            {
                "run_id": run_id,
                "manifest": summary.manifest,
                "execution": summary.execution,
                "traces": traces,
                "calls_used": summary.calls_used,
                "artifacts": str(out.relative_to(Path(work_dir))),
                "error": summary.error[:600],
            },
            headline,
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def evaluate_candidate(
        policy_path: str = "policy",
        note: str = "",
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Evaluate a policy on the training split.

        Args:
            policy_path: Workspace-relative path to the policy directory.
            note: A short label to help you tell runs apart later.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with accuracy, delta over baseline, and the path to run artifacts.
        """
        ledger = _ledger(inference_state)
        policy_dir = _resolve(work_dir, policy_path)
        run_id = f"exp-{ledger.experiments + 1}"
        out = _run_dir(work_dir, run_id)
        questions = out / "questions.jsonl"
        n_items = _write_questions(benchmark, "train", questions)
        budgeted = min(
            n_items * max_calls_per_question,
            ledger.remaining()["student_calls"],
        )
        ledger.reserve(experiments=1, calls=budgeted)

        results: dict[str, Any] = {}
        deltas = []
        for model in models:
            run_spec = _spec_for(
                work_dir, run_id, policy_dir, questions, model, budgeted, "train"
            )
            summary = PolicyEvaluator().run(run_spec)
            predictions_path = Path(run_spec.predictions_path)
            predictions = (
                [
                    json.loads(line)
                    for line in predictions_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                if predictions_path.is_file()
                else []
            )
            outcome = read_outcomes(run_spec.log_dir, predictions)
            accuracy = outcome.n_correct / n_items if n_items else 0.0
            baseline_value = float(
                (config.get("baselines_train") or config.get("baselines") or {}).get(
                    model, 0.0
                )
            )
            delta = accuracy - baseline_value
            deltas.append(delta)
            results[model] = {
                "accuracy": round(accuracy, 4),
                "baseline": round(baseline_value, 4),
                "delta": round(delta, 4),
                "calls_used": summary.calls_used,
                "n_crashed": summary.n_crashed,
                "n_unparseable": summary.n_unparseable,
                "budget_exhausted_at": summary.budget_exhausted_at,
                "error": summary.error[:300],
                "by_topic": outcome.by_category(),
            }

        worst = min(deltas) if deltas else 0.0
        record = RunRecord(
            run_id=run_id,
            kind="experiment",
            policy_dir=policy_path,
            score=round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            delta=round(worst, 4),
            n_items=n_items,
            calls_used=sum(r["calls_used"] for r in results.values()),
            note=note[:200],
        )
        ledger.record_run(record)
        if ledger.best_run_id == run_id:
            try:
                loaded = discover_policy(policy_dir)
                _stage_submission(
                    work_dir,
                    policy_dir,
                    {"name": loaded.manifest.name},
                    record.__dict__,
                    "auto",
                )
            except PolicyError:
                pass

        verdicts = "; ".join(
            f"{model}: {value['delta']:+.3f}" for model, value in results.items()
        )
        return _compact(
            {
                "run_id": run_id,
                "results": results,
                "artifacts": str(out.relative_to(Path(work_dir))),
                "staged_as_best": ledger.best_run_id == run_id,
            },
            f"Experiment {run_id} on {n_items} train questions - {verdicts}",
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def inspect_failures(
        run_id: str = "last",
        only: str = "wrong",
        limit: int = 5,
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Inspect prediction records from an earlier run.

        Args:
            run_id: Which run to inspect, or "last" for the most recent.
            only: Filter - "wrong", "right", "errored", or "all".
            limit: Maximum number of questions to return.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with per-question answers, logs, call counts and errors.
        """
        ledger = _ledger(inference_state)
        runs_root = Path(work_dir) / "runs"
        if run_id == "last":
            candidates = sorted(runs_root.glob("*"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                return _compact({}, "No runs yet - use dry_run_policy first.", ledger)
            run_id = candidates[-1].name

        rows: list[dict[str, Any]] = []
        for model_dir in sorted((runs_root / run_id).glob("*")):
            predictions = model_dir / "predictions.jsonl"
            if not predictions.is_file():
                continue
            records = [
                json.loads(line)
                for line in predictions.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            outcome = read_outcomes(model_dir / "log", records)
            correctness = {item.item_id: item for item in outcome.items}
            for record in records:
                item = correctness.get(record["item_id"])
                correct = item.correct if item else None
                keep = (
                    only == "all"
                    or (only == "wrong" and correct is False)
                    or (only == "right" and correct is True)
                    or (only == "errored" and record["error"])
                )
                if keep:
                    rows.append(
                        {
                            "item_id": record["item_id"],
                            "model": model_dir.name,
                            "answer": record["answer"],
                            "target": item.target if item else "",
                            "correct": correct,
                            "calls": record["calls"],
                            "error": record["error"],
                            "log": record["log"],
                        }
                    )
        rows = rows[: max(1, int(limit))]
        return _compact(
            {"run_id": run_id, "filter": only, "questions": rows},
            f"{len(rows)} question(s) from {run_id} matching {only!r}.",
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def compare_runs(work_dir: str = "", inference_state: Any = None) -> str:
        """Compare completed policy runs.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the run ledger and which run is currently staged to submit.
        """
        ledger = _ledger(inference_state)
        experiments = [run for run in ledger.runs if run.get("kind") == "experiment"]
        ordered = sorted(
            experiments, key=lambda run: run.get("delta") or -9.9, reverse=True
        )
        n_items = int(config.get("n_train", 30))
        resolution = 2 * (0.5 / max(1, n_items) ** 0.5)
        return _compact(
            {
                "runs": ordered,
                "best_run_id": ledger.best_run_id,
                "note": (
                    f"With {n_items} questions, two deltas differing by less than "
                    f"about {resolution:.2f} are not distinguishable. Prefer a policy "
                    "whose gain comes from many changed questions, not two lucky ones."
                ),
            },
            f"{len(ordered)} experiment(s); best is {ledger.best_run_id or 'none'}.",
            ledger,
        )

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def get_budget(work_dir: str = "", inference_state: Any = None) -> str:
        """Return the remaining policy budget.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with used and remaining experiments, dry runs, student calls and
            reveals.
        """
        ledger = _ledger(inference_state)
        return _compact(ledger.snapshot(), ledger.advice(), ledger)

    @tool(hidden_args=["work_dir", "inference_state"], trusted=True)
    def submit_policy(
        policy_path: str = "policy",
        rationale: str = "",
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Validate and stage the final policy submission.

        Args:
            policy_path: Workspace-relative path to the policy directory.
            rationale: A short note on why this is your best policy.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON confirming what was staged, and the literal string to submit.
        """
        ledger = _ledger(inference_state)
        policy_dir = _resolve(work_dir, policy_path)
        try:
            loaded = discover_policy(policy_dir)
        except PolicyError as exc:
            return _compact({"error": str(exc)}, f"NOT staged: {exc}", ledger)

        best = ledger.best_run()
        _stage_submission(
            work_dir,
            policy_dir,
            {
                "name": loaded.manifest.name,
                "memory": loaded.manifest.memory,
            },
            best,
            "agent",
            rationale,
        )
        return _compact(
            {
                "staged": True,
                "policy_dir": policy_path,
                "manifest": loaded.manifest.name,
                "submit_answer_with": "submission.json",
            },
            "Staged. Now call submit_answer with exactly: submission.json",
            ledger,
        )

    built = (
        get_baseline,
        reveal_train_questions,
        query_student,
        dry_run_policy,
        evaluate_candidate,
        inspect_failures,
        compare_runs,
        get_budget,
        submit_policy,
    )
    return {item.name: item for item in built}
