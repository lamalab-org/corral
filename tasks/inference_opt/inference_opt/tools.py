"""The agent's tools. They run in the controller and never import the policy."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corral.core.tool import Tool, tool
from corral.core.transition import CORRAL_PRIVATE_ARTIFACTS_ARGUMENT
from inference_opt import datasets
from inference_opt.budget import LEDGER_RESOURCE, BudgetSpec, RunRecord, StateLedger
from inference_opt.client import probe_student
from inference_opt.errors import error_line, error_tail
from inference_opt.runner import ModelRun, run_policy
from inference_opt.score import HarnessError, final_result

__all__ = ["create_tools"]

#: Below this fraction of any budget, every tool result carries a submit reminder.
NAG_THRESHOLD = 0.2

_TOOL = {
    "hidden_args": ["work_dir", LEDGER_RESOURCE],
    "workspace_args": ("work_dir",),
    "trusted": True,
    "resources": (LEDGER_RESOURCE,),
}
#: Tools that run a policy also keep each run's graded logs with the trial.
_RUN_TOOL = {
    **_TOOL,
    "hidden_args": [*_TOOL["hidden_args"], CORRAL_PRIVATE_ARTIFACTS_ARGUMENT],
}


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


def _saver(private_artifacts: Any) -> Callable[[Path], str] | None:
    """Corral's private store when the tool runs under Corral, else nothing."""
    return None if private_artifacts is None else private_artifacts.save


def _idle_warning(runs: list[ModelRun]) -> str:
    """Name a run that never reached the student or answered nothing, else ``""``.

    Such a run does not crash, so without this its reply reads as a success.
    """
    problems = []
    if not any(run.calls_used for run in runs):
        problems.append("made no student calls")
    if not any(answer.strip() for run in runs for answer in run.answers.values()):
        problems.append("gave no non-empty answer")
    return f"WARNING: the policy ran but {' and '.join(problems)}. " if problems else ""


def _resolve(work_dir: str, policy_path: str) -> Path:
    root = Path(work_dir).resolve()
    candidate = (root / policy_path).resolve()
    # Keep the agent inside its workspace.
    if not candidate.is_relative_to(root):
        raise ValueError(f"policy path must be inside the workspace: {policy_path}")
    return candidate


def _policy_hash(policy_dir: Path) -> str:
    """A fingerprint of every file in the policy folder, so a submission is traceable."""
    digest = hashlib.sha256()
    for path in sorted(p for p in policy_dir.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(policy_dir)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def create_tools(config: dict[str, Any], work_dir: str) -> dict[str, Tool]:
    """Build the tool set for one task.

    ``config`` is the task's immutable ``initial_input``. The ledger is a Corral
    resource: restored before each call and committed after it.
    """
    del work_dir  # Corral passes the task's own folder on every call instead
    benchmark = str(config["benchmark"])
    models = list(config["models"])
    spec = BudgetSpec.from_mapping(config.get("budget"))
    model_specs = dict(config.get("model_specs") or {})
    baselines_train = dict(config.get("baselines_train") or config.get("baselines") or {})

    def _ledger(inference_state: Any) -> StateLedger:
        return StateLedger({} if inference_state is None else inference_state, spec)

    def _examples(ledger: StateLedger) -> list[dict[str, Any]]:
        """The revealed train questions with their answers, rebuilt from the ledger."""
        revealed = set(ledger.revealed_ids)
        if not revealed:
            return []
        targets = datasets.load_targets(benchmark, "train")
        baseline_items = dict((config.get("baseline_items") or {}).get(models[0], {}))
        completions = dict(config.get("baseline_completions") or {})
        return [
            {
                **datasets.public_record(item),
                "target": targets.get(item.item_id, ""),
                "baseline_correct": bool(baseline_items.get(item.item_id, False)),
                "baseline_completion": str(completions.get(item.item_id, "")),
            }
            for item in datasets.load_items(benchmark, "train")
            if item.item_id in revealed
        ]

    # -- reading the task -------------------------------------------------------

    @tool(**_TOOL)
    def get_baseline(work_dir: str = "", inference_state: Any = None) -> str:
        """Return the measured zero-shot baseline.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the train-split baseline accuracy per model and per topic.
        """
        ledger = _ledger(inference_state)
        payload = {
            "benchmark": benchmark,
            "models": models,
            "baseline_accuracy_train": baselines_train,
            "per_topic": config.get("baseline_topics", {}),
            "n_train": int(config.get("n_train", 30)),
            "n_test": int(config.get("n_test", 30)),
        }
        best = ", ".join(f"{m}={baselines_train.get(m, 0.0):.3f}" for m in models)
        return _compact(
            payload, f"Zero-shot baseline on {benchmark} (train split): {best}.", ledger
        )

    @tool(**_TOOL)
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
        baseline_items = dict((config.get("baseline_items") or {}).get(models[0], {}))
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
        new_records = []
        for item in chosen:
            record = datasets.public_record(item)
            record["target"] = targets.get(item.item_id, "")
            record["baseline_correct"] = bool(baseline_items.get(item.item_id, False))
            record["baseline_completion"] = str(
                (config.get("baseline_completions") or {}).get(item.item_id, "")
            )
            new_records.append(record)
        ledger.mark_revealed([item.item_id for item in chosen])

        # A copy for the agent to reread. Runs rebuild the examples from the
        # ledger, so editing or deleting this file changes nothing.
        store = Path(work_dir) / "revealed" / "train_revealed.jsonl"
        datasets.write_jsonl(store, _examples(ledger))
        return _compact(
            {
                "revealed": new_records,
                "total_revealed": len(ledger.revealed_ids),
                "reveals_left": ledger.remaining()["reveals"],
                "stored_at": str(store.relative_to(Path(work_dir))),
            },
            f"Revealed {len(new_records)} train question(s) using strategy "
            f"{strategy!r}. Every run's ctx.examples holds all of them.",
            ledger,
        )

    @tool(**_TOOL)
    def query_student(
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        n: int = 1,
        work_dir: str = "",
        inference_state: Any = None,
    ) -> str:
        """Query the student without recording an evaluation prediction.

        Args:
            prompt: The user message to send.
            system: Optional system message.
            temperature: Sampling temperature; 0.0 is deterministic.
            max_tokens: Maximum tokens to generate, reasoning included.
            n: How many samples to draw.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the completions and why each ended ("stop", or "length"
            when max_tokens ran out; then the completion may be empty).
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
            # Drop only the provider prefix: "vllm/Qwen/X" is served as "Qwen/X".
            served_name=model_specs.get(models[0], "").partition("/")[2] or None,
        )
        payload: dict[str, Any] = {
            "model": "student",
            "completions": [c.text for c in completions],
            "finish_reasons": [c.finish_reason for c in completions],
        }
        summary = f"Student returned {len(completions)} completion(s)."
        truncated = [c for c in completions if c.finish_reason == "length"]
        if truncated:
            summary += (
                f" {len(truncated)} hit max_tokens={max_tokens} before finishing; an "
                "empty completion means the student was still reasoning (reasoning "
                "counts toward max_tokens). Raise max_tokens to see the answer."
            )
            # The end of the cut-off reasoning shows how far the student got.
            payload["reasoning_tail"] = [
                c.reasoning[-300:] if not c.text and c.reasoning else "" for c in completions
            ]
        return _compact(payload, summary, ledger)

    # -- running a policy -------------------------------------------------------

    @tool(**_RUN_TOOL)
    def dry_run_policy(
        policy_path: str = "policy",
        question_ids: str = "",
        work_dir: str = "",
        inference_state: Any = None,
        corral_private_artifacts: Any = None,
    ) -> str:
        """Run a policy through the evaluation pipeline on a small sample.

        Args:
            policy_path: Workspace-relative path to the policy directory.
            question_ids: Optional comma-separated item ids; defaults to two
                revealed train questions.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with per-question answers, whether each was right, and any errors.
        """
        ledger = _ledger(inference_state)
        policy_dir = _resolve(work_dir, policy_path)
        wanted = [item.strip() for item in question_ids.split(",") if item.strip()]
        if not wanted:
            train = datasets.load_items(benchmark, "train")
            revealed = set(ledger.revealed_ids)
            preferred = [i.item_id for i in train if i.item_id in revealed][:2]
            wanted = preferred or [item.item_id for item in train[:2]]

        run_id = ledger.next_run_id("dry")
        budget = min(32, ledger.remaining()["student_calls"])
        ledger.reserve(debug_runs=1, calls=budget)
        run = run_policy(
            config=config,
            work_dir=Path(work_dir),
            run_id=run_id,
            policy_dir=policy_dir,
            split="train",
            models=models[:1],
            budget_per_model=budget,
            examples=_examples(ledger),
            item_ids=wanted,
            save_private=_saver(corral_private_artifacts),
        )[models[0]]
        if run.load_error:
            ledger.refund(calls=budget, debug_runs=1)
            return _compact(
                {"error": run.load_error},
                f"Policy could not be loaded: {run.load_error}",
                ledger,
            )
        ledger.refund(calls=max(0, budget - run.calls_used))
        traces = [
            {
                "item_id": item_id,
                "answer": run.answers.get(item_id, ""),
                "correct": run.correct.get(item_id),
                "calls": run.per_question.get(item_id, {}).get("calls", 0),
                "error": run.errors.get(item_id, ""),
                "log": run.logs.get(item_id, []),
            }
            for item_id in run.question_ids
        ]
        ledger.record_run(
            RunRecord(
                run_id=run_id,
                kind="dry_run",
                policy_dir=policy_path,
                n_items=run.n_items,
                n_correct=run.n_correct,
                calls_used=run.calls_used,
                error=error_line(run.error or run.infrastructure_error),
                private_artifact=run.private_artifact,
            )
        )
        failed = run.error or run.infrastructure_error
        if failed:
            headline = f"Dry run FAILED: {error_line(failed)}"
        elif warning := _idle_warning([run]):
            headline = f"Dry run on {len(traces)} question(s): {warning}"
        else:
            headline = (
                f"Dry run OK on {len(traces)} question(s); the policy runs end to end."
            )
        return _compact(
            {
                "run_id": run_id,
                "manifest": run.manifest,
                "traces": traces,
                "calls_used": run.calls_used,
                "artifacts": f"runs/{run_id}",
                "error": error_tail(failed, 2000),
            },
            headline,
            ledger,
        )

    @tool(**_RUN_TOOL)
    def evaluate_candidate(
        policy_path: str = "policy",
        note: str = "",
        work_dir: str = "",
        inference_state: Any = None,
        corral_private_artifacts: Any = None,
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
        run_id = ledger.next_run_id("exp")
        model_budget = ledger.remaining()["student_calls"] // len(models)
        reserved = model_budget * len(models)
        ledger.reserve(experiments=1, calls=reserved)
        runs = run_policy(
            config=config,
            work_dir=Path(work_dir),
            run_id=run_id,
            policy_dir=policy_dir,
            split="train",
            models=models,
            budget_per_model=model_budget,
            examples=_examples(ledger),
            save_private=_saver(corral_private_artifacts),
        )
        load_error = next((run.load_error for run in runs.values() if run.load_error), "")
        if load_error:
            ledger.refund(calls=reserved, experiments=1)
            return _compact(
                {"error": load_error}, f"Policy could not be loaded: {load_error}", ledger
            )
        used = sum(run.calls_used for run in runs.values())
        ledger.refund(calls=max(0, reserved - used))

        results = {model: _train_result(run) for model, run in runs.items()}
        deltas = [value["delta"] for value in results.values()]
        record = RunRecord(
            run_id=run_id,
            kind="experiment",
            policy_dir=policy_path,
            score=round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            delta=round(min(deltas), 4) if deltas else 0.0,
            n_items=next(iter(runs.values())).n_items,
            calls_used=used,
            note=note[:200],
            private_artifact=next(iter(runs.values())).private_artifact,
        )
        ledger.record_run(record)
        verdicts = "; ".join(
            f"{model}: {value['delta']:+.3f}" for model, value in results.items()
        )
        return _compact(
            {
                "run_id": run_id,
                "results": results,
                "artifacts": f"runs/{run_id}",
                "best_so_far": ledger.best_run_id == run_id,
            },
            f"{_idle_warning(list(runs.values()))}"
            f"Experiment {run_id} on {record.n_items} train questions - {verdicts}",
            ledger,
        )

    def _train_result(run: ModelRun) -> dict[str, Any]:
        baseline_value = float(baselines_train.get(run.model, 0.0))
        delta = run.accuracy - baseline_value
        by_topic: dict[str, dict[str, Any]] = {}
        categories = {
            item.item_id: item.category or "(none)"
            for item in datasets.load_items(benchmark, "train")
        }
        for item_id in run.question_ids:
            bucket = by_topic.setdefault(categories.get(item_id, "(none)"), {"n": 0, "n_correct": 0})
            bucket["n"] += 1
            bucket["n_correct"] += int(run.correct.get(item_id, False))
        return {
            "accuracy": round(run.accuracy, 4),
            "baseline": round(baseline_value, 4),
            "delta": round(delta, 4),
            "calls_used": run.calls_used,
            "n_submitted": len(run.answers),
            "n_crashed": run.n_crashed,
            "budget_exhausted": run.budget_exhausted,
            "error": error_tail(run.error or run.infrastructure_error or run.grading_error, 1000),
            "by_topic": by_topic,
        }

    @tool(**_TOOL)
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
            JSON with per-question answers, logs, call counts and errors. The gold
            answer is shown only for questions you have revealed.
        """
        ledger = _ledger(inference_state)
        runs_root = Path(work_dir) / "runs"
        if run_id == "last":
            candidates = sorted(
                (path for path in runs_root.glob("*") if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                return _compact({}, "No runs yet - use dry_run_policy first.", ledger)
            run_id = candidates[-1].name
        folder = (runs_root / run_id).resolve()
        if not folder.is_relative_to(runs_root.resolve()):
            raise ValueError(f"unknown run {run_id!r}")

        revealed = set(ledger.revealed_ids)
        targets = datasets.load_targets(benchmark, "train") if revealed else {}
        rows: list[dict[str, Any]] = []
        for model_dir in sorted(path for path in folder.glob("*") if path.is_dir()):
            predictions = model_dir / "predictions.jsonl"
            if not predictions.is_file():
                continue
            for line in predictions.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                correct = record.get("correct")
                keep = (
                    only == "all"
                    or (only == "wrong" and correct is False)
                    or (only == "right" and correct is True)
                    or (only == "errored" and record.get("error"))
                )
                if keep:
                    item_id = str(record.get("item_id", ""))
                    rows.append(
                        {
                            "item_id": item_id,
                            "model": model_dir.name,
                            "answer": record.get("answer", ""),
                            "target": targets.get(item_id) if item_id in revealed else None,
                            "correct": correct,
                            "calls": record.get("calls", 0),
                            "error": record.get("error", ""),
                            "log": record.get("log", []),
                        }
                    )
        rows = rows[: max(1, int(limit))]
        return _compact(
            {"run_id": run_id, "filter": only, "questions": rows},
            f"{len(rows)} question(s) from {run_id} matching {only!r}.",
            ledger,
        )

    @tool(**_TOOL)
    def compare_runs(work_dir: str = "", inference_state: Any = None) -> str:
        """Compare completed policy runs.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with the experiment ledger and the best run so far.
        """
        ledger = _ledger(inference_state)
        experiments = [
            {key: value for key, value in run.items() if key != "private_artifact"}
            for run in ledger.runs
            if run.get("kind") == "experiment"
        ]
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

    @tool(**_TOOL)
    def get_budget(work_dir: str = "", inference_state: Any = None) -> str:
        """Return the remaining policy budget.

        Args:
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON with used and remaining experiments, dry runs, student calls,
            reveals and submissions.
        """
        ledger = _ledger(inference_state)
        return _compact(ledger.snapshot(), ledger.advice(), ledger)

    @tool(**_RUN_TOOL)
    def submit_policy(
        policy_path: str = "policy",
        rationale: str = "",
        work_dir: str = "",
        inference_state: Any = None,
        corral_private_artifacts: Any = None,
    ) -> str:
        """Submit the final policy: it runs once on the held-out test questions.

        Args:
            policy_path: Workspace-relative path to the policy directory.
            rationale: A short note on why this is your best policy.
            work_dir: Task workspace, injected by the environment.

        Returns:
            JSON confirming the submission, and the literal string to submit. The
            test result is not shown; it is used when the episode is scored.
        """
        ledger = _ledger(inference_state)
        policy_dir = _resolve(work_dir, policy_path)
        used = int(ledger.state["submissions"])
        if used >= spec.max_submissions:
            return _compact(
                {"error": "no submissions left"},
                f"NOT submitted: all {spec.max_submissions} submissions are used. "
                "The last one stands; call submit_answer with exactly: submission.json",
                ledger,
            )
        policy_hash = _policy_hash(policy_dir)
        run_id = ledger.next_run_id("final")
        runs = run_policy(
            config=config,
            work_dir=Path(work_dir),
            run_id=run_id,
            policy_dir=policy_dir,
            split="test",
            models=models,
            budget_per_model=int(config["final_max_student_calls"]),
            examples=_examples(ledger),
            write_copy=False,
            save_private=_saver(corral_private_artifacts),
        )
        load_error = next((run.load_error for run in runs.values() if run.load_error), "")
        if load_error:
            # Import fails before any test question is seen; it costs nothing.
            return _compact(
                {"error": load_error}, f"NOT submitted: {load_error}", ledger
            )
        final: dict[str, Any] = {
            "policy_hash": policy_hash,
            "policy_dir": policy_path,
            "submission": used + 1,
            "run_id": run_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "private_artifact": next(iter(runs.values())).private_artifact,
        }
        try:
            final["result"] = final_result(config, runs)
        except HarnessError as exc:
            final["harness_error"] = str(exc)
        ledger.state["final"] = final
        ledger.state["submissions"] = used + 1
        (Path(work_dir) / "submission.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "policy_dir": policy_path,
                    "policy_hash": policy_hash,
                    "submission": used + 1,
                    "rationale": rationale[:2000],
                    "submitted_at": final["submitted_at"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return _compact(
            {
                "submitted": True,
                "policy_dir": policy_path,
                "policy_hash": policy_hash[:12],
                "submissions_left": spec.max_submissions - used - 1,
                "submit_answer_with": "submission.json",
            },
            "Submitted. Your policy ran on the held-out test questions; that result "
            "is kept hidden until the episode is scored, and a later submission "
            "replaces it. Now call submit_answer with exactly: submission.json",
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
