"""Run one policy evaluation with Inspect."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from inference_opt.api import BudgetExhausted, LabeledExample, Question
from inference_opt.policy import PolicyError, discover_policy
from inference_opt.eval_runner.runtime import RunRuntime
from inference_opt.eval_runner.solver import policy_solver
from inference_opt.eval_runner.spec import RunSpec, RunSummary
from inference_opt.scoring_specs import spec_for

#: Environment variables that may contain provider credentials.
_SCRUBBED_PREFIXES = ("OPENAI_", "ANTHROPIC_", "GOOGLE_", "AZURE_", "AWS_", "HF_")
_SCRUBBED_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
_KEEP = {"VLLM_BASE_URL", "INSPECT_EVAL_MODEL"}


def _is_credential(name: str) -> bool:
    return name not in _KEEP and (
        name.startswith(_SCRUBBED_PREFIXES) or name.endswith(_SCRUBBED_SUFFIXES)
    )


def scrub_environment() -> list[str]:
    """Remove provider credentials from this process. Returns what was removed."""
    removed: list[str] = []
    for name in list(os.environ):
        if _is_credential(name):
            os.environ.pop(name, None)
            removed.append(name)
    return removed


@contextmanager
def scrubbed_environment():
    """Run code without provider credentials in its environment."""
    saved = {name: value for name, value in os.environ.items() if _is_credential(name)}
    scrub_environment()
    try:
        yield
    finally:
        for name in list(os.environ):
            if _is_credential(name):
                os.environ.pop(name, None)
        os.environ.update(saved)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _to_sample(
    record: dict[str, Any], index: int, total: int, target: str | list[str] = ""
) -> Any:
    from inspect_ai.dataset import Sample

    answer_format = str(record.get("answer_format", "text"))
    options = record.get("options") or None
    if answer_format == "mcq_multi" and isinstance(target, str):
        target = [part.strip() for part in target.split(",") if part.strip()]
    return Sample(
        id=str(record["item_id"]),
        input=str(record["question"]),
        choices=list(options) if options else None,
        target=target,
        metadata={
            "question": str(record["question"]),
            "benchmark": record.get("benchmark", ""),
            "answer_format": answer_format,
            "answer_type": (
                "mcq"
                if answer_format.startswith("mcq")
                else "numeric"
                if answer_format == "numeric"
                else "text"
            ),
            "task_type": (
                "mae"
                if record.get("benchmark") == "chembench"
                and answer_format == "numeric"
                else "multiple_choice"
                if record.get("benchmark") == "chembench"
                and answer_format.startswith("mcq")
                else None
            ),
            "category": record.get("category"),
            "index": index,
            "total": total,
        },
    )


def _revealed_examples(path: str | None) -> tuple[LabeledExample, ...]:
    """Load the labeled train examples the agent unlocked, for ``setup()``."""
    if not path or not Path(path).is_file():
        return ()
    examples = []
    for record in _read_jsonl(Path(path)):
        answer_format = str(record.get("answer_format", "text"))
        options = record.get("options") or None
        question = Question(
            id=str(record["item_id"]),
            text=str(record["question"]),
            benchmark=str(record.get("benchmark", "")),
            answer_type=(
                "mcq"
                if answer_format.startswith("mcq")
                else "numeric"
                if answer_format == "numeric"
                else "text"
            ),
            choices=tuple(options) if options else None,
            choice_labels=(
                tuple(chr(ord("A") + i) for i in range(len(options)))
                if options
                else None
            ),
            topic=record.get("category"),
        )
        examples.append(
            LabeledExample(
                question=question,
                answer=str(record.get("target", "")),
                baseline_completion=str(record.get("baseline_completion", "")),
                baseline_correct=bool(record.get("baseline_correct", False)),
            )
        )
    return tuple(examples)


def run(spec: RunSpec, targets: dict[str, str] | None = None) -> RunSummary:
    """Execute one policy over one question set and write the run artifacts."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.model import get_model
    from inspect_ai.util._display import init_display_type

    started = time.monotonic()
    summary = RunSummary(run_id=spec.run_id)
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_root = Path(spec.policy_dir)
    try:
        policy = discover_policy(policy_root)
    except PolicyError as exc:
        summary.error = f"policy could not be loaded: {exc}"
        summary.write(spec.summary_path)
        return summary

    manifest = policy.manifest
    summary.manifest = {
        "name": manifest.name,
        "execution": "sequential",
        "memory": manifest.memory,
        "max_calls_per_question": manifest.max_calls_per_question,
        "setup_calls": manifest.setup_calls,
        "components": [component.name for component in manifest.components],
    }

    records = _read_jsonl(Path(spec.questions_path))
    if targets is None:
        targets = {
            str(record["item_id"]): str(record["target"])
            for record in records
            if "target" in record
        }
    total = len(records)
    summary.n_questions = total

    init_display_type("plain")
    model = get_model(
        spec.model_spec,
        **(
            {"base_url": spec.base_url, "api_key": spec.api_key or "none"}
            if spec.base_url
            else {}
        ),
    )

    per_question_cap = min(manifest.max_calls_per_question, spec.max_calls_per_question)
    runtime = RunRuntime(
        model=model,
        questions=total,
        total_calls=max(0, spec.total_calls - spec.setup_calls),
        max_calls_per_question=per_question_cap,
        max_tokens_per_call=min(manifest.max_tokens_per_call, spec.max_tokens_per_call),
        memory_enabled=manifest.memory == "shared",
        predictions_path=Path(spec.predictions_path),
        benchmark=spec.benchmark,
        split=spec.split,
        policy_api=spec.policy_api,
    )

    # -- setup(), charged against its own allowance --------------------------
    if policy.has_setup and spec.setup_calls > 0:
        from inference_opt.budget import QuestionAllocator
        from inference_opt.eval_runner.runtime import QuestionMeter

        setup_allocator = QuestionAllocator(
            total_calls=spec.setup_calls, questions=1, per_question_cap=spec.setup_calls
        )
        setup_meter = QuestionMeter(setup_allocator, "__setup__")
        context, setup_log_lines = runtime.make_setup_context(
            setup_meter,
            policy_root,
            _revealed_examples(spec.revealed_path),
            dict(manifest.config),
        )
        try:
            with scrubbed_environment():
                policy.setup(context)
        except BudgetExhausted as exc:
            summary.first_tracebacks.append(f"setup budget exhausted: {exc}")
        except Exception:
            summary.first_tracebacks.append(traceback.format_exc(limit=8))
        summary.setup_calls_used = setup_meter.used
        summary.setup_log = setup_log_lines[:50]

    # -- one task per answer format; a branching scorer is where misgrading hides
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("answer_format", "text")), []).append(record)

    tasks = []
    for answer_format, group in sorted(groups.items()):
        benchmark = str(group[0].get("benchmark", spec.benchmark))
        resolved = spec_for(benchmark, answer_format)
        samples = [
            _to_sample(
                record,
                record.get("index", position),
                total,
                targets.get(str(record["item_id"]), ""),
            )
            for position, record in enumerate(group)
        ]
        tasks.append(
            Task(
                name=f"{benchmark or 'run'}-{answer_format}",
                dataset=MemoryDataset(samples=samples, name=f"{benchmark}:{spec.split}"),
                solver=policy_solver(policy, runtime, policy_root),
                scorer=resolved.build_scorer(),
                epochs=spec.epochs,
                fail_on_error=False,
            )
        )

    # Sequential by default so memory, budgets, and artifacts are stable; the
    # stateless zero-shot baseline opts into concurrency via spec.max_connections.
    summary.execution = "sequential" if spec.max_connections <= 1 else "concurrent"

    try:
        inspect_eval(
            tasks,
            model=model,
            max_samples=spec.max_connections,
            max_connections=spec.max_connections,
            max_tasks=1,
            log_dir=spec.log_dir,
            log_format="json",
            log_realtime=False,
            score_display=False,
            display="plain",
            retry_on_error=2,
            fail_on_error=False,
            time_limit=spec.time_limit_s,
        )
        summary.ok = True
    except Exception:
        summary.error = traceback.format_exc(limit=12)

    written = runtime.flush_predictions()
    predictions = runtime.predictions
    summary.n_answered = written
    summary.n_crashed = sum(
        1 for row in predictions if row["error"] and "budget_exhausted" not in row["error"]
    )
    summary.n_unparseable = sum(
        1 for row in predictions if str(row["answer"]).endswith("NOANSWER")
    )
    summary.calls_used = runtime.calls_used
    summary.output_tokens = runtime.output_tokens
    summary.budget_exhausted_at = runtime.exhausted_at
    summary.questions_after_exhaustion = runtime.questions_after_exhaustion
    summary.per_component = runtime.per_component
    summary.seconds = round(time.monotonic() - started, 3)
    for row in predictions:
        if row["error"] and len(summary.first_tracebacks) < 3:
            summary.first_tracebacks.append(f"{row['item_id']}: {row['error']}")
    summary.write(spec.summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m inference_opt.eval_runner <spec.json>", file=sys.stderr)
        return 2
    spec = RunSpec.read(argv[0])
    api_key = spec.api_key or os.environ.get("VLLM_API_KEY")
    spec = replace(spec, api_key=api_key)
    try:
        with scrubbed_environment():
            summary = run(spec)
    except Exception:
        summary = RunSummary(run_id=spec.run_id, ok=False, error=traceback.format_exc())
        summary.write(spec.summary_path)
    # One machine-readable line on stdout for local/container diagnostics.
    print(json.dumps({"run_id": summary.run_id, "ok": summary.ok,
                      "error": summary.error[:400], "calls_used": summary.calls_used,
                      "summary_path": spec.summary_path}))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
