"""Grade one run's answers with Inspect. Gold targets arrive on stdin."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from inference_opt.grading.solver import replay_solver
from inference_opt.grading.spec import GradeSpec, GradeSummary
from inference_opt.scoring_specs import spec_for


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _to_sample(record: dict[str, Any], target: str | list[str]) -> Any:
    from inspect_ai.dataset import Sample

    answer_format = str(record.get("answer_format", "text"))
    benchmark = str(record.get("benchmark", ""))
    options = record.get("options") or None
    # Inspect's choice() wants one letter per correct option. inspect_evals'
    # chembench scorer wants the comma-joined string: given a list, it would
    # credit any single correct letter and reject the full answer.
    if answer_format == "mcq_multi" and isinstance(target, str) and benchmark != "chembench":
        target = [part.strip() for part in target.split(",") if part.strip()]
    return Sample(
        id=str(record["item_id"]),
        input=str(record["question"]),
        choices=list(options) if options else None,
        target=target,
        metadata={
            "benchmark": benchmark,
            "answer_format": answer_format,
            # inspect_evals' chembench scorer reads these two.
            "task_type": (
                ("mae" if answer_format == "numeric" else "multiple_choice")
                if benchmark == "chembench"
                else None
            ),
            "subset_name": record.get("category") or benchmark,
            "category": record.get("category"),
        },
    )


def grade(spec: GradeSpec, targets: dict[str, str]) -> GradeSummary:
    """Score every question once; the log holds the targets, so it stays private."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.util._display import init_display_type

    summary = GradeSummary(run_id=spec.run_id)
    records = _read_jsonl(Path(spec.questions_path))
    answers = json.loads(Path(spec.answers_path).read_text(encoding="utf-8"))
    summary.n_questions = len(records)

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("answer_format", "text")), []).append(record)
    tasks = []
    for answer_format, group in sorted(groups.items()):
        benchmark = str(group[0].get("benchmark", spec.benchmark))
        tasks.append(
            Task(
                name=f"{benchmark or 'run'}-{answer_format}",
                dataset=MemoryDataset(
                    samples=[
                        _to_sample(record, targets.get(str(record["item_id"]), ""))
                        for record in group
                    ],
                    name=f"{benchmark}:{spec.split}",
                ),
                solver=replay_solver(answers),
                scorer=spec_for(benchmark, answer_format).build_scorer(),
                fail_on_error=False,
            )
        )

    init_display_type("plain")
    try:
        inspect_eval(
            tasks,
            model="mockllm/model",
            max_tasks=1,
            log_dir=spec.log_dir,
            log_format="json",
            log_realtime=False,
            score_display=False,
            display="plain",
            fail_on_error=False,
        )
        summary.ok = True
    except Exception:
        summary.error = traceback.format_exc(limit=12)
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    spec = GradeSpec.read(argv[0])
    targets = json.loads(sys.stdin.read())
    try:
        summary = grade(spec, targets)
    except Exception:
        summary = GradeSummary(run_id=spec.run_id, error=traceback.format_exc())
    summary.write(spec.summary_path)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
