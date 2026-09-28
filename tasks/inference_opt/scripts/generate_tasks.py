"""Generate the level 1 and level 2 task definitions.

Usage::

    uv run python scripts/generate_tasks.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: A deterministic namespace, so regenerating does not churn every uuid.
NAMESPACE = uuid.UUID("6f1d5a52-0f6e-4a1d-9a27-6b6c9f0e1a10")

BENCHMARKS = ("gsm8k", "mmlu_pro", "gpqa_diamond", "bbh", "chembench", "arc_challenge")
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "mmlu_pro": "MMLU-Pro",
    "gpqa_diamond": "GPQA-Diamond",
    "bbh": "BIG-Bench Hard",
    "chembench": "ChemBench",
    "arc_challenge": "ARC-Challenge",
}

LEVEL_1_MODELS = [["student_a"], ["student_b"]]
LEVEL_2_MODELS = [["student_a", "student_b"], ["student_c", "student_d"]]

TOOLS = [
    "get_baseline",
    "reveal_train_questions",
    "query_student",
    "dry_run_policy",
    "evaluate_candidate",
    "inspect_failures",
    "compare_runs",
    "get_budget",
    "submit_policy",
]

N_TRAIN = 30
N_TEST = 30


def _task(benchmark: str, models: list[str], level: int) -> dict:
    joint = len(models) > 1
    suffix = "".join(model.rsplit("_", 1)[-1] for model in models)
    task_id = f"{benchmark}_{suffix}"
    calls_per_question = 8
    final_calls = N_TEST * 12

    if joint:
        name = f"Joint {benchmark} improvement on two student models"
        description = (
            f"Build and submit one inference-time policy for "
            f"{BENCHMARK_LABELS[benchmark]}. The same policy is evaluated against "
            "two frozen student models. You are scored on the smaller improvement, "
            "so the policy must help both models."
        )
    else:
        name = f"Improve {benchmark}"
        description = (
            f"Build and submit an inference-time policy for "
            f"{BENCHMARK_LABELS[benchmark]}."
        )

    return {
        "id": task_id,
        "uuid": str(uuid.uuid5(NAMESPACE, f"inference-opt-l{level}-{task_id}")),
        "name": name,
        "description": description,
        "tools": TOOLS,
        "submission_format": (
            "The literal string 'submission.json' - stage it with submit_policy first."
        ),
        "initial_input": {
            "benchmark": benchmark,
            "models": models,
            "joint": joint,
            "policy_api": "primitive",
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            # Filled in by scripts/measure_baselines.py --write-tasks.
            "baselines": dict.fromkeys(models, 0.0),
            "baselines_train": dict.fromkeys(models, 0.0),
            "baseline_items": {},
            "model_specs": {},
            "base_urls": {},
            "final_max_student_calls": final_calls,
            "final_setup_calls": 0,
            "setup_calls": 20,
            "scale": 1.0,
            "budget": {
                "max_experiments": 20,
                "max_debug_runs": 10,
                "max_student_calls": 20 * N_TRAIN * calls_per_question // 4,
                "max_reveals": 4,
                "max_probe_calls": 40,
                "reveal_batch": 5,
            },
        },
    }


def main() -> None:
    for level, rosters in ((1, LEVEL_1_MODELS), (2, LEVEL_2_MODELS)):
        tasks = [
            _task(benchmark, models, level)
            for benchmark in BENCHMARKS
            for models in rosters
        ]
        target = ROOT / "environments" / f"level_{level}" / "tasks_json"
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("tasks.json"):
            old.unlink()
        for index, task in enumerate(tasks, start=1):
            path = target / f"task_{index}.json"
            path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"level {level}: wrote {len(tasks)} task files to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
