"""Generate the level 1 and level 2 task definitions.

Level 1 pairs one MATH subset with one student. The task passes if the student,
with the agent's policy, matches the zero-shot test accuracy of a larger model
of its own family (``REFERENCES``). The pairs are those where that model leads
the student by at least 0.13 on test and also leads on train.

Level 2 keeps the earlier design, two students per task against a fixed bar
(``--pass-rule``), until it is replaced.

Baselines, reference results and model specs are read from
``inference_opt/data/baselines/<version>.json``, so run ``measure_baselines.py``
for the students and the reference models first.

Usage::

    uv run python scripts/generate_tasks.py [--pass-rule threshold|headroom]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_opt import datasets  # noqa: E402
from inference_opt.score import DEFAULT_PASS_RULE  # noqa: E402

#: A deterministic namespace, so regenerating does not churn every uuid.
NAMESPACE = uuid.UUID("6f1d5a52-0f6e-4a1d-9a27-6b6c9f0e1a10")

BENCHMARK_LABELS = {
    "math_algebra": "MATH Level 5 algebra",
    "math_prealgebra": "MATH Level 5 prealgebra",
    "math_number_theory": "MATH Level 5 number theory",
    "math_counting_probability": "MATH Level 5 counting & probability",
    "math_geometry": "MATH Level 5 geometry",
}

#: Each student's larger model of the same family: the name in the baselines
#: file and the name shown to the agent.
REFERENCES = {
    "student_a": ("reference_a", "Qwen 2.5 72B Instruct"),
    "student_b": ("reference_b", "Llama 3.1 70B Instruct"),
    "student_d": ("reference_d", "Ministral 3 14B"),
}

#: (benchmark, students) per task.
LEVEL_1 = [
    ("math_algebra", ["student_b"]),
    ("math_algebra", ["student_d"]),
    ("math_prealgebra", ["student_b"]),
    ("math_prealgebra", ["student_d"]),
    ("math_number_theory", ["student_b"]),
    ("math_number_theory", ["student_a"]),
    ("math_counting_probability", ["student_b"]),
    ("math_counting_probability", ["student_a"]),
    ("math_geometry", ["student_d"]),
    ("math_geometry", ["student_a"]),
]
LEVEL_2 = [
    ("math_algebra", ["student_a", "student_b"]),
    ("math_algebra", ["student_c", "student_d"]),
    ("math_prealgebra", ["student_a", "student_d"]),
    ("math_prealgebra", ["student_b", "student_c"]),
    ("math_number_theory", ["student_b", "student_c"]),
    ("math_number_theory", ["student_a", "student_c"]),
    ("math_counting_probability", ["student_c", "student_d"]),
    ("math_counting_probability", ["student_b", "student_d"]),
    ("math_geometry", ["student_c", "student_d"]),
    ("math_geometry", ["student_a", "student_d"]),
]

#: ``threshold`` (default): every student reaches the mean MATH score of the 50
#: models on a public leaderboard (``data/reference/math_leaderboard.csv``). ``headroom``: every
#: student closes at least half the gap between its baseline and a perfect score.
PASS_RULES = {"threshold": DEFAULT_PASS_RULE, "headroom": {"kind": "headroom", "min_closed": 0.5}}


def _reference_rule(benchmark: str, models: list[str], measured: dict) -> dict:
    """Pass if every student matches its larger model's zero-shot test accuracy."""
    references = {}
    for model in models:
        name, label = REFERENCES[model]
        result = measured["results"][name][benchmark]
        references[model] = {
            "name": label,
            "model_spec": measured["models"][name],
            "test_accuracy": result["test"]["accuracy"],
            "train_accuracy": result["train"]["accuracy"],
        }
    return {"kind": "reference", "references": references}

#: Requests one run keeps in flight per student. OpenRouter's rate limits are
#: per account and per provider: at 64 requests at once, Parasail (student_c)
#: refused about half, CoreWeave (student_b) a fifth, Phala and Mistral none.
STUDENT_CONCURRENCY = {"student_a": 32, "student_b": 32, "student_c": 8, "student_d": 32}

TOOLS = [
    "get_baseline",
    "reveal_examples",
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


def _task(
    benchmark: str, models: list[str], level: int, measured: dict, pass_rule: dict
) -> dict:
    joint = len(models) > 1
    suffix = "".join(model.rsplit("_", 1)[-1] for model in models)
    task_id = f"{benchmark}_{suffix}"
    calls_per_question = 8
    label = BENCHMARK_LABELS[benchmark]

    if joint:
        name = f"Joint {benchmark} improvement on two student models"
        description = (
            f"Build and submit one inference-time policy for {label}. The same "
            "policy is evaluated against two frozen student models. It must help "
            "both models to pass."
        )
    else:
        name = f"Improve {benchmark}"
        description = f"Build and submit an inference-time policy for {label}."

    results = measured["results"]
    return {
        "id": task_id,
        "uuid": str(uuid.uuid5(NAMESPACE, f"inference-opt-l{level}-{task_id}")),
        "name": name,
        "description": description,
        "tools": TOOLS,
        "submission_format": (
            "Any text; the score comes from the policy submitted with submit_policy."
        ),
        "initial_input": {
            "benchmark": benchmark,
            "models": models,
            "joint": joint,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "baselines": {m: results[m][benchmark]["test"]["accuracy"] for m in models},
            "baselines_train": {
                m: results[m][benchmark]["train"]["accuracy"] for m in models
            },
            "model_specs": {m: measured["models"][m] for m in models},
            "student_options": {
                m: measured.get("student_options", {}).get(m, {}) for m in models
            },
            "base_urls": {},
            "final_max_student_calls": N_TEST * 12,
            "student_concurrency": {m: STUDENT_CONCURRENCY[m] for m in models},
            "scale": 1.0,
            "pass_rule": pass_rule,
            "budget": {
                "max_experiments": 20,
                "max_debug_runs": 10,
                "max_student_calls": 20 * N_TRAIN * calls_per_question // 4,
                "max_reveals": 4,
                "max_probe_calls": 40,
                "reveal_batch": 5,
                "max_submissions": 1,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-rule", choices=sorted(PASS_RULES), default="threshold")
    args = parser.parse_args()
    baselines = ROOT / "inference_opt" / "data" / "baselines" / f"{datasets.DATASET_VERSION}.json"
    measured = json.loads(baselines.read_text(encoding="utf-8"))
    for level, roster in ((1, LEVEL_1), (2, LEVEL_2)):
        tasks = [
            _task(
                benchmark, models, level, measured,
                _reference_rule(benchmark, models, measured)
                if level == 1 else PASS_RULES[args.pass_rule],
            )
            for benchmark, models in roster
        ]
        target = ROOT / "environments" / f"level_{level}" / "tasks_json"
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("*.json"):
            old.unlink()
        for index, task in enumerate(tasks, start=1):
            path = target / f"task_{index}.json"
            path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"level {level}: wrote {len(tasks)} task files to {target.relative_to(ROOT)}")
        for task in tasks:
            config = task["initial_input"]
            rule = config["pass_rule"]
            bar = (
                {m: r["test_accuracy"] for m, r in rule["references"].items()}
                if rule["kind"] == "reference" else rule.get("min_accuracy")
            )
            print(f"   {task['id']:32} test baselines {config['baselines']}  bar {bar}")


if __name__ == "__main__":
    main()
