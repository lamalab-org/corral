"""Generate the level 1 and level 2 task definitions.

Every task pairs one benchmark with one student, and passes if the student, with
the agent's policy, matches a larger model of its own family run zero-shot on
the same test questions with the same prompt and scorer.

- Level 1 (accuracy): MATH Level 5 subjects; the student must match the larger
  model's test accuracy.
- Level 2 (robustness): GSM-Symbolic, GSM-P1 and GSM-P2; the student must match
  the larger model's template consistency, the share of templates whose every
  test instance is answered correctly, without its accuracy falling below its
  own zero-shot accuracy.

The pairs are those where the larger model leads by a clear margin on test (at
least 0.13 on MATH, 0.10 on GSM) and also leads on train. Baselines, the larger
models' results and model specs are read from
``inference_opt/data/baselines/<version>.json``, so run ``measure_baselines.py``
for the students and the larger models first.

Usage::

    uv run python scripts/generate_tasks.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_opt import datasets  # noqa: E402
from inference_opt.score import template_consistency  # noqa: E402

#: A deterministic namespace, so regenerating does not churn every uuid.
NAMESPACE = uuid.UUID("6f1d5a52-0f6e-4a1d-9a27-6b6c9f0e1a10")

BENCHMARK_LABELS = {
    "math_algebra": "MATH Level 5 algebra",
    "math_prealgebra": "MATH Level 5 prealgebra",
    "math_number_theory": "MATH Level 5 number theory",
    "math_counting_probability": "MATH Level 5 counting & probability",
    "math_geometry": "MATH Level 5 geometry",
    "gsm_symbolic": "GSM-Symbolic",
    "gsm_p1": "GSM-P1 (GSM-Symbolic with one added clause)",
    "gsm_p2": "GSM-P2 (GSM-Symbolic with two added clauses)",
}

#: Each student's larger model of the same family: the name in the baselines
#: file and the name shown to the agent. Llama 3.2 3B has no larger Llama 3.2
#: text model, so it is held to Llama 3.1 70B.
REFERENCES = {
    "student_a": ("reference_a", "Qwen 2.5 72B Instruct"),
    "student_b": ("reference_b", "Llama 3.1 70B Instruct"),
    "student_c": ("reference_b", "Llama 3.1 70B Instruct"),
    "student_d": ("reference_d", "Ministral 3 14B"),
    "student_e": ("reference_e", "Qwen 3.5 27B (thinking off)"),
}

#: (benchmark, student) per task.
LEVEL_1 = [
    ("math_algebra", "student_b"),
    ("math_algebra", "student_d"),
    ("math_prealgebra", "student_b"),
    ("math_prealgebra", "student_d"),
    ("math_number_theory", "student_b"),
    ("math_number_theory", "student_a"),
    ("math_counting_probability", "student_b"),
    ("math_counting_probability", "student_a"),
    ("math_geometry", "student_d"),
    ("math_geometry", "student_a"),
]
LEVEL_2 = [
    ("gsm_symbolic", "student_b"),
    ("gsm_p1", "student_b"),
    ("gsm_p2", "student_b"),
    ("gsm_symbolic", "student_d"),
    ("gsm_p1", "student_d"),
    ("gsm_p2", "student_d"),
    ("gsm_p2", "student_a"),
    ("gsm_p2", "student_e"),
    ("gsm_symbolic", "student_c"),
    ("gsm_p1", "student_c"),
]
METRICS = {1: "accuracy", 2: "template_consistency"}

#: Student calls a level 2 agent may spend in total, about 21 per train question
#: on GSM-Symbolic and GSM-P1 and 24 on GSM-P2. Level 1 gets 40 per question.
LEVEL_2_BUDGET = {"gsm_symbolic": 3200, "gsm_p1": 3200, "gsm_p2": 2400}

#: Requests one run keeps in flight per student. OpenRouter's rate limits are
#: per account and per provider: at 64 requests at once, Parasail (student_c)
#: refused about half, CoreWeave (student_b) a fifth, Phala and Mistral none;
#: Together (student_e) took 16 without refusals.
STUDENT_CONCURRENCY = {
    "student_a": 32, "student_b": 32, "student_c": 8, "student_d": 32, "student_e": 16,
}

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


def _score(result: dict, metric: str) -> float:
    """A split's score on ``metric``, from a measured result."""
    if metric == "accuracy":
        return result["accuracy"]
    per_item = result["per_item"]
    return round(template_consistency(per_item, list(per_item)), 4)


def _pass_rule(benchmark: str, model: str, metric: str, measured: dict) -> dict:
    """Pass if the student matches its larger model's zero-shot test score on ``metric``."""
    name, label = REFERENCES[model]
    result = measured["results"][name][benchmark]
    rule = {
        "kind": "reference",
        "metric": metric,
        "references": {
            model: {
                "name": label,
                "model_spec": measured["models"][name],
                "train": _score(result["train"], metric),
                "test": _score(result["test"], metric),
            }
        },
    }
    if metric == "template_consistency":
        rule["no_regression"] = True
    return rule


def _task(benchmark: str, model: str, level: int, measured: dict) -> dict:
    metric = METRICS[level]
    task_id = f"{benchmark}_{model.rsplit('_', 1)[-1]}"
    label = BENCHMARK_LABELS[benchmark]
    n_train = len(datasets.load_items(benchmark, "train"))
    n_test = len(datasets.load_items(benchmark, "test"))
    calls_per_question = 8
    own = measured["results"][model][benchmark]
    if metric == "template_consistency":
        description = (
            f"Build and submit an inference-time policy that makes the student "
            f"answer {label} consistently: right on every renamed and renumbered "
            "instance of a problem, not just some."
        )
    else:
        description = f"Build and submit an inference-time policy for {label}."
    config = {
        "benchmark": benchmark,
        "models": [model],
        "joint": False,
        "n_train": n_train,
        "n_test": n_test,
        "baselines": {model: own["test"]["accuracy"]},
        "baselines_train": {model: own["train"]["accuracy"]},
        "model_specs": {model: measured["models"][model]},
        "student_options": {model: measured.get("student_options", {}).get(model, {})},
        "base_urls": {},
        "final_max_student_calls": n_test * 12,
        "student_concurrency": {model: STUDENT_CONCURRENCY[model]},
        "scale": 1.0,
        "pass_rule": _pass_rule(benchmark, model, metric, measured),
        "budget": {
            "max_experiments": 20,
            "max_debug_runs": 10,
            "max_student_calls": LEVEL_2_BUDGET.get(
                benchmark, 20 * n_train * calls_per_question // 4
            ),
            "max_reveals": 4,
            "max_probe_calls": 40,
            "reveal_batch": 5,
            "max_submissions": 1,
        },
    }
    if metric == "template_consistency":
        config["baseline_template_consistency_train"] = {
            model: _score(own["train"], metric)
        }
    return {
        "id": task_id,
        "uuid": str(uuid.uuid5(NAMESPACE, f"inference-opt-l{level}-{task_id}")),
        "name": f"Improve {benchmark}",
        "description": description,
        "tools": TOOLS,
        "submission_format": (
            "Any text; the score comes from the policy submitted with submit_policy."
        ),
        "initial_input": config,
    }


def main() -> None:
    baselines = ROOT / "inference_opt" / "data" / "baselines" / f"{datasets.DATASET_VERSION}.json"
    measured = json.loads(baselines.read_text(encoding="utf-8"))
    for level, roster in ((1, LEVEL_1), (2, LEVEL_2)):
        tasks = [_task(benchmark, model, level, measured) for benchmark, model in roster]
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
            ((model, reference),) = rule["references"].items()
            own_test = _score(
                measured["results"][model][config["benchmark"]]["test"], rule["metric"]
            )
            print(f"   {task['id']:30} {rule['metric']:20} student {own_test:.2f} -> "
                  f"{reference['name']} {reference['test']:.2f}   "
                  f"({config['n_train']}/{config['n_test']} questions)")


if __name__ == "__main__":
    main()
