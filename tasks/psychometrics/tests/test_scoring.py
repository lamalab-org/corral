#!/usr/bin/env python3
"""Check that each task scores the way it is meant to.

For every task: the right answer scores 1.0, every wrong one scores 0.0 and says
which stage it failed at, and malformed or dishonest submissions are rejected.

    uv run --with numpy --with pandas --with scipy --with semopy \
      python tests/test_scoring.py
"""

from __future__ import annotations

import importlib
import json
import logging
import warnings

import pandas as pd
import semopy

from corral_psychometrics import paths
from corral_psychometrics.score import score_model_criteria

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

_BIASED = ["HSNS1", "HSNS10", "HSNS5", "HSNS8"]
ROOT = paths.task_root()

# (level, task, the submission that must score 1.0). Every file a task needs
# follows from that numbering -- see corral_psychometrics.paths.
TASKS = [
    (1, 1, "two_correlated_factors"),
    (1, 2, "bifactor_general_plus_specifics"),
    (1, 3, "unidimensional_with_correlated_residuals"),
    (1, 4, "DD, bifactor (CORRECT)"),
    (1, 5, "correct"),
    (1, 6, "joint latent model (CORRECT)"),
    (1, 7, "correct"),
    (1, 8, "correct"),
    (1, 9, "correct"),
    (1, 10, "correct"),
    (2, 1, "correct"),
    (2, 2, "correct"),
    (2, 3, "correct"),
    (2, 4, "correct"),
    (2, 5, "correct"),
    (2, 6, "correct"),
    (2, 7, "correct"),
    (2, 8, "correct"),
    (2, 9, "correct"),
    (2, 10, "correct"),
]


def load_generator(level, number):
    """Import the generator that produced one task's data."""
    return importlib.import_module(paths.find(level, number).generator)


def build_submission(spec, X, items):
    """The submission an agent would make if it really did fit this model."""
    if "gender" in items:
        return _group_submission(spec, X, items, _BIASED)
    model = semopy.Model(spec)
    model.fit(X[items])
    ins = model.inspect(std_est=True)
    load = ins[ins.op == "~"].copy()
    load["abs"] = pd.to_numeric(load["Est. Std"], errors="coerce").abs()
    best = load.loc[load.groupby("lval")["abs"].idxmax()]
    cov = ins[
        (ins.op == "~~")
        & (ins.lval != ins.rval)
        & ins.lval.isin(["F1", "F2"])
        & ins.rval.isin(["F1", "F2"])
    ]
    phi = float(pd.to_numeric(cov["Est. Std"], errors="coerce").iloc[0]) if len(cov) else None
    return {
        "model_syntax": spec,
        "loadings": {r.lval: round(float(r.abs), 3) for r in best.itertuples()},
        "factor_correlation": phi,
    }


def _group_submission(spec, X, items, truth=None):
    """A submission for the tasks that compare two groups."""
    used = [c for c in X.columns if c in spec]
    model = semopy.Model(spec)
    model.fit(X[used + ["gender"]] if "gender" not in used else X[used])
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == "gender") & ~ins.lval.isin(X.columns)]
    value = float(pd.to_numeric(rows["Est. Std"], errors="coerce").iloc[0]) if len(rows) else 0.0
    return {
        "model_syntax": spec,
        "latent_difference": round(value, 4),
        "biased_items": list(truth or []),
    }


def run_task(level, number, expected_winner):
    gen = load_generator(level, number)
    params = json.loads(paths.find(level, number).definition.read_text())[0]["scoring_params"]
    if params.get("task_type") == "population_classification":
        return run_population_task(gen, params, expected_winner)
    if params.get("task_type") == "gender_item_integrity":
        return run_gender_integrity_task(gen, params, expected_winner)
    if params.get("task_type") == "behavioral_validity":
        return run_behavioral_validity_task(gen, params, expected_winner)
    if params.get("task_type") == "misfit_replication":
        return run_misfit_replication_task(gen, params, expected_winner)
    if params.get("task_type") == "adaptive_bank_choice":
        return run_adaptive_bank_choice_task(gen, params, expected_winner)
    if params.get("task_type") == "model_identification":
        return run_model_identification_task(gen, params, expected_winner)
    items = params["items"]
    data_path = paths.resolve(params["data_dir"]) / params["dataset"]
    data = pd.read_csv(data_path, sep="\t")
    keep = params["syntax_whitelist"]["items"]
    for col, val in params.get("subset", {}).items():
        data = data[data[col].isin(val)] if isinstance(val, list) else data[data[col] == val]
    X = data[[c for c in keep if c in data.columns]]
    X = X[(X != 0).all(axis=1)].astype(float)

    failures = []
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    # Some tasks answer with a model, others with a judgement; ask the generator.
    if hasattr(gen, "candidate_submissions"):
        cases = gen.candidate_submissions(X)
    else:
        cases = {
            name: build_submission(spec, X, items) for name, spec in gen.candidate_models().items()
        }
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        ok = result["score_binary"] == want
        failures += [] if ok else [f"{name}: expected {want}, got {result['score_binary']}"]
        print(
            f"  {name:38s} {result['score_binary']:6.1f} "
            f"{result['score_partial']:7.2f}  {result['reason']}"
        )

    good = cases.get("correct") or build_submission(gen.reference_syntax(), X, items)
    pooled_X = pd.read_csv(data_path, sep="\t")
    if "gender" in items:
        pooled_X = pooled_X[pooled_X.gender.isin([1, 2])]
    pooled_X = pooled_X[items]
    pooled_X = pooled_X[(pooled_X != 0).all(axis=1)].astype(float)
    key = (
        "item_quality"
        if "item_quality" in good
        else "scoring"
        if "scoring" in good
        else "replication"
        if "replication" in good
        else "correlations"
        if "correlations" in good
        else "comparisons"
        if "comparisons" in good
        else "measurement_conclusion"
        if "measurement_conclusion" in good
        else "recommendation"
        if "recommendation" in good
        else "affected_items"
        if "affected_items" in good
        else "holdout_conclusions"
        if "holdout_conclusions" in good
        else "diagnosis"
        if "diagnosis" in good
        else "latent_difference"
        if "gender" in items
        else "loadings"
    )
    fake = (
        {k: "sound" for k in good["item_quality"]}
        if key == "item_quality"
        else {k: "total_only" for k in good["scoring"]}
        if key == "scoring"
        else {i: {c: "exact" for c in v} for i, v in good["replication"].items()}
        if key == "replication"
        else [[a, b, 0.3] for a, b, _ in good["correlations"]]
        if key == "correlations"
        else {k: True for k in good["comparisons"]}
        if key == "comparisons"
        else {k: True for k in good["measurement_conclusion"]}
        if key == "measurement_conclusion"
        else {k: "supported" for k in good["recommendation"]}
        if key == "recommendation"
        else list(items)
        if key == "affected_items"
        else {k: "generalizes" for k in good["holdout_conclusions"]}
        if key == "holdout_conclusions"
        else "repeated_delivery_only"
        if key == "diagnosis"
        else 0.9
        if key == "latent_difference"
        else {k: 0.55 for k in items}
    )
    adversarial = {
        **(
            {}
            if key in ("comparisons", "correlations", "replication", "scoring", "item_quality")
            else {
                "pooled sample (no US filter)": build_submission(
                    gen.reference_syntax(), pooled_X, items
                )
            }
        ),
        f"{key} fabricated": {**good, key: fake},
        f"{key} omitted": {k: v for k, v in good.items() if k != key},
        "code injection": {**good, "model_syntax": "import os"},
        "unknown variable": {**good, "model_syntax": f"F1 =~ {'+'.join(items)}+GHOST"},
        "not JSON": "the model is two factors",
    }
    print(f"  {'-' * 70}")
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        ok = result["score_binary"] == 0.0
        failures += [] if ok else [f"adversarial {name} scored {result['score_binary']}"]
        print(
            f"  {name:38s} {result['score_binary']:6.1f} "
            f"{result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_population_task(gen, params, expected_winner):
    failures = []
    cases = gen.candidate_submissions()
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    good = cases["correct"]
    adversarial = {
        "missing classifications": {},
        "unknown population": {
            "classifications": {**good["classifications"], "person_1": ["population_z"]}
        },
        "extra person": {
            "classifications": {**good["classifications"], "person_6": ["population_a"]}
        },
        "not JSON": "the populations are unclear",
    }
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        if result["score_binary"] != 0.0:
            failures.append(f"adversarial {name} scored {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_gender_integrity_task(gen, params, expected_winner):
    failures = []
    cases = gen.candidate_submissions()
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    good = cases["correct"]
    adversarial = {
        "item diagnoses omitted": {k: v for k, v in good.items() if k != "item_diagnoses"},
        "comparison omitted": {k: v for k, v in good.items() if k != "comparison"},
        "model omitted": {k: v for k, v in good.items() if k != "model_syntax"},
        "not JSON": "the comparison is uncertain",
    }
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        if result["score_binary"] != 0.0:
            failures.append(f"adversarial {name} scored {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_behavioral_validity_task(gen, params, expected_winner):
    failures = []
    cases = gen.candidate_submissions()
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    good = cases["correct"]
    adversarial = {
        "association omitted": {k: v for k, v in good.items() if k != "association"},
        "unstable items omitted": {k: v for k, v in good.items() if k != "unstable_items"},
        "model omitted": {k: v for k, v in good.items() if k != "model_syntax"},
        "not JSON": "the association is uncertain",
    }
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        if result["score_binary"] != 0.0:
            failures.append(f"adversarial {name} scored {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_misfit_replication_task(gen, params, expected_winner):
    failures = []
    cases = gen.candidate_submissions()
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    good = cases["correct"]
    adversarial = {
        "findings omitted": {k: v for k, v in good.items() if k != "findings"},
        "replication omitted": {k: v for k, v in good.items() if k != "replication"},
        "model omitted": {k: v for k, v in good.items() if k != "model_syntax"},
        "not JSON": "the modifications are uncertain",
    }
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        if result["score_binary"] != 0.0:
            failures.append(f"adversarial {name} scored {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_adaptive_bank_choice_task(gen, params, expected_winner):
    failures = []
    cases = gen.candidate_submissions()
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    good = cases["correct"]
    adversarial = {
        "bank choice omitted": {k: v for k, v in good.items() if k != "bank_choice"},
        "dependent pairs omitted": {k: v for k, v in good.items() if k != "dependent_pairs"},
        "vendor flags taken as given": {**good, "unsupported_vendor_flags": []},
        "recommendation omitted": {k: v for k, v in good.items() if k != "recommendation"},
        "not JSON": "the screened bank seems preferable",
    }
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        if result["score_binary"] != 0.0:
            failures.append(f"adversarial {name} scored {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def run_model_identification_task(gen, params, expected_winner):
    cases = dict(gen.candidate_submissions())
    good = cases["correct"]
    cases["one dataset omitted"] = {"assignments": dict(list(good["assignments"].items())[:-1])}
    cases["not JSON"] = "the models are unclear"
    failures = []
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        if result["score_binary"] != want:
            failures.append(f"{name}: expected {want}, got {result['score_binary']}")
        print(
            f"  {name:38s} {result['score_binary']:6.1f} {result['score_partial']:7.2f}  {result['reason']}"
        )
    return failures


def main(tasks=TASKS):
    failures = []
    for task in tasks:
        failures += run_task(*task)
    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("all tasks score as intended")
    return 0


def test_all_tasks_score_as_intended():
    """Expose the integration suite to pytest-based CI as one test."""
    assert main() == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int)
    parser.add_argument("--tasks", type=int, nargs="+")
    args = parser.parse_args()
    selected = [
        task
        for task in TASKS
        if (args.level is None or task[0] == args.level)
        and (args.tasks is None or task[1] in args.tasks)
    ]
    raise SystemExit(main(selected))
