#!/usr/bin/env python3
"""Generate Level 2 Task 04 artifacts and scoring metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l2_t04_hsns_population_classification"
UUID = "7e3d1b42-8a69-4c5f-b2d7-6f0a913ce854"
SEED = 20260927
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_04"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_04.json"

ITEMS = C.HSNS_ITEMS
F1 = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
F2 = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
FACTOR_OF = {item: ("F1" if item in F1 else "F2") for item in ITEMS}
LOADINGS = {
    "HSNS1": 0.55,
    "HSNS4": 0.42,
    "HSNS5": 0.70,
    "HSNS6": 0.58,
    "HSNS8": 0.68,
    "HSNS10": 0.66,
    "HSNS2": 0.70,
    "HSNS3": 0.58,
    "HSNS7": 0.69,
    "HSNS9": 0.52,
}

N_REFERENCE = 3_000
CASE_SIZE = 300
CASE_SOURCES = ["population_a", "population_c", "population_d", "population_e", "population_f"]
COMPATIBILITY_MARGIN = 0.04
RESIDUAL_PAIR = ("HSNS5", "HSNS10")

PROMPT = (
    "You are given large response samples from six anonymised HSNS populations and five "
    "case samples, each containing {case_size} respondents drawn from one of those populations. "
    "The population identifiers say nothing about how the populations differ.\n\n"
    "Work out what distinguishes the reference populations, then decide which of them each "
    "case sample could have come from. A population is compatible when the case's responses "
    "are no more than {margin} units of log-likelihood per respondent less likely under it "
    "than under the best-fitting population. Some cases have one compatible population and "
    "some have more than one; do not force a unique answer.\n"
).format(case_size=CASE_SIZE, margin=COMPATIBILITY_MARGIN)

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "classifications": {
    "case_1": ["population_a"],
    "case_2": ["population_b", "population_c"]
  }
}

Include every case exactly once. Each value is a list of compatible population
identifiers. Order does not matter. No model or working is required.
"""


def ordinal_block(rng, n, factors, phi, loadings, residual_pair=None, residual=0.0):
    eta = rng.multivariate_normal(np.zeros(len(factors)), phi, n)
    index = {name: i for i, name in enumerate(factors)}
    shared = rng.normal(size=n)
    out = {}
    for item in ITEMS:
        score = sum(loading * eta[:, index[factor]] for factor, loading in loadings[item])
        explained = sum(loading**2 for _, loading in loadings[item])
        if residual_pair and item in residual_pair:
            score = score + np.sqrt(residual) * shared
            explained += residual
        noise = rng.normal(0, np.sqrt(max(1 - explained, 0.08)), n)
        out[item] = C.categorize(score + noise, C.THRESHOLDS[item])
    return pd.DataFrame(out)[ITEMS]


def two_factor(rng, n, loading_scale=1.0, phi=0.35, cross_loading=None):
    loading = {
        item: [(factor, value * loading_scale)]
        for item, (factor, value) in {
            item: (FACTOR_OF[item], value) for item, value in LOADINGS.items()
        }.items()
    }
    if cross_loading:
        item, factor, value = cross_loading
        loading[item].append((factor, value))
    return ordinal_block(rng, n, ["F1", "F2"], [[1, phi], [phi, 1]], loading)


def one_factor(rng, n):
    return ordinal_block(rng, n, ["G"], [[1.0]], {item: [("G", 0.62)] for item in ITEMS})


def bifactor(rng, n):
    loading = {item: [("G", 0.25), (FACTOR_OF[item], 0.62)] for item in ITEMS}
    return ordinal_block(rng, n, ["G", "F1", "F2"], np.eye(3), loading)


def residual_dependence(rng, n):
    return ordinal_block(
        rng,
        n,
        ["F1", "F2"],
        [[1, 0.35], [0.35, 1]],
        {item: [(FACTOR_OF[item], value)] for item, value in LOADINGS.items()},
        residual_pair=RESIDUAL_PAIR,
        residual=0.70,
    )


POPULATIONS = {
    "population_a": lambda rng, n: two_factor(rng, n),
    "population_b": one_factor,
    "population_c": bifactor,
    "population_d": lambda rng, n: two_factor(
        rng, n, loading_scale=1.10, phi=0.72, cross_loading=("HSNS9", "F1", 0.35)
    ),
    "population_e": lambda rng, n: two_factor(rng, n, loading_scale=0.99, phi=0.37),
    "population_f": residual_dependence,
}


def reference_models(references):
    return {
        name: (
            frame[ITEMS].values.astype(float).mean(axis=0),
            np.cov(frame[ITEMS].values.T.astype(float)) + np.eye(len(ITEMS)) * 1e-6,
        )
        for name, frame in references.items()
    }


def log_likelihood(case, models):
    from scipy.stats import multivariate_normal

    values = case[ITEMS].values.astype(float)
    return {
        name: float(
            multivariate_normal(mean, covariance, allow_singular=True).logpdf(values).mean()
        )
        for name, (mean, covariance) in models.items()
    }


def compatible(case, models):
    scores = log_likelihood(case, models)
    best = max(scores.values())
    return sorted(name for name, value in scores.items() if best - value <= COMPATIBILITY_MARGIN)


def build_references(rng):
    return {name: recipe(rng, N_REFERENCE) for name, recipe in POPULATIONS.items()}


def build_cases(rng):
    return {
        f"case_{i}": POPULATIONS[source](rng, CASE_SIZE) for i, source in enumerate(CASE_SOURCES, 1)
    }


def write_codebook(path):
    path.write_text(
        "# Codebook - anonymised HSNS population comparison\n\n"
        "Every file is tab-separated. Responses run from 1 (Disagree) to 5 (Agree); all responses are complete.\n\n"
        "The reference files hold large samples from six populations, named only by identifier. How they differ is not supplied. `cases.csv` holds five case samples of "
        f"{CASE_SIZE} respondents each.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n"
    )


def build_task_json(shas):
    return [
        {
            "id": TASK_ID,
            "name": "Which HSNS population did each case sample come from?",
            "uuid": UUID,
            "keywords": ["psychometrics", "population classification", "HSNS", "model comparison"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                "reference_datasets": {name: f"{name}.csv" for name in POPULATIONS},
                "cases": "cases.csv",
                "codebook": "codebook.md",
                "data_sha256": shas,
            },
            "tools": [],
            "scoring_function": "score_population_classification",
            "scoring_params": {
                "task_type": "population_classification",
                "truth_path": "artifacts/level_2/task_04/truth.json",
                "items": ITEMS,
                "populations": list(POPULATIONS),
                "reference_datasets": {name: f"{name}.csv" for name in POPULATIONS},
                "cases": "cases.csv",
                "compatibility_margin": COMPATIBILITY_MARGIN,
                "case_size": CASE_SIZE,
            },
        }
    ]


def candidate_submissions():
    expected = json.loads((OUT_DIR / "truth.json").read_text())["scored"]["classifications"]
    overlapping = next(k for k, value in expected.items() if len(value) > 1)
    return {
        "correct": {"classifications": expected},
        "forces a unique answer everywhere": {
            "classifications": {k: [v[0]] for k, v in expected.items()}
        },
        "names every population for every case": {
            "classifications": {k: list(POPULATIONS) for k in expected}
        },
        "misses the one genuine overlap": {
            "classifications": {k: ([v[0]] if k == overlapping else v) for k, v in expected.items()}
        },
    }


def verify(references, expected):
    models = reference_models(references)
    rng = np.random.default_rng(SEED + 7717)
    trials = 25
    stability, contains = {}, {}
    for name, recipe in POPULATIONS.items():
        sets = [tuple(compatible(recipe(rng, CASE_SIZE), models)) for _ in range(trials)]
        modal = max(set(sets), key=sets.count)
        stability[name] = sets.count(modal) / trials
        contains[name] = sum(name in value for value in sets) / trials
    close = set(compatible(references["population_a"].iloc[:CASE_SIZE], models))
    unique_count = sum(
        len(compatible(POPULATIONS[name](rng, CASE_SIZE), models)) == 1 for name in POPULATIONS
    )
    return C.report(
        [
            ("every held-out case includes its own population", min(contains.values()) == 1.0),
            ("the compatible set is stable across redraws", min(stability.values()) >= 0.90),
            ("the close populations remain inseparable", close == {"population_a", "population_e"}),
            ("the other populations are identified uniquely", unique_count == 4),
            (
                "the delivered cases include unique and overlapping answers",
                0 < sum(len(v) > 1 for v in expected.values()) < len(expected),
            ),
        ]
    )


def naive(references, cases, expected):
    means = {
        name: frame[ITEMS].values.astype(float).mean(axis=0) for name, frame in references.items()
    }
    wrong = 0
    for case_id, frame in cases.items():
        target = frame[ITEMS].values.astype(float).mean(axis=0)
        closest = min(means, key=lambda name: float(np.abs(means[name] - target).mean()))
        wrong += closest not in expected[case_id]
    rng = np.random.default_rng(SEED + 4242)
    hits = 0
    for name, recipe in POPULATIONS.items():
        for _ in range(20):
            target = recipe(rng, CASE_SIZE)[ITEMS].values.astype(float).mean(axis=0)
            closest = min(
                means, key=lambda candidate: float(np.abs(means[candidate] - target).mean())
            )
            hits += closest == name
    accuracy = hits / (len(POPULATIONS) * 20)
    return C.report(
        [
            ("matching item means is near chance", accuracy < 2 / len(POPULATIONS)),
            ("matching item means misclassifies delivered cases", wrong > 0),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    references = build_references(rng)
    cases = build_cases(rng)
    models = reference_models(references)
    expected = {case_id: compatible(frame, models) for case_id, frame in cases.items()}
    if action == "verify":
        return verify(references, expected)
    if action == "naive":
        return naive(references, cases, expected)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, frame in references.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", sep="\t", index=False)
        shas[f"{name}.csv"] = C.file_sha256(OUT_DIR / f"{name}.csv")
    delivered = pd.concat(
        [frame.assign(case_id=case_id) for case_id, frame in cases.items()], ignore_index=True
    )
    delivered[["case_id"] + ITEMS].to_csv(OUT_DIR / "cases.csv", sep="\t", index=False)
    shas["cases.csv"] = C.file_sha256(OUT_DIR / "cases.csv")
    write_codebook(OUT_DIR / "codebook.md")
    truth = {
        "task_id": TASK_ID,
        "scored": {"classifications": expected},
        "generative_parameters": {
            "case_size": CASE_SIZE,
            "compatibility_margin": COMPATIBILITY_MARGIN,
            "case_sources": dict(zip(expected, CASE_SOURCES)),
        },
        "provenance": C.provenance(Path(__file__).name, SEED, len(delivered), shas["cases.csv"]),
    }
    (OUT_DIR / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    TASK_JSON.parent.mkdir(parents=True, exist_ok=True)
    TASK_JSON.write_text(json.dumps(build_task_json(shas), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
