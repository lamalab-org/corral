#!/usr/bin/env python3
"""Generate Level 2 Task 04 artifacts and scoring metadata.

Four HSNS populations, each with its own structure, and three case samples
drawn from some of them.

    population_a   two correlated factors
    population_b   two correlated factors plus a residual covariance on HSNS5, HSNS10
    population_c   the same model as population_a, sampled independently
    population_d   two factors, stronger loadings, phi 0.72, HSNS9 cross-loads

The agent describes each population with a model and places each case. a and c
are one population under two identifiers, so a case from either could have come
from both. b and d each differ from a by one detail, which the submitted model
has to carry: every model is refitted on its population's reference sample and
has to fit as well as the generating model does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t04_hsns_population_classification"
UUID = "7e3d1b42-8a69-4c5f-b2d7-6f0a913ce854"
SEED = 20260927
TASK = paths.task(__file__)

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
CASE_SOURCES = ["population_b", "population_c", "population_d"]
# The answer comes from the generating models, not from this margin. verify()
# uses it to confirm the data support that answer: a population is compatible
# when the case's mean Gaussian log-likelihood under it is within this margin of
# the best-fitting population's. The agent is not told it.
COMPATIBILITY_MARGIN = 0.04
RESIDUAL_PAIR = ("HSNS5", "HSNS10")

PROMPT = (
    "You have large HSNS response samples from four reference populations, `population_a` to "
    "`population_d`, and three case samples of {case_size} respondents, each drawn from one of "
    "those populations. The population identifiers carry no information.\n\n"
    "Describe the structure of each population with a measurement model, then list, for each "
    "case, every population it could have come from.\n"
).format(case_size=CASE_SIZE)

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "models": {
    "<population_id>": "<semopy model for that population>"
  },
  "classifications": {
    "<case_id>": ["<population_id>", ...]
  }
}

Give a complete, runnable model for every population so its fit on that
population's reference sample can be reproduced.
Include every case exactly once; each value is a list of compatible population
identifiers, in any order.
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


# a and c share one model; the rest are separable.
POPULATIONS = {
    "population_a": two_factor,
    "population_b": residual_dependence,
    "population_c": two_factor,
    "population_d": lambda rng, n: two_factor(
        rng, n, loading_scale=1.10, phi=0.72, cross_loading=("HSNS9", "F1", 0.35)
    ),
}
STRUCTURES = {
    "population_a": "two correlated factors, phi .35",
    "population_b": f"population_a plus residual covariance .70 on {RESIDUAL_PAIR}",
    "population_c": "the same model as population_a, sampled independently",
    "population_d": "two factors, loadings x1.10, phi .72, HSNS9 cross-loads .35 on F1",
}

TWO_FACTORS = f"F1 =~ {'+'.join(F1)}\nF2 =~ {'+'.join(F2)}"
# The generating model of each population, in semopy syntax: the floor every
# submitted model is held to on that population's reference sample.
MODEL_SYNTAX = {
    "population_a": TWO_FACTORS,
    "population_b": f"{TWO_FACTORS}\n{RESIDUAL_PAIR[0]} ~~ {RESIDUAL_PAIR[1]}",
    "population_c": TWO_FACTORS,
    "population_d": f"F1 =~ {'+'.join(F1)}+HSNS9\nF2 =~ {'+'.join(F2)}",
}
# Models an analyst might give instead: one detail short of the generating
# model, or one too many. verify() confirms which of them the floor admits.
ALTERNATIVE_SYNTAX = {
    "one factor": f"G =~ {'+'.join(ITEMS)}",
    "two factors": TWO_FACTORS,
    "two factors + residual": MODEL_SYNTAX["population_b"],
    "two factors + cross-loading": MODEL_SYNTAX["population_d"],
}
# The same margins Level 1 holds a model to, without the population-matrix term.
TIER_2 = [
    {"key": "CFI", "direction": "higher", "eps": 0.005},
    {"key": "RMSEA", "direction": "lower", "eps": 0.005},
    {"key": "SRMR", "direction": "lower", "eps": 0.005},
    {"key": "BIC", "direction": "lower", "eps": 10.0},
]
MAX_FACTORS = 3


def reference_criteria(references):
    """Fit of each population's generating model on its own reference sample."""
    from corral_psychometrics.score import evaluate_model

    return {
        name: evaluate_model(MODEL_SYNTAX[name], frame[ITEMS].astype(float), ITEMS)[0]
        for name, frame in references.items()
    }


def model_verdict(spec, frame, floor):
    """Whether a model passes the scorer's checks against one population's floor."""
    from corral_psychometrics.score import check_constraints, check_dominance, evaluate_model

    criteria, model = evaluate_model(spec, frame[ITEMS].astype(float), ITEMS)
    constraints, _ = check_constraints(model, criteria, {})
    dominance = check_dominance(criteria, floor, TIER_2)
    return all(constraints.values()) and all(v for v in dominance.values() if v is not None)


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


def same_model(source):
    """Populations generated by the same model as `source`, which a case cannot tell apart."""
    return sorted(name for name, recipe in POPULATIONS.items() if recipe is POPULATIONS[source])


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
        "The reference files hold large samples from four populations, named only by identifier. How they differ is not supplied. `cases.csv` holds three case samples of "
        f"{CASE_SIZE} respondents each, identified by `case_id`; each case comes from one of the four populations.\n\n"
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
                # Exactly the files copied into the agent's workspace.
                "public_inputs": ["cases.csv", "codebook.md"]
                + [f"{name}.csv" for name in POPULATIONS],
                "reference_datasets": {name: f"{name}.csv" for name in POPULATIONS},
                "cases": "cases.csv",
                "codebook": "codebook.md",
                "data_sha256": shas,
            },
            "tools": [],
            "scoring_function": "score_population_classification",
            "scoring_params": {
                "task_type": "population_classification",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "items": ITEMS,
                "populations": list(POPULATIONS),
                "reference_datasets": {name: f"{name}.csv" for name in POPULATIONS},
                "cases": "cases.csv",
                "case_size": CASE_SIZE,
                "max_factors": MAX_FACTORS,
                "tier_2_comparative": TIER_2,
            },
        }
    ]


def candidate_submissions():
    expected = json.loads(TASK.truth.read_text())["scored"]["classifications"]
    overlapping = next(k for k, value in expected.items() if len(value) > 1)
    correct = {"models": dict(MODEL_SYNTAX), "classifications": expected}
    return {
        "correct": correct,
        "forces a unique answer everywhere": {
            **correct,
            "classifications": {k: [v[0]] for k, v in expected.items()},
        },
        "names every population for every case": {
            **correct,
            "classifications": {k: list(POPULATIONS) for k in expected},
        },
        "misses the one genuine overlap": {
            **correct,
            "classifications": {
                k: ([v[0]] if k == overlapping else v) for k, v in expected.items()
            },
        },
        "the textbook model for every population": {
            **correct,
            "models": {name: TWO_FACTORS for name in POPULATIONS},
        },
        "misses the residual covariance": {
            **correct,
            "models": {**MODEL_SYNTAX, "population_b": TWO_FACTORS},
        },
        "misses the cross-loading": {
            **correct,
            "models": {**MODEL_SYNTAX, "population_d": TWO_FACTORS},
        },
        "right classifications, no models": {"classifications": expected},
    }


def verify(references, cases, expected):
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

    from corral_psychometrics.score import evaluate_model

    floors = reference_criteria(references)
    admitted, bic_best_admitted = {}, True
    print("models on each reference sample (PASS = admitted by the scorer's floor):")
    for name, frame in references.items():
        admitted[name] = {
            label: model_verdict(spec, frame, floors[name])
            for label, spec in ALTERNATIVE_SYNTAX.items()
        }
        bic = {
            label: evaluate_model(spec, frame[ITEMS].astype(float), ITEMS)[0]["BIC"]
            for label, spec in ALTERNATIVE_SYNTAX.items()
        }
        best = min(bic, key=bic.get)
        bic_best_admitted &= admitted[name][best]
        print(
            f"  {name}: "
            + ", ".join(
                f"{label} {'PASS' if ok else 'fail'}" for label, ok in admitted[name].items()
            )
            + f"  (lowest BIC: {best})"
        )
    short_rejected = (
        not admitted["population_a"]["one factor"]
        and not admitted["population_b"]["two factors"]
        and not admitted["population_d"]["two factors"]
    )
    return C.report(
        [
            (
                "every generating model passes its own floor",
                all(
                    model_verdict(MODEL_SYNTAX[name], frame, floors[name])
                    for name, frame in references.items()
                ),
            ),
            ("the lowest-BIC candidate is always admitted", bic_best_admitted),
            ("a model one detail short of the truth is rejected", short_rejected),
            ("every held-out case includes its own population", min(contains.values()) == 1.0),
            ("the compatible set is stable across redraws", min(stability.values()) >= 0.90),
            (
                "the rule reproduces the answer for every delivered case",
                all(
                    compatible(cases[case_id], models) == members
                    for case_id, members in expected.items()
                ),
            ),
            (
                "the shared-model pair is genuinely inseparable",
                close == {"population_a", "population_c"},
            ),
            (
                "the other populations are identified uniquely",
                unique_count == len(POPULATIONS) - 2,
            ),
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
    expected = {
        case_id: same_model(source) for case_id, source in zip(cases, CASE_SOURCES, strict=True)
    }
    if action == "verify":
        return verify(references, cases, expected)
    if action == "naive":
        return naive(references, cases, expected)

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, frame in references.items():
        frame.to_csv(TASK.artifacts / f"{name}.csv", sep="\t", index=False)
        shas[f"{name}.csv"] = C.file_sha256(TASK.artifacts / f"{name}.csv")
    delivered = pd.concat(
        [frame.assign(case_id=case_id) for case_id, frame in cases.items()], ignore_index=True
    )
    delivered[["case_id"] + ITEMS].to_csv(TASK.artifacts / "cases.csv", sep="\t", index=False)
    shas["cases.csv"] = C.file_sha256(TASK.artifacts / "cases.csv")
    write_codebook(TASK.artifacts / "codebook.md")
    truth = {
        "task_id": TASK_ID,
        "scored": {"classifications": expected},
        "scoring_reference": {
            "case_size": CASE_SIZE,
            "rule": (
                "A case's answer is every population generated by the same model as its "
                "source. population_a and population_c share one model. Each submitted "
                "model must fit its population's reference sample as well as the "
                "generating model does."
            ),
            "population_models": {
                name: {
                    "reference_model_syntax": MODEL_SYNTAX[name],
                    "reference_criteria": criteria,
                }
                for name, criteria in reference_criteria(references).items()
            },
        },
        "generative_parameters": {
            "population_structures": STRUCTURES,
            "case_sources": dict(zip(expected, CASE_SOURCES)),
        },
        "provenance": C.provenance(Path(__file__).name, SEED, len(delivered), shas["cases.csv"]),
    }
    C.write_json(TASK.truth, truth)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(TASK.definition, build_task_json(shas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
