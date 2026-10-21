#!/usr/bin/env python3
"""Generate Level 2 Task 03 artifacts and scoring metadata.

Four Dirty Dozen populations, each with its own structure, and three case
samples drawn from some of them.

    population_a   three correlated factors, loadings .68, r .25
    population_b   two factors, psychopathy and narcissism merged, loadings .70, r .30
    population_c   population_a plus a residual covariance on DDP1, DDP2
    population_d   the same model as population_a, sampled independently

The agent describes each population with a model and places each case. a and d
are one population under two identifiers, so a case from either could have come
from both and the answer for it is a list, not one name. b and c each differ
from a by one detail, which the submitted model has to carry: every model is
refitted on its population's reference sample and has to fit as well as the
generating model does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t03_population_classification"
UUID = "c4a71f28-9b30-4e6d-8f52-17ad0c9e4b63"
SEED = 20260926
TASK = paths.task(__file__)

ITEMS = C.DD_ITEMS
SUBSCALE = {
    **{item: "P" for item in ITEMS[0:4]},
    **{item: "N" for item in ITEMS[4:8]},
    **{item: "M" for item in ITEMS[8:12]},
}

N_REFERENCE = 3_000
CASE_SIZE = 200
CASE_SOURCES = ["population_d", "population_c", "population_b"]

# The answer comes from the generating models, not from this margin. verify()
# uses it to confirm the data support that answer: a population is compatible
# when the case's mean Gaussian log-likelihood under it is within this margin of
# the best-fitting population's. The agent is not told it.
COMPATIBILITY_MARGIN = 0.06

RESIDUAL_PAIR = ("DDP1", "DDP2")
RESIDUAL_SHARE = 0.40

PROMPT = (
    "You have large Dirty Dozen response samples from four reference populations, "
    "`population_a` to `population_d`, and three case samples of {case_size} respondents, each "
    "drawn from one of those populations. The population identifiers carry no information.\n\n"
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

Give a model for every population. Each is re-fitted on that population's
reference sample during evaluation, so it must be complete and runnable.
Include every case exactly once; each value is a list of compatible population
identifiers, in any order.
"""


def ordinal_block(rng, n, factors, phi, loadings, residual_pair=None, residual=0.0):
    """Draw ordinal responses from a latent factor model.

    factors       latent variable names, in the order `phi` uses
    phi           correlations between the latent variables
    loadings      item -> list of (factor, loading)
    residual_pair two items sharing variance the factors do not explain
    residual      how much variance they share
    """
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


def correlated_factors(rng, n, loading, phi, **kwargs):
    """One factor per subscale, equally correlated."""
    matrix = np.full((3, 3), phi)
    np.fill_diagonal(matrix, 1.0)
    return ordinal_block(
        rng, n, ["P", "N", "M"], matrix, {i: [(SUBSCALE[i], loading)] for i in ITEMS}, **kwargs
    )


def two_factors(rng, n, loading, phi):
    """Two subscales merged against the third."""
    loadings = {i: [("A" if SUBSCALE[i] in ("P", "N") else "B", loading)] for i in ITEMS}
    return ordinal_block(rng, n, ["A", "B"], np.array([[1, phi], [phi, 1]]), loadings)


def three_factors(rng, n):
    return correlated_factors(rng, n, 0.68, 0.25)


# Four populations. a and d share one model; the rest are separable.
POPULATIONS = {
    "population_a": three_factors,
    "population_b": lambda rng, n: two_factors(rng, n, 0.70, 0.30),
    "population_c": lambda rng, n: correlated_factors(
        rng, n, 0.68, 0.25, residual_pair=RESIDUAL_PAIR, residual=RESIDUAL_SHARE
    ),
    "population_d": three_factors,
}
STRUCTURES = {
    "population_a": "three correlated factors, loadings .68, r .25",
    "population_b": "two factors, psychopathy and narcissism merged, loadings .70, r .30",
    "population_c": f"population_a plus residual covariance {RESIDUAL_SHARE} on {RESIDUAL_PAIR}",
    "population_d": "the same model as population_a, sampled independently",
}

_P, _N, _M = ("+".join(item for item in ITEMS if SUBSCALE[item] == s) for s in ("P", "N", "M"))
THREE_FACTORS = f"P =~ {_P}\nN =~ {_N}\nM =~ {_M}"
# The generating model of each population, in semopy syntax: the floor every
# submitted model is held to on that population's reference sample.
MODEL_SYNTAX = {
    "population_a": THREE_FACTORS,
    "population_b": f"PN =~ {_P}+{_N}\nM =~ {_M}",
    "population_c": f"{THREE_FACTORS}\n{RESIDUAL_PAIR[0]} ~~ {RESIDUAL_PAIR[1]}",
    "population_d": THREE_FACTORS,
}
# Models an analyst might give instead: one detail short of the generating
# model, or one too many. verify() confirms which of them the floor admits.
ALTERNATIVE_SYNTAX = {
    "one factor": f"G =~ {_P}+{_N}+{_M}",
    "three factors": THREE_FACTORS,
    "two factors": MODEL_SYNTAX["population_b"],
    "three factors + residual": MODEL_SYNTAX["population_c"],
    "bifactor": (
        f"G =~ {_P}+{_N}+{_M}\nSP =~ {_P}\nSN =~ {_N}\nSM =~ {_M}\n"
        "G ~~ 0*SP\nG ~~ 0*SN\nG ~~ 0*SM\nSP ~~ 0*SN\nSP ~~ 0*SM\nSN ~~ 0*SM"
    ),
}
# The same margins Level 1 holds a model to, without the population-matrix term.
TIER_2 = [
    {"key": "CFI", "direction": "higher", "eps": 0.005},
    {"key": "RMSEA", "direction": "lower", "eps": 0.005},
    {"key": "SRMR", "direction": "lower", "eps": 0.005},
    {"key": "BIC", "direction": "lower", "eps": 10.0},
]
MAX_FACTORS = 4


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
    """Mean and covariance of each reference sample, which is what a case is judged against."""
    return {
        name: (
            frame[ITEMS].values.astype(float).mean(axis=0),
            np.cov(frame[ITEMS].values.T.astype(float)) + np.eye(len(ITEMS)) * 1e-6,
        )
        for name, frame in references.items()
    }


def log_likelihood(case, models):
    """Average log-likelihood of a case sample under each reference population."""
    from scipy.stats import multivariate_normal

    x = case[ITEMS].values.astype(float)
    return {
        name: float(multivariate_normal(mean, cov, allow_singular=True).logpdf(x).mean())
        for name, (mean, cov) in models.items()
    }


def compatible(case, models, margin=COMPATIBILITY_MARGIN):
    """Populations the case could have come from, by the stated rule."""
    scores = log_likelihood(case, models)
    best = max(scores.values())
    return sorted(name for name, value in scores.items() if best - value <= margin)


def same_model(source):
    """Populations generated by the same model as `source`, which a case cannot tell apart."""
    return sorted(name for name, recipe in POPULATIONS.items() if recipe is POPULATIONS[source])


def build_references(rng):
    return {name: recipe(rng, N_REFERENCE) for name, recipe in POPULATIONS.items()}


def build_cases(rng):
    """One case sample per entry in CASE_SOURCES, drawn fresh from that population."""
    return {
        f"case_{i}": POPULATIONS[source](rng, CASE_SIZE)
        for i, source in enumerate(CASE_SOURCES, start=1)
    }


def write_codebook(path):
    path.write_text(
        "# Codebook - anonymised population comparison\n\n"
        "Every file is tab-separated. Responses run from 1 (Disagree) to 5 (Agree); all "
        "responses are complete.\n\n"
        "The reference files hold large samples from four populations, named only by "
        "identifier. How they differ is not supplied. `cases.csv` holds three case samples of "
        f"{CASE_SIZE} respondents each, identified by `case_id`; each case comes from one of "
        "the four populations.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n"
    )


def build_task_json(expected, shas):
    return [
        {
            "id": TASK_ID,
            "name": "Which population did each case sample come from?",
            "uuid": UUID,
            "keywords": ["psychometrics", "population classification", "model comparison"],
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
    """The intended answer and the ways a model or a classification goes wrong."""
    expected = json.loads(TASK.truth.read_text())["scored"]["classifications"]
    overlapping = next(k for k, v in expected.items() if len(v) > 1)
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
            "models": {name: THREE_FACTORS for name in POPULATIONS},
        },
        "misses the residual covariance": {
            **correct,
            "models": {**MODEL_SYNTAX, "population_c": THREE_FACTORS},
        },
        "one factor everywhere": {
            **correct,
            "models": {name: ALTERNATIVE_SYNTAX["one factor"] for name in POPULATIONS},
        },
        "right classifications, no models": {"classifications": expected},
    }


def verify(references, cases, expected):
    """Confirm the rule gives the same answer on independently drawn case samples."""
    models = reference_models(references)
    rng = np.random.default_rng(SEED + 7717)
    trials = 25

    print(f"reference samples: {N_REFERENCE:,} each. case samples: {CASE_SIZE} respondents.")
    print(f"compatible set = within {COMPATIBILITY_MARGIN} log-likelihood per respondent\n")
    print(f"  held-out cases, {trials} freshly drawn per population:")
    print(f"    {'source':12s} {'modal compatible set':28s} {'stable':>7s} {'source in set':>14s}")

    stability, contains = {}, {}
    for name, recipe in POPULATIONS.items():
        sets = [tuple(compatible(recipe(rng, CASE_SIZE), models)) for _ in range(trials)]
        modal = max(set(sets), key=sets.count)
        stability[name] = sets.count(modal) / trials
        contains[name] = sum(name in s for s in sets) / trials
        print(
            f"    {name:12s} {'{' + ','.join(p[-1] for p in modal) + '}':28s} "
            f"{stability[name]:6.0%} {contains[name]:13.0%}"
        )

    print("\n  the delivered cases:")
    for case_id, members in expected.items():
        print(f"    {case_id}: {members}")

    floors = reference_criteria(references)
    print("\n  models on each reference sample (PASS = admitted by the scorer's floor):")
    admitted, bic_best_admitted, short_rejected = {}, True, True
    from corral_psychometrics.score import evaluate_model

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
            f"    {name}: "
            + ", ".join(
                f"{label} {'PASS' if ok else 'fail'}" for label, ok in admitted[name].items()
            )
            + f"  (lowest BIC: {best})"
        )
    short_rejected = (
        not admitted["population_b"]["one factor"]
        and not admitted["population_b"]["three factors"]
        and not admitted["population_c"]["three factors"]
        and not admitted["population_a"]["two factors"]
    )

    overlaps = [k for k, v in expected.items() if len(v) > 1]
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
                set(compatible(references["population_a"].iloc[:CASE_SIZE], models))
                == {"population_a", "population_d"},
            ),
            (
                "the other populations are identified uniquely",
                sum(
                    1
                    for name in POPULATIONS
                    if len(compatible(POPULATIONS[name](rng, CASE_SIZE), models)) == 1
                )
                == len(POPULATIONS) - 2,
            ),
            (
                "the delivered cases include both a unique and an overlapping answer",
                0 < len(overlaps) < len(expected),
            ),
        ]
    )


def naive(references, cases, expected):
    """Confirm matching on item means alone cannot do the job."""
    models = reference_models(references)
    means = {
        name: frame[ITEMS].values.astype(float).mean(axis=0) for name, frame in references.items()
    }

    print("  a weaker method: match each case to the population with the closest item means,")
    print("  ignoring how the items relate to one another.\n")
    print(f"    {'case':8s} {'closest by item means':24s} {'compatible (the rule)':24s}")
    wrong = 0
    for case_id, frame in cases.items():
        target = frame[ITEMS].values.astype(float).mean(axis=0)
        closest = min(means, key=lambda n: float(np.abs(means[n] - target).mean()))
        truth = expected[case_id]
        if closest not in truth:
            wrong += 1
        print(f"    {case_id:8s} {closest:24s} {str(truth):24s}")

    rng = np.random.default_rng(SEED + 4242)
    hits = 0
    for name, recipe in POPULATIONS.items():
        for _ in range(20):
            target = recipe(rng, CASE_SIZE)[ITEMS].values.astype(float).mean(axis=0)
            hits += min(means, key=lambda n: float(np.abs(means[n] - target).mean())) == name
    accuracy = hits / (len(POPULATIONS) * 20)
    print(f"\n  over 120 held-out cases it identifies the source {accuracy:.0%} of the time")
    print(f"  (chance is {1 / len(POPULATIONS):.0%}). The populations share their item")
    print("  distributions and differ only in the relationships between items.")

    return C.report(
        [
            ("matching on item means is near chance", accuracy < 2 / len(POPULATIONS)),
            ("and it misclassifies delivered cases", wrong > 0),
            (
                "while the likelihood rule places every case with its own population",
                all(
                    source in compatible(cases[case_id], models)
                    for case_id, source in zip(expected, CASE_SOURCES)
                ),
            ),
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
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in references.items():
        frame.to_csv(TASK.artifacts / f"{name}.csv", sep="\t", index=False)
    delivered = pd.concat(
        [frame.assign(case_id=case_id) for case_id, frame in cases.items()], ignore_index=True
    )
    delivered[["case_id"] + ITEMS].to_csv(TASK.artifacts / "cases.csv", sep="\t", index=False)
    write_codebook(TASK.artifacts / "codebook.md")

    shas = {f"{name}.csv": C.file_sha256(TASK.artifacts / f"{name}.csv") for name in POPULATIONS}
    shas["cases.csv"] = C.file_sha256(TASK.artifacts / "cases.csv")
    truth = {
        "task_id": TASK_ID,
        "scored": {"classifications": expected},
        "scoring_reference": {
            "case_size": CASE_SIZE,
            "rule": (
                "A case's answer is every population generated by the same model as its "
                "source. population_a and population_d share one model. Each submitted "
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
            "reference_rows": N_REFERENCE,
            "items": ITEMS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, len(delivered), shas["cases.csv"]),
    }
    C.write_json(TASK.truth, truth)
    C.write_json(TASK.definition, build_task_json(expected, shas))

    print(f"\n{len(delivered):,} case rows -> {TASK.artifacts / 'cases.csv'}")
    for name in ("codebook.md",):
        print(f"           {TASK.artifacts / name}")
    print(f"           {TASK.definition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
