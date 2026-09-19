#!/usr/bin/env python3
"""Generate Level 2 Task 03 artifacts and scoring metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l2_t03_population_classification"
UUID = "c4a71f28-9b30-4e6d-8f52-17ad0c9e4b63"
SEED = 20260926
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_03"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_03.json"

ITEMS = C.DD_ITEMS
SUBSCALE = {
    **{item: "P" for item in ITEMS[0:4]},
    **{item: "N" for item in ITEMS[4:8]},
    **{item: "M" for item in ITEMS[8:12]},
}

N_REFERENCE = 3_000
CASE_SIZE = 200
CASE_SOURCES = ["population_a", "population_c", "population_d", "population_e", "population_f"]

# A population is compatible with a case when the case's responses are almost as
# likely under it as under the best-fitting population. "Almost as likely" is
# this many units of log-likelihood per respondent, and the agent is told it.
COMPATIBILITY_MARGIN = 0.06

RESIDUAL_PAIR = ("DDP1", "DDP2")
RESIDUAL_SHARE = 0.40

PROMPT = (
    "You are given large response samples from six anonymised reference populations and five "
    "case samples, each of {case_size} respondents drawn from one of those populations. The "
    "questionnaire is the Dirty Dozen. The population identifiers say nothing about how the "
    "populations differ.\n\n"
    "Work out what distinguishes the reference populations, then decide which of them each "
    "case sample could have come from. A population counts as compatible with a case when the "
    "case's responses are no more than {margin} units of log-likelihood per respondent less "
    "likely under it than under the best-fitting population. Some cases have one compatible "
    "population and some have more than one; do not force a unique answer.\n"
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


def unidimensional(rng, n, loading):
    """A single factor behind every item."""
    return ordinal_block(rng, n, ["G"], np.array([[1.0]]), {i: [("G", loading)] for i in ITEMS})


def two_factors(rng, n, loading, phi):
    """Two subscales merged against the third."""
    loadings = {i: [("A" if SUBSCALE[i] in ("P", "N") else "B", loading)] for i in ITEMS}
    return ordinal_block(rng, n, ["A", "B"], np.array([[1, phi], [phi, 1]]), loadings)


def bifactor(rng, n, general, specific):
    """A general factor across all items plus one narrow factor per subscale."""
    loadings = {i: [("G", general), (SUBSCALE[i], specific)] for i in ITEMS}
    return ordinal_block(rng, n, ["G", "P", "N", "M"], np.eye(4), loadings)


# Six populations. a and e are deliberately close; the rest are separable.
POPULATIONS = {
    "population_a": lambda rng, n: correlated_factors(rng, n, 0.68, 0.25),
    "population_b": lambda rng, n: unidimensional(rng, n, 0.62),
    "population_c": lambda rng, n: bifactor(rng, n, 0.62, 0.38),
    "population_d": lambda rng, n: two_factors(rng, n, 0.70, 0.30),
    "population_e": lambda rng, n: correlated_factors(rng, n, 0.66, 0.29),
    "population_f": lambda rng, n: correlated_factors(
        rng, n, 0.68, 0.25, residual_pair=RESIDUAL_PAIR, residual=RESIDUAL_SHARE
    ),
}
STRUCTURES = {
    "population_a": "three correlated factors, loadings .68, r .25",
    "population_b": "one factor, loadings .62",
    "population_c": "bifactor, general .62, specific .38",
    "population_d": "two factors, loadings .70, r .30",
    "population_e": "three correlated factors, loadings .66, r .29",
    "population_f": f"population_a plus residual covariance {RESIDUAL_SHARE} on {RESIDUAL_PAIR}",
}


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
        "The reference files hold large samples from six populations, named only by "
        "identifier. How they differ is not supplied. `cases.csv` holds five case samples of "
        f"{CASE_SIZE} respondents each, identified by `case_id`; each case comes from one of "
        "the six populations.\n\n"
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
                "reference_datasets": {name: f"{name}.csv" for name in POPULATIONS},
                "cases": "cases.csv",
                "codebook": "codebook.md",
                "data_sha256": shas,
            },
            "tools": [],
            "scoring_function": "score_population_classification",
            "scoring_params": {
                "task_type": "population_classification",
                "truth_path": "artifacts/level_2/task_03/truth.json",
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
    """The intended answer and the ways a classification goes wrong."""
    expected = json.loads((OUT_DIR / "truth.json").read_text())["scored"]["classifications"]
    overlapping = next(k for k, v in expected.items() if len(v) > 1)
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

    print("\n  the five delivered cases:")
    for case_id, members in expected.items():
        print(f"    {case_id}: {members}")

    overlaps = [k for k, v in expected.items() if len(v) > 1]
    return C.report(
        [
            ("every held-out case includes its own population", min(contains.values()) == 1.0),
            ("the compatible set is stable across redraws", min(stability.values()) >= 0.90),
            (
                "the deliberately close pair is genuinely inseparable",
                set(compatible(references["population_a"].iloc[:CASE_SIZE], models))
                == {"population_a", "population_e"},
            ),
            (
                "the other populations are identified uniquely",
                sum(
                    1
                    for name in POPULATIONS
                    if len(compatible(POPULATIONS[name](rng, CASE_SIZE), models)) == 1
                )
                == 4,
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
                "while the stated rule places every case with its own population",
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
    models = reference_models(references)
    expected = {case_id: compatible(frame, models) for case_id, frame in cases.items()}

    if action == "verify":
        return verify(references, cases, expected)
    if action == "naive":
        return naive(references, cases, expected)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_JSON.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in references.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", sep="\t", index=False)
    delivered = pd.concat(
        [frame.assign(case_id=case_id) for case_id, frame in cases.items()], ignore_index=True
    )
    delivered[["case_id"] + ITEMS].to_csv(OUT_DIR / "cases.csv", sep="\t", index=False)
    write_codebook(OUT_DIR / "codebook.md")

    shas = {f"{name}.csv": C.file_sha256(OUT_DIR / f"{name}.csv") for name in POPULATIONS}
    shas["cases.csv"] = C.file_sha256(OUT_DIR / "cases.csv")
    truth = {
        "task_id": TASK_ID,
        "scored": {"classifications": expected},
        "scoring_reference": {
            "compatibility_margin": COMPATIBILITY_MARGIN,
            "case_size": CASE_SIZE,
            "rule": (
                "A population is compatible when its mean log-likelihood for the case is "
                "within the margin of the best-fitting population's."
            ),
        },
        "generative_parameters": {
            "population_structures": STRUCTURES,
            "case_sources": dict(zip(expected, CASE_SOURCES)),
            "reference_rows": N_REFERENCE,
            "items": ITEMS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, len(delivered), shas["cases.csv"]),
    }
    (OUT_DIR / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    TASK_JSON.write_text(json.dumps(build_task_json(expected, shas), indent=2) + "\n")

    print(f"\n{len(delivered):,} case rows -> {OUT_DIR / 'cases.csv'}")
    for name in ("codebook.md", "truth.json"):
        print(f"           {OUT_DIR / name}")
    print(f"           {TASK_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
