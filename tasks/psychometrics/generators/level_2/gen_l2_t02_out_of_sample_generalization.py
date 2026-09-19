#!/usr/bin/env python3
"""Generate Level 2 Task 02 artifacts and scoring metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l2_t02_out_of_sample_generalization"
SEED = 20260922
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_02"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_02.json"

ITEMS = C.HSNS_ITEMS
F1, F2 = "F1", "F2"
LOADINGS = {
    "HSNS1": (F1, 0.55),
    "HSNS4": (F1, 0.30),
    "HSNS5": (F1, 0.70),
    "HSNS6": (F1, 0.48),
    "HSNS8": (F1, 0.71),
    "HSNS10": (F1, 0.66),
    "HSNS2": (F2, 0.76),
    "HSNS3": (F2, 0.58),
    "HSNS7": (F2, 0.69),
    "HSNS9": (F2, 0.52),
}
PHI = 0.35

# Two F1 items share wording, so a model without this covariance is rejected in
# the training data. It is part of the structure carried into every holdout.
RESIDUAL_CORR = (("HSNS5", "HSNS10"), 0.12)

# What each holdout does to the population the training model was built on.
HOLDOUTS = {
    "holdout_a": {"phi": PHI, "extra_cross": 0.0},
    "holdout_b": {"phi": 0.78, "extra_cross": 0.0},
    "holdout_c": {"phi": PHI, "extra_cross": 0.45},
}
HOLDOUT_DECISIONS = {
    "holdout_a": "generalizes",
    "holdout_b": "measurement_structure_holds_relations_differ",
    "holdout_c": "measurement_structure_fails",
}

# How far a standardised loading and the factor correlation may move before the
# holdout counts as a different population. The agent is told both numbers.
THRESHOLDS = {"loading": 0.10, "factor_correlation": 0.10}

N_TRAIN, N_HOLDOUT = 6_000, 4_000
POP_REFERENCE_N = 200_000

PROMPT = (
    "You are given a labelled training dataset and three anonymised holdout datasets "
    "containing the same questionnaire. A measurement model can look convincing in the data "
    "used to develop it while failing in a new population.\n\n"
    "Develop a model in the training data, then refit that model, unchanged, in each holdout "
    "and decide what happens to it there:\n\n"
    "  generalizes\n"
    "      every standardised loading stays within {loading} of its training value, and so "
    "does the correlation between the factors.\n"
    "  measurement_structure_holds_relations_differ\n"
    "      the loadings stay within {loading}, but the correlation between the factors moves "
    "by more than {factor_correlation}.\n"
    "  measurement_structure_fails\n"
    "      at least one loading moves by more than {loading}.\n\n"
    "Return the model and one decision for each holdout.\n"
).format(**THRESHOLDS)

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "F1 =~ ...\\nF2 =~ ...\\nF1 ~~ F2",
  "holdout_conclusions": {
    "holdout_a": "generalizes|measurement_structure_holds_relations_differ|measurement_structure_fails",
    "holdout_b": "...",
    "holdout_c": "..."
  }
}

The model must include every HSNS item. It is refitted in each holdout exactly
as submitted, and the loadings and factor correlation it produces there are read
from that refit, so the decisions must match what your own model does.
"""


def reference_syntax():
    """The generating model: two correlated traits and one residual covariance."""
    f1 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == F1)
    f2 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == F2)
    pair = RESIDUAL_CORR[0]
    return f"F1 =~ {f1}\nF2 =~ {f2}\nF1 ~~ F2\n{pair[0]} ~~ {pair[1]}"


def cross_loading_syntax():
    """A model that absorbs holdout C's change before validation."""
    f1 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == F1)
    f2 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == F2)
    pair = RESIDUAL_CORR[0]
    return f"F1 =~ {f1}+HSNS9\nF2 =~ {f2}\nF1 ~~ F2\n{pair[0]} ~~ {pair[1]}"


def simulate(rng, n, phi=PHI, extra_cross=0.0, label=None):
    """Draw responses from one population.

    n            respondents
    phi          correlation between the two traits
    extra_cross  cross-loading added to HSNS9, which breaks the structure
    label        value for the country column; omitted when None
    """
    eta = rng.multivariate_normal([0.0, 0.0], [[1.0, phi], [phi, 1.0]], n)
    shared = rng.normal(size=n)
    out = {}
    for item, (factor, loading) in LOADINGS.items():
        index = 0 if factor == F1 else 1
        common = loading * eta[:, index]
        explained = loading**2
        if item == "HSNS9" and extra_cross:
            common += extra_cross * eta[:, 0]
            explained += extra_cross**2 + 2 * extra_cross * loading * phi
        if item in RESIDUAL_CORR[0]:
            common += np.sqrt(RESIDUAL_CORR[1]) * shared
            explained += RESIDUAL_CORR[1]
        y = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        out[item] = C.categorize(y, C.THRESHOLDS[item])
    frame = pd.DataFrame(out)
    frame["age"] = np.clip(rng.normal(35, 11, n).round(), 18, 78).astype(int)
    frame["gender"] = rng.choice([1, 2], size=n, p=[0.61, 0.39])
    frame["accuracy"] = np.clip(rng.beta(8, 1.5, n) * 100, 1, 100).round().astype(int)
    frame[ITEMS] = frame[ITEMS].mask(rng.random((n, len(ITEMS))) < 0.01, 0)
    columns = ITEMS + ["age", "gender", "accuracy"]
    if label is not None:
        frame["country"] = label
        columns += ["country"]
    return frame[columns]


def build_holdouts(rng):
    """The three anonymised holdouts, with no column identifying the population."""
    return {name: simulate(rng, N_HOLDOUT, **spec) for name, spec in HOLDOUTS.items()}


def analysis_sample(df):
    """Complete responses."""
    return df[ITEMS][(df[ITEMS] != 0).all(axis=1)].astype(float)


def population_matrix(rng):
    """Correlations a correctly specified model reproduces, from a large draw."""
    big = simulate(rng, POP_REFERENCE_N)
    return np.corrcoef(big[ITEMS].values.T.astype(float)).round(3)


def standardised(X, syntax=None):
    """Loadings, factor correlation and fit a model gives on `X`.

    X       one row per respondent
    syntax  model to fit; the frozen training model by default
    """
    import semopy

    model = C.fit(syntax or reference_syntax(), X, ITEMS)
    estimates = model.inspect(std_est=True)
    load = estimates[(estimates.op == "~") & estimates.lval.isin(ITEMS)]
    cov = estimates[
        (estimates.op == "~~") & (estimates.lval != estimates.rval) & ~estimates.lval.isin(ITEMS)
    ]
    stats = semopy.calc_stats(model).iloc[0]
    return {
        "n": len(X),
        "loadings": {row["lval"]: float(row["Est. Std"]) for _, row in load.iterrows()},
        "factor_correlation": float(cov["Est. Std"].iloc[0]),
        "CFI": float(stats["CFI"]),
        "RMSEA": float(stats["RMSEA"]),
    }


def travel(train, holdout):
    """How far the frozen model moves between the training data and a holdout."""
    return {
        "max_loading_shift": max(
            abs(holdout["loadings"][item] - train["loadings"][item]) for item in ITEMS
        ),
        "factor_correlation_shift": abs(
            holdout["factor_correlation"] - train["factor_correlation"]
        ),
    }


def classify(moved):
    """The decision the stated thresholds imply."""
    if moved["max_loading_shift"] > THRESHOLDS["loading"]:
        return "measurement_structure_fails"
    if moved["factor_correlation_shift"] > THRESHOLDS["factor_correlation"]:
        return "measurement_structure_holds_relations_differ"
    return "generalizes"


def write_codebook(path, frames):
    path.write_text(
        "# Codebook - out-of-sample validation\n\n"
        "Each tab-separated file contains responses to the same ten HSNS items. "
        "Ratings run from 1 (Disagree) to 5 (Agree); 0 denotes a missing response.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n\n| variable | description |\n|---|---|\n"
        "| `age` | respondent age in years |\n"
        "| `gender` | 1 = Male, 2 = Female |\n"
        "| `accuracy` | self-rated response accuracy, 0-100 |\n"
        "| `country` | collection country; present in the training file only |\n\n"
        "The holdout files carry no column identifying their population.\n\n"
        "| file | rows |\n|---|---:|\n"
        + "\n".join(f"| `{name}` | {len(frame):,} |" for name, frame in frames.items())
        + "\n"
    )


def build_truth(floor, pop, data_sha, rows):
    return {
        "task_id": TASK_ID,
        "scored": {"holdout_conclusions": HOLDOUT_DECISIONS},
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": ITEMS,
        },
        "generative_parameters": {
            "training": {"phi": PHI, "residual_correlation": RESIDUAL_CORR},
            "holdouts": HOLDOUTS,
            "decision_thresholds": THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    contract = C.scoring_contract(
        "artifacts/level_2/task_02/truth.json",
        ITEMS,
        [
            {
                "key": "holdout_conclusions",
                "fn": "score_label_panel",
                "truth_key": "scored.holdout_conclusions",
                "criterion": "out_of_sample_generalization",
            },
            {
                "key": "holdout_diagnosis",
                "fn": "score_label_panel",
                "truth_key": "scored.holdout_conclusions",
                "derive_from": "refit_holdouts",
                "criterion": "out_of_sample_generalization",
            },
        ],
    )
    contract["holdout_datasets"] = {name: f"{name}.csv" for name in HOLDOUTS}
    contract["holdout_thresholds"] = THRESHOLDS
    return [
        {
            "id": TASK_ID,
            "name": "Does the HSNS model generalize out of sample?",
            "uuid": "7f8f969f-8b7d-45ab-97eb-8a8c49cc9af2",
            "keywords": ["psychometrics", "validation", "generalization", "replication"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                "training_dataset": "data.csv",
                "holdout_datasets": [f"{name}.csv" for name in HOLDOUTS],
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


def candidate_submissions(X):
    """The intended answer and the ways a validation stops short of it."""
    correct = {"model_syntax": reference_syntax(), "holdout_conclusions": dict(HOLDOUT_DECISIONS)}
    fit_only = dict(HOLDOUT_DECISIONS)
    fit_only["holdout_b"] = "generalizes"
    return {
        "correct": correct,
        "judges on fit alone": {**correct, "holdout_conclusions": fit_only},
        "assumes everything generalizes": {
            **correct,
            "holdout_conclusions": {name: "generalizes" for name in HOLDOUTS},
        },
        "rejects everything": {
            **correct,
            "holdout_conclusions": {name: "measurement_structure_fails" for name in HOLDOUTS},
        },
        "misses the residual covariance": {
            "model_syntax": "\n".join(reference_syntax().splitlines()[:3]),
            "holdout_conclusions": dict(HOLDOUT_DECISIONS),
        },
        "pre-absorbs the holdout cross-loading": {
            "model_syntax": cross_loading_syntax(),
            "holdout_conclusions": dict(HOLDOUT_DECISIONS),
        },
        "uses a one-factor model": {
            "model_syntax": "G =~ " + "+".join(ITEMS),
            "holdout_conclusions": dict(HOLDOUT_DECISIONS),
        },
    }


def verify(train, holdouts):
    """Confirm each holdout lands on its intended decision, by the stated rule."""
    X = analysis_sample(train)
    train_fit = standardised(X)
    without = standardised(X, "\n".join(reference_syntax().splitlines()[:3]))
    print(
        f"training: n={train_fit['n']:,}, factor correlation={train_fit['factor_correlation']:.3f}"
    )
    print(
        f"  dropping the residual covariance costs {train_fit['CFI'] - without['CFI']:.3f} CFI "
        f"and raises RMSEA from {train_fit['RMSEA']:.3f} to {without['RMSEA']:.3f}"
    )
    print("\n  the frozen model refitted in each holdout:")
    print(f"    {'':11s} {'CFI':>6s} {'RMSEA':>7s} {'max dloading':>13s} {'dphi':>7s}  decision")

    decisions, moves = {}, {}
    for name, frame in holdouts.items():
        fit = standardised(analysis_sample(frame))
        moved = travel(train_fit, fit)
        decisions[name] = classify(moved)
        moves[name] = moved
        print(
            f"    {name:11s} {fit['CFI']:6.3f} {fit['RMSEA']:7.3f} "
            f"{moved['max_loading_shift']:13.3f} {moved['factor_correlation_shift']:7.3f}  "
            f"{decisions[name]}"
        )

    return C.report(
        [
            ("the training sample is substantial", train_fit["n"] > 5_000),
            (
                "a model that misses the residual covariance fits clearly worse",
                train_fit["CFI"] - without["CFI"] > 0.01,
            ),
            ("the stated thresholds give the intended decisions", decisions == HOLDOUT_DECISIONS),
            (
                "holdout A moves on neither measure",
                moves["holdout_a"]["max_loading_shift"] < 0.07
                and moves["holdout_a"]["factor_correlation_shift"] < 0.07,
            ),
            (
                "holdout B keeps its loadings but moves its factor correlation",
                moves["holdout_b"]["max_loading_shift"] < 0.07
                and moves["holdout_b"]["factor_correlation_shift"] > 0.25,
            ),
            (
                "holdout C moves a loading well past the threshold",
                moves["holdout_c"]["max_loading_shift"] > 0.18,
            ),
        ]
    )


def naive(train, holdouts):
    """Confirm fit indices alone accept the holdout that does not replicate."""
    train_fit = standardised(analysis_sample(train))
    print("  validating on global fit alone, the usual rule of CFI > .95 and RMSEA < .06:\n")
    print(f"    {'':11s} {'CFI':>6s} {'RMSEA':>7s}  fit-only verdict   truth")
    verdicts = {}
    for name, frame in holdouts.items():
        fit = standardised(analysis_sample(frame))
        passes = fit["CFI"] > 0.95 and fit["RMSEA"] < 0.06
        verdicts[name] = "generalizes" if passes else "measurement_structure_fails"
        print(
            f"    {name:11s} {fit['CFI']:6.3f} {fit['RMSEA']:7.3f}  "
            f"{verdicts[name]:18s} {HOLDOUT_DECISIONS[name]}"
        )
    wrong = [name for name in holdouts if verdicts[name] != HOLDOUT_DECISIONS[name]]
    print(f"\n  fit alone gets {len(wrong)} of {len(holdouts)} wrong: {', '.join(wrong)}")
    print("  holdout_b fits beautifully and is not the same population.")
    del train_fit

    return C.report(
        [
            ("fit alone accepts holdout B", verdicts["holdout_b"] == "generalizes"),
            ("but holdout B is not a replication", HOLDOUT_DECISIONS["holdout_b"] != "generalizes"),
            ("so a fit-only validation reports a wrong decision", len(wrong) > 0),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    train = simulate(rng, N_TRAIN, label="US")
    holdouts = build_holdouts(rng)
    if action == "verify":
        return verify(train, holdouts)
    if action == "naive":
        return naive(train, holdouts)
    pop = population_matrix(np.random.default_rng(SEED + 1001))
    floor = C.evaluate_model(reference_syntax(), analysis_sample(train), ITEMS, pop)
    C.write_artifacts(
        OUT_DIR,
        TASK_JSON,
        train,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(train)),
        build_task_json,
    )
    for name, frame in holdouts.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", sep="\t", index=False)
    write_codebook(
        OUT_DIR / "codebook.md",
        {"data.csv": train, **{f"{name}.csv": frame for name, frame in holdouts.items()}},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
