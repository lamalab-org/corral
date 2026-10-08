#!/usr/bin/env python3
"""Generate Task 09 artifacts and scoring metadata.

Generating model, United States: five correlated factors, two from the HSNS
(E, V) and three from the Dirty Dozen (M, P, N).

Within an instrument the dimensions are moderately related; across the two they
are barely related at all (0.12 to 0.30), and that is the finding the task is
after. It is hidden by respondents who were not reading: 8% answer every item
identically and a further 4% click at random, which inflates every correlation
that spans the two instruments. Outside the United States the instruments
really are related (0.45), so pooling inflates the same correlations again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t09_careless_responding"
SEED = 20260921
TASK = paths.task(__file__)

HSNS = C.HSNS_ITEMS
DD = C.DD_ITEMS
ITEMS = C.ALL_ITEMS

E = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
V = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
FACTORS = ["E", "V", "M", "P", "N"]
ITEMS_OF = {"E": E, "V": V, **{k: [i for i in DD if i[2] == k] for k in "MPN"}}
FACTOR_OF = {i: f for f, its in ITEMS_OF.items() for i in its}

LOADINGS = {
    "HSNS1": 0.64,
    "HSNS2": 0.70,
    "HSNS3": 0.63,
    "HSNS4": 0.60,
    "HSNS5": 0.66,
    "HSNS6": 0.58,
    "HSNS7": 0.67,
    "HSNS8": 0.68,
    "HSNS9": 0.61,
    "HSNS10": 0.62,
    "DDP1": 0.70,
    "DDP2": 0.66,
    "DDP3": 0.69,
    "DDP4": 0.58,
    "DDN1": 0.68,
    "DDN2": 0.66,
    "DDN3": 0.70,
    "DDN4": 0.64,
    "DDM1": 0.70,
    "DDM2": 0.68,
    "DDM3": 0.62,
    "DDM4": 0.72,
}
# Within each instrument the dimensions are moderately related; across the two
# they are barely related at all, which is the finding the contamination hides.
PHI = {
    ("E", "V"): 0.35,
    ("M", "P"): 0.45,
    ("M", "N"): 0.35,
    ("P", "N"): 0.30,
    ("E", "M"): 0.25,
    ("E", "P"): 0.20,
    ("E", "N"): 0.30,
    ("V", "M"): 0.15,
    ("V", "P"): 0.12,
    ("V", "N"): 0.22,
}
CROSS_PAIRS = [("E", "M"), ("E", "P"), ("E", "N"), ("V", "M"), ("V", "P"), ("V", "N")]

# Respondents who were not reading. Straight-liners give one answer to all 22
# items; the rest click at random. The proportions are ordinary for web panels.
STRAIGHT_LINE_RATE = 0.08
RANDOM_RATE = 0.04
# What a straight-liner picks. Weighted towards the middle of the scale, as
# observed response sets are.
STRAIGHT_VALUE_P = [0.10, 0.12, 0.38, 0.22, 0.18]

# Outside the United States the two instruments really are strongly related, so
# pooling inflates the same correlations a second time.
NON_US_CROSS_PHI = 0.45

CORRELATION_TOLERANCE = 0.06
POP_REFERENCE_N = 250_000

PROMPT = """\
You are given responses from an online personality survey run in several countries. \
The codebook describes every column.

The survey contains two instruments. Using the participants from the United States, \
establish a measurement model for each instrument, then report how strongly each \
dimension of one instrument is related to each dimension of the other, free of \
measurement error.

Before estimating those relationships, inspect response quality. Some respondents \
give the same answer to every item. The self-rated `accuracy` field is available, \
but it is only another piece of evidence, not a guaranteed indicator of careless \
responding. Use the response patterns to decide which rows should be excluded.

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<a model covering both instruments, in lavaan notation>",
  "correlations": [["<dimension>", "<dimension>", <float>], ...]
}

Name your dimensions whatever you like and refer to them by those names; they are
matched by which items they cover. Report one correlation for every pair of
dimensions that spans the two instruments. `model_syntax` is re-fitted during
evaluation, so it must be complete and runnable.
"""


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def phi_matrix(us):
    """Latent correlations among the five dimensions."""
    out = np.eye(len(FACTORS))
    for (a, b), value in PHI.items():
        i, j = FACTORS.index(a), FACTORS.index(b)
        same = (a in "EV") == (b in "EV")
        out[i, j] = out[j, i] = value if (us or same) else NON_US_CROSS_PHI
    return out


def honest_responses(n, rng, us):
    """Responses from respondents who read the questions."""
    eta = rng.multivariate_normal(np.zeros(len(FACTORS)), phi_matrix(us), size=n)
    cols = {}
    for item in ITEMS:
        lam = LOADINGS[item]
        ystar = lam * eta[:, FACTORS.index(FACTOR_OF[item])] + rng.normal(0, np.sqrt(1 - lam**2), n)
        cols[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(cols)[ITEMS]


def contaminate(frame, rng):
    """Overwrite a slice of respondents with response sets and random clicking.

    Returns the frame and the row positions of each kind, so the generator can
    record who was spoiled without ever putting a flag in the dataset.
    """
    n = len(frame)
    order = rng.permutation(n)
    n_straight = int(round(n * STRAIGHT_LINE_RATE))
    n_random = int(round(n * RANDOM_RATE))
    straight = order[:n_straight]
    clicking = order[n_straight : n_straight + n_random]
    values = frame.to_numpy(copy=True)
    values[straight] = rng.choice([1, 2, 3, 4, 5], size=(n_straight, 1), p=STRAIGHT_VALUE_P)
    values[clicking] = rng.integers(1, 6, size=(n_random, len(ITEMS)))
    return pd.DataFrame(values, columns=ITEMS), set(straight), set(clicking)


def build_dataset(rng):
    """Build the full survey: every country, both instruments, contamination."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        block, _, _ = contaminate(honest_responses(n, rng, country == "US"), rng)
        frames.append(pd.concat([block, C.demographics(n, rng, country)], axis=1))
    return C.finalize(frames, rng, SEED)


def analysis_sample(df, country="US", screen=None):
    """One country's complete responses, optionally screened.

    `screen` is "spread" to drop respondents whose answers never vary, or
    "accuracy" to drop the ones who rated their own answers poorly.
    """
    block = df[df.country == country] if country else df
    X = block[ITEMS]
    keep = (X != 0).all(axis=1)
    if screen == "spread":
        keep &= X.std(axis=1) > 0
    elif screen == "accuracy":
        keep &= block.accuracy >= 50
    return X[keep].astype(float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def reference_syntax():
    """The joint model: five dimensions across the two instruments."""
    return "\n".join(f"{f} =~ {'+'.join(ITEMS_OF[f])}" for f in FACTORS)


def latent_correlations(X):
    """Correlations between the five dimensions, net of measurement error."""
    import semopy

    model = semopy.Model(reference_syntax())
    model.fit(X[ITEMS])
    ins = model.inspect(std_est=True)
    rows = ins[
        (ins.op == "~~") & (ins.lval != ins.rval) & ins.lval.isin(FACTORS) & ins.rval.isin(FACTORS)
    ]
    return {frozenset((r["lval"], r["rval"])): float(r["Est. Std"]) for _, r in rows.iterrows()}


def population_reference(rng):
    """Truth from a large clean draw, and the matrix the scorer compares against.

    The targets come from uncontaminated responses - that is the answer - while
    the population matrix is contaminated, because that is the data every
    submission is re-fitted on.
    """
    clean = honest_responses(POP_REFERENCE_N, rng, True)
    targets = latent_correlations(clean.astype(float))
    spoiled, _, _ = contaminate(clean.copy(), rng)
    return targets, C.population_matrix(spoiled, ITEMS)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def build_truth(targets, floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "correlations": [
                {
                    "a": sorted(ITEMS_OF[a]),
                    "b": sorted(ITEMS_OF[b]),
                    "r": round(targets[frozenset((a, b))], 3),
                }
                for a, b in CROSS_PAIRS
            ],
            "calibration_country": "US",
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": ITEMS,
        },
        "generative_parameters": {
            "loadings": LOADINGS,
            "items_of_dimension": ITEMS_OF,
            "phi": {f"{a}-{b}": v for (a, b), v in PHI.items()},
            "non_us_cross_phi": NON_US_CROSS_PHI,
            "contamination": {
                "straight_line_rate": STRAIGHT_LINE_RATE,
                "random_rate": RANDOM_RATE,
                "straight_value_probabilities": STRAIGHT_VALUE_P,
                "detection": "zero spread across the 22 items",
            },
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        TASK,
        ITEMS,
        [
            {
                "key": "correlations",
                "fn": "score_correlations_by_composition",
                "truth_key": "scored.correlations",
                "tol": CORRELATION_TOLERANCE,
                "criterion": "latent_association",
            },
        ],
    )
    contract["syntax_whitelist"]["max_factors"] = 6
    return [
        {
            "id": TASK_ID,
            "name": "How strongly are the two instruments related?",
            "uuid": "3d47e9b1-0c62-4a75-b8e3-5619fa07c2d4",
            "keywords": [
                "psychometrics",
                "data quality",
                "careless responding",
                "discriminant validity",
                "measurement error",
            ],
            "metrics": ["binary", "partial"],
            "level": 1,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "codebook.md",
                    "data.csv",
                ],
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


def _submission(values):
    return {
        "model_syntax": reference_syntax(),
        "correlations": [[a, b, round(values[frozenset((a, b))], 3)] for a, b in CROSS_PAIRS],
    }


def candidate_submissions(X):
    """Answers an analyst might give, for the scoring tests."""
    del X
    rng = np.random.default_rng(SEED)
    df = build_dataset(rng)
    return {
        "correct": _submission(latent_correlations(analysis_sample(df, screen="spread"))),
        "unscreened": _submission(latent_correlations(analysis_sample(df))),
        "screened on self-rated accuracy": _submission(
            latent_correlations(analysis_sample(df, screen="accuracy"))
        ),
        "all countries, screened": _submission(
            latent_correlations(analysis_sample(df, country=None, screen="spread"))
        ),
    }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def verify(df, targets):
    """Confirm the contamination is invisible to the model and fatal to the answer."""
    import semopy

    raw = analysis_sample(df)
    clean = analysis_sample(df, screen="spread")
    by_accuracy = analysis_sample(df, screen="accuracy")
    print(
        f"US sample {len(raw):,}   after the spread screen {len(clean):,}   "
        f"after the accuracy screen {len(by_accuracy):,}\n"
    )

    fits = {}
    for label, X in [("unscreened", raw), ("screened", clean)]:
        model = semopy.Model(reference_syntax())
        model.fit(X[ITEMS])
        stats = semopy.calc_stats(model)
        fits[label] = (float(stats["CFI"].iloc[0]), float(stats["RMSEA"].iloc[0]))
        print(
            f"  the five-dimension model on {label:11s} data: "
            f"CFI {fits[label][0]:.4f}  RMSEA {fits[label][1]:.4f}"
        )

    got = {
        label: latent_correlations(X)
        for label, X in [("unscreened", raw), ("accuracy", by_accuracy), ("screened", clean)]
    }
    print(f"\n{'pair':14s} {'truth':>7s} {'unscreened':>11s} {'accuracy':>9s} {'screened':>9s}")
    worst = {k: 0.0 for k in got}
    for a, b in CROSS_PAIRS:
        key = frozenset((a, b))
        print(
            f"  {a} x {b:9s} {targets[key]:7.3f} {got['unscreened'][key]:11.3f} "
            f"{got['accuracy'][key]:9.3f} {got['screened'][key]:9.3f}"
        )
        for label in got:
            worst[label] = max(worst[label], abs(got[label][key] - targets[key]))
    print(
        f"\n  largest error: unscreened {worst['unscreened']:.3f}, "
        f"accuracy-screened {worst['accuracy']:.3f}, "
        f"screened {worst['screened']:.3f}  (tolerance "
        f"{CORRELATION_TOLERANCE})"
    )

    us = df[df.country == "US"]
    spoiled = us[ITEMS].std(axis=1) == 0
    print(
        f"  the spread screen removes {int(spoiled.sum()):,} of {len(us):,} "
        f"US rows; self-rated accuracy below 50 covers "
        f"{int((spoiled & (us.accuracy < 50)).sum()):,} of them"
    )

    pooled = latent_correlations(analysis_sample(df, country=None, screen="spread"))
    pooled_worst = max(abs(pooled[frozenset(p)] - targets[frozenset(p)]) for p in CROSS_PAIRS)

    checks = [
        ("the contaminated data fit the right model well", fits["unscreened"][0] > 0.95),
        (
            "so nothing in the fit warns that anything is wrong",
            fits["unscreened"][0] > 0.95 and fits["unscreened"][1] < 0.06,
        ),
        (
            "yet the cross-instrument correlations are badly wrong",
            worst["unscreened"] > 2 * CORRELATION_TOLERANCE,
        ),
        (
            "screening on self-rated accuracy does not help",
            worst["accuracy"] > CORRELATION_TOLERANCE,
        ),
        ("screening on response spread does", worst["screened"] <= CORRELATION_TOLERANCE),
        (
            "and skipping the US filter is wrong even after screening",
            pooled_worst > CORRELATION_TOLERANCE,
        ),
    ]
    return C.report(checks)


def naive(df, targets):
    """Confirm the ordinary analysis overstates how related the instruments are."""
    got = latent_correlations(analysis_sample(df))
    print("Fitting the right model to every complete response:\n")
    print(f"  {'pair':14s} {'reported':>9s} {'truth':>7s} {'error':>7s}")
    overstated = 0
    for a, b in CROSS_PAIRS:
        key = frozenset((a, b))
        error = got[key] - targets[key]
        overstated += error > CORRELATION_TOLERANCE
        print(f"  {a} x {b:9s} {got[key]:9.3f} {targets[key]:7.3f} {error:+7.3f}")
    print(
        f"\n  {overstated} of {len(CROSS_PAIRS)} associations overstated "
        f"by more than {CORRELATION_TOLERANCE}"
    )
    return C.report(
        [
            ("the ordinary analysis overstates the association", overstated >= 4),
            (
                "including the pair that is really near zero",
                got[frozenset(("V", "P"))] - targets[frozenset(("V", "P"))] > CORRELATION_TOLERANCE,
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)
    print(f"Fitting the reference (population draw N={POP_REFERENCE_N:,}) ...")
    targets, pop = population_reference(np.random.default_rng(SEED + 999))

    if action != "build":
        return verify(df, targets) if action == "verify" else naive(df, targets)

    floor = C.evaluate_model(reference_syntax(), analysis_sample(df), ITEMS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(targets, floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
