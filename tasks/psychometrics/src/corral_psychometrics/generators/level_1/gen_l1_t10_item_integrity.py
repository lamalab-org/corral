#!/usr/bin/env python3
"""Generate Task 10 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t10_item_integrity"
SEED = 20260922
TASK = paths.task(__file__)

HSNS = C.HSNS_ITEMS
DD = C.DD_ITEMS
ITEMS = HSNS

LOADINGS = {
    "HSNS1": 0.64,
    "HSNS2": 0.70,
    "HSNS3": 0.61,
    "HSNS4": 0.62,
    "HSNS5": 0.66,
    "HSNS6": 0.48,
    "HSNS7": 0.66,
    "HSNS8": 0.68,
    "HSNS9": 0.59,
    "HSNS10": 0.63,
}

# Three recording faults and one genuinely poor item.
MIS_KEYED = "HSNS4"  # stored with the scale reversed
NEUTRAL_CODED = "HSNS7"  # non-responses written as the midpoint
NEUTRAL_RATE = 0.45
TRUNCATED = "HSNS2"  # the top of the scale was never recorded
TRUNCATED_AT = 4
WEAK = "HSNS6"  # the data are fine; the item measures poorly

TRUTH = {
    MIS_KEYED: "mis_keyed",
    NEUTRAL_CODED: "missing_as_neutral",
    TRUNCATED: "truncated_scale",
    WEAK: "weak_item",
    **{i: "sound" for i in HSNS if i not in (MIS_KEYED, NEUTRAL_CODED, TRUNCATED, WEAK)},
}

# Outside the United States every item is measured less well, and one sound item
# was mistranslated and runs backwards - so pooling makes that item, not the
# real one, look like the weak link.
NON_US_LOADING_SCALE = 0.72
NON_US_BROKEN = {"HSNS9": -0.45}

POP_REFERENCE_N = 250_000

PROMPT = """\
You are given responses from an online personality survey run in several countries. \
The codebook describes every column.

The survey contains two instruments. Using the participants from the United States, \
work through the Hypersensitive Narcissism Scale item by item and classify each of \
its items as exactly one of:

  sound                the item is doing its job
  mis_keyed            the item was stored with its scale reversed
  missing_as_neutral   non-responses were recorded as the middle category
  truncated_scale      the upper part of the response scale was never recorded
  weak_item            nothing is wrong with the data; the item measures poorly

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<a model for the instrument, in lavaan notation>",
  "item_quality": {"<item>": "<class>", ...}
}

Give a class for every item of the instrument. `model_syntax` is re-fitted during
evaluation, so it must be complete and runnable.
"""


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def honest_responses(n, rng, scale):
    """Responses to all 22 items from one trait per instrument."""
    eta = rng.normal(size=n)
    dark = rng.normal(size=n)
    cols = {}
    for item in HSNS:
        lam = (
            LOADINGS[item] * scale
            if scale == 1.0
            else NON_US_BROKEN.get(item, LOADINGS[item] * scale)
        )
        cols[item] = C.categorize(
            lam * eta + rng.normal(0, np.sqrt(1 - lam**2), n), C.THRESHOLDS[item]
        )
    for item in DD:
        lam = 0.62 * scale
        cols[item] = C.categorize(
            lam * dark + rng.normal(0, np.sqrt(1 - lam**2), n), C.THRESHOLDS[item]
        )
    return pd.DataFrame(cols)[C.ALL_ITEMS]


def corrupt(frame, rng):
    """Apply the three recording faults. They came from the survey platform, so
    they affect every country alike."""
    frame = frame.copy()
    frame[MIS_KEYED] = 6 - frame[MIS_KEYED]
    frame.loc[rng.random(len(frame)) < NEUTRAL_RATE, NEUTRAL_CODED] = 3
    frame[TRUNCATED] = frame[TRUNCATED].clip(upper=TRUNCATED_AT)
    return frame


def build_dataset(rng):
    """Build the full survey: every country, both instruments, the faults."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        scale = 1.0 if country == "US" else NON_US_LOADING_SCALE
        block = corrupt(honest_responses(n, rng, scale), rng)
        frames.append(pd.concat([block, C.demographics(n, rng, country)], axis=1))
    return C.finalize(frames, rng, SEED)


def analysis_sample(df, country="US", repair=True):
    """One country's complete HSNS responses, with the reversal undone."""
    block = df[df.country == country] if country else df
    X = block[ITEMS]
    X = X[(X != 0).all(axis=1)].astype(float)
    if repair:
        X = X.copy()
        X[MIS_KEYED] = 6 - X[MIS_KEYED]
    return X


# --------------------------------------------------------------------------
# Models and diagnostics
# --------------------------------------------------------------------------
def reference_syntax():
    """One trait behind the ten items."""
    return "F =~ " + "+".join(ITEMS)


def fitted_loadings(X):
    """Standardised loading of each item on the single trait."""
    import semopy

    model = semopy.Model(reference_syntax())
    model.fit(X[ITEMS])
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == "F")]
    return {r["lval"]: float(r["Est. Std"]) for _, r in rows.iterrows()}


def excess_neutral(X):
    """How much more often each item is answered 3 than a normal trait allows."""
    out = {}
    for item in ITEMS:
        tau = C.THRESHOLDS[item]
        out[item] = float((X[item] == 3).mean() - (norm.cdf(tau[2]) - norm.cdf(tau[1])))
    return out


def population_correlation_matrix():
    """Correlation matrix a perfectly specified, repaired US model reproduces."""
    rng = np.random.default_rng(SEED + 999)
    big = corrupt(honest_responses(POP_REFERENCE_N, rng, 1.0), rng)
    big = big[ITEMS].astype(float)
    big[MIS_KEYED] = 6 - big[MIS_KEYED]
    return np.corrcoef(big.values.T).round(3)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {"item_quality": TRUTH, "calibration_country": "US"},
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": ITEMS,
            # Undone before fitting so that a submission is judged on the same
            # responses whether or not it spotted the reversal.
            "repairs": {"reverse_scored": {MIS_KEYED: 6}},
        },
        "generative_parameters": {
            "loadings": LOADINGS,
            "faults": {
                "mis_keyed": MIS_KEYED,
                "missing_as_neutral": {"item": NEUTRAL_CODED, "rate": NEUTRAL_RATE},
                "truncated_scale": {"item": TRUNCATED, "top": TRUNCATED_AT},
                "weak_item": WEAK,
            },
            "non_us_loading_scale": NON_US_LOADING_SCALE,
            "non_us_broken_items": NON_US_BROKEN,
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
                "key": "item_quality",
                "fn": "score_label_panel",
                "truth_key": "scored.item_quality",
                "criterion": "item_integrity",
            },
        ],
    )
    contract["syntax_whitelist"]["max_factors"] = 4
    return [
        {
            "id": TASK_ID,
            "name": "Which items are bad, and which is the data?",
            "uuid": "9a51c7e0-4bd8-4f36-83a2-1e6cd4907b55",
            "keywords": [
                "psychometrics",
                "data quality",
                "item analysis",
                "measurement",
                "screening",
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


def candidate_submissions(X):
    """Answers an analyst might give, for the scoring tests."""
    del X

    def sub(quality):
        return {"model_syntax": reference_syntax(), "item_quality": quality}

    drop_the_weak = {i: ("weak_item" if TRUTH[i] != "sound" else "sound") for i in ITEMS}
    missed_the_quiet = {i: (TRUTH[i] if i in (MIS_KEYED, WEAK) else "sound") for i in ITEMS}
    swapped = {**TRUTH, NEUTRAL_CODED: "weak_item", WEAK: "missing_as_neutral"}
    return {
        "correct": sub(dict(TRUTH)),
        "everything odd is a weak item": sub(drop_the_weak),
        "only the visible faults found": sub(missed_the_quiet),
        "corrupted and weak items confused": sub(swapped),
    }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def verify(df, pop):
    """Confirm each fault is findable and that the look-alikes need the data."""
    del pop
    X = analysis_sample(df)
    raw = analysis_sample(df, repair=False)
    loads = fitted_loadings(X)
    raw_loads = fitted_loadings(raw)
    excess = excess_neutral(X)

    print(f"US sample: {len(X):,}\n")
    print(
        f"{'item':8s} {'generated':>10s} {'as stored':>10s} {'repaired':>9s} "
        f"{'max':>4s} {'excess P(3)':>12s}  truth"
    )
    for item in ITEMS:
        print(
            f"  {item:6s} {LOADINGS[item]:10.2f} {raw_loads[item]:10.3f} "
            f"{loads[item]:9.3f} {int(X[item].max()):4d} {excess[item]:+12.3f}"
            f"  {TRUTH[item]}"
        )

    sound = [loads[i] for i in ITEMS if TRUTH[i] == "sound"]
    gap = abs(loads[WEAK] - loads[NEUTRAL_CODED])
    pooled = fitted_loadings(analysis_sample(df, country=None))
    pooled_sound_min = min(pooled[i] for i in ITEMS if TRUTH[i] == "sound")
    pooled_weakest = min((i for i in ITEMS), key=lambda i: pooled[i])

    print(
        f"\n  the two look-alikes differ by {gap:.3f} in loading "
        f"({WEAK} {loads[WEAK]:.3f}, {NEUTRAL_CODED} {loads[NEUTRAL_CODED]:.3f})"
    )
    print(
        f"  the truncated item loads {loads[TRUNCATED]:.3f}, inside the sound "
        f"range {min(sound):.3f}-{max(sound):.3f}"
    )
    print(
        f"  pooling all countries makes {pooled_weakest} the weakest item at "
        f"{pooled[pooled_weakest]:.3f}; in the United States the weakest is "
        f"{WEAK} at {loads[WEAK]:.3f}"
    )

    checks = [
        (
            "the mis-keyed item announces itself and recodes cleanly",
            raw_loads[MIS_KEYED] < -0.10 and abs(loads[MIS_KEYED] - LOADINGS[MIS_KEYED]) < 0.10,
        ),
        ("the corrupted and the weak item are indistinguishable by loading", gap < 0.05),
        (
            "but the corrupted one is unmistakable in its responses",
            excess[NEUTRAL_CODED] > 0.20
            and max(excess[i] for i in ITEMS if i != NEUTRAL_CODED) < 0.05,
        ),
        ("the truncated item is invisible in the model", loads[TRUNCATED] > min(sound)),
        (
            "and unmistakable in the data",
            X[TRUNCATED].max() < 5 and all(X[i].max() == 5 for i in ITEMS if i != TRUNCATED),
        ),
        (
            "every sound item stays clear of the weak one",
            min(sound) - max(loads[WEAK], loads[NEUTRAL_CODED]) > 0.05,
        ),
        (
            "skipping the US filter makes a sound item look like the weak one",
            pooled_sound_min < pooled[WEAK],
        ),
    ]
    return C.report(checks)


def naive(df, pop):
    """Confirm that the model output alone cannot classify the items."""
    del pop
    X = analysis_sample(df, repair=False)
    loads = fitted_loadings(X)
    print("Classifying each item from its loading alone:\n")
    print(f"  {'item':8s} {'loading':>8s}  {'verdict':>20s}  truth")
    wrong = []
    for item in ITEMS:
        verdict = (
            "mis_keyed"
            if loads[item] < -0.10
            else "weak_item"
            if abs(loads[item]) < 0.50
            else "sound"
        )
        if verdict != TRUTH[item]:
            wrong.append(item)
        print(
            f"  {item:8s} {loads[item]:8.3f}  {verdict:>20s}  {TRUTH[item]}"
            f"{'   <- wrong' if verdict != TRUTH[item] else ''}"
        )
    print(f"\n  the loadings alone misclassify {len(wrong)} of {len(ITEMS)} items: {wrong}")
    return C.report(
        [
            ("a corrupted item is mistaken for a weak one", NEUTRAL_CODED in wrong),
            ("a truncated item passes as sound", TRUNCATED in wrong),
            ("so the model output alone cannot answer the task", len(wrong) >= 2),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)

    if action != "build":
        return verify(df, None) if action == "verify" else naive(df, None)

    print(f"Fitting the scoring reference (population draw N={POP_REFERENCE_N:,}) ...")
    pop = population_correlation_matrix()
    floor = C.evaluate_model(reference_syntax(), analysis_sample(df), ITEMS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
