#!/usr/bin/env python3
"""Generate the HSNS item-integrity task with recording faults and a weak item."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
NEUTRAL_CODED = "HSNS7"  # real answers overwritten with zero (the neutral value)
NEUTRAL_RATE = 0.45
TRUNCATED = "HSNS2"  # responses were clipped at an upper limit
TRUNCATED_AT = 0.8
WEAK = "HSNS6"  # the data are fine; the item measures poorly

TRUTH = {
    MIS_KEYED: "mis_keyed",
    NEUTRAL_CODED: "inserted_neutral",
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
classify each item of the Hypersensitive Narcissism Scale as exactly one of:

  sound                the item is doing its job
  mis_keyed            the item was stored with its scale reversed
  inserted_neutral     recorded answers were overwritten with zero (the neutral value)
  truncated_scale      one end of the response scale was never recorded
  duplicated_item      the column repeats another item's responses
  weak_item            nothing is wrong with the data; the item measures poorly
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<a model for the instrument, in semopy syntax>",
  "item_quality": {"<item>": "<class>", ...}
}

Give a class for every item of the instrument. `model_syntax` must be complete
and runnable so the fit can be reproduced.
"""


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
        cols[item] = C.continuous_response(
            lam * eta + rng.normal(0, np.sqrt(1 - lam**2), n), C.INTERCEPTS[item]
        )
    for item in DD:
        lam = 0.62 * scale
        cols[item] = C.continuous_response(
            lam * dark + rng.normal(0, np.sqrt(1 - lam**2), n), C.INTERCEPTS[item]
        )
    return pd.DataFrame(cols)[C.ALL_ITEMS]


def corrupt(frame, rng):
    """Apply the three recording faults. They came from the survey platform, so
    they affect every country alike."""
    frame = frame.copy()
    frame[MIS_KEYED] = -frame[MIS_KEYED]
    frame.loc[rng.random(len(frame)) < NEUTRAL_RATE, NEUTRAL_CODED] = 0
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
    X = X[X.notna().all(axis=1)].astype(float)
    if repair:
        X = X.copy()
        X[MIS_KEYED] = -X[MIS_KEYED]
    return X


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
    """Point mass at zero: continuous honest responses have no exact-zero mass."""
    return {item: float((X[item] == 0).mean()) for item in ITEMS}


def population_correlation_matrix():
    """Correlation matrix a perfectly specified, repaired US model reproduces."""
    rng = np.random.default_rng(SEED + 999)
    big = corrupt(honest_responses(POP_REFERENCE_N, rng, 1.0), rng)
    big = big[ITEMS].astype(float)
    big[MIS_KEYED] = -big[MIS_KEYED]
    return np.corrcoef(big.values.T).round(3)


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
            "repairs": {"reverse_scored": {MIS_KEYED: 0}},
        },
        "generative_parameters": {
            "loadings": LOADINGS,
            "faults": {
                "mis_keyed": MIS_KEYED,
                "inserted_neutral": {"item": NEUTRAL_CODED, "rate": NEUTRAL_RATE},
                "truncated_scale": {"item": TRUNCATED, "top": TRUNCATED_AT},
                "weak_item": WEAK,
            },
            "non_us_loading_scale": NON_US_LOADING_SCALE,
            "non_us_broken_items": NON_US_BROKEN,
            "item_intercepts": C.INTERCEPTS,
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
        """Build an item-quality submission using the reference model."""
        return {"model_syntax": reference_syntax(), "item_quality": quality}

    drop_the_weak = {i: ("weak_item" if TRUTH[i] != "sound" else "sound") for i in ITEMS}
    missed_the_quiet = {i: (TRUTH[i] if i in (MIS_KEYED, WEAK) else "sound") for i in ITEMS}
    swapped = {**TRUTH, NEUTRAL_CODED: "weak_item", WEAK: "inserted_neutral"}
    return {
        "correct": sub(dict(TRUTH)),
        "everything odd is a weak item": sub(drop_the_weak),
        "only the visible faults found": sub(missed_the_quiet),
        "corrupted and weak items confused": sub(swapped),
    }


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
        f"{'max':>4s} {'P(exact zero)':>12s}  truth"
    )
    for item in ITEMS:
        print(
            f"  {item:6s} {LOADINGS[item]:10.2f} {raw_loads[item]:10.3f} "
            f"{loads[item]:9.3f} {X[item].max():4.1f} {excess[item]:+12.3f}"
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
            X[TRUNCATED].max() == TRUNCATED_AT
            and all(X[i].max() > TRUNCATED_AT for i in ITEMS if i != TRUNCATED),
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
    """Generate task files or run the requested validation mode."""
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
