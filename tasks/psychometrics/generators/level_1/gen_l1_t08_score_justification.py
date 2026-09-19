#!/usr/bin/env python3
"""Generate Task 08 artifacts and scoring metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l1_t08_score_justification"
SEED = 20260918
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_1" / "task_08"
TASK_JSON = C.PKG_ROOT / "environments" / "level_1" / "tasks_json" / "task_08.json"

HSNS = C.HSNS_ITEMS
DD = C.DD_ITEMS
ITEMS = C.ALL_ITEMS
SPECIFIC_OF = {item: item[2] for item in DD}
SUBSCALES = ["M", "P", "N"]

# One factor, modest and fairly even loadings: a sound total score with an
# unremarkable alpha.
HSNS_LOADINGS = {
    "HSNS1": 0.48,
    "HSNS2": 0.52,
    "HSNS3": 0.44,
    "HSNS4": 0.40,
    "HSNS5": 0.50,
    "HSNS6": 0.42,
    "HSNS7": 0.49,
    "HSNS8": 0.45,
    "HSNS9": 0.47,
    "HSNS10": 0.43,
}
# Two items are worded alike and agree a little beyond the trait. Small enough
# to leave the total score sound, large enough that the model is an
# approximation rather than an exact match.
HSNS_RESIDUAL_CORR = (("HSNS2", "HSNS7"), 0.10)

# A weak broad trait under three strong narrow ones. The balance between the
# two varies from item to item within a subscale. Without that variation the
# model would be three related traits written a different way.
DD_GENERAL = {
    "DDM1": 0.46,
    "DDM2": 0.38,
    "DDM3": 0.22,
    "DDM4": 0.44,
    "DDP1": 0.40,
    "DDP2": 0.30,
    "DDP3": 0.44,
    "DDP4": 0.24,
    "DDN1": 0.26,
    "DDN2": 0.34,
    "DDN3": 0.42,
    "DDN4": 0.30,
}
DD_SPECIFIC = {
    "DDM1": 0.62,
    "DDM2": 0.74,
    "DDM3": 0.80,
    "DDM4": 0.66,
    "DDP1": 0.70,
    "DDP2": 0.78,
    "DDP3": 0.64,
    "DDP4": 0.76,
    "DDN1": 0.78,
    "DDN2": 0.72,
    "DDN3": 0.66,
    "DDN4": 0.74,
}

# Outside the United States both instruments behave differently, in the
# direction that makes each one's verdict come out wrong.
NON_US_HSNS_F1 = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
NON_US_HSNS_F2 = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
NON_US_HSNS_LOADING = 0.60
NON_US_HSNS_PHI = 0.25
NON_US_DD_GENERAL = 0.70
NON_US_DD_SPECIFIC = 0.25

TRUTH = {"hsns": "total_only", "dirty_dozen": "subscales_only"}
OMEGA_H_FLOOR = 0.70

POP_REFERENCE_N = 400_000

PROMPT = """\
You are given responses from an online personality survey run in several countries. \
The codebook describes every column.

The survey contains two instruments. Using the participants from the United States, \
establish a measurement model for each instrument and decide what it entitles you to \
score. Classify each instrument as exactly one of:

  total_only           one score over all of that instrument's items
  subscales_only       separate subscale scores, but no defensible total
  total_and_subscales  both
  none                 neither

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<the model you calibrated on the United States, covering both
                    instruments, in lavaan notation>",
  "scoring": {"<instrument>": "<class>", "<instrument>": "<class>"}
}

Name the two instruments `hsns` and `dirty_dozen`. `model_syntax` is re-fitted on the
United States sample during evaluation, so it must be complete and runnable.
"""


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_hsns(n, rng, us):
    """One factor in the United States; two weakly correlated factors elsewhere."""
    if us:
        return C.correlated_block(
            n,
            rng,
            {i: ("F", HSNS_LOADINGS[i]) for i in HSNS},
            np.array([[1.0]]),
            ["F"],
            C.THRESHOLDS,
            resid_corr=HSNS_RESIDUAL_CORR,
        )[HSNS]
    phi = np.array([[1.0, NON_US_HSNS_PHI], [NON_US_HSNS_PHI, 1.0]])
    loadings = {i: ("A" if i in NON_US_HSNS_F1 else "B", NON_US_HSNS_LOADING) for i in HSNS}
    return C.correlated_block(n, rng, loadings, phi, ["A", "B"], C.THRESHOLDS)[HSNS]


def simulate_dd(n, rng, us):
    """A weak general factor under strong specifics in the United States; a
    dominant general factor with little left over elsewhere."""
    general = DD_GENERAL if us else {i: NON_US_DD_GENERAL for i in DD}
    specific = DD_SPECIFIC if us else {i: NON_US_DD_SPECIFIC for i in DD}
    return C.bifactor_block(n, rng, general, specific, SPECIFIC_OF, DD, C.THRESHOLDS)


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        us = country == "US"
        frames.append(
            pd.concat(
                [
                    simulate_hsns(n, rng, us),
                    simulate_dd(n, rng, us),
                    C.demographics(n, rng, country),
                ],
                axis=1,
            )
        )
    return C.finalize(frames, rng, SEED)


# --------------------------------------------------------------------------
# Models and reliability
# --------------------------------------------------------------------------
def hsns_syntax():
    return "H =~ " + "+".join(HSNS)


def dd_bifactor():
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in SUBSCALES}
    lines = ["G =~ " + "+".join(DD)]
    lines += [f"S{k} =~ {'+'.join(by[k])}" for k in SUBSCALES]
    lines += [f"G ~~ 0*S{k}" for k in SUBSCALES]
    lines += ["SM ~~ 0*SP", "SM ~~ 0*SN", "SP ~~ 0*SN"]
    return "\n".join(lines)


def dd_three_factor():
    by = {k: [i for i in DD if SPECIFIC_OF[i] == k] for k in SUBSCALES}
    return "\n".join(f"F{k} =~ {'+'.join(by[k])}" for k in SUBSCALES)


def reference_syntax():
    """The joint model calibrated on the United States, covering both instruments."""
    return hsns_syntax() + "\n" + dd_bifactor()


def omega(general, specific, items):
    """Proportion of a sum score's variance due to the general factor, and to all
    common factors: omega_hierarchical and omega_total."""
    from_general = sum(general[i] for i in items) ** 2
    from_specific = sum(
        sum(specific[i] for i in items if SPECIFIC_OF[i] == k) ** 2
        for k in dict.fromkeys(SPECIFIC_OF[i] for i in items)
    )
    unique = sum(1 - general[i] ** 2 - specific[i] ** 2 for i in items)
    total = from_general + from_specific + unique
    return from_general / total, (from_general + from_specific) / total


def hsns_omega():
    """Omega of the HSNS total. With one factor it is both omega_h and omega_total."""
    common = sum(HSNS_LOADINGS.values()) ** 2
    return common / (common + sum(1 - v**2 for v in HSNS_LOADINGS.values()))


def omega_h_from_fit(X):
    """Omega_hierarchical of the Dirty Dozen total, from a fitted bifactor model."""
    import semopy

    model = semopy.Model(dd_bifactor())
    model.fit(X[DD])
    ins = model.inspect(std_est=True)
    load = ins[ins.op == "~"]
    general = {r["lval"]: float(r["Est. Std"]) for _, r in load.iterrows() if r["rval"] == "G"}
    specific = {r["lval"]: float(r["Est. Std"]) for _, r in load.iterrows() if r["rval"] != "G"}
    return omega(general, specific, DD)[0]


def population_correlation_matrix():
    """Correlation matrix a perfectly specified US model would reproduce."""
    rng = np.random.default_rng(SEED + 999)
    big = pd.concat(
        [simulate_hsns(POP_REFERENCE_N, rng, True), simulate_dd(POP_REFERENCE_N, rng, True)], axis=1
    )
    return C.population_matrix(big, ITEMS)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    omega_h, omega_total = omega(DD_GENERAL, DD_SPECIFIC, DD)
    return {
        "task_id": TASK_ID,
        "scored": {
            "scoring": TRUTH,
            "item_assignment": {
                **{i: "H" for i in HSNS},
                **{i: f"S{SPECIFIC_OF[i]}" for i in DD},
            },
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
            "hsns": {
                "loadings": HSNS_LOADINGS,
                "omega": round(hsns_omega(), 3),
                "residual_correlation": {
                    "items": list(HSNS_RESIDUAL_CORR[0]),
                    "covariance": HSNS_RESIDUAL_CORR[1],
                },
                "non_us_loading": NON_US_HSNS_LOADING,
                "non_us_phi": NON_US_HSNS_PHI,
                "non_us_f1_items": NON_US_HSNS_F1,
                "non_us_f2_items": NON_US_HSNS_F2,
            },
            "dirty_dozen": {
                "general_loadings": DD_GENERAL,
                "specific_loadings": DD_SPECIFIC,
                "specific_of_item": SPECIFIC_OF,
                "omega_hierarchical": round(omega_h, 3),
                "omega_total": round(omega_total, 3),
                "non_us_general": NON_US_DD_GENERAL,
                "non_us_specific": NON_US_DD_SPECIFIC,
            },
            "omega_h_floor": OMEGA_H_FLOOR,
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        "artifacts/level_1/task_08/truth.json",
        ITEMS,
        [
            {
                "key": "scoring",
                "fn": "score_label_panel",
                "truth_key": "scored.scoring",
                "criterion": "score_justification",
            },
            {
                "key": "item_assignment",
                "fn": "score_partition",
                "derive_from": "refit_primary_loadings",
                "truth_key": "scored.item_assignment",
                "criterion": "structure",
            },
        ],
    )
    contract["syntax_whitelist"]["max_factors"] = 6
    return [
        {
            "id": TASK_ID,
            "name": "What may each instrument's scores be used for?",
            "uuid": "1f959f96-36f1-490b-b6e9-9178773f2944",
            "keywords": [
                "psychometrics",
                "reliability",
                "dimensionality",
                "scoring",
                "measurement",
            ],
            "metrics": ["binary", "partial"],
            "level": 1,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
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

    def sub(hsns, dd, syntax=None):
        return {
            "model_syntax": syntax or reference_syntax(),
            "scoring": {"hsns": hsns, "dirty_dozen": dd},
        }

    return {
        "correct": sub("total_only", "subscales_only"),
        "alpha-led: both totals fine": sub("total_only", "total_and_subscales"),
        "instruments swapped": sub("subscales_only", "total_only"),
        "Dirty Dozen total kept": sub("total_only", "total_and_subscales"),
        "right answer, three correlated factors": sub(
            "total_only", "subscales_only", hsns_syntax() + "\n" + dd_three_factor()
        ),
    }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def verify(df, pop):
    """Confirm alpha ranks the instruments backwards and omega_h puts them right."""
    del pop
    import semopy

    us = C.analysis_sample(df, ITEMS)
    pooled = df[ITEMS][(df[ITEMS] != 0).all(axis=1)].astype(float)
    print(f"US calibration sample: {len(us):,}   whole file: {len(pooled):,}\n")

    a_hsns, a_dd = C.alpha(us[HSNS]), C.alpha(us[DD])
    w_hsns = hsns_omega()
    w_dd_h, w_dd_total = omega(DD_GENERAL, DD_SPECIFIC, DD)
    print(
        f"{'instrument':14s} {'alpha':>7s} {'omega_h':>9s} {'omega_total':>12s} "
        f"{'verdict':>20s}"
    )
    print(f"{'HSNS':14s} {a_hsns:7.3f} {w_hsns:9.3f} {w_hsns:12.3f} " f"{TRUTH['hsns']:>20s}")
    print(
        f"{'Dirty Dozen':14s} {a_dd:7.3f} {w_dd_h:9.3f} {w_dd_total:12.3f} "
        f"{TRUTH['dirty_dozen']:>20s}"
    )

    print("\nDirty Dozen subscales")
    for k in SUBSCALES:
        block = [i for i in DD if SPECIFIC_OF[i] == k]
        h, total = omega(DD_GENERAL, DD_SPECIFIC, block)
        print(
            f"  {k}: alpha {C.alpha(us[block]):.3f}  omega_total {total:.3f}  "
            f"specific share {total - h:.3f}"
        )

    print("\nModel comparison on the United States")
    print(
        f"{'instrument':14s} {'model':22s} {'df':>4s} {'CFI':>8s} {'RMSEA':>7s} "
        f"{'BIC':>10s} {'phi':>7s} {'chi2 p':>9s}"
    )
    fits = {}
    for instrument, items, models in [
        (
            "HSNS",
            HSNS,
            [
                ("unidimensional", hsns_syntax()),
                (
                    "two_correlated",
                    f"A =~ {'+'.join(NON_US_HSNS_F1)}\n" f"B =~ {'+'.join(NON_US_HSNS_F2)}",
                ),
            ],
        ),
        ("Dirty Dozen", DD, [("three_correlated", dd_three_factor()), ("bifactor", dd_bifactor())]),
    ]:
        for name, spec in models:
            model = semopy.Model(spec)
            model.fit(us[items])
            stats = semopy.calc_stats(model)
            ins = model.inspect(std_est=True)
            cov = ins[
                (ins.op == "~~")
                & (ins.lval != ins.rval)
                & ~ins.lval.isin(items)
                & ~ins.rval.isin(items)
            ]
            free = pd.to_numeric(cov["Est. Std"], errors="coerce").abs()
            phi = float(free.max()) if len(free) and free.notna().any() else float("nan")
            bic = float(stats["chi2"].iloc[0]) + len(model.param_vals) * np.log(len(us))
            p_value = float(stats["chi2 p-value"].iloc[0])
            fits[name] = (float(stats["CFI"].iloc[0]), bic, phi, p_value)
            print(
                f"{instrument:14s} {name:22s} {stats['DoF'].iloc[0]:4.0f} "
                f"{stats['CFI'].iloc[0]:8.4f} {stats['RMSEA'].iloc[0]:7.4f} "
                f"{bic:10.1f} {phi:7.3f} {p_value:9.2e}"
            )

    pooled_cfi = float(semopy.calc_stats(_fitted(hsns_syntax(), pooled, HSNS))["CFI"].iloc[0])
    pooled_omega_h = omega_h_from_fit(pooled)
    print(
        f"\nSkipping the United States filter: HSNS unidimensional CFI "
        f"{fits['unidimensional'][0]:.4f} -> {pooled_cfi:.4f}, "
        f"Dirty Dozen omega_h {w_dd_h:.3f} -> {pooled_omega_h:.3f}, "
        f"alpha {a_dd:.3f} -> {C.alpha(pooled[DD]):.3f}"
    )

    checks = [
        ("alpha ranks the Dirty Dozen above the HSNS", a_dd - a_hsns > 0.05),
        ("omega_h ranks them the other way, decisively", w_hsns > OMEGA_H_FLOOR and w_dd_h < 0.50),
        ("the Dirty Dozen total is reliable yet uninterpretable", w_dd_total > 0.85),
        (
            "its subscales are worth scoring",
            all(
                omega(DD_GENERAL, DD_SPECIFIC, [i for i in DD if SPECIFIC_OF[i] == k])[1] > 0.80
                for k in SUBSCALES
            ),
        ),
        (
            "the bifactor is not a relabelling of three correlated factors",
            fits["three_correlated"][1] - fits["bifactor"][1] > 100,
        ),
        (
            "a second HSNS factor is redundant rather than merely unhelpful",
            fits["two_correlated"][2] > 0.90,
        ),
        (
            "skipping the US filter breaks both verdicts",
            pooled_cfi < 0.95 and pooled_omega_h > 0.50,
        ),
        (
            "neither generating model recovers its data exactly",
            fits["unidimensional"][3] < 0.001 and fits["bifactor"][3] < 0.001,
        ),
    ]
    return C.report(checks)


def _fitted(spec, X, items):
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    return model


def naive(df, pop):
    """Confirm that reporting alpha, as most papers do, picks the wrong instrument."""
    del pop
    us = C.analysis_sample(df, ITEMS)
    a_hsns, a_dd = C.alpha(us[HSNS]), C.alpha(us[DD])
    leader = "Dirty Dozen" if a_dd > a_hsns else "HSNS"
    print("Judging each total score by Cronbach's alpha:\n")
    print(f"  HSNS         alpha {a_hsns:.3f}  -> total_only")
    print(f"  Dirty Dozen  alpha {a_dd:.3f}  -> total_only")
    print(f"\n  alpha calls the {leader} the better-measured instrument")
    print(f"  truth: {TRUTH['hsns']} and {TRUTH['dirty_dozen']}")
    wrong = sum(TRUTH[k] != "total_only" for k in TRUTH)
    print(f"  alpha gets {wrong} of {len(TRUTH)} instruments wrong")
    return C.report(
        [
            ("alpha prefers the instrument whose total means least", a_dd > a_hsns),
            ("and so licenses a score the data do not support", wrong >= 1),
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
    floor = C.evaluate_model(reference_syntax(), C.analysis_sample(df, ITEMS), ITEMS, pop)
    C.write_artifacts(
        OUT_DIR,
        TASK_JSON,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
