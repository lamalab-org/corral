#!/usr/bin/env python3
"""Generate Task 01 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t01_hsns_structure"
SEED = 20260901
TASK = paths.task(__file__)

ITEMS = C.HSNS_ITEMS
F1, F2 = "egocentrism", "oversensitivity"

# Item -> (trait, loading). A loading is how strongly the item tracks the
# trait, 0 to 1. HSNS4 is a weak item by design.
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

# Two small extras, so that the true model is a close approximation of the
# data rather than an exact match. Real data never match a model exactly.
CROSS_LOADING = ("HSNS9", F1, 0.15)
RESIDUAL_CORR = (("HSNS5", "HSNS10"), 0.10)

# Participants outside the US come from a weaker, less differentiated population.
LOADING_SCALE_NON_US = 0.72
PHI_NON_US = 0.68

# These two items are answered slightly differently by women at the same trait
# level. Too small to change the answer to this task.
GENDER_DIF = {"HSNS3": -0.18, "HSNS8": 0.18}

# The Dark Triad items are in the file but are not part of this task.
DD_LOADINGS = {
    "DDM1": ("mach", 0.74),
    "DDM2": ("mach", 0.71),
    "DDM3": ("mach", 0.58),
    "DDM4": ("mach", 0.77),
    "DDP1": ("psych", 0.69),
    "DDP2": ("psych", 0.73),
    "DDP3": ("psych", 0.66),
    "DDP4": ("psych", 0.48),
    "DDN1": ("narc", 0.78),
    "DDN2": ("narc", 0.81),
    "DDN3": ("narc", 0.70),
    "DDN4": ("narc", 0.62),
}
DD_PHI = {("mach", "psych"): 0.58, ("mach", "narc"): 0.42, ("psych", "narc"): 0.39}

POP_REFERENCE_N = 2_000_000

PROMPT = """\
You are given responses from an online personality survey. The codebook describes every column.\

Read the data and select participants from the United States. Identify the \
theoretically plausible psychometric models for the Hypersensitive Narcissism Scale (HSNS). \
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<complete lavaan/semopy model>",
  "loadings": {"<item>": <float>, ...},
  "factor_correlation": <float, or null if your model has no two oblique factors>
}

`model_syntax` uses lavaan notation (`=~` loadings, `~~` (co)variances, `0*` to fix a parameter to zero).

`loadings` gives, for each item you analysed, the largest absolute standardised loading that item has on any factor in your model.
"""


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        is_us = country == "US"
        demo = C.demographics(n, rng, country)
        phi = PHI if is_us else PHI_NON_US
        hsns = C.correlated_block(
            n,
            rng,
            LOADINGS,
            np.array([[1.0, phi], [phi, 1.0]]),
            [F1, F2],
            C.THRESHOLDS,
            scale=1.0 if is_us else LOADING_SCALE_NON_US,
            cross=CROSS_LOADING if is_us else None,
            resid_corr=RESIDUAL_CORR if is_us else None,
            dif=GENDER_DIF if is_us else None,
            female=demo.gender == 2,
        )
        names = ["mach", "psych", "narc"]
        phi_dd = np.eye(3)
        for (a, b), v in DD_PHI.items():
            phi_dd[names.index(a), names.index(b)] = v
            phi_dd[names.index(b), names.index(a)] = v
        dd = C.correlated_block(n, rng, DD_LOADINGS, phi_dd, names, C.THRESHOLDS)
        frames.append(pd.concat([hsns, dd, demo], axis=1))
    return C.finalize(frames, rng, SEED)


def reference_syntax():
    """lavaan syntax for the generating model, used as the scoring floor."""
    f1 = "+".join(k for k, v in LOADINGS.items() if v[0] == F1)
    f2 = "+".join(k for k, v in LOADINGS.items() if v[0] == F2)
    return f"F1 =~ {f1}\nF2 =~ {f2}"


def candidate_models():
    """The rival structures a competent analyst would fit to these items."""
    f1 = "+".join(k for k, v in LOADINGS.items() if v[0] == F1)
    f2 = "+".join(k for k, v in LOADINGS.items() if v[0] == F2)
    every = "+".join(ITEMS)
    return {
        "unidimensional": f"G =~ {every}",
        "two_correlated_factors": reference_syntax(),
        "two_orthogonal_factors": f"F1 =~ {f1}\nF2 =~ {f2}\nF1 ~~ 0*F2",
        "three_correlated_factors": (
            "F1 =~ HSNS1+HSNS8+HSNS6\nF2 =~ HSNS4+HSNS5+HSNS10\nF3 =~ HSNS2+HSNS3+HSNS7+HSNS9"
        ),
        "unidimensional_with_correlated_residuals": (
            f"G =~ {every}\nHSNS2 ~~ HSNS7\nHSNS5 ~~ HSNS10\nHSNS1 ~~ HSNS8"
        ),
        "bifactor_general_plus_specifics": (
            f"G =~ {every}\nS1 =~ {f1}\nS2 =~ {f2}\nG ~~ 0*S1\nG ~~ 0*S2\nS1 ~~ 0*S2"
        ),
    }


def population_correlation_matrix():
    """Correlation matrix a perfectly specified model would reproduce."""
    rng = np.random.default_rng(SEED + 999)
    big = C.correlated_block(
        POP_REFERENCE_N,
        rng,
        LOADINGS,
        np.array([[1.0, PHI], [PHI, 1.0]]),
        [F1, F2],
        C.THRESHOLDS,
        cross=CROSS_LOADING,
        resid_corr=RESIDUAL_CORR,
    )[ITEMS]
    return C.population_matrix(big, list(big.columns))


def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "model_family": "two_correlated_factors",
            "n_factors": 2,
            "loadings": {k: v[1] for k, v in LOADINGS.items()},
            "factor_correlation": PHI,
            "item_assignment": {k: ("F1" if v[0] == F1 else "F2") for k, v in LOADINGS.items()},
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
            "us": {
                "loadings": {k: v[1] for k, v in LOADINGS.items()},
                "factor_of_item": {k: v[0] for k, v in LOADINGS.items()},
                "phi": PHI,
                "cross_loading": list(CROSS_LOADING),
                "residual_correlation": [list(RESIDUAL_CORR[0]), RESIDUAL_CORR[1]],
            },
            "non_us": {"loading_scale": LOADING_SCALE_NON_US, "phi": PHI_NON_US},
            "gender_dif": GENDER_DIF,
            "dark_triad": {
                "loadings": {k: v[1] for k, v in DD_LOADINGS.items()},
                "phi": {f"{a}_{b}": v for (a, b), v in DD_PHI.items()},
            },
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    return [
        {
            "id": TASK_ID,
            "name": "HSNS factor structure discovery",
            "uuid": "b1f7c2ae-4a63-4a0e-9b5f-2c7c3e5a1d01",
            "keywords": ["psychometrics", "factor analysis", "model selection", "HSNS"],
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
            "scoring_params": C.scoring_contract(
                TASK,
                ITEMS,
                [
                    {
                        "key": "loadings",
                        "fn": "score_vector",
                        "truth_key": "scored.loadings",
                        "tol": 0.08,
                        "all_must_be_within": True,
                        "criterion": "parameter_quality",
                    },
                    {
                        "key": "factor_correlation",
                        "fn": "score_scalar",
                        "truth_key": "scored.factor_correlation",
                        "tol": 0.06,
                        "applicable_if": "model_has_two_oblique_factors",
                        "criterion": "parameter_quality",
                    },
                    {
                        "key": "item_assignment",
                        "fn": "score_partition",
                        "derive_from": "refit_primary_loadings",
                        "truth_key": "scored.item_assignment",
                        "criterion": "structure",
                    },
                ],
            ),
        }
    ]


def verify(df, pop):
    """Confirm the intended answer wins and each rival fails for its own reason."""
    from scipy.stats import chi2 as chi2_dist

    X = C.analysis_sample(df, ITEMS)
    print(f"US complete-case N = {len(X):,}\n")

    criteria, rows = {}, []
    for name, spec in candidate_models().items():
        crit = C.evaluate_model(spec, X, ITEMS, pop)
        criteria[name] = crit
        rows.append(
            {
                "model": name,
                "df": crit["df"],
                "p": 1 - chi2_dist.cdf(crit["chi2"], crit["df"]),
                "CFI": crit["CFI"],
                "RMSEA": crit["RMSEA"],
                "BIC": crit["BIC"],
                "sigma": crit["sigma_max_abs_deviation"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    truth = criteria["two_correlated_factors"]
    checks = [
        (
            "chi-square rejects the generating model",
            1 - chi2_dist.cdf(truth["chi2"], truth["df"]) < 0.01,
        ),
        ("approximate fit of the generating model is good", truth["CFI"] > 0.97),
        (
            "a rival beats it on fit, so fit alone cannot decide",
            any(
                c["CFI"] > truth["CFI"]
                for n, c in criteria.items()
                if n != "two_correlated_factors"
            ),
        ),
        (
            "unidimensional is clearly rejected",
            criteria["unidimensional"]["CFI"] < 0.90,
        ),
    ]
    return C.report(checks)


def naive(df, pop):
    """Confirm that skipping the US filter changes the reported answer."""
    del pop
    spec = reference_syntax()
    print("Effect of the analysis sample on the reported factor correlation\n")
    print(f"{'sample':22s} {'N':>7s} {'phi':>7s}  within 0.35 +/- 0.06?")
    outcome = {}
    for label, sub in (
        ("US only (correct)", df[df.country == "US"]),
        ("pooled (no filter)", df),
    ):
        X = sub[ITEMS]
        X = X[(X != 0).all(axis=1)].astype(float)
        import semopy

        model = semopy.Model(spec)
        model.fit(X)
        ins = model.inspect(std_est=True)
        cov = ins[
            (ins.op == "~~")
            & (ins.lval != ins.rval)
            & ins.lval.isin(["F1", "F2"])
            & ins.rval.isin(["F1", "F2"])
        ]
        phi = float(pd.to_numeric(cov["Est. Std"], errors="coerce").iloc[0])
        outcome[label] = abs(phi - PHI) <= 0.06
        print(f"{label:22s} {len(X):7,} {phi:7.3f}  {'yes' if outcome[label] else 'NO'}")
    return C.report(
        [
            (
                "the pooled analysis reports a phi outside tolerance",
                outcome["US only (correct)"] and not outcome["pooled (no filter)"],
            )
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)

    if action != "build":
        pop = population_correlation_matrix()
        return verify(df, pop) if action == "verify" else naive(df, pop)

    print(f"Fitting the scoring reference (population draw N={POP_REFERENCE_N:,}) ...")
    pop = population_correlation_matrix()
    floor = C.evaluate_model(reference_syntax(), C.analysis_sample(df, ITEMS), ITEMS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
