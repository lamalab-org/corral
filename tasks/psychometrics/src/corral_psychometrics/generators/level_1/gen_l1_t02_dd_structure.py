#!/usr/bin/env python3
"""Generate Task 02 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t02_dd_structure"
SEED = 20260902
TASK = paths.task(__file__)

ITEMS = C.DD_ITEMS

# Every item measures one broad trait and, on top of that, the narrower trait
# its subscale is named for. The two are unrelated to each other, so what an
# item explains is split between them.
SPECIFIC_OF = {
    "DDM1": "M",
    "DDM2": "M",
    "DDM3": "M",
    "DDM4": "M",
    "DDP1": "P",
    "DDP2": "P",
    "DDP3": "P",
    "DDP4": "P",
    "DDN1": "N",
    "DDN2": "N",
    "DDN3": "N",
    "DDN4": "N",
}
GENERAL_LOADINGS = {
    "DDM1": 0.58,
    "DDM2": 0.52,
    "DDM3": 0.48,
    "DDM4": 0.60,
    "DDP1": 0.55,
    "DDP2": 0.50,
    "DDP3": 0.53,
    "DDP4": 0.38,
    "DDN1": 0.42,
    "DDN2": 0.40,
    "DDN3": 0.50,
    "DDN4": 0.52,
}
SPECIFIC_LOADINGS = {
    "DDM1": 0.48,
    "DDM2": 0.44,
    "DDM3": 0.42,
    "DDM4": 0.50,
    "DDP1": 0.52,
    "DDP2": 0.48,
    "DDP3": 0.55,
    "DDP4": 0.35,
    "DDN1": 0.58,
    "DDN2": 0.56,
    "DDN3": 0.45,
    "DDN4": 0.30,
}

# Outside the US there is no broad trait at all, only three related ones, and
# all of them are measured worse. Analysing everyone together recovers
# neither picture.
NON_US_LOADING = {"M": 0.52, "P": 0.48, "N": 0.50}
NON_US_PHI = {("M", "P"): 0.35, ("M", "N"): 0.25, ("P", "N"): 0.20}

# The HSNS items are in the file but are not part of this task.
HSNS_LOADINGS = {
    "HSNS1": ("ego", 0.55),
    "HSNS4": ("ego", 0.30),
    "HSNS5": ("ego", 0.70),
    "HSNS6": ("ego", 0.48),
    "HSNS8": ("ego", 0.71),
    "HSNS10": ("ego", 0.66),
    "HSNS2": ("sens", 0.76),
    "HSNS3": ("sens", 0.58),
    "HSNS7": ("sens", 0.69),
    "HSNS9": ("sens", 0.52),
}
HSNS_PHI = 0.35

POP_REFERENCE_N = 1_500_000

PROMPT = """\
You are given responses from an online personality survey. The codebook describes \
every column.

Read the data and select participants from the United States. Report the one \
model for the Dirty Dozen you would defend to a reviewer.\
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<complete lavaan/semopy model>",
  "loadings": {"<item>": <float>, ...}
}

`model_syntax` uses lavaan notation (`=~` loadings, `~~` (co)variances, `0*` to fix a parameter to zero).

`loadings` gives, for each item you analysed, the largest absolute standardised loading that item has on any factor in your model.
"""


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        demo = C.demographics(n, rng, country)
        if country == "US":
            dd = C.bifactor_block(
                n,
                rng,
                GENERAL_LOADINGS,
                SPECIFIC_LOADINGS,
                SPECIFIC_OF,
                ITEMS,
                C.THRESHOLDS,
            )
        else:
            names = ["M", "P", "N"]
            phi = np.eye(3)
            for (a, b), v in NON_US_PHI.items():
                phi[names.index(a), names.index(b)] = v
                phi[names.index(b), names.index(a)] = v
            dd = C.correlated_block(
                n,
                rng,
                {i: (SPECIFIC_OF[i], NON_US_LOADING[SPECIFIC_OF[i]]) for i in ITEMS},
                phi,
                names,
                C.THRESHOLDS,
            )[ITEMS]
        hsns = C.correlated_block(
            n,
            rng,
            HSNS_LOADINGS,
            np.array([[1.0, HSNS_PHI], [HSNS_PHI, 1.0]]),
            ["ego", "sens"],
            C.THRESHOLDS,
        )
        frames.append(pd.concat([hsns, dd, demo], axis=1))
    return C.finalize(frames, rng, SEED)


def reference_syntax():
    """lavaan syntax for the generating model, used as the scoring floor."""
    by = {k: [i for i in ITEMS if SPECIFIC_OF[i] == k] for k in "MPN"}
    lines = [f"G =~ {'+'.join(ITEMS)}"]
    lines += [f"S{k} =~ {'+'.join(by[k])}" for k in "MPN"]
    lines += [f"G ~~ 0*S{k}" for k in "MPN"]
    lines += ["SM ~~ 0*SP", "SM ~~ 0*SN", "SP ~~ 0*SN"]
    return "\n".join(lines)


def candidate_models():
    """The rival structures a competent analyst would fit to these items."""
    by = {k: [i for i in ITEMS if SPECIFIC_OF[i] == k] for k in "MPN"}
    three = "\n".join(f"F{k} =~ {'+'.join(by[k])}" for k in "MPN")
    return {
        "unidimensional": f"G =~ {'+'.join(ITEMS)}",
        "three_correlated_factors": three,
        "second_order": three + "\nDT =~ FM+FP+FN",
        "bifactor_general_plus_specifics": reference_syntax(),
    }


def population_correlation_matrix():
    """Correlation matrix a perfectly specified model would reproduce."""
    rng = np.random.default_rng(SEED + 999)
    big = C.bifactor_block(
        POP_REFERENCE_N,
        rng,
        GENERAL_LOADINGS,
        SPECIFIC_LOADINGS,
        SPECIFIC_OF,
        ITEMS,
        C.THRESHOLDS,
    )
    return C.population_matrix(big, list(big.columns))


def primary_loadings():
    """Largest absolute loading generated for each item, across all factors."""
    return {i: round(max(GENERAL_LOADINGS[i], SPECIFIC_LOADINGS[i]), 3) for i in ITEMS}


def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "model_family": "bifactor_general_plus_specifics",
            "n_factors": 4,
            "loadings": primary_loadings(),
            "item_assignment": {
                i: ("G" if GENERAL_LOADINGS[i] >= SPECIFIC_LOADINGS[i] else f"S{SPECIFIC_OF[i]}")
                for i in ITEMS
            },
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
                "general_loadings": GENERAL_LOADINGS,
                "specific_loadings": SPECIFIC_LOADINGS,
                "specific_of_item": SPECIFIC_OF,
                "factors_orthogonal": True,
            },
            "non_us": {
                "structure": "three_correlated_factors",
                "loadings": NON_US_LOADING,
                "phi": {f"{a}_{b}": v for (a, b), v in NON_US_PHI.items()},
            },
            "hsns": {
                "loadings": {k: v[1] for k, v in HSNS_LOADINGS.items()},
                "phi": HSNS_PHI,
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
            "name": "Dirty Dozen factor structure discovery",
            "uuid": "3d2a91c7-58be-4f16-a0c4-7e91b2d6c402",
            "keywords": [
                "psychometrics",
                "factor analysis",
                "model selection",
                "bifactor",
                "Dirty Dozen",
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

    truth = criteria["bifactor_general_plus_specifics"]
    three, second = criteria["three_correlated_factors"], criteria["second_order"]
    checks = [
        (
            "the three-factor model looks acceptable on its own (CFI > .95)",
            three["CFI"] > 0.95,
        ),
        ("but is beaten by the generating model on fit", truth["CFI"] > three["CFI"]),
        ("on parsimony", truth["BIC"] < three["BIC"] - 10),
        (
            "and on accuracy",
            truth["sigma_max_abs_deviation"] < three["sigma_max_abs_deviation"] - 0.01,
        ),
        # With exactly three subscales these two are the same model written two
        # ways, so no fit measure can separate them. The tolerance allows for
        # the solver stopping in slightly different places.
        (
            "second-order is statistically equivalent to three correlated factors",
            abs(second["CFI"] - three["CFI"]) < 1e-3
            and abs(second["chi2"] - three["chi2"]) < 1.0
            and second["df"] == three["df"],
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
    truth = primary_loadings()
    spec = reference_syntax()
    print("Effect of the analysis sample on the reported loadings\n")
    print(f"{'sample':22s} {'N':>7s} {'worst |error|':>14s}  within 0.08?")
    outcome = {}
    for label, sub in (
        ("US only (correct)", df[df.country == "US"]),
        ("pooled (no filter)", df),
    ):
        X = sub[ITEMS]
        X = X[(X != 0).all(axis=1)].astype(float)
        fitted = {k: abs(v) for k, v in C.loadings(C.fit(spec, X, ITEMS)).items()}
        worst = max(abs(fitted[i] - truth[i]) for i in ITEMS)
        outcome[label] = worst <= 0.08
        print(f"{label:22s} {len(X):7,} {worst:14.3f}  {'yes' if outcome[label] else 'NO'}")
    return C.report(
        [
            (
                "the pooled analysis reports loadings outside tolerance",
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
