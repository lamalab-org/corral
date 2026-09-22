#!/usr/bin/env python3
"""Generate Task 05 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t05_defensible_comparisons"
SEED = 20260905
TASK = paths.task(__file__)

HSNS = C.HSNS_ITEMS
OTHER = ["DDN1", "DDN2", "DDN3", "DDN4"]  # the second trait
MODEL_VARS = HSNS + OTHER
ALL_VARS = MODEL_VARS + ["gender"]

# Identical in both groups: the items measure the trait the same way.
HSNS_LOADINGS = {
    "HSNS1": 0.55,
    "HSNS2": 0.76,
    "HSNS3": 0.58,
    "HSNS4": 0.52,
    "HSNS5": 0.70,
    "HSNS6": 0.48,
    "HSNS7": 0.69,
    "HSNS8": 0.71,
    "HSNS9": 0.60,
    "HSNS10": 0.66,
}
OTHER_LOADINGS = {"DDN1": 0.78, "DDN2": 0.81, "DDN3": 0.70, "DDN4": 0.62}

# Six items are answered differently at the same trait level, in both
# directions. Four are clean - but nothing in the data says which four, so no
# reference set can be defended and the latent means are not identified.
ITEM_SHIFTS = {
    "HSNS1": +0.40,
    "HSNS3": -0.45,
    "HSNS4": +0.35,
    "HSNS7": -0.50,
    "HSNS8": +0.45,
    "HSNS10": -0.40,
}
CLEAN_ITEMS = [i for i in HSNS if i not in ITEM_SHIFTS]

# Real, and not recoverable from these data.
LATENT_DIFFERENCE = 0.30

# How the two traits relate, per group. This IS comparable.
CORRELATION = {"men": 0.50, "women": 0.30}

LOADING_SCALE_NON_US = 0.80
POP_REFERENCE_N = 600_000

COMPARISONS = {
    "factor_structure": True,
    "loadings": True,
    "factor_variances": True,
    "association_with_other_trait": True,
    "latent_means": False,
    "observed_score_means": False,
}

PROMPT = """\
You are given responses from an online personality survey. The codebook describes \
every column.

Read the data and select participants from the United States who reported male or \
female. Investigate how far the Hypersensitive Narcissism Scale (HSNS) measures the \
same thing in men and in women, and on that basis decide which of the following \
comparisons between the two groups are psychometrically defensible:

  factor_structure               do the same items measure the same factor
  loadings                       does the trait relate to its items the same way
  factor_variances               is the spread of the trait comparable
  association_with_other_trait   is its relationship with the Dark Triad narcissism
                                 items comparable
  latent_means                   is the average level of the trait comparable
  observed_score_means           are the average total scores comparable

Carry out the comparisons you judge defensible, and report the correlation between \
the two traits in each group.

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<complete lavaan/semopy model>",
  "comparisons": {
    "factor_structure": <true|false>, "loadings": <true|false>,
    "factor_variances": <true|false>, "association_with_other_trait": <true|false>,
    "latent_means": <true|false>, "observed_score_means": <true|false>
  },
  "trait_correlation_men": <float>,
  "trait_correlation_women": <float>
}

`model_syntax` is the measurement model you used, in lavaan notation (`=~` loadings,
`~~` (co)variances). It is re-fitted during evaluation, so it must be complete and
runnable.
"""


def simulate(n, rng, female, scale=1.0, us=True):
    """Both traits, with item shifts and a group-specific trait correlation."""
    rho = np.where(female, CORRELATION["women"], CORRELATION["men"]) if us else np.full(n, 0.40)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    eta_h = z1 + (np.where(female, LATENT_DIFFERENCE, 0.0) if us else 0.0)
    eta_o = rho * z1 + np.sqrt(np.clip(1 - rho**2, 0, None)) * z2

    out = {}
    for item, base in HSNS_LOADINGS.items():
        lam = base * scale
        ystar = lam * eta_h + rng.normal(0, np.sqrt(1 - lam**2), n)
        tau = np.asarray(C.THRESHOLDS[item])
        if us and item in ITEM_SHIFTS:
            out[item] = np.where(
                female, C.categorize(ystar, tau + ITEM_SHIFTS[item]), C.categorize(ystar, tau)
            )
        else:
            out[item] = C.categorize(ystar, tau)
    for item, base in OTHER_LOADINGS.items():
        lam = base * scale
        ystar = lam * eta_o + rng.normal(0, np.sqrt(1 - lam**2), n)
        out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[MODEL_VARS]


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        us = country == "US"
        demo = C.demographics(n, rng, country)
        block = simulate(
            n, rng, (demo.gender == 2).to_numpy(), 1.0 if us else LOADING_SCALE_NON_US, us
        )
        filler = C.correlated_block(
            n,
            rng,
            {i: ("x", 0.6) for i in C.DD_ITEMS if i not in OTHER},
            np.array([[1.0]]),
            ["x"],
            C.THRESHOLDS,
        )
        frames.append(pd.concat([block, filler, demo], axis=1))
    return C.finalize(frames, rng, SEED)


def comparison_sample(df):
    """US respondents reporting male or female, with complete answers."""
    sub = df[(df.country == "US") & df.gender.isin([1, 2])]
    X = sub[ALL_VARS]
    return X[(X[MODEL_VARS] != 0).all(axis=1)].astype(float)


def reference_syntax():
    """The joint measurement model: the two traits, correlated."""
    return f"F1 =~ {'+'.join(HSNS)}\n" f"F2 =~ {'+'.join(OTHER)}"


def trait_correlation(X):
    """Correlation between the two latent traits in one group."""
    import semopy

    model = semopy.Model(reference_syntax())
    model.fit(X[MODEL_VARS])
    ins = model.inspect(std_est=True)
    row = ins[
        (ins.op == "~~")
        & (ins.lval != ins.rval)
        & ins.lval.isin(["F1", "F2"])
        & ins.rval.isin(["F1", "F2"])
    ]
    return float(pd.to_numeric(row["Est. Std"], errors="coerce").iloc[0])


def latent_mean_difference(X, anchors):
    """Latent mean difference implied by treating `anchors` as unbiased."""
    import semopy

    freed = [i for i in HSNS if i not in anchors]
    spec = f"F1 =~ {'+'.join(HSNS)}\nF1 ~ gender" + "".join(f"\n{i} ~ gender" for i in freed)
    model = semopy.Model(spec)
    model.fit(X[HSNS + ["gender"]])
    ins = model.inspect(std_est=True)
    row = ins[(ins.op == "~") & (ins.lval == "F1") & (ins.rval == "gender")]
    return float(pd.to_numeric(row["Est. Std"], errors="coerce").iloc[0])


def population_reference():
    """Per-group trait correlations and the population matrix, at large N."""
    rng = np.random.default_rng(SEED + 999)
    targets = {}
    for label, female in (("men", False), ("women", True)):
        big = simulate(POP_REFERENCE_N, rng, np.full(POP_REFERENCE_N, female))
        targets[label] = trait_correlation(big)
    rng2 = np.random.default_rng(SEED + 1001)
    female = rng2.random(POP_REFERENCE_N) < 0.377
    pooled = simulate(POP_REFERENCE_N, rng2, female)
    pooled["gender"] = np.where(female, 2, 1)
    pop = np.corrcoef(pooled[MODEL_VARS].values.T.astype(float))
    return targets, pop.round(3)


def build_truth(targets, floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "comparisons": COMPARISONS,
            "trait_correlation_men": round(targets["men"], 3),
            "trait_correlation_women": round(targets["women"], 3),
            "highest_invariance_level": "metric",
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": MODEL_VARS,
        },
        "generative_parameters": {
            "us": {
                "hsns_loadings": HSNS_LOADINGS,
                "other_loadings": OTHER_LOADINGS,
                "loadings_invariant_across_gender": True,
                "item_threshold_shifts_for_women": ITEM_SHIFTS,
                "clean_items": CLEAN_ITEMS,
                "latent_difference_women_minus_men": LATENT_DIFFERENCE,
                "latent_difference_identified": False,
                "trait_correlation": CORRELATION,
            },
            "non_us": {
                "loading_scale": LOADING_SCALE_NON_US,
                "item_threshold_shifts_for_women": {},
                "trait_correlation": 0.40,
            },
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    """Assemble the Corral task definition, including the scoring contract."""
    contract = C.scoring_contract(
        TASK,
        MODEL_VARS,
        [
            {
                "key": "comparisons",
                "fn": "score_boolean_panel",
                "truth_key": "scored.comparisons",
                "criterion": "defensibility",
            },
            {
                "key": "trait_correlation_men",
                "fn": "score_scalar",
                "truth_key": "scored.trait_correlation_men",
                "tol": 0.06,
                "criterion": "group_comparison",
            },
            {
                "key": "trait_correlation_women",
                "fn": "score_scalar",
                "truth_key": "scored.trait_correlation_women",
                "tol": 0.06,
                "criterion": "group_comparison",
            },
        ],
    )
    contract["subset"] = {"country": "US", "gender": [1, 2]}
    contract["syntax_whitelist"]["items"] = ALL_VARS
    return [
        {
            "id": TASK_ID,
            "name": "Which gender comparisons are defensible?",
            "uuid": "e91c6b74-3a85-4d02-b1f7-6c2e8a9d4505",
            "keywords": [
                "psychometrics",
                "measurement invariance",
                "item bias",
                "group comparison",
                "validity",
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
    men, women = X[X.gender == 1], X[X.gender == 2]
    base = {
        "model_syntax": reference_syntax(),
        "trait_correlation_men": round(trait_correlation(men), 3),
        "trait_correlation_women": round(trait_correlation(women), 3),
    }
    pooled = round(trait_correlation(X), 3)
    return {
        "correct": {**base, "comparisons": dict(COMPARISONS)},
        "compares the means too": {
            **base,
            "comparisons": {**COMPARISONS, "latent_means": True, "observed_score_means": True},
        },
        "refuses everything": {**base, "comparisons": {k: False for k in COMPARISONS}},
        "accepts everything": {**base, "comparisons": {k: True for k in COMPARISONS}},
        "right panel, pooled correlations": {
            **base,
            "comparisons": dict(COMPARISONS),
            "trait_correlation_men": pooled,
            "trait_correlation_women": pooled,
        },
    }


def verify(df, targets, pop):
    """Confirm what is comparable is recoverable and what is not is unstable."""
    import semopy

    X = comparison_sample(df)
    men, women = X[X.gender == 1], X[X.gender == 2]
    print(f"US sample: {len(men):,} men, {len(women):,} women\n")

    # loadings identical across groups -> metric invariance holds
    def loadings(sub):
        m = semopy.Model(reference_syntax())
        m.fit(sub[MODEL_VARS])
        ins = m.inspect(std_est=True)
        lo = ins[ins.op == "~"]
        return lo.assign(v=pd.to_numeric(lo["Est. Std"], errors="coerce")).set_index("lval")["v"]

    lm, lw = loadings(men), loadings(women)
    worst = max(abs(lm[i] - lw[i]) for i in HSNS)
    print(f"  largest loading difference across groups   {worst:.3f}")

    r_men, r_women = trait_correlation(men), trait_correlation(women)
    print(
        f"  trait correlation, men                     {r_men:+.3f} "
        f"(target {targets['men']:+.3f})"
    )
    print(
        f"  trait correlation, women                   {r_women:+.3f} "
        f"(target {targets['women']:+.3f})"
    )

    # the latent mean depends entirely on which items are assumed unbiased
    anchor_sets = {
        "the truly clean four": CLEAN_ITEMS,
        "four shifted one way": ["HSNS1", "HSNS4", "HSNS8", "HSNS2"],
        "four shifted the other": ["HSNS3", "HSNS7", "HSNS10", "HSNS5"],
        "all ten (assume no bias)": HSNS,
    }
    print("\n  latent mean difference, by which items are assumed unbiased:")
    estimates = {}
    for label, anchors in anchor_sets.items():
        estimates[label] = latent_mean_difference(X, anchors)
        print(f"    {label:28s} {estimates[label]:+.3f}")
    spread = max(estimates.values()) - min(estimates.values())
    print(f"    spread across anchor choices {spread:.3f}")

    checks = [
        ("loadings are invariant, so associations are comparable", worst < 0.08),
        (
            "the trait correlations are recoverable in each group",
            abs(r_men - targets["men"]) <= 0.06 and abs(r_women - targets["women"]) <= 0.06,
        ),
        ("and they really do differ between the groups", r_men - r_women > 0.12),
        (
            "the latent mean difference is not identified "
            "(anchor choice moves it by more than 0.20)",
            spread > 0.20,
        ),
    ]
    return C.report(checks)


def naive(df, targets, pop):
    """Confirm the default analysis reports a mean difference it cannot support."""
    del pop
    X = comparison_sample(df)
    men, women = X[X.gender == 1], X[X.gender == 2]
    tm, tw = men[HSNS].sum(1), women[HSNS].sum(1)
    pooled = np.sqrt(
        ((len(tm) - 1) * tm.var() + (len(tw) - 1) * tw.var()) / (len(tm) + len(tw) - 2)
    )
    d = (tw.mean() - tm.mean()) / pooled
    from scipy import stats

    t, p = stats.ttest_ind(tw, tm, equal_var=False)
    print(f"  t-test on total scores: d = {d:+.3f}, p = {p:.2e}")
    print("  a significant difference is available to report, and it is not")
    print("  interpretable: the items are not answered the same way by the groups.")
    return C.report(
        [
            ("the observed difference is statistically significant", p < 0.05),
            (
                "so the tempting comparison is the one that must be refused",
                COMPARISONS["observed_score_means"] is False,
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    print("Simulating ...")
    df = build_dataset(rng)
    print(f"Calibrating targets (population draw N={POP_REFERENCE_N:,}) ...")
    targets, pop = population_reference()

    if action != "build":
        return verify(df, targets, pop) if action == "verify" else naive(df, targets, pop)

    floor = C.evaluate_model(reference_syntax(), comparison_sample(df), MODEL_VARS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(targets, floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
