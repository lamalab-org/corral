#!/usr/bin/env python3
"""Generate Task 03 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t03_local_dependence"
SEED = 20260903
TASK = paths.task(__file__)

ITEMS = C.HSNS_ITEMS
TRAIT = "vulnerability"

# HSNS1 and HSNS8 track the trait strongly. That alone makes them one of the
# most correlated pairs, so sorting raw correlations points at them first.
LOADINGS = {
    "HSNS1": (TRAIT, 0.82),
    "HSNS2": (TRAIT, 0.58),
    "HSNS3": (TRAIT, 0.66),
    "HSNS4": (TRAIT, 0.55),
    "HSNS5": (TRAIT, 0.49),
    "HSNS6": (TRAIT, 0.61),
    "HSNS7": (TRAIT, 0.64),
    "HSNS8": (TRAIT, 0.80),
    "HSNS9": (TRAIT, 0.60),
    "HSNS10": (TRAIT, 0.53),
}

# Pairs that agree beyond what the trait explains. Both read as near-
# paraphrases: HSNS2 and HSNS7 are about other people's remarks, HSNS5 and
# HSNS10 about other people's troubles.
RESIDUAL_PAIRS = [(("HSNS2", "HSNS7"), 0.38), (("HSNS5", "HSNS10"), 0.33)]

# HSNS1 and HSNS8 read as the most similar pair of all and have no such
# agreement between them. They are the decoy.
DECOY_PAIR = ("HSNS1", "HSNS8")

# Outside the US the items are measured worse and no pair agrees beyond the
# trait, so analysing everyone together weakens every loading.
LOADING_SCALE_NON_US = 0.75

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

POP_REFERENCE_N = 1_500_000

PROMPT = """\
You are given responses from an online personality survey. The codebook describes \
every column.

Read the data and select participants from the United States. Identify the \
theoretically plausible psychometric models for the Hypersensitive Narcissism Scale \
(HSNS).

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<complete lavaan/semopy model>",
  "loadings": {"<item>": <float>, ...},
  "factor_correlation": <float, or null if your model has no two oblique factors>
}

`model_syntax` uses lavaan notation (`=~` loadings, `~~` (co)variances, `0*` to fix a
parameter to zero).

`loadings` gives, for each item you analysed, the largest absolute standardised loading that item has on any factor in your model.
"""


def simulate_hsns(n, rng, scale=1.0, dependent=True):
    """Draw HSNS responses from one trait, optionally with the duplicate pairs."""
    eta = rng.normal(size=n)
    shared = {pair: rng.normal(size=n) for pair, _ in RESIDUAL_PAIRS}
    out = {}
    for item in ITEMS:
        lam = LOADINGS[item][1] * scale
        common, explained = lam * eta, lam**2
        if dependent:
            for pair, cov in RESIDUAL_PAIRS:
                if item in pair:
                    common = common + np.sqrt(cov) * shared[pair]
                    explained += cov
        ystar = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[ITEMS]


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        is_us = country == "US"
        hsns = simulate_hsns(n, rng, scale=1.0 if is_us else LOADING_SCALE_NON_US, dependent=is_us)
        names = ["mach", "psych", "narc"]
        phi = np.eye(3)
        for (a, b), v in DD_PHI.items():
            phi[names.index(a), names.index(b)] = v
            phi[names.index(b), names.index(a)] = v
        dd = C.correlated_block(n, rng, DD_LOADINGS, phi, names, C.THRESHOLDS)
        frames.append(pd.concat([hsns, dd, C.demographics(n, rng, country)], axis=1))
    return C.finalize(frames, rng, SEED)


def reference_syntax():
    """lavaan syntax for the generating model, used as the scoring floor."""
    lines = [f"G =~ {'+'.join(ITEMS)}"]
    lines += [f"{a} ~~ {b}" for (a, b), _ in RESIDUAL_PAIRS]
    return "\n".join(lines)


def candidate_models():
    """The rival structures a competent analyst would fit to these items."""
    every = "+".join(ITEMS)
    two_factor = "F1 =~ HSNS2+HSNS7+HSNS3+HSNS9\nF2 =~ HSNS1+HSNS4+HSNS5+HSNS6+HSNS8+HSNS10"
    return {
        "unidimensional": f"G =~ {every}",
        "two_correlated_factors": two_factor,
        "bifactor_general_plus_specifics": (
            f"G =~ {every}\nS1 =~ HSNS2+HSNS7+HSNS3+HSNS9\n"
            "S2 =~ HSNS1+HSNS4+HSNS5+HSNS6+HSNS8+HSNS10\n"
            "G ~~ 0*S1\nG ~~ 0*S2\nS1 ~~ 0*S2"
        ),
        "unidimensional_with_correlated_residuals": reference_syntax(),
        "unidim_with_decoy_pair_freed": (
            reference_syntax() + f"\n{DECOY_PAIR[0]} ~~ {DECOY_PAIR[1]}"
        ),
        "unidim_top_two_raw_correlations": (
            f"G =~ {every}\nHSNS2 ~~ HSNS7\n{DECOY_PAIR[0]} ~~ {DECOY_PAIR[1]}"
        ),
    }


def population_correlation_matrix():
    """Correlation matrix a perfectly specified model would reproduce."""
    rng = np.random.default_rng(SEED + 999)
    return np.corrcoef(simulate_hsns(POP_REFERENCE_N, rng).values.T.astype(float)).round(3)


def build_truth(floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    return {
        "task_id": TASK_ID,
        "scored": {
            "model_family": "unidimensional_with_correlated_residuals",
            "n_factors": 1,
            "loadings": {k: v[1] for k, v in LOADINGS.items()},
            "residual_pairs": [list(pair) for pair, _ in RESIDUAL_PAIRS],
            "item_assignment": {k: "G" for k in ITEMS},
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
                "residual_covariances": {f"{a}_{b}": v for (a, b), v in RESIDUAL_PAIRS},
                "decoy_pair": list(DECOY_PAIR),
                "decoy_residual_covariance": 0.0,
            },
            "non_us": {
                "loading_scale": LOADING_SCALE_NON_US,
                "residual_covariances": {},
            },
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
            "name": "HSNS dimensionality versus item redundancy",
            "uuid": "6c40b18e-9d2f-4a77-93e1-58aa0c7f5d03",
            "keywords": [
                "psychometrics",
                "factor analysis",
                "local dependence",
                "model selection",
                "HSNS",
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
                    # The answer: which pairs agree beyond the trait. Read from the
                    # submitted model rather than asked for separately.
                    {
                        "key": "residual_pairs",
                        "fn": "score_pair_set",
                        "derive_from": "refit_residual_covariances",
                        "truth_key": "scored.residual_pairs",
                        "criterion": "local_dependence",
                    },
                    {
                        "key": "factor_correlation",
                        "fn": "score_scalar",
                        "truth_key": "scored.factor_correlation",
                        "tol": 0.06,
                        "applicable_if": "model_has_two_oblique_factors",
                        "criterion": "parameter_quality",
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

    truth = criteria["unidimensional_with_correlated_residuals"]
    corr = pd.DataFrame(np.corrcoef(X.values.T), index=ITEMS, columns=ITEMS)
    ranked = sorted(
        ((a, b, corr.loc[a, b]) for i, a in enumerate(ITEMS) for b in ITEMS[i + 1 :]),
        key=lambda t: -t[2],
    )
    decoy_rank = [(a, b) for a, b, _ in ranked].index(DECOY_PAIR) + 1
    residual = corr.values - _implied(reference_syntax().split("\n")[0], X)
    res = pd.DataFrame(residual, index=ITEMS, columns=ITEMS)

    print("\ntop raw correlations: " + "  ".join(f"{a}-{b} {r:.3f}" for a, b, r in ranked[:3]))
    print(
        "residual correlations under one factor: "
        + "  ".join(f"{a}-{b} {res.loc[a, b]:+.3f}" for (a, b), _ in RESIDUAL_PAIRS)
        + f"   decoy {DECOY_PAIR[0]}-{DECOY_PAIR[1]} "
        f"{res.loc[DECOY_PAIR[0], DECOY_PAIR[1]]:+.3f}"
    )

    checks = [
        ("the generating model fits well", truth["CFI"] > 0.99),
        (
            "the two-factor reading fits badly",
            criteria["two_correlated_factors"]["CFI"] < truth["CFI"] - 0.05,
        ),
        (
            "plain unidimensional fits badly",
            criteria["unidimensional"]["CFI"] < truth["CFI"] - 0.05,
        ),
        ("the decoy pair ranks in the top 3 raw correlations", decoy_rank <= 3),
        (
            "but its residual correlation is an order of magnitude smaller",
            abs(res.loc[DECOY_PAIR[0], DECOY_PAIR[1]]) * 5
            < min(abs(res.loc[a, b]) for (a, b), _ in RESIDUAL_PAIRS),
        ),
    ]
    return C.report(checks)


def _implied(unidim_spec, X):
    """Model-implied correlation matrix of the plain one-factor model."""
    import semopy

    model = semopy.Model(unidim_spec)
    model.fit(X[ITEMS])
    sigma = model.calc_sigma()[0]
    order = list(model.vars["observed"])
    pick = [order.index(i) for i in ITEMS]
    sigma = sigma[np.ix_(pick, pick)]
    scale = np.sqrt(np.diag(sigma))
    return sigma / np.outer(scale, scale)


def naive(df, pop):
    """Confirm the default dimensionality check and the US filter both matter."""
    del pop
    from factor_analyzer import FactorAnalyzer

    X = C.analysis_sample(df, ITEMS)
    corr = np.corrcoef(X.values.T)
    eigen = np.linalg.eigvalsh(corr)[::-1]
    rng = np.random.default_rng(0)
    random_eigen = [
        np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(X.shape).T))[::-1] for _ in range(30)
    ]
    n_factors = int((eigen > np.percentile(random_eigen, 95, axis=0)).sum())
    print(
        f"parallel analysis on the US sample -> {n_factors} factors "
        f"(eigenvalues {np.round(eigen[:3], 2)})"
    )
    FactorAnalyzer(n_factors=2, rotation="oblimin", method="minres").fit(X.values)

    truth = {k: v[1] for k, v in LOADINGS.items()}
    print(f"\n{'sample':22s} {'N':>7s} {'worst |error|':>14s}  within 0.08?")
    outcome = {}
    for label, sub in (
        ("US only (correct)", df[df.country == "US"]),
        ("pooled (no filter)", df),
    ):
        Xs = sub[ITEMS]
        Xs = Xs[(Xs != 0).all(axis=1)].astype(float)
        fitted = {k: abs(v) for k, v in C.loadings(C.fit(reference_syntax(), Xs, ITEMS)).items()}
        worst = max(abs(fitted[i] - truth[i]) for i in ITEMS)
        outcome[label] = worst <= 0.08
        print(f"{label:22s} {len(Xs):7,} {worst:14.3f}  " f"{'yes' if outcome[label] else 'NO'}")

    return C.report(
        [
            ("the usual dimensionality check reports two factors", n_factors == 2),
            (
                "the pooled analysis reports loadings outside tolerance",
                outcome["US only (correct)"] and not outcome["pooled (no filter)"],
            ),
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
