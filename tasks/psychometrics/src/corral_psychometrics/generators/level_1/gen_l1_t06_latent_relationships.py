#!/usr/bin/env python3
"""Generate Task 06 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l1_t06_latent_relationships"
SEED = 20260906
TASK = paths.task(__file__)

FACTORS = ["VULN", "EGO", "MACH", "PSYCH", "NARC"]
ITEMS_OF = {
    "VULN": ["HSNS2", "HSNS3", "HSNS7", "HSNS9"],
    "EGO": ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"],
    "MACH": ["DDM1", "DDM2", "DDM3", "DDM4"],
    "PSYCH": ["DDP1", "DDP2", "DDP3", "DDP4"],
    "NARC": ["DDN1", "DDN2", "DDN3", "DDN4"],
}
ITEMS = [i for f in FACTORS for i in ITEMS_OF[f]]

# Egocentrism is measured poorly, so relationships involving it shrink furthest
# when the instruments are related through scale scores.
LOADINGS = {
    "HSNS2": 0.76,
    "HSNS3": 0.58,
    "HSNS7": 0.69,
    "HSNS9": 0.60,
    "HSNS1": 0.45,
    "HSNS4": 0.42,
    "HSNS5": 0.48,
    "HSNS6": 0.40,
    "HSNS8": 0.47,
    "HSNS10": 0.44,
    "DDM1": 0.74,
    "DDM2": 0.71,
    "DDM3": 0.58,
    "DDM4": 0.77,
    "DDP1": 0.69,
    "DDP2": 0.73,
    "DDP3": 0.66,
    "DDP4": 0.48,
    "DDN1": 0.78,
    "DDN2": 0.81,
    "DDN3": 0.70,
    "DDN4": 0.62,
}

# Egocentrism and Dark-Triad narcissism are very nearly the same construct.
PHI = {
    ("VULN", "EGO"): 0.35,
    ("VULN", "MACH"): 0.28,
    ("VULN", "PSYCH"): 0.40,
    ("VULN", "NARC"): 0.30,
    ("EGO", "MACH"): 0.52,
    ("EGO", "PSYCH"): 0.34,
    ("EGO", "NARC"): 0.86,
    ("MACH", "PSYCH"): 0.58,
    ("MACH", "NARC"): 0.42,
    ("PSYCH", "NARC"): 0.39,
}

# The relationships the task asks about: HSNS dimensions against Dirty Dozen ones.
CROSS_PAIRS = [
    ("VULN", "MACH"),
    ("VULN", "PSYCH"),
    ("VULN", "NARC"),
    ("EGO", "MACH"),
    ("EGO", "PSYCH"),
    ("EGO", "NARC"),
]

# Above this, two traits count as indistinguishable. The prompt gives the same
# number, so the judgement has one right answer.
REDUNDANCY_THRESHOLD = 0.80

LOADING_SCALE_NON_US = 0.80
POP_REFERENCE_N = 600_000

PROMPT = """\
You are given responses from an online personality survey. The codebook describes \
every column.

Read the data and select participants from the United States. The survey contains two \
instruments, each covering several dimensions. Examine how the dimensions of one \
relate to the dimensions of the other, accounting for the fact that no dimension is \
measured perfectly.

Report the correlation between each pair of dimensions across the two instruments, and \
say which of those pairs are too closely related to be treated as distinct - take a \
correlation of 0.80 or above to mean the two dimensions cannot be told apart.

Items are five-point ordinal ratings and 0 denotes a missing response.
"""

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "<complete lavaan/semopy model>",
  "correlations": [["<factor>", "<factor>", <float>], ...],
  "not_distinguishable": [["<factor>", "<factor>"], ...]
}

`model_syntax` is the model you used, in lavaan notation (`=~` loadings, `~~`
(co)variances). It is re-fitted during evaluation, so it must be complete and runnable.

Name your factors whatever you like and refer to them by those names: dimensions are
matched by which items they cover. Report one correlation for every pair of dimensions
that spans the two instruments.
"""


def simulate(n, rng, scale=1.0):
    """Draw both instruments from correlated latent dimensions."""
    phi = np.eye(len(FACTORS))
    for (a, b), v in PHI.items():
        phi[FACTORS.index(a), FACTORS.index(b)] = v
        phi[FACTORS.index(b), FACTORS.index(a)] = v
    eta = rng.multivariate_normal(np.zeros(len(FACTORS)), phi, size=n)
    out = {}
    for factor in FACTORS:
        for item in ITEMS_OF[factor]:
            lam = LOADINGS[item] * scale
            ystar = lam * eta[:, FACTORS.index(factor)] + rng.normal(0, np.sqrt(1 - lam**2), n)
            out[item] = C.categorize(ystar, C.THRESHOLDS[item])
    return pd.DataFrame(out)[ITEMS]


def build_dataset(rng):
    """Build the full survey: every country, both instruments, demographics."""
    frames = []
    for country, n in C.N_BY_COUNTRY.items():
        block = simulate(n, rng, 1.0 if country == "US" else LOADING_SCALE_NON_US)
        frames.append(pd.concat([block, C.demographics(n, rng, country)], axis=1))
    return C.finalize(frames, rng, SEED)


def reference_syntax():
    """The joint model: every dimension latent, estimated together."""
    return "\n".join(f"{f} =~ {'+'.join(ITEMS_OF[f])}" for f in FACTORS)


def latent_correlations(X):
    """Correlations between the latent dimensions of the joint model."""
    import semopy

    model = semopy.Model(reference_syntax())
    model.fit(X[ITEMS])
    ins = model.inspect(std_est=True)
    rows = ins[
        (ins.op == "~~") & (ins.lval != ins.rval) & ins.lval.isin(FACTORS) & ins.rval.isin(FACTORS)
    ]
    return {frozenset((r["lval"], r["rval"])): float(r["Est. Std"]) for _, r in rows.iterrows()}


def scale_score_correlations(X):
    """What relating the instruments through summed scale scores gives instead."""
    scores = pd.DataFrame({f: X[ITEMS_OF[f]].sum(1) for f in FACTORS})
    return {frozenset((a, b)): scores[a].corr(scores[b]) for a, b in CROSS_PAIRS}


def population_reference():
    """Correlations at large N, plus the population matrix."""
    rng = np.random.default_rng(SEED + 999)
    big = simulate(POP_REFERENCE_N, rng)
    targets = latent_correlations(big)
    pop = np.corrcoef(big[ITEMS].values.T.astype(float))
    return targets, pop.round(3)


def build_truth(targets, floor, pop, data_sha, rows):
    """Assemble the hidden ground truth, scoring reference and provenance."""
    cross = [
        {"a": ITEMS_OF[a], "b": ITEMS_OF[b], "r": round(targets[frozenset((a, b))], 3)}
        for a, b in CROSS_PAIRS
    ]
    redundant = [
        [ITEMS_OF[a], ITEMS_OF[b]]
        for a, b in CROSS_PAIRS
        if targets[frozenset((a, b))] >= REDUNDANCY_THRESHOLD
    ]
    return {
        "task_id": TASK_ID,
        "scored": {
            "correlations": cross,
            "not_distinguishable": redundant,
            "redundancy_threshold": REDUNDANCY_THRESHOLD,
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
                "loadings": LOADINGS,
                "items_of_dimension": ITEMS_OF,
                "phi": {f"{a}_{b}": v for (a, b), v in PHI.items()},
                "near_redundant_pair": ["EGO", "NARC"],
            },
            "non_us": {"loading_scale": LOADING_SCALE_NON_US},
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
                "tol": 0.06,
                "criterion": "measurement_error",
            },
            {
                "key": "not_distinguishable",
                "fn": "score_pair_set",
                "match": "composition",
                "truth_key": "scored.not_distinguishable",
                "criterion": "discriminant_validity",
            },
        ],
    )
    contract["syntax_whitelist"]["max_factors"] = 8
    return [
        {
            "id": TASK_ID,
            "name": "Relationships between the instruments' dimensions",
            "uuid": "5b83d1f6-47ae-4c39-9d20-3f7154e8c606",
            "keywords": [
                "psychometrics",
                "measurement error",
                "attenuation",
                "discriminant validity",
                "structural equation model",
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
    latent = latent_correlations(X)
    scores = scale_score_correlations(X)

    def payload(source):
        corr = [[a, b, round(source[frozenset((a, b))], 3)] for a, b in CROSS_PAIRS]
        redundant = [
            [a, b] for a, b in CROSS_PAIRS if source[frozenset((a, b))] >= REDUNDANCY_THRESHOLD
        ]
        return {
            "model_syntax": reference_syntax(),
            "correlations": corr,
            "not_distinguishable": redundant,
        }

    correct = payload(latent)
    return {
        "joint latent model (CORRECT)": correct,
        "scale-score correlations": payload(scores),
        "latent values, misses the redundancy": {**correct, "not_distinguishable": []},
        "latent values, over-flags redundancy": {
            **correct,
            "not_distinguishable": [[a, b] for a, b in CROSS_PAIRS[-3:]],
        },
    }


def verify(df, targets, pop):
    """Confirm the joint model recovers the relationships and scale scores do not."""
    X = C.analysis_sample(df, ITEMS)
    latent = latent_correlations(X)
    scores = scale_score_correlations(X)
    print(f"US complete-case N = {len(X):,}\n")
    print(
        f"  {'pair':14s} {'true':>6s} {'joint model':>12s} {'scale scores':>13s}"
        f" {'shrinkage':>10s}"
    )
    worst_latent, worst_scores = 0.0, 0.0
    for a, b in CROSS_PAIRS:
        true = PHI[(a, b)]
        lv, sv = latent[frozenset((a, b))], scores[frozenset((a, b))]
        worst_latent = max(worst_latent, abs(lv - targets[frozenset((a, b))]))
        worst_scores = max(worst_scores, abs(sv - true))
        print(f"  {a + '-' + b:14s} {true:6.2f} {lv:12.3f} {sv:13.3f} {sv - true:+10.3f}")

    ego_narc_latent = latent[frozenset(("EGO", "NARC"))]
    ego_narc_scores = scores[frozenset(("EGO", "NARC"))]
    checks = [
        ("the joint model recovers every relationship", worst_latent <= 0.06),
        ("scale scores do not", worst_scores > 0.06),
        (
            "the shrinkage is uneven, so the picture is distorted not just shrunk",
            max(abs(scores[frozenset(p)] - PHI[p]) for p in CROSS_PAIRS)
            - min(abs(scores[frozenset(p)] - PHI[p]) for p in CROSS_PAIRS)
            > 0.10,
        ),
        ("one pair is genuinely not distinguishable", ego_narc_latent >= REDUNDANCY_THRESHOLD),
        ("but looks distinguishable through scale scores", ego_narc_scores < REDUNDANCY_THRESHOLD),
    ]
    return C.report(checks)


def naive(df, targets, pop):
    """Confirm the scale-score analysis gives a uniform and wrong picture."""
    del targets, pop
    X = C.analysis_sample(df, ITEMS)
    scores = scale_score_correlations(X)
    values = [scores[frozenset(p)] for p in CROSS_PAIRS]
    true = [PHI[p] for p in CROSS_PAIRS]
    print(f"  scale-score correlations span {min(values):.2f} to {max(values):.2f}")
    print(f"  the real ones span            {min(true):.2f} to {max(true):.2f}")
    print(
        f"  every pair is understated, by {min(t - v for t, v in zip(true, values)):.2f}"
        f" to {max(t - v for t, v in zip(true, values)):.2f}"
    )
    return C.report(
        [
            (
                "every scale-score correlation is outside tolerance",
                all(abs(scores[frozenset(p)] - PHI[p]) > 0.06 for p in CROSS_PAIRS),
            ),
            (
                "and the real spread is wider than the apparent one",
                (max(true) - min(true)) > (max(values) - min(values)),
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

    floor = C.evaluate_model(reference_syntax(), C.analysis_sample(df, ITEMS), ITEMS, pop)
    C.write_artifacts(
        TASK,
        df,
        lambda sha: build_truth(targets, floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
