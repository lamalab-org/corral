#!/usr/bin/env python3
"""Generate Level 2 Task 09 artifacts and scoring metadata.

Generating model: a two-parameter logistic item response model over 20 binary
items, difficulties evenly spaced from -2 to +2.

    P(correct) = 1 / (1 + exp(-slope * (theta - difficulty)))

Three items should leave the bank, and the vendor's review finds only one of
them. CAT06 and CAT07 are redundant with each other and the vendor flags them;
CAT13 and CAT14 are equally redundant and it misses them; CAT12's
discrimination collapses from 1.15 to 0.30 in the holdout. CAT03 is flagged and
is fine. All four dependent items have high discrimination, so an adaptive test
reaches for them first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t09_adaptive_bank_choice"
UUID = "e9f4c6b2-3a71-4d85-9c20-8b6e1a7f5342"
SEED = 20261001
TASK = paths.task(__file__)

ITEMS = [f"CAT{i:02d}" for i in range(1, 21)]
N_CALIBRATION = 4_000
N_HOLDOUT = 2_000
CAT_LENGTH = 8

DIFFICULTY = dict(zip(ITEMS, np.linspace(-2.0, 2.0, len(ITEMS))))
DISCRIMINATION = dict(
    zip(
        ITEMS,
        [
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            1.60,
            1.65,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.55,
            1.60,
            1.20,
            1.25,
            1.30,
            1.35,
            1.40,
            1.45,
        ],
    )
)

# Two dependent pairs, both made of high-discrimination items so that an
# adaptive test prefers them. The vendor found one of them.
FLAGGED_PAIR = ("CAT06", "CAT07")
MISSED_PAIR = ("CAT13", "CAT14")
FLAGGED_SHARE = 1.90
MISSED_SHARE = 1.85
FALSE_FLAG = "CAT03"
UNSTABLE_ITEM = "CAT12"
UNSTABLE_HOLDOUT_DISCRIMINATION = 0.30

# The screened bank drops the redundant half of each pair and the item whose
# discrimination does not hold up.
SCREENED_OUT = [FLAGGED_PAIR[1], MISSED_PAIR[1], UNSTABLE_ITEM]
VENDOR_FLAGS = sorted([*FLAGGED_PAIR, FALSE_FLAG])

NODES = np.linspace(-4.0, 4.0, 31)
PRIOR = norm.pdf(NODES) / norm.pdf(NODES).sum()

PROMPT = """You are choosing an item bank for a short computerized adaptive test. The test administers a small number of items per respondent, always selecting the item that is most informative at the respondent's current ability estimate, and must estimate ability across the whole scale.

You have calibration responses, an independent holdout sample, and the vendor's item review. The vendor recommends its full bank because it carries the most item information. The review marks some items for attention, but it is a review of the vendor's own process and has not been checked against the responses.

Decide which bank to use and report what the responses say about the items: which item pairs are locally dependent, which vendor flags the data do not support, and which items do not behave the same way in the holdout sample. Item information computed from a calibration fit assumes responses are independent given ability; where that assumption fails, an adaptive test keeps selecting the same information twice."""

SUBMISSION_FORMAT = """A single JSON object:

{
  "bank_choice": "vendor_bank|screened_bank",
  "dependent_pairs": [["CATxx", "CATyy"]],
  "unsupported_vendor_flags": ["CATxx"],
  "unstable_items": ["CATxx"],
  "recommendation": "use_for_adaptive_testing|do_not_use_without_recalibration"
}

`screened_bank` means the vendor bank with the redundant half of each dependent
pair and every unstable item removed. `dependent_pairs` lists every locally
dependent pair the responses show, whether or not the vendor flagged it.
`unsupported_vendor_flags` lists items the vendor flagged that the responses do
not justify. No model syntax or working is required."""


def _probability(discrimination, difficulty, theta):
    """2PL response probability, as nodes by items."""
    return 1.0 / (1.0 + np.exp(-np.outer(theta, discrimination) + discrimination * difficulty))


def fit_2pl(X, rounds=30):
    """Marginal maximum likelihood estimates of discrimination and difficulty.

    X  respondents by items, scored 0 or 1

    Expectation-maximisation over a fixed ability grid: each round works out
    where respondents sit, then refits each item against that.
    """
    n_items = X.shape[1]
    a = np.ones(n_items)
    b = np.zeros(n_items)
    for _ in range(rounds):
        P = _probability(a, b, NODES)
        logL = X @ np.log(P.T + 1e-12) + (1 - X) @ np.log(1 - P.T + 1e-12)
        weight = np.exp(logL - logL.max(axis=1, keepdims=True)) * PRIOR
        weight /= weight.sum(axis=1, keepdims=True)
        at_node = weight.sum(axis=0)
        correct = weight.T @ X
        for i in range(n_items):

            def negative_loglik(par, i=i):
                slope, location = par
                p = np.clip(1.0 / (1.0 + np.exp(-slope * (NODES - location))), 1e-9, 1 - 1e-9)
                return -(
                    correct[:, i] * np.log(p) + (at_node - correct[:, i]) * np.log(1 - p)
                ).sum()

            best = minimize(
                negative_loglik,
                [a[i], b[i]],
                method="L-BFGS-B",
                bounds=[(0.15, 3.5), (-4.0, 4.0)],
            )
            a[i], b[i] = best.x
    return a, b


def ability(X, a, b):
    """Expected ability of each respondent given their answers."""
    P = _probability(a, b, NODES)
    logL = X @ np.log(P.T + 1e-12) + (1 - X) @ np.log(1 - P.T + 1e-12)
    weight = np.exp(logL - logL.max(axis=1, keepdims=True)) * PRIOR
    weight /= weight.sum(axis=1, keepdims=True)
    return weight @ NODES


def nominal_information(a):
    """Total information the calibration fit claims the bank provides."""
    return float((a**2 * 0.25).sum())


def run_adaptive_test(X, a, b, length=CAT_LENGTH):
    """Administer `length` items each, always the most informative one left.

    Returns each respondent's ability estimate and how often each item was used.
    """
    n, k = X.shape
    P = _probability(a, b, NODES)
    contribution = X[:, None, :] * np.log(P[None, :, :] + 1e-12) + (1 - X)[:, None, :] * np.log(
        1 - P[None, :, :] + 1e-12
    )
    taken = np.zeros((n, k), dtype=bool)
    theta = np.zeros(n)
    logL = np.zeros((n, len(NODES)))
    used = np.zeros(k)
    for _ in range(length):
        p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
        information = a[None, :] ** 2 * p * (1 - p)
        information[taken] = -1.0
        pick = information.argmax(axis=1)
        taken[np.arange(n), pick] = True
        used += np.bincount(pick, minlength=k)
        logL += contribution[np.arange(n), :, pick]
        weight = np.exp(logL - logL.max(axis=1, keepdims=True)) * PRIOR
        weight /= weight.sum(axis=1, keepdims=True)
        theta = weight @ NODES
    return theta, used / n


def simulate(rng, n, holdout=False):
    """Draw responses. Only the unstable item behaves differently in the holdout."""
    theta = rng.normal(size=n)
    shared_flagged = rng.normal(size=n)
    shared_missed = rng.normal(size=n)
    out = {}
    for item in ITEMS:
        slope = DISCRIMINATION[item]
        if holdout and item == UNSTABLE_ITEM:
            slope = UNSTABLE_HOLDOUT_DISCRIMINATION
        z = slope * (theta - DIFFICULTY[item])
        if item in FLAGGED_PAIR:
            z = z + FLAGGED_SHARE * shared_flagged
        if item in MISSED_PAIR:
            z = z + MISSED_SHARE * shared_missed
        out[item] = rng.binomial(1, 1.0 / (1.0 + np.exp(-z)))
    frame = pd.DataFrame(out)[ITEMS]
    frame.insert(0, "participant_id", [f"{'H' if holdout else 'C'}{i:05d}" for i in range(n)])
    return frame, theta


def responses(frame):
    """The response matrix on its own."""
    return frame[ITEMS].values.astype(float)


def residual_correlations(X, a, b):
    """Correlations left between items once ability is accounted for."""
    theta = ability(X, a, b)
    expected = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    residual = X - expected
    return np.corrcoef(residual.T)


def dependent_pairs(X, a, b, cutoff=0.15):
    """Item pairs whose responses stay correlated after ability is removed."""
    matrix = residual_correlations(X, a, b)
    found = []
    for i in range(len(ITEMS)):
        for j in range(i + 1, len(ITEMS)):
            if matrix[i, j] >= cutoff:
                found.append([ITEMS[i], ITEMS[j]])
    return sorted(found)


def bank_index(bank):
    return [ITEMS.index(item) for item in bank]


def evaluate_bank(bank, calibration, holdout, true_theta):
    """Fit a bank, then see how well it measures under adaptive selection."""
    index = bank_index(bank)
    a, b = fit_2pl(responses(calibration)[:, index])
    estimate, used = run_adaptive_test(responses(holdout)[:, index], a, b)
    return {
        "items": len(bank),
        "nominal_information": round(nominal_information(a), 3),
        "ability_rmse": round(float(np.sqrt(((estimate - true_theta) ** 2).mean())), 4),
        "ability_correlation": round(float(np.corrcoef(estimate, true_theta)[0, 1]), 4),
        "most_used": [bank[i] for i in np.argsort(-used)[:4]],
        "discrimination": dict(zip(bank, np.round(a, 3))),
    }


def write_vendor_review(path):
    """The vendor's own process notes. They describe review, not measurement."""
    rows = []
    for item in ITEMS:
        flagged = item in VENDOR_FLAGS
        rows.append(
            {
                "item": item,
                "review_batch": "B2" if flagged else "B1",
                "reviewed_on": "2026-04-18" if flagged else "2026-04-11",
                "vendor_flag": "flagged" if flagged else "cleared",
                "bank_version": 3,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def write_codebook(path):
    path.write_text(
        "# Codebook - adaptive item bank\n\n"
        "`data.csv` and `holdout.csv` are tab-separated and hold binary responses to the "
        "same twenty items, one row per respondent. The holdout sample was collected "
        "separately from the calibration sample.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `participant_id` | respondent identifier, unique within each file |\n"
        "| `CAT01`-`CAT20` | item response, 0 = incorrect, 1 = correct |\n\n"
        "`vendor_review.csv` records the vendor's review of its own item bank.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `item` | item identifier |\n"
        "| `review_batch` | batch the item was reviewed in |\n"
        "| `reviewed_on` | date of that review |\n"
        "| `vendor_flag` | whether the vendor marked the item for attention |\n"
        "| `bank_version` | version of the bank the review applies to |\n"
    )


def write_memo(path):
    path.write_text(
        "# Preliminary analysis\n\n"
        "The vendor recommends its full twenty-item bank. Its case is that the bank "
        "carries more item information than any reduced version, and that the items it "
        "marked for attention in the latest review are the ones to watch.\n\n"
        "The recommendation rests on information computed from the calibration fit. No "
        "one has checked the vendor's flags against the responses, looked for items that "
        "behave differently in the holdout sample, or simulated the adaptive test itself.\n"
    )


def build_task_json(data_sha):
    return [
        {
            "id": TASK_ID,
            "name": "Which item bank should the adaptive test use?",
            "uuid": UUID,
            "keywords": ["psychometrics", "adaptive testing", "item response theory"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "codebook.md",
                    "data.csv",
                    "holdout.csv",
                    "preliminary_analysis.md",
                    "vendor_review.csv",
                ],
                "calibration_dataset": "data.csv",
                "holdout_dataset": "holdout.csv",
                "vendor_review": "vendor_review.csv",
                "codebook": "codebook.md",
                "preliminary_analysis": "preliminary_analysis.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_adaptive_bank_choice",
            "scoring_params": {
                "task_type": "adaptive_bank_choice",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "dataset": "data.csv",
                "holdout_dataset": "holdout.csv",
                "items": ITEMS,
            },
        }
    ]


def truth(data_sha, rows, vendor, screened):
    return {
        "task_id": TASK_ID,
        "scored": {
            "bank_choice": "screened_bank",
            "dependent_pairs": sorted([list(FLAGGED_PAIR), list(MISSED_PAIR)]),
            "unsupported_vendor_flags": [FALSE_FLAG],
            "unstable_items": [UNSTABLE_ITEM],
            "recommendation": "use_for_adaptive_testing",
        },
        "measured": {"vendor_bank": vendor, "screened_bank": screened},
        "generative_parameters": {
            "difficulty": {k: round(float(v), 3) for k, v in DIFFICULTY.items()},
            "discrimination": DISCRIMINATION,
            "dependent_pairs": {
                "flagged_by_vendor": list(FLAGGED_PAIR),
                "missed_by_vendor": list(MISSED_PAIR),
            },
            "false_vendor_flag": FALSE_FLAG,
            "unstable_item": UNSTABLE_ITEM,
            "screened_out": SCREENED_OUT,
            "adaptive_test_length": CAT_LENGTH,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def candidate_submissions():
    correct = {
        "bank_choice": "screened_bank",
        "dependent_pairs": sorted([list(FLAGGED_PAIR), list(MISSED_PAIR)]),
        "unsupported_vendor_flags": [FALSE_FLAG],
        "unstable_items": [UNSTABLE_ITEM],
        "recommendation": "use_for_adaptive_testing",
    }
    return {
        "correct": correct,
        "follows the vendor review": {
            **correct,
            "bank_choice": "vendor_bank",
            "dependent_pairs": [list(FLAGGED_PAIR)],
            "unsupported_vendor_flags": [],
        },
        "chooses on nominal information": {**correct, "bank_choice": "vendor_bank"},
        "misses the unflagged pair": {**correct, "dependent_pairs": [list(FLAGGED_PAIR)]},
        "keeps the item that does not transport": {**correct, "unstable_items": []},
    }


def verify(calibration, holdout, theta_holdout):
    """Confirm every conclusion follows from the responses, not the vendor file."""
    X_cal, X_hold = responses(calibration), responses(holdout)
    a_cal, b_cal = fit_2pl(X_cal)
    a_hold, b_hold = fit_2pl(X_hold)

    found = dependent_pairs(X_cal, a_cal, b_cal)
    matrix = residual_correlations(X_cal, a_cal, b_cal)
    pair_value = lambda p: float(matrix[ITEMS.index(p[0]), ITEMS.index(p[1])])  # noqa: E731
    false_flag_worst = max(
        abs(matrix[ITEMS.index(FALSE_FLAG), j]) for j in range(len(ITEMS)) if ITEMS[j] != FALSE_FLAG
    )
    drop = {item: round(float(a_cal[i] - a_hold[i]), 3) for i, item in enumerate(ITEMS)}

    vendor = evaluate_bank(ITEMS, calibration, holdout, theta_holdout)
    screened = evaluate_bank(
        [i for i in ITEMS if i not in SCREENED_OUT], calibration, holdout, theta_holdout
    )

    print("residual correlation after ability is accounted for:")
    print(
        f"  {FLAGGED_PAIR[0]}~{FLAGGED_PAIR[1]} (vendor flagged this pair) {pair_value(FLAGGED_PAIR):+.3f}"
    )
    print(
        f"  {MISSED_PAIR[0]}~{MISSED_PAIR[1]} (vendor missed it)           {pair_value(MISSED_PAIR):+.3f}"
    )
    print(f"  largest involving {FALSE_FLAG} (vendor flagged it)      {false_flag_worst:+.3f}")
    print("\ndiscrimination lost between calibration and holdout:")
    for item in sorted(drop, key=lambda k: -drop[k])[:3]:
        print(
            f"  {item}: {a_cal[ITEMS.index(item)]:.2f} -> {a_hold[ITEMS.index(item)]:.2f}  ({drop[item]:+.3f})"
        )
    print(f"\n{'bank':14s} {'items':>5s} {'nominal info':>12s} {'ability rmse':>12s} {'corr':>7s}")
    for name, result in (("vendor", vendor), ("screened", screened)):
        print(
            f"  {name:12s} {result['items']:5d} {result['nominal_information']:12.2f} "
            f"{result['ability_rmse']:12.4f} {result['ability_correlation']:7.4f}"
        )
    print(f"  the vendor bank administers {', '.join(vendor['most_used'][:3])} most often")

    return C.report(
        [
            (
                "both planted pairs stay correlated once ability is removed",
                min(pair_value(FLAGGED_PAIR), pair_value(MISSED_PAIR)) > 0.15,
            ),
            (
                "the responses show the pair the vendor missed",
                list(MISSED_PAIR) in found,
            ),
            (
                "and show no dependence for the item the vendor flagged",
                false_flag_worst < 0.10,
            ),
            (
                "only the unstable item loses discrimination in the holdout",
                drop[UNSTABLE_ITEM] > 0.40
                and max(v for k, v in drop.items() if k != UNSTABLE_ITEM) < 0.30,
            ),
            (
                "the vendor bank claims more information",
                vendor["nominal_information"] > screened["nominal_information"],
            ),
            (
                "and measures ability worse under adaptive selection",
                screened["ability_rmse"] < vendor["ability_rmse"],
            ),
            (
                "because it keeps administering a dependent pair",
                len({*FLAGGED_PAIR, *MISSED_PAIR} & set(vendor["most_used"])) >= 2,
            ),
            (
                "the responses find exactly the two planted pairs",
                found == sorted([list(FLAGGED_PAIR), list(MISSED_PAIR)]),
            ),
        ]
    )


def naive(calibration, holdout, theta_holdout):
    """Confirm the vendor review and the information count both mislead."""
    X_cal = responses(calibration)
    a_cal, b_cal = fit_2pl(X_cal)
    found = dependent_pairs(X_cal, a_cal, b_cal)
    from_vendor = [list(FLAGGED_PAIR)]

    vendor = evaluate_bank(ITEMS, calibration, holdout, theta_holdout)
    screened = evaluate_bank(
        [i for i in ITEMS if i not in SCREENED_OUT], calibration, holdout, theta_holdout
    )

    print(f"  the vendor review flags {', '.join(VENDOR_FLAGS)}")
    print(f"  the responses show     {', '.join('~'.join(p) for p in found)}")
    print(f"  taking the review at face value misses {MISSED_PAIR[0]}~{MISSED_PAIR[1]}")
    print(f"  and excludes {FALSE_FLAG}, which the responses do not implicate\n")
    print("  choosing the bank with more information gives the vendor bank:")
    print(
        f"    nominal information {vendor['nominal_information']:.2f} against "
        f"{screened['nominal_information']:.2f}"
    )
    print(
        f"    ability rmse        {vendor['ability_rmse']:.4f} against "
        f"{screened['ability_rmse']:.4f}"
    )

    return C.report(
        [
            ("the vendor review is not what the responses show", found != from_vendor),
            (
                "it misses a real dependent pair",
                list(MISSED_PAIR) in found and list(MISSED_PAIR) not in from_vendor,
            ),
            (
                "choosing on information alone picks the worse bank",
                vendor["nominal_information"] > screened["nominal_information"]
                and vendor["ability_rmse"] > screened["ability_rmse"],
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    calibration, _ = simulate(rng, N_CALIBRATION)
    holdout, theta_holdout = simulate(rng, N_HOLDOUT, holdout=True)
    if action == "verify":
        return verify(calibration, holdout, theta_holdout)
    if action == "naive":
        return naive(calibration, holdout, theta_holdout)

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(TASK.artifacts / "data.csv", sep="\t", index=False)
    holdout.to_csv(TASK.artifacts / "holdout.csv", sep="\t", index=False)
    write_vendor_review(TASK.artifacts / "vendor_review.csv")
    write_codebook(TASK.artifacts / "codebook.md")
    write_memo(TASK.artifacts / "preliminary_analysis.md")
    vendor = evaluate_bank(ITEMS, calibration, holdout, theta_holdout)
    screened = evaluate_bank(
        [i for i in ITEMS if i not in SCREENED_OUT], calibration, holdout, theta_holdout
    )
    data_sha = C.file_sha256(TASK.artifacts / "data.csv")
    C.write_json(TASK.truth, truth(data_sha, len(calibration), vendor, screened))
    C.write_json(TASK.definition, build_task_json(data_sha))
    print(
        f"\n{len(calibration):,} calibration and {len(holdout):,} holdout rows -> {TASK.artifacts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
