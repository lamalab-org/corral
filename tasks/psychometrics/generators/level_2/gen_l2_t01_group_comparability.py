#!/usr/bin/env python3
"""Generate Level 2 Task 01 artifacts and scoring metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l2_t01_group_comparability"
SEED = 20260921
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_01"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_01.json"

ITEMS = C.HSNS_ITEMS
ALL_VARS = ITEMS + ["gender"]
LOADINGS = {
    "HSNS1": ("egocentrism", 0.55),
    "HSNS4": ("egocentrism", 0.30),
    "HSNS5": ("egocentrism", 0.70),
    "HSNS6": ("egocentrism", 0.48),
    "HSNS8": ("egocentrism", 0.71),
    "HSNS10": ("egocentrism", 0.66),
    "HSNS2": ("oversensitivity", 0.76),
    "HSNS3": ("oversensitivity", 0.58),
    "HSNS7": ("oversensitivity", 0.69),
    "HSNS9": ("oversensitivity", 0.52),
}

# Group 2 clears these four items' thresholds more easily at any trait level.
# Two sit on each factor, so both traits look different until the items are freed.
ITEM_DIF = {"HSNS1": -0.65, "HSNS8": -0.65, "HSNS3": -0.65, "HSNS7": -0.65}

# The groups are identical on both traits. Every point of the observed gap is bias.
LATENT_DIFFERENCE = {"egocentrism": 0.0, "oversensitivity": 0.0}

# What a model that frees the four items should find left on the traits.
ADJUSTED_DIFFERENCE = 0.0
ADJUSTED_TOLERANCE = 0.06

PHI = 0.35
CROSS_LOADING = ("HSNS9", 0.15)
RESIDUAL_CORR = (("HSNS5", "HSNS10"), 0.10)
N_GROUP_1, N_GROUP_2 = 6_000, 6_000
POP_REFERENCE_N = 200_000

RECOMMENDATION = {"status": "not_supported"}

PROMPT = (
    "You are investigating whether the Hypersensitive Narcissism Scale (HSNS) can support a "
    "comparison between the two groups identified in the codebook. The groups have visibly "
    "different response patterns, and a preliminary memo recommends reporting a group "
    "difference. It is unclear whether that recommendation reflects a difference in the "
    "underlying traits, the way some items behave, or both.\n\n"
    "Assess the memo's recommendation using the responses. Carry out whatever analyses you "
    "judge necessary and return a reproducible model, one overall recommendation, and the "
    "group difference that remains on each trait once the model accounts for any item that "
    "does not behave the same way in both groups.\n"
)

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "F1 =~ ...\\nF2 =~ ...\\nF1 ~ gender\\nF2 ~ gender",
  "recommendation": {"status": "supported|not_supported|supported_only_after_reanalysis"},
  "affected_items": []
}

The model must include every HSNS item and `gender`. Use lavaan-style syntax
(`=~`, `~~`, and `~`). Give `gender` a path to each trait. An item that does not
behave the same way in both groups is declared by giving it its own path from
`gender`, and listed in `affected_items`.

The remaining group difference is read from the model you submit, so it must be
the model your recommendation rests on.
"""


def reference_syntax():
    """The generating model: two correlated traits, plus the four biased items."""
    f1 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == "egocentrism")
    f2 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == "oversensitivity")
    paths = "\n".join(f"{item} ~ gender" for item in sorted(ITEM_DIF))
    return f"F1 =~ {f1}\nF2 =~ {f2}\nF1 ~ gender\nF2 ~ gender\n{paths}"


def unadjusted_syntax():
    """The same model with no item allowed to differ: the trap."""
    f1 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == "egocentrism")
    f2 = "+".join(item for item, (factor, _) in LOADINGS.items() if factor == "oversensitivity")
    return f"F1 =~ {f1}\nF2 =~ {f2}\nF1 ~ gender\nF2 ~ gender"


def responses(rng, n_1, n_2, apply_dif=True):
    """Draw item responses for both groups.

    n_1, n_2   respondents per group
    apply_dif  shift the four biased items' thresholds for group 2
    """
    gender = np.r_[np.ones(n_1), np.full(n_2, 2)]
    eta = rng.multivariate_normal([0.0, 0.0], [[1.0, PHI], [PHI, 1.0]], len(gender))
    eta += np.where(
        gender[:, None] == 2,
        [LATENT_DIFFERENCE["egocentrism"], LATENT_DIFFERENCE["oversensitivity"]],
        0.0,
    )
    shared = rng.normal(size=len(gender))
    out = {}
    for item, (factor, loading) in LOADINGS.items():
        factor_index = 0 if factor == "egocentrism" else 1
        common = loading * eta[:, factor_index]
        explained = loading**2
        if item == CROSS_LOADING[0]:
            cross_loading = CROSS_LOADING[1]
            common += cross_loading * eta[:, 0]
            explained += cross_loading**2 + 2 * cross_loading * loading * PHI
        if item in RESIDUAL_CORR[0]:
            residual = RESIDUAL_CORR[1]
            common += np.sqrt(residual) * shared
            explained += residual
        y = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), len(gender))
        tau = np.asarray(C.THRESHOLDS[item])
        shift = ITEM_DIF.get(item, 0.0) if apply_dif else 0.0
        values = np.empty(len(y), dtype=int)
        for group, threshold in ((1, tau), (2, tau + shift)):
            mask = gender == group
            values[mask] = C.categorize(y[mask], threshold)
        out[item] = values
    frame = pd.DataFrame(out)
    frame["gender"] = gender.astype(int)
    return frame


def simulate(rng):
    """The delivered dataset."""
    frame, eta = responses_with_latents(rng, N_GROUP_1, N_GROUP_2)
    frame["age"] = np.clip(rng.normal(35, 11, len(frame)).round(), 18, 78).astype(int)
    frame["accuracy"] = np.clip(rng.beta(8, 1.5, len(frame)) * 100, 1, 100).round().astype(int)
    frame["country"] = "US"
    frame[ITEMS] = frame[ITEMS].mask(rng.random((len(frame), len(ITEMS))) < 0.01, 0)
    frame = frame[ITEMS + ["age", "gender", "accuracy", "country"]]
    frame.insert(0, "participant_id", [f"P{i:05d}" for i in range(1, len(frame) + 1)])
    behavior = pd.DataFrame(
        {
            "participant_id": frame["participant_id"],
            "gender": frame["gender"],
            "credit_allocation": 50 + 10 * eta[:, 0] + rng.normal(0, 8, len(frame)),
            "feedback_reactivity": 50 + 10 * eta[:, 1] + rng.normal(0, 8, len(frame)),
        }
    )
    return frame, behavior


def responses_with_latents(rng, n_1, n_2, apply_dif=True):
    """Draw responses and retain the latent traits for the auxiliary file."""
    gender = np.r_[np.ones(n_1), np.full(n_2, 2)]
    eta = rng.multivariate_normal([0.0, 0.0], [[1.0, PHI], [PHI, 1.0]], len(gender))
    shared = rng.normal(size=len(gender))
    out = {}
    for item, (factor, loading) in LOADINGS.items():
        factor_index = 0 if factor == "egocentrism" else 1
        common = loading * eta[:, factor_index]
        explained = loading**2
        if item == CROSS_LOADING[0]:
            cross_loading = CROSS_LOADING[1]
            common += cross_loading * eta[:, 0]
            explained += cross_loading**2 + 2 * cross_loading * loading * PHI
        if item in RESIDUAL_CORR[0]:
            residual = RESIDUAL_CORR[1]
            common += np.sqrt(residual) * shared
            explained += residual
        y = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), len(gender))
        tau = np.asarray(C.THRESHOLDS[item])
        shift = ITEM_DIF.get(item, 0.0) if apply_dif else 0.0
        values = np.empty(len(y), dtype=int)
        for group, threshold in ((1, tau), (2, tau + shift)):
            mask = gender == group
            values[mask] = C.categorize(y[mask], threshold)
        out[item] = values
    frame = pd.DataFrame(out)
    frame["gender"] = gender.astype(int)
    return frame, eta


def population_matrix(rng):
    """Correlations a correctly specified model reproduces, from a very large draw."""
    big = responses(rng, POP_REFERENCE_N // 2, POP_REFERENCE_N // 2)
    return np.corrcoef(big[ALL_VARS].values.T.astype(float)).round(3)


def analysis_sample(df):
    """Complete responses from both groups."""
    sub = df[df.gender.isin([1, 2])]
    return sub[ALL_VARS][(sub[ITEMS] != 0).all(axis=1)].astype(float)


def group_difference(X, syntax):
    """The group difference each trait keeps under a given model."""
    model = C.fit(syntax, X, ALL_VARS)
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == "gender") & ins.lval.isin(["F1", "F2"])]
    return {row["lval"]: round(float(row["Est. Std"]), 3) for _, row in rows.iterrows()}


def raw_effect_size(X):
    """Cohen's d between the groups' total scores."""
    t1, t2 = (X[X.gender == g][ITEMS].sum(axis=1) for g in (1, 2))
    pooled = np.sqrt(
        ((len(t1) - 1) * t1.var() + (len(t2) - 1) * t2.var()) / (len(t1) + len(t2) - 2)
    )
    return round(float((t2.mean() - t1.mean()) / pooled), 3)


def write_task_memo(path):
    path.write_text(
        "# Preliminary analysis memo\n\n"
        "Total HSNS scores were compared between the two groups. Group 2 scored "
        "higher, the difference was large and highly significant, and the scale's "
        "internal consistency was acceptable in both groups. On that basis the "
        "group difference should be reported as a trait difference.\n"
    )


def write_codebook(path, df):
    lines = [
        "# Codebook - group comparability study",
        "",
        "Tab-separated, one row per respondent. Questionnaire responses are ratings "
        "from 1 (Disagree) to 5 (Agree); 0 denotes a missing response.",
        "",
        "The separate `behavior.csv` file contains two independent behavioural "
        "indicators for the same participants.",
        "",
        "| item | text |",
        "|---|---|",
        *[f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS],
        "",
        "| variable | description |",
        "|---|---|",
        "| `age` | respondent age in years |",
        "| `gender` | 1 = Group 1, 2 = Group 2 |",
        "| `accuracy` | self-rated response accuracy, 0-100 |",
        "| `country` | collection country; all rows are US |",
        "",
        f"Rows: {len(df):,}. Complete responses are not guaranteed.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_behavior(path, behavior):
    behavior.to_csv(path, sep="\t", index=False)


def write_behavior_codebook(path):
    path.write_text(
        "# Codebook - behavioural indicators\n\n"
        "`behavior.csv` is tab-separated and uses the same `participant_id` and "
        "`gender` values as `data.csv`. The two continuous variables are scores "
        "from separate behavioural tasks; higher values indicate more of the "
        "behaviour recorded by that task.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `participant_id` | participant key for joining the files |\n"
        "| `gender` | 1 = Group 1, 2 = Group 2 |\n"
        "| `credit_allocation` | score from a joint-credit allocation task |\n"
        "| `feedback_reactivity` | score from a feedback-response task |\n"
    )


def build_truth(floor, pop, data_sha, rows):
    return {
        "task_id": TASK_ID,
        "scored": {
            "recommendation": RECOMMENDATION,
            "affected_items": sorted(ITEM_DIF),
            "adjusted_group_difference": ADJUSTED_DIFFERENCE,
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "population_reference_n": POP_REFERENCE_N,
            "item_order": ALL_VARS,
        },
        "generative_parameters": {
            "items": ITEMS,
            "loadings": LOADINGS,
            "group_latent_difference": LATENT_DIFFERENCE,
            "item_threshold_shifts_for_group_2": ITEM_DIF,
            "metric_invariance": True,
            "scalar_invariance": False,
            "cross_loading": CROSS_LOADING,
            "residual_correlation": RESIDUAL_CORR,
            "thresholds": C.THRESHOLDS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    contract = C.scoring_contract(
        "artifacts/level_2/task_01/truth.json",
        ALL_VARS,
        [
            {
                "key": "recommendation",
                "fn": "score_label_panel",
                "truth_key": "scored.recommendation",
                "criterion": "overall_recommendation",
            },
            {
                "key": "adjusted_group_difference",
                "fn": "score_scalar",
                "truth_key": "scored.adjusted_group_difference",
                "derive_from": "refit_latent_group_difference",
                "covariate": "gender",
                "tol": ADJUSTED_TOLERANCE,
                "criterion": "group_difference",
            },
            {
                "key": "affected_items",
                "fn": "score_item_set",
                "truth_key": "scored.affected_items",
                "covariate": "gender",
                "criterion": "item_level_bias",
            },
            {
                "key": "model_affected_items",
                "fn": "score_item_set",
                "truth_key": "scored.affected_items",
                "derive_from": "refit_covariate_paths",
                "covariate": "gender",
                "criterion": "model_item_level_bias",
            },
        ],
    )
    contract["subset"] = {"country": "US", "gender": [1, 2]}
    return [
        {
            "id": TASK_ID,
            "name": "Can this questionnaire compare the two groups?",
            "uuid": "3b8e4b4c-57a0-4a1e-9a8d-0e62cf0c5101",
            "keywords": ["psychometrics", "measurement invariance", "DIF", "group comparison"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "preliminary_memo": "preliminary_analysis.md",
                "behavior_data": "behavior.csv",
                "behavior_codebook": "behavior_codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


def candidate_submissions(X):
    """The intended answer and the ways an analysis stops short of it."""
    correct = {
        "model_syntax": reference_syntax(),
        "recommendation": RECOMMENDATION,
        "affected_items": sorted(ITEM_DIF),
    }
    flat = "F1 =~ " + "+".join(ITEMS) + "\nF1 ~ gender"
    return {
        "correct": correct,
        "trusts the memo": {
            "model_syntax": flat,
            "recommendation": {"status": "supported"},
            "affected_items": [],
        },
        "fits a latent model but frees no item": {
            "model_syntax": unadjusted_syntax(),
            "recommendation": {"status": "supported"},
            "affected_items": [],
        },
        "frees the items but keeps the memo's conclusion": {
            **correct,
            "recommendation": {"status": "supported"},
        },
        "hedges instead of concluding": {
            **correct,
            "recommendation": {"status": "supported_only_after_reanalysis"},
        },
        "finds only half the biased items": {
            "model_syntax": unadjusted_syntax() + "\nHSNS1 ~ gender\nHSNS8 ~ gender",
            "recommendation": RECOMMENDATION,
            "affected_items": ["HSNS1", "HSNS8"],
        },
        "frees every item": {
            "model_syntax": unadjusted_syntax()
            + "\n"
            + "\n".join(f"{item} ~ gender" for item in ITEMS),
            "recommendation": RECOMMENDATION,
            "affected_items": ITEMS,
        },
    }


def verify(df, behavior):
    """Confirm the conclusion reverses only when the biased items are freed."""
    X = analysis_sample(df)
    raw = raw_effect_size(X)
    unadjusted = group_difference(X, unadjusted_syntax())
    adjusted = group_difference(X, reference_syntax())

    print(f"complete cases: {len(X):,}")
    print("\n  the same question, asked three ways:")
    print(f"    total scores                      d = {raw:+.3f}")
    print(
        f"    latent model, no item freed       F1 = {unadjusted['F1']:+.3f}, "
        f"F2 = {unadjusted['F2']:+.3f}"
    )
    print(
        f"    latent model, four items freed    F1 = {adjusted['F1']:+.3f}, "
        f"F2 = {adjusted['F2']:+.3f}"
    )
    print("\n  the groups were generated with no trait difference at all.")

    first_pass = item_gender_effects(X, ITEMS)
    anchors = purified_anchors(X)
    flagged = sorted(set(ITEMS) - set(anchors))
    final = item_gender_effects(X, anchors)
    gap = min(abs(final[i]) for i in flagged) / max(abs(final[i]) for i in anchors)

    ranked = sorted(ITEMS, key=lambda item: -abs(first_pass[item]))
    intruder = next(item for item in ranked if item in anchors)
    print("\n  finding the biased items:")
    print(
        f"    against the raw total, the clean item {intruder} ranks "
        f"{ranked.index(intruder) + 1}th of 10, above a biased one"
    )
    print(f"    against purified anchors, the two sets separate by {gap:.0f}x")
    print(
        f"    behavioural indicators: credit={behavior.credit_allocation.mean():.2f}, "
        f"feedback={behavior.feedback_reactivity.mean():.2f}"
    )

    return C.report(
        [
            (
                "both groups have substantial complete samples",
                all((X.gender == g).sum() > 4_000 for g in [1, 2]),
            ),
            ("the total-score comparison shows a large difference", raw > 0.30),
            (
                "so does a latent model that frees no item",
                min(unadjusted.values()) > 0.15,
            ),
            (
                "freeing the four items leaves no trait difference",
                max(abs(v) for v in adjusted.values()) <= ADJUSTED_TOLERANCE,
            ),
            (
                "so the conclusion reverses rather than merely weakens",
                min(unadjusted.values()) - max(abs(v) for v in adjusted.values()) > 0.15,
            ),
            (
                "a scan against the raw total does not separate them",
                max(abs(first_pass[i]) for i in anchors) > min(abs(first_pass[i]) for i in flagged),
            ),
            (
                "purifying the anchors recovers exactly the biased items",
                flagged == sorted(ITEM_DIF),
            ),
            ("and separates them from the rest by a wide margin", gap > 5),
        ]
    )


def item_gender_effects(X, anchors):
    """Each item's group effect, holding the anchor items' total score constant.

    anchors  items assumed to behave the same way in both groups
    """
    total = X[anchors].sum(axis=1).values
    design = np.c_[np.ones(len(X)), (X.gender == 2).astype(float).values, total]
    return {
        item: round(float(np.linalg.lstsq(design, X[item].values, rcond=None)[0][1]), 3)
        for item in ITEMS
    }


def purified_anchors(X, stop_at=0.15):
    """Drop the most deviant item and rescore, until the rest agree.

    Scoring every item against a total that includes the biased ones spreads
    their effect over the whole scale. Removing them one at a time and
    rescoring on what is left undoes that.

    stop_at  leave the remaining items alone once none exceeds this
    """
    anchors = list(ITEMS)
    while len(anchors) > 3:
        effects = item_gender_effects(X, anchors)
        worst = max(anchors, key=lambda item: abs(effects[item]))
        if abs(effects[worst]) < stop_at:
            break
        anchors.remove(worst)
    return anchors


def naive(df):
    """Confirm the memo's analysis reaches the opposite of the right answer."""
    from scipy import stats

    X = analysis_sample(df)
    totals = X[ITEMS].sum(axis=1)
    t, p = stats.ttest_ind(totals[X.gender == 2], totals[X.gender == 1], equal_var=False)
    print(f"  total-score comparison: d = {raw_effect_size(X):+.3f}, t = {t:.2f}, p = {p:.2e}")
    print(f"  internal consistency:   alpha = {C.alpha(X[ITEMS]):.3f}")
    print("  everything the memo checked looks healthy, and its conclusion is wrong:")
    print("  the groups do not differ on either trait.")

    return C.report(
        [
            ("the memo's comparison is large and significant", p < 0.05 and abs(t) > 10),
            (
                "but the groups were generated with no trait difference",
                set(LATENT_DIFFERENCE.values()) == {0.0},
            ),
            (
                "so the memo's recommendation is the opposite of the right answer",
                RECOMMENDATION["status"] == "not_supported",
            ),
        ]
    )


def main():
    action = C.mode()
    df, behavior = simulate(np.random.default_rng(SEED))
    if action == "verify":
        return verify(df, behavior)
    if action == "naive":
        return naive(df)
    pop = population_matrix(np.random.default_rng(SEED + 1001))
    floor = C.evaluate_model(reference_syntax(), analysis_sample(df), ALL_VARS, pop)
    C.write_artifacts(
        OUT_DIR,
        TASK_JSON,
        df,
        lambda sha: build_truth(floor, pop.tolist(), sha, len(df)),
        build_task_json,
    )
    write_codebook(OUT_DIR / "codebook.md", df)
    write_behavior(OUT_DIR / "behavior.csv", behavior)
    write_behavior_codebook(OUT_DIR / "behavior_codebook.md")
    write_task_memo(OUT_DIR / "preliminary_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
