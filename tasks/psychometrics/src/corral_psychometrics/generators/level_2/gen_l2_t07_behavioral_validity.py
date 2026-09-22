#!/usr/bin/env python3
"""Generate Level 2 Task 07 artifacts and scoring metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t07_behavioral_validity"
UUID = "d7c3b5a1-2e84-4c9f-8d16-6b0a7e5f3921"
SEED = 20260929
TASK = paths.task(__file__)

ITEMS = C.HSNS_ITEMS
OUTCOME = "behavior"
GROUP = "gender"
N_TRAIN = 6_000
N_HOLDOUT = 4_000
LATENT_GROUP_SHIFT = 0.80
LATENT_EFFECT = 0.18
GROUP_EFFECT = 0.40
TRAIN_ITEM_EFFECT = 0.28
ITEM_WITHOUT_REPLICATION = "HSNS7"
MISSING_RATE = 0.012

# These are classification rules, not hidden judgement calls.
MIN_EFFECT = 0.10
EFFECT_TOLERANCE = 0.06
ITEM_TRAIN_MIN = 0.10
ITEM_HOLDOUT_MAX = 0.05

LOADINGS = {
    "HSNS1": 0.64,
    "HSNS2": 0.70,
    "HSNS3": 0.61,
    "HSNS4": 0.62,
    "HSNS5": 0.66,
    "HSNS6": 0.48,
    "HSNS7": 0.68,
    "HSNS8": 0.66,
    "HSNS9": 0.59,
    "HSNS10": 0.63,
}

PROMPT = f"""You are evaluating whether the Hypersensitive Narcissism Scale predicts an independently measured behavioural outcome. A preliminary memo reports an association in a labelled training sample. A second, held-out sample contains the same questionnaire, group variable, and outcome.

Investigate whether the association is only a difference between groups or whether it also appears within groups. Assess whether the group-adjusted latent association generalizes to the holdout. Examine whether any item contributes an outcome association in training that does not replicate in the holdout.

Use these decision rules for the reported claims:

  - `supported`: the group-adjusted latent coefficient has absolute value at least {MIN_EFFECT:.2f} in both samples.
  - `replicates`: the latent coefficient has absolute value at least {MIN_EFFECT:.2f} in both groups and the two within-group estimates differ by no more than {EFFECT_TOLERANCE:.2f}.
  - `generalizes`: the group-adjusted latent coefficient changes by no more than {EFFECT_TOLERANCE:.2f} between training and holdout.

For the item check, identify an item whose direct outcome association is at least {ITEM_TRAIN_MIN:.2f} in training but no more than {ITEM_HOLDOUT_MAX:.2f} in the holdout after accounting for the latent trait and group. The response file and the outcome file are already joined by `participant_id`; do not infer a mechanism from a pooled coefficient alone.
"""

SUBMISSION_FORMAT = """A single JSON object:

{
  "model_syntax": "F =~ HSNS1+...+HSNS10\\nbehavior ~ F + gender",
  "association": {
    "group_adjusted": "supported|not_supported",
    "within_group": "replicates|does_not_replicate",
    "holdout": "generalizes|does_not_generalize"
  },
  "unstable_items": ["<item identified from the evidence>"]
}

The model must cover every HSNS item, include `behavior`, and include both the
latent factor and `gender` in the behavioural regression. The scorer refits the
submitted measurement model in the training and holdout samples and performs
the within-group and item checks from that model. The submission records
conclusions, not a required analysis workflow.
"""


def reference_syntax():
    return f"F =~ {'+'.join(ITEMS)}\n{OUTCOME} ~ F + {GROUP} + {ITEM_WITHOUT_REPLICATION}"


def simulate(rng, n, item_effect):
    """Generate responses and a behavioural outcome from a one-factor trait."""
    group = rng.choice([1, 2], size=n)
    group01 = (group == 2).astype(float)
    eta = rng.normal(size=n) + LATENT_GROUP_SHIFT * group01
    signals = {}
    responses = {}
    for item, loading in LOADINGS.items():
        signal = loading * eta + rng.normal(0, np.sqrt(1 - loading**2), n)
        signals[item] = signal
        responses[item] = C.categorize(signal, C.THRESHOLDS[item])

    outcome = (
        LATENT_EFFECT * eta
        + GROUP_EFFECT * group01
        + item_effect * signals[ITEM_WITHOUT_REPLICATION]
        + rng.normal(0, 0.70, n)
    )
    frame = pd.DataFrame(responses)
    frame[GROUP] = group
    frame[OUTCOME] = outcome
    frame[ITEMS] = frame[ITEMS].mask(rng.random((n, len(ITEMS))) < MISSING_RATE, 0)
    frame.insert(0, "participant_id", [f"{'T' if item_effect else 'H'}{i:05d}" for i in range(n)])
    return frame[["participant_id", *ITEMS, GROUP, OUTCOME]]


def write_codebook(path):
    path.write_text(
        "# Codebook - behavioural validity\n\n"
        "The training and holdout files are tab-separated and already joined by "
        "`participant_id`. Ratings run from 1 (Disagree) to 5 (Agree); 0 denotes a "
        "missing item response. The behavioural outcome is continuous.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `participant_id` | participant identifier, unique within each file |\n"
        "| `gender` | 1 = Group 1, 2 = Group 2 |\n"
        "| `behavior` | independently measured behavioural outcome; higher values indicate more of the behaviour |\n\n"
        "The training file is `data.csv`; `holdout.csv` was collected separately "
        "and has no training/holdout indicator in the data.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n"
    )


def write_memo(path):
    path.write_text(
        "# Preliminary analysis\n\n"
        "The training sample shows a statistically clear association between the "
        "HSNS total and the behavioural outcome. The analyst recommends describing "
        "the HSNS as a behavioural predictor.\n\n"
        "The analysis did not adjust for group membership, separate the association "
        "within groups, or test the result in the second sample. It also treated all "
        "items as interchangeable contributors to the total.\n"
    )


def build_task_json(data_sha):
    return [
        {
            "id": TASK_ID,
            "name": "Does the HSNS predict behaviour beyond group membership?",
            "uuid": UUID,
            "keywords": ["psychometrics", "criterion validity", "prediction", "replication"],
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
                ],
                "training_dataset": "data.csv",
                "holdout_dataset": "holdout.csv",
                "codebook": "codebook.md",
                "preliminary_analysis": "preliminary_analysis.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_behavioral_validity",
            "scoring_params": {
                "task_type": "behavioral_validity",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "dataset": "data.csv",
                "holdout_dataset": "holdout.csv",
                "items": ITEMS,
                "outcome": OUTCOME,
                "covariate": GROUP,
                "max_factors": 3,
                "min_effect": MIN_EFFECT,
                "effect_tolerance": EFFECT_TOLERANCE,
                "item_train_min": ITEM_TRAIN_MIN,
                "item_holdout_max": ITEM_HOLDOUT_MAX,
            },
        }
    ]


def truth(data_sha, rows):
    return {
        "task_id": TASK_ID,
        "scored": {
            "association": {
                "group_adjusted": "supported",
                "within_group": "replicates",
                "holdout": "generalizes",
            },
            "unstable_items": [ITEM_WITHOUT_REPLICATION],
        },
        "generative_parameters": {
            "latent_effect": LATENT_EFFECT,
            "latent_group_shift": LATENT_GROUP_SHIFT,
            "group_effect": GROUP_EFFECT,
            "training_item_effect": TRAIN_ITEM_EFFECT,
            "item_without_replication": ITEM_WITHOUT_REPLICATION,
            "thresholds": {
                "min_effect": MIN_EFFECT,
                "effect_tolerance": EFFECT_TOLERANCE,
                "item_train_min": ITEM_TRAIN_MIN,
                "item_holdout_max": ITEM_HOLDOUT_MAX,
            },
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def candidate_submissions():
    correct = {
        "model_syntax": reference_syntax(),
        "association": {
            "group_adjusted": "supported",
            "within_group": "replicates",
            "holdout": "generalizes",
        },
        "unstable_items": [ITEM_WITHOUT_REPLICATION],
    }
    return {
        "correct": correct,
        "pooled_only": {
            **correct,
            "association": {**correct["association"], "within_group": "does_not_replicate"},
        },
        "ignores_holdout": {
            **correct,
            "association": {**correct["association"], "holdout": "does_not_generalize"},
        },
        "misses_item_instability": {**correct, "unstable_items": []},
        "raw_total_only": {**correct, "model_syntax": "behavior ~ gender"},
    }


def _fit_coefficients(frame, syntax):
    import semopy

    model = semopy.Model(syntax)
    model.fit(frame[ITEMS + [GROUP, OUTCOME]])
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.lval == OUTCOME)]
    return {
        row.rval: float(row["Est. Std"]) for _, row in rows.iterrows() if row.rval in {"F", GROUP}
    }


def _measurement_syntax():
    return f"F =~ {'+'.join(ITEMS)}"


def _within(frame, gender):
    syntax = _measurement_syntax() + f"\n{OUTCOME} ~ F + {ITEM_WITHOUT_REPLICATION}"
    return _fit_coefficients(frame[frame[GROUP] == gender], syntax)


def _item_effect(frame, item):
    syntax = _measurement_syntax() + f"\n{OUTCOME} ~ F + {GROUP} + {item}"
    import semopy

    model = semopy.Model(syntax)
    model.fit(frame[ITEMS + [GROUP, OUTCOME]])
    ins = model.inspect(std_est=True)
    row = ins[(ins.op == "~") & (ins.lval == OUTCOME) & (ins.rval == item)]
    return float(row["Est. Std"].iloc[0])


def verify(train, holdout):
    train = train[(train[ITEMS] != 0).all(axis=1)]
    holdout = holdout[(holdout[ITEMS] != 0).all(axis=1)]
    pooled = _fit_coefficients(
        train, _measurement_syntax() + f"\n{OUTCOME} ~ F + {ITEM_WITHOUT_REPLICATION}"
    )["F"]
    adjusted = _fit_coefficients(train, reference_syntax())["F"]
    holdout_adjusted = _fit_coefficients(holdout, reference_syntax())["F"]
    within = {_group: _within(train, _group)["F"] for _group in (1, 2)}
    item_train = _item_effect(train, ITEM_WITHOUT_REPLICATION)
    item_holdout = _item_effect(holdout, ITEM_WITHOUT_REPLICATION)
    print(f"pooled/group-adjusted latent coefficient: {pooled:+.3f} / {adjusted:+.3f}")
    print(f"holdout group-adjusted coefficient:       {holdout_adjusted:+.3f}")
    print(f"within groups:                             {within[1]:+.3f}, {within[2]:+.3f}")
    print(
        f"{ITEM_WITHOUT_REPLICATION} direct effect:                 {item_train:+.3f} / {item_holdout:+.3f}"
    )
    return C.report(
        [
            ("the pooled association contains a between-group component", pooled - adjusted > 0.04),
            ("the adjusted association is supported", abs(adjusted) >= MIN_EFFECT),
            (
                "both within-group associations are supported",
                min(abs(v) for v in within.values()) >= MIN_EFFECT,
            ),
            ("the within-group estimates agree", abs(within[1] - within[2]) <= EFFECT_TOLERANCE),
            (
                "the latent association generalizes",
                abs(adjusted - holdout_adjusted) <= EFFECT_TOLERANCE,
            ),
            ("the training-only item effect is visible", abs(item_train) >= ITEM_TRAIN_MIN),
            ("the item effect does not replicate", abs(item_holdout) <= ITEM_HOLDOUT_MAX),
        ]
    )


def naive(train, holdout):
    train = train[(train[ITEMS] != 0).all(axis=1)]
    holdout = holdout[(holdout[ITEMS] != 0).all(axis=1)]
    pooled = _fit_coefficients(train, reference_syntax())["F"]
    print(f"pooled coefficient: {pooled:+.3f}")
    print("A pooled coefficient alone cannot distinguish a between-group association,")
    print("a within-group association, or a training-only item contribution.")
    return C.report(
        [
            (
                "the pooled analysis does not answer the within-group question",
                abs(pooled) > MIN_EFFECT,
            ),
            ("the holdout is not interchangeable with the training sample", len(holdout) > 3_000),
        ]
    )


def main():
    action = C.mode()
    train = simulate(np.random.default_rng(SEED), N_TRAIN, TRAIN_ITEM_EFFECT)
    holdout = simulate(np.random.default_rng(SEED + 1), N_HOLDOUT, 0.0)
    if action == "verify":
        return verify(train, holdout)
    if action == "naive":
        return naive(train, holdout)

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TASK.artifacts / "data.csv", sep="\t", index=False)
    holdout.to_csv(TASK.artifacts / "holdout.csv", sep="\t", index=False)
    write_codebook(TASK.artifacts / "codebook.md")
    write_memo(TASK.artifacts / "preliminary_analysis.md")
    C.write_json(TASK.truth, truth(C.file_sha256(TASK.artifacts / "data.csv"), len(train)))
    C.write_json(TASK.definition, build_task_json(C.file_sha256(TASK.artifacts / "data.csv")))
    print(f"{len(train):,} training rows and {len(holdout):,} holdout rows -> {TASK.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
