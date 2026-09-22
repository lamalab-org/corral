#!/usr/bin/env python3
"""Generate Level 2 Task 06 artifacts and scoring metadata.

Generating model: one HSNS factor, with group 2 genuinely 0.25 higher on the
trait, and five kinds of item planted on top.

    HSNS2 HSNS3   export wrote a middle response over the real answer, group 2
    HSNS5 HSNS9   real differential item functioning - the trait is the same,
                  the response function is not
    HSNS7         mis-keyed
    HSNS6         no fault; the item simply measures poorly
    the rest      sound

The delivery audit flags the re-sent batch as HSNS2, HSNS3 and HSNS4 - HSNS4
rode along undamaged, and neither genuinely biased item was in that batch, so
the audit alone gives the wrong answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t06_gender_item_integrity"
UUID = "5f2d8b71-6d4b-4db9-a3b8-0c9e7f1a624d"
SEED = 20260928
TASK = paths.task(__file__)

ITEMS = C.HSNS_ITEMS
N_PER_GROUP = 5_000
LATENT_DIFFERENCE = 0.25
MISSING_RATE = 0.012
EFFECT_TOLERANCE = 0.05

LOADINGS = {
    "HSNS1": 0.64,
    "HSNS2": 0.70,
    "HSNS3": 0.61,
    "HSNS4": 0.62,
    "HSNS5": 0.66,
    "HSNS6": 0.28,
    "HSNS7": 0.68,
    "HSNS8": 0.66,
    "HSNS9": 0.59,
    "HSNS10": 0.63,
}

# The export wrote a middle response over the real answer, for group 2 only.
INSERTED_NEUTRAL = {"HSNS2": 0.35, "HSNS3": 0.30}
# Genuine differential item functioning: the respondent's own trait, a different
# response function in group 2. HSNS5 widens the middle category, HSNS9 shifts.
WIDENED = ("HSNS5", 0.60, -0.40)
SHIFTED = ("HSNS9", -0.90)
MIS_KEYED = "HSNS7"
WEAK = "HSNS6"

DIAGNOSES = {
    "HSNS1": "sound",
    "HSNS2": "inserted_neutral",
    "HSNS3": "inserted_neutral",
    "HSNS4": "sound",
    "HSNS5": "gender_dif",
    "HSNS6": "weak_item",
    "HSNS7": "mis_keyed",
    "HSNS8": "sound",
    "HSNS9": "gender_dif",
    "HSNS10": "sound",
}
LABELS = ["sound", "mis_keyed", "inserted_neutral", "gender_dif", "weak_item"]

# Items whose response function differs between the groups, whatever the cause.
# A model that compares the groups has to let these differ.
GROUP_DEPENDENT = sorted(set(INSERTED_NEUTRAL) | {WIDENED[0], SHIFTED[0]})

# The audit flags the batch that was re-sent. HSNS4 rode along undamaged, and
# neither DIF item was in it.
RETRANSMITTED = ["HSNS2", "HSNS3", "HSNS4"]
FIELD_WIDTH = {
    "HSNS1": 3,
    "HSNS2": 2,
    "HSNS3": 2,
    "HSNS4": 2,
    "HSNS5": 3,
    "HSNS6": 2,
    "HSNS7": 3,
    "HSNS8": 3,
    "HSNS9": 2,
    "HSNS10": 3,
}

PROMPT = """You are investigating a reported difference between two groups on the Hypersensitive Narcissism Scale. You have the responses, a preliminary memo, and an audit describing how the response file was exported.

Several items do not behave the same way in the two groups. An item can look that way because the export wrote values over the real answers, because the item genuinely functions differently in the two groups, because it was stored with its scale reversed, or because it is simply a weak item. The audit describes transmission, not measurement: an item it flags may be undamaged, and an item it does not flag may still be faulty.

One check is worth knowing about. If a value was written in by the export it tells you nothing about the respondent, so the people carrying it look like an arbitrary slice of the sample on every other item. If the same value is a genuine answer, the people giving it sit where their remaining responses say they should. Comparing a response category against the rest of the questionnaire separates the two.

Diagnose every item using exactly these labels: `sound`, `mis_keyed`, `inserted_neutral`, `gender_dif`, `weak_item`. Any label may apply to no items, one item, or several. Then repair what can be repaired, fit a model that supports a group comparison, and say whether the comparison is reportable."""

SUBMISSION_FORMAT = """A single JSON object:

{
  "model_syntax": "<complete model with the required group paths>",
  "item_diagnoses": {"<item>": "<label>", ...},
  "comparison": "reportable_after_repair|not_reportable"
}

Give one diagnosis for every HSNS item. The model must cover every item and give
`gender` a path to the latent factor. An item that does not behave the same way
in both groups is declared by giving it its own path from `gender`.

Items diagnosed `mis_keyed` are reversed before the model is refitted, and the
remaining group difference is read from that refit, so the diagnoses and the
model have to agree with each other."""


def thresholds_for(item, gender):
    """Where this item's response categories fall for this group."""
    tau = list(C.THRESHOLDS[item])
    if gender == 2 and item == WIDENED[0]:
        tau[1] -= WIDENED[1]
        tau[2] += WIDENED[1]
        tau = [t + WIDENED[2] for t in tau]
    if gender == 2 and item == SHIFTED[0]:
        tau = [t + SHIFTED[1] for t in tau]
    return tau


def responses(rng, n, gender):
    """One group's answers, from one latent trait per respondent."""
    eta = rng.normal(0.0, 1.0, n) + (LATENT_DIFFERENCE if gender == 2 else 0.0)
    out = {}
    for item, loading in LOADINGS.items():
        noise = rng.normal(0, np.sqrt(1 - loading**2), n)
        out[item] = C.categorize(loading * eta + noise, thresholds_for(item, gender))
    return pd.DataFrame(out)[ITEMS]


def build_data(rng):
    """The delivered file: real answers, then what the export did to them."""
    frames = []
    for gender in (1, 2):
        block = responses(rng, N_PER_GROUP, gender)
        if gender == 2:
            for item, rate in INSERTED_NEUTRAL.items():
                block.loc[rng.random(len(block)) < rate, item] = 3
        block[MIS_KEYED] = 6 - block[MIS_KEYED]
        block["gender"] = gender
        frames.append(block)
    data = pd.concat(frames, ignore_index=True)
    data[ITEMS] = data[ITEMS].mask(rng.random(data[ITEMS].shape) < MISSING_RATE, 0)
    data["country"] = "US"
    return data[ITEMS + ["gender", "country"]]


def analysis_sample(df, repair_mis_key=True):
    """Complete responses, optionally with the reversed item put back."""
    X = df[(df[ITEMS] != 0).all(axis=1)][ITEMS + ["gender"]].astype(float).copy()
    if repair_mis_key:
        X[MIS_KEYED] = 6 - X[MIS_KEYED]
    return X


def reference_syntax():
    """One trait, a group path, and a free path for every group-dependent item."""
    paths = "\n".join(f"{item} ~ gender" for item in GROUP_DEPENDENT)
    return "F =~ " + "+".join(ITEMS) + "\nF ~ gender\n" + paths


def gender_effect(X, syntax):
    """The group difference the model leaves on the trait."""
    model = C.fit(syntax, X, ITEMS + ["gender"])
    estimates = model.inspect(std_est=True)
    row = estimates[(estimates.op == "~") & (estimates.rval == "gender") & (estimates.lval == "F")]
    return round(float(row["Est. Std"].iloc[0]), 3)


def neutral_profile(X, item):
    """How much the people giving a middle response vary on everything else.

    A value written in by the export carries no information, so the people
    holding it spread as widely as the whole group. A genuine middle answer
    picks out a narrow band of the trait.
    """
    rest = X[[i for i in ITEMS if i != item]].sum(axis=1)
    middle = rest[X[item] == 3]
    return round(float(middle.std() / rest.std()), 3) if len(middle) > 30 else None


def write_codebook(path):
    path.write_text(
        "# Codebook - group comparison\n\n"
        "Tab-separated, one row per respondent. Responses are five-point ratings from "
        "1 (Disagree) to 5 (Agree); `0` denotes a missing response.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n\n| variable | description |\n|---|---|\n"
        "| `gender` | 1 = Group 1, 2 = Group 2 |\n"
        "| `country` | collection country; all rows are US |\n"
    )


def write_audit(out_dir):
    """Transmission records. They say how the file moved, not what is in it."""
    rows = []
    for item in ITEMS:
        resent = item in RETRANSMITTED
        rows.append(
            {
                "item": item,
                "export_batch": "B2" if resent else "B1",
                "transmission_window": "2026-03-04 22:10-22:40"
                if resent
                else "2026-03-04 09:00-11:30",
                "field_width": FIELD_WIDTH[item],
                "rows_written": N_PER_GROUP * 2,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "export_audit.csv", sep="\t", index=False)
    (out_dir / "export_audit_codebook.md").write_text(
        "# Codebook - export audit\n\n"
        "One row per item, from the export system's own logs. These are transmission "
        "records: they describe how each column was written out, not what the responses "
        "mean.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `item` | HSNS item |\n"
        "| `export_batch` | batch the column was written in |\n"
        "| `transmission_window` | when that batch was sent |\n"
        "| `field_width` | characters allocated to the column |\n"
        "| `rows_written` | rows the export reported writing |\n\n"
        "Batch `B2` was re-sent the same evening after the first attempt was "
        "interrupted. The operator recorded no detail about what, if anything, "
        "differed in the second attempt.\n"
    )
    (out_dir / "preliminary_analysis.md").write_text(
        "# Preliminary analysis\n\n"
        "Group 2 scored higher on the HSNS total and the difference was highly "
        "significant, so the analyst recommended reporting it as a group difference.\n\n"
        "Two observations were left unresolved. One item correlates negatively with "
        "the total until it is recoded. Several items sit at the middle response far "
        "more often in Group 2 than in Group 1, and the analyst assumed these were the "
        "columns named in the export audit.\n"
    )


def build_task_json(shas, effect):
    return [
        {
            "id": TASK_ID,
            "name": "Is the group difference real or an export artifact?",
            "uuid": UUID,
            "keywords": ["psychometrics", "item integrity", "DIF", "data cleaning"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "codebook.md",
                    "data.csv",
                    "export_audit.csv",
                    "export_audit_codebook.md",
                    "preliminary_analysis.md",
                ],
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "preliminary_analysis": "preliminary_analysis.md",
                "export_audit": "export_audit.csv",
                "export_audit_codebook": "export_audit_codebook.md",
                "data_sha256": shas["data.csv"],
            },
            "tools": [],
            "scoring_function": "score_gender_item_integrity",
            "scoring_params": {
                "task_type": "gender_item_integrity",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "dataset": "data.csv",
                "items": ITEMS,
                "covariate": "gender",
                "labels": LABELS,
                "max_factors": 4,
                "effect_tolerance": EFFECT_TOLERANCE,
            },
        }
    ]


def candidate_submissions():
    """The intended answer and the ways this investigation goes wrong."""
    correct = {
        "model_syntax": reference_syntax(),
        "item_diagnoses": DIAGNOSES,
        "comparison": "reportable_after_repair",
    }
    trusting = {**DIAGNOSES, "HSNS4": "inserted_neutral", "HSNS5": "sound", "HSNS9": "sound"}
    by_count = {**DIAGNOSES, "HSNS3": "sound", "HSNS5": "inserted_neutral"}
    return {
        "correct": correct,
        "trusts the audit's batch": {
            **correct,
            "item_diagnoses": trusting,
            "model_syntax": "F =~ "
            + "+".join(ITEMS)
            + "\nF ~ gender\n"
            + "\n".join(f"{i} ~ gender" for i in RETRANSMITTED),
        },
        "calls the most middle responses corrupted": {**correct, "item_diagnoses": by_count},
        "never repairs the reversed item": {
            **correct,
            "item_diagnoses": {**DIAGNOSES, MIS_KEYED: "sound"},
        },
        "reports the raw difference": {
            **correct,
            "model_syntax": "F =~ " + "+".join(ITEMS) + "\nF ~ gender",
            "comparison": "not_reportable",
        },
        "hedges on the comparison": {**correct, "comparison": "not_reportable"},
    }


def verify(data):
    """Confirm the faults are separable only by the conditional check."""
    X = analysis_sample(data)
    group2 = X[X.gender == 2]
    effect = gender_effect(X, reference_syntax())
    base = "F =~ " + "+".join(ITEMS) + "\nF ~ gender"
    naive_effect = gender_effect(X, base)
    audit_effect = gender_effect(X, base + "\n" + "\n".join(f"{i} ~ gender" for i in RETRANSMITTED))

    print("group difference left on the trait, by which items the model frees:")
    print(f"  the four that behave differently:  {effect:+.3f}")
    print(f"  none at all:                       {naive_effect:+.3f}")
    print(f"  the three the audit flags:         {audit_effect:+.3f}")

    print("\n  group 2, middle responses and what they tell you:")
    print(f"    {'item':8s} {'% at 3':>7s} {'spread ratio':>13s}  audit  truth")
    profile = {}
    for item in ITEMS:
        share = float((group2[item] == 3).mean())
        profile[item] = neutral_profile(group2, item)
        flag = "B2" if item in RETRANSMITTED else "  "
        print(f"    {item:8s} {share:7.2f} {profile[item]:13.3f}   {flag}   {DIAGNOSES[item]}")

    candidates = [i for i in ITEMS if float((group2[i] == 3).mean()) > 0.25]
    inserted = [profile[i] for i in candidates if i in INSERTED_NEUTRAL]
    genuine = [profile[i] for i in candidates if i not in INSERTED_NEUTRAL]
    r1, r2 = X[X.gender == 1], X[X.gender == 2]
    rest = lambda sub, item: sub[[i for i in ITEMS if i != item]].sum(axis=1)  # noqa: E731
    dif_kept = {
        item: (
            round(float(r1[item].corr(rest(r1, item))), 3),
            round(float(r2[item].corr(rest(r2, item))), 3),
        )
        for item in (WIDENED[0], SHIFTED[0])
    }
    print("\n  DIF items, item-rest correlation by group (the trait link must survive):")
    for item, (a, b) in dif_kept.items():
        shift = float((r2[item].mean() - r1[item].mean()))
        print(f"    {item}: group1 {a:+.3f}  group2 {b:+.3f}   mean shift {shift:+.3f}")

    return C.report(
        [
            (
                "every item with many middle responses is a candidate",
                set(candidates) == set(INSERTED_NEUTRAL) | {WIDENED[0]},
            ),
            (
                "and the middle response is uninformative only on the overwritten ones",
                min(inserted) > max(genuine) + 0.04,
            ),
            (
                "and the overwritten items are not the ones with the most middle responses",
                float((group2[WIDENED[0]] == 3).mean())
                > max(float((group2[i] == 3).mean()) for i in INSERTED_NEUTRAL),
            ),
            (
                "DIF keeps each item tied to the trait in both groups",
                all(
                    min(pair) > 0.30 and abs(pair[0] - pair[1]) < 0.12 for pair in dif_kept.values()
                ),
            ),
            (
                "while changing how the group answers it",
                abs(float(r2[SHIFTED[0]].mean() - r1[SHIFTED[0]].mean())) > 0.15,
            ),
            (
                "the audit flags a clean item and misses both DIF items",
                DIAGNOSES["HSNS4"] == "sound"
                and "HSNS4" in RETRANSMITTED
                and not {WIDENED[0], SHIFTED[0]} & set(RETRANSMITTED),
            ),
            (
                "freeing the group-dependent items changes the group difference",
                abs(naive_effect - effect) > EFFECT_TOLERANCE,
            ),
            (
                "following the audit instead changes it the other way",
                abs(audit_effect - effect) > EFFECT_TOLERANCE,
            ),
            (
                "three labels cover more than one item, so counts give nothing away",
                sum(
                    1
                    for label in set(DIAGNOSES.values())
                    if list(DIAGNOSES.values()).count(label) > 1
                )
                >= 3,
            ),
        ]
    )


def naive(data):
    """Confirm following the audit, or counting middle responses, gets it wrong."""
    X = analysis_sample(data)
    group2 = X[X.gender == 2]

    audit_says = {
        item: ("inserted_neutral" if item in RETRANSMITTED else "sound") for item in ITEMS
    }
    shares = {item: float((group2[item] == 3).mean()) for item in ITEMS}
    top = sorted(shares, key=shares.get, reverse=True)[: len(INSERTED_NEUTRAL)]
    count_says = {item: ("inserted_neutral" if item in top else "sound") for item in ITEMS}

    audit_wrong = [i for i in ITEMS if audit_says[i] != DIAGNOSES[i]]
    count_wrong = [i for i in ITEMS if count_says[i] != DIAGNOSES[i]]
    print(f"  believing the audit's batch flags:      {len(audit_wrong)} of 10 items wrong")
    print(f"    {', '.join(audit_wrong)}")
    print(f"  calling the most middle-heavy items corrupted: {len(count_wrong)} of 10 wrong")
    print(f"    {', '.join(count_wrong)}")
    print(f"\n  the most middle-heavy item is {top[0]}, and it is {DIAGNOSES[top[0]]}.")

    return C.report(
        [
            ("the audit's flags do not identify the faults", "HSNS4" in audit_wrong),
            ("and they miss the items that really differ", WIDENED[0] in audit_wrong),
            (
                "counting middle responses picks a genuine item first",
                DIAGNOSES[top[0]] != "inserted_neutral",
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    data = build_data(rng)
    if action == "verify":
        return verify(data)
    if action == "naive":
        return naive(data)

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(TASK.artifacts / "data.csv", sep="\t", index=False)
    write_codebook(TASK.artifacts / "codebook.md")
    write_audit(TASK.artifacts)
    effect = gender_effect(analysis_sample(data), reference_syntax())
    shas = {"data.csv": C.file_sha256(TASK.artifacts / "data.csv")}
    truth = {
        "task_id": TASK_ID,
        "scored": {
            "item_diagnoses": DIAGNOSES,
            "comparison": "reportable_after_repair",
            "group_dependent_items": GROUP_DEPENDENT,
            "repaired_gender_effect": effect,
        },
        "repairs": {"reverse_scored": {MIS_KEYED: 6}},
        "generative_parameters": {
            "latent_difference": LATENT_DIFFERENCE,
            "loadings": LOADINGS,
            "inserted_neutral_rates": INSERTED_NEUTRAL,
            "widened_middle_category": {WIDENED[0]: [WIDENED[1], WIDENED[2]]},
            "threshold_shift": {SHIFTED[0]: SHIFTED[1]},
            "mis_keyed": MIS_KEYED,
            "weak_item": WEAK,
            "retransmitted_batch": RETRANSMITTED,
            "items": ITEMS,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, len(data), shas["data.csv"]),
    }
    C.write_json(TASK.truth, truth)
    C.write_json(TASK.definition, build_task_json(shas, effect))
    print(f"\n{len(data):,} rows -> {TASK.artifacts / 'data.csv'}")
    for name in ("codebook.md", "export_audit.csv"):
        print(f"           {TASK.artifacts / name}")
    print(f"           {TASK.definition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
