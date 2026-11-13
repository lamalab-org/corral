#!/usr/bin/env python3
"""Generate Level 2 Task 08 artifacts and scoring metadata.

Generating model: two correlated HSNS factors (phi 0.28), with four problems
planted in both samples and one planted in the development sample alone.

    Matrix-page factor   four items, variance 0.22       replicates
    HSNS9 -> F1         cross-loading 0.38              replicates
    HSNS3               differential item functioning   replicates
    HSNS6               loading 0.30, too weak to keep  replicates
    Second page factor   HSNS1/HSNS2, variance 0.36      development only

The development-only pair is the largest single improvement to development fit
and is absent from the replication sample, so chasing fit alone gets it wrong.
The stable format crosses the personality dimensions; its four indicators allow
the shared method pattern to be tested separately from personality cross-loadings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t08_misfit_replication"
UUID = "b8e4c6a2-1f73-4d95-9c20-7a5e3b8d6142"
SEED = 20260930
TASK = paths.task(__file__)

ITEMS = C.HSNS_ITEMS
GROUP = "gender"
N_DEVELOPMENT = 6_000
N_REPLICATION = 4_000
MISSING_RATE = 0.012

F1_ITEMS = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
F2_ITEMS = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
LOADINGS = {
    "HSNS1": ("F1", 0.64),
    "HSNS2": ("F2", 0.70),
    "HSNS3": ("F2", 0.61),
    "HSNS4": ("F1", 0.62),
    "HSNS5": ("F1", 0.66),
    "HSNS6": ("F1", 0.22),
    "HSNS7": ("F2", 0.68),
    "HSNS8": ("F1", 0.66),
    "HSNS9": ("F2", 0.59),
    "HSNS10": ("F1", 0.63),
}
FACTOR_CORRELATION = 0.28
LATENT_GROUP_DIFFERENCE = 0.20

# Presentation groups span the personality dimensions. Factors are orthogonal
# to the traits and each other, with unit loadings and estimated variances.
METHOD_ITEMS = ("HSNS4", "HSNS5", "HSNS7", "HSNS10")
METHOD_VARIANCE = 0.22
DEVELOPMENT_METHOD_ITEMS = ("HSNS1", "HSNS2")
DEVELOPMENT_METHOD_VARIANCE = 0.36

CROSS_LOADING = ("HSNS9", "F1", 0.38)
STABLE_DIF = "HSNS3"
DIF_SHIFT = 0.60
WEAK_ITEM = "HSNS6"

# A modification counts as supported in a sample when removing it costs at least
# this much CFI there.
REPLICATION_GAIN = 0.010
WEAK_LOADING = 0.30

FINDINGS = {
    "local_dependence": [list(METHOD_ITEMS)],
    "cross_loadings": [CROSS_LOADING[0]],
    "dif_items": [STABLE_DIF],
    "poor_items": [WEAK_ITEM],
}
REPLICATION = {
    "local_dependence": "replicates",
    "cross_loadings": "replicates",
    "dif_items": "replicates",
    "poor_items": "replicates",
    "rejected_development_modifications": [list(DEVELOPMENT_METHOD_ITEMS)],
}

PROMPT = """You are reviewing a two-factor HSNS measurement model. A development sample shows several item-level signs of misfit. An independently collected replication sample has the same questionnaire and group variable.

Decide which of the apparent problems are real, and classify each as a shared presentation effect among an item cluster, a cross-loading on the second factor, an item the two groups answer differently at the same trait level, or an item too weak to keep. An item may fall under more than one of these.

Return a final model, your findings, and for each kind of finding whether it holds up in the replication sample. Also list any modification the development sample supports that you rejected.
"""

SUBMISSION_FORMAT = """A single JSON object:

{
  "model_syntax": "<two trait factors, supported method factors and group paths>",
  "findings": {
    "local_dependence": [["<items sharing a method effect>"], ...],
    "cross_loadings": ["<item>", ...],
    "dif_items": ["<item>", ...],
    "poor_items": ["<item>", ...]
  },
  "replication": {
    "local_dependence": "does_not_replicate|replicates",
    "cross_loadings": "does_not_replicate|replicates",
    "dif_items": "does_not_replicate|replicates",
    "poor_items": "does_not_replicate|replicates",
    "rejected_development_modifications": [["<item>", "<item>"], ...]
  }
}

The model may include orthogonal method factors, cross-loadings and item paths from
`gender`. Give `gender` a path to each personality factor. Method factors must be
orthogonal to the personality factors and each other; fix their loadings to 1
and estimate their variances, or use another identified parameterization. An item reported as a
poor item may be left out of the measurement model; no other item may be.

Each replication verdict is recomputed by removing that modification from your
model and refitting in the replication sample, so the verdicts have to match what
your own model does there. A kind of finding `replicates` when removing each of its
modifications lowers CFI in the replication sample by at least 0.01. Poor items
replicate when every one still has a standardised loading below 0.30 there. A
rejected modification is written as the items that would share an additional
method factor; it must raise CFI by at least 0.01 in the development sample and by
less than 0.01 in the replication sample. The final model must reach CFI >= 0.95
and RMSEA <= 0.08 in the replication sample."""


def baseline_syntax():
    """Two correlated factors and a group path to each, with nothing added."""
    return build_syntax(cross_loading=False, method=False, dif=False, drop_weak=False)


def build_syntax(cross_loading=True, method=True, dif=True, drop_weak=True, extra=()):
    """Assemble a model from the modifications it does and does not contain.

    cross_loading  let the cross-loading item load on F1 as well
    method         include the stable presentation factor
    dif            free the group path on the DIF item
    drop_weak      leave the weak item out of the measurement model
    extra          further lines to append
    """
    f1 = [item for item in F1_ITEMS if not (drop_weak and item == WEAK_ITEM)]
    if cross_loading:
        f1 = f1 + [CROSS_LOADING[0]]
    lines = [
        f"F1 =~ {'+'.join(f1)}",
        f"F2 =~ {'+'.join(F2_ITEMS)}",
        "F1 ~~ F2",
        f"F1 ~ {GROUP}",
        f"F2 ~ {GROUP}",
    ]
    if method:
        lines.append(C.method_syntax("Format", METHOD_ITEMS, ["F1", "F2"]))
    if dif:
        lines.append(f"{STABLE_DIF} ~ {GROUP}")
    return "\n".join(lines + list(extra))


def reference_syntax():
    """The four real modifications, and the weak item dropped."""
    return build_syntax()


def decoy_syntax():
    """The reference model with the development-only pair added."""
    return build_syntax(
        extra=(C.method_syntax("Transient", DEVELOPMENT_METHOD_ITEMS, ["F1", "F2", "Format"]),)
    )


def simulate(rng, n, development):
    """One sample. The decoy dependence exists only in the development one."""
    gender = rng.choice([1, 2], size=n)
    eta = rng.multivariate_normal(
        [0.0, 0.0], [[1.0, FACTOR_CORRELATION], [FACTOR_CORRELATION, 1.0]], n
    ) + np.where(gender[:, None] == 2, LATENT_GROUP_DIFFERENCE, 0.0)
    shared_stable = rng.normal(size=n)
    shared_decoy = rng.normal(size=n)
    out = {}
    for item, (factor, loading) in LOADINGS.items():
        index = 0 if factor == "F1" else 1
        signal = loading * eta[:, index]
        explained = loading**2
        if item == CROSS_LOADING[0]:
            extra = CROSS_LOADING[2]
            signal += extra * eta[:, 0]
            explained += extra**2 + 2 * extra * loading * FACTOR_CORRELATION
        if item in METHOD_ITEMS:
            signal += np.sqrt(METHOD_VARIANCE) * shared_stable
            explained += METHOD_VARIANCE
        if development and item in DEVELOPMENT_METHOD_ITEMS:
            signal += np.sqrt(DEVELOPMENT_METHOD_VARIANCE) * shared_decoy
            explained += DEVELOPMENT_METHOD_VARIANCE
        y = signal + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        intercept = np.asarray(C.INTERCEPTS[item], dtype=float)
        if item == STABLE_DIF:
            out[item] = np.where(
                gender == 2,
                C.continuous_response(y, intercept + DIF_SHIFT),
                C.continuous_response(y, intercept),
            )
        else:
            out[item] = C.continuous_response(y, intercept)
    frame = pd.DataFrame(out)[ITEMS]
    frame[GROUP] = gender
    frame[ITEMS] = frame[ITEMS].mask(rng.random((n, len(ITEMS))) < MISSING_RATE, np.nan)
    prefix = "D" if development else "R"
    frame.insert(0, "participant_id", [f"{prefix}{i:05d}" for i in range(n)])
    return frame[["participant_id", *ITEMS, GROUP]]


def complete(frame, items=None):
    """Rows answering every item the model uses."""
    columns = items or ITEMS
    return frame[frame[columns].notna().all(axis=1)][columns + [GROUP]].astype(float)


def cfi(frame, syntax, items=None):
    """CFI of a model in one sample."""
    import semopy

    model = C.fit(syntax, complete(frame, items), (items or ITEMS) + [GROUP])
    return float(semopy.calc_stats(model)["CFI"].iloc[0])


def without(syntax, line):
    """The same model with one line taken out."""
    return syntax.replace(line, "")


def modification_gain(frame, syntax, line, items=None):
    """How much CFI a single modification is worth in one sample."""
    return round(cfi(frame, syntax, items) - cfi(frame, without(syntax, line), items), 4)


def write_codebook(path):
    path.write_text(
        "# Codebook - misfit replication\n\n"
        "The development and replication files are tab-separated and hold the same ten "
        "HSNS items. Responses are continuous and unbounded; empty fields denote a missing "
        "response.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `participant_id` | identifier, unique within each file |\n"
        f"| `{GROUP}` | 1 = Group 1, 2 = Group 2 |\n\n"
        "Presentation log (the same format in both samples): "
        + ", ".join(METHOD_ITEMS)
        + " appeared together on one matrix page; "
        + ", ".join(DEVELOPMENT_METHOD_ITEMS)
        + " shared a separate page. "
        "All other items appeared individually. Whether a shared response influence "
        "exists or replicates must be checked in the responses.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n"
    )


def write_memo(path):
    path.write_text(
        "# Preliminary analysis\n\n"
        "The two-factor model fits the development sample poorly. Several item-level "
        "modification indices are large and the analyst proposes adding whichever paths "
        "improve fit, starting with the largest. On that rule the model reaches an "
        "excellent development fit.\n\n"
        "The proposed changes have not been separated into residual dependence, "
        "cross-loading, group-dependent item behaviour or weak item quality, and the "
        "replication file has not been opened. Treat the development fit as a set of "
        "hypotheses rather than a finished model.\n"
    )


def build_task_json(data_sha):
    return [
        {
            "id": TASK_ID,
            "name": "Which model modifications survive replication?",
            "uuid": UUID,
            "keywords": ["psychometrics", "model fit", "replication", "DIF"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "codebook.md",
                    "data.csv",
                    "preliminary_analysis.md",
                    "replication.csv",
                ],
                "development_dataset": "data.csv",
                "replication_dataset": "replication.csv",
                "codebook": "codebook.md",
                "preliminary_analysis": "preliminary_analysis.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_misfit_replication",
            "scoring_params": {
                "task_type": "misfit_replication",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "dataset": "data.csv",
                "replication_dataset": "replication.csv",
                "items": ITEMS,
                "factor_items": {"F1": F1_ITEMS, "F2": F2_ITEMS},
                "covariate": GROUP,
                "max_factors": 5,
                "replication_gain": REPLICATION_GAIN,
                "weak_loading": WEAK_LOADING,
            },
        }
    ]


def truth(data_sha, rows):
    return {
        "task_id": TASK_ID,
        "scored": {"findings": FINDINGS, "replication": REPLICATION},
        "generative_parameters": {
            "stable_method": {"items": list(METHOD_ITEMS), "variance": METHOD_VARIANCE},
            "development_only_method": {
                "items": list(DEVELOPMENT_METHOD_ITEMS),
                "variance": DEVELOPMENT_METHOD_VARIANCE,
            },
            "cross_loading": list(CROSS_LOADING),
            "stable_dif_item": STABLE_DIF,
            "dif_shift": DIF_SHIFT,
            "weak_item": WEAK_ITEM,
            "latent_group_difference": LATENT_GROUP_DIFFERENCE,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def candidate_submissions():
    """The intended answer and the ways a modification search goes wrong."""
    correct = {
        "model_syntax": reference_syntax(),
        "findings": FINDINGS,
        "replication": REPLICATION,
    }
    return {
        "correct": correct,
        "keeps every development modification": {
            "model_syntax": decoy_syntax(),
            "findings": {
                **FINDINGS,
                "local_dependence": [list(METHOD_ITEMS), list(DEVELOPMENT_METHOD_ITEMS)],
            },
            "replication": {**REPLICATION, "rejected_development_modifications": []},
        },
        "modifies nothing": {
            "model_syntax": baseline_syntax(),
            "findings": {key: [] for key in FINDINGS},
            "replication": {**REPLICATION, "rejected_development_modifications": []},
        },
        "calls the cross-loading a dependence": {
            **correct,
            "model_syntax": build_syntax(
                cross_loading=False, dif=False, extra=(f"{CROSS_LOADING[0]} ~~ HSNS8",)
            ),
            "findings": {
                **FINDINGS,
                "cross_loadings": [],
                "dif_items": [],
                "local_dependence": [list(METHOD_ITEMS), [CROSS_LOADING[0], "HSNS8"]],
            },
        },
        "keeps the weak item": {
            "model_syntax": build_syntax(drop_weak=False),
            "findings": {**FINDINGS, "poor_items": []},
            "replication": REPLICATION,
        },
    }


def verify(development, replication):
    """Confirm every intended finding is recoverable and the decoy is not."""
    kept = [item for item in ITEMS if item != WEAK_ITEM]
    ref, dec = reference_syntax(), decoy_syntax()
    lines = {
        "cross_loadings": None,
        "local_dependence": C.method_syntax("Format", METHOD_ITEMS, ["F1", "F2"]),
        "dif_items": f"{STABLE_DIF} ~ {GROUP}",
    }
    decoy_line = C.method_syntax("Transient", DEVELOPMENT_METHOD_ITEMS, ["F1", "F2", "Format"])

    print("what each modification is worth, in CFI, in each sample\n")
    print(f"  {'modification':34s} {'development':>12s} {'replication':>12s}")
    gains = {}
    for name, line in lines.items():
        if line is None:
            no_cross = build_syntax(cross_loading=False)
            gains[name] = (
                round(cfi(development, ref, kept) - cfi(development, no_cross, kept), 4),
                round(cfi(replication, ref, kept) - cfi(replication, no_cross, kept), 4),
            )
        else:
            gains[name] = (
                modification_gain(development, ref, line, kept),
                modification_gain(replication, ref, line, kept),
            )
        print(f"  {name:34s} {gains[name][0]:+12.4f} {gains[name][1]:+12.4f}")
    gains["decoy"] = (
        modification_gain(development, dec, decoy_line, kept),
        modification_gain(replication, dec, decoy_line, kept),
    )
    print(
        f"  {'the development-only pair':34s} {gains['decoy'][0]:+12.4f} {gains['decoy'][1]:+12.4f}"
    )

    weak_dev = C.loadings(C.fit(baseline_syntax(), complete(development), ITEMS + [GROUP]))[
        WEAK_ITEM
    ]
    weak_rep = C.loadings(C.fit(baseline_syntax(), complete(replication), ITEMS + [GROUP]))[
        WEAK_ITEM
    ]
    ref_rep, dec_rep = cfi(replication, ref, kept), cfi(replication, dec, kept)
    ref_dev, dec_dev = cfi(development, ref, kept), cfi(development, dec, kept)
    print(
        f"\n  weak item {WEAK_ITEM} loading: development {weak_dev:.3f}, replication {weak_rep:.3f}"
    )
    print(f"  whole-model CFI  development: correct {ref_dev:.4f}  with decoy {dec_dev:.4f}")
    print(f"                   replication: correct {ref_rep:.4f}  with decoy {dec_rep:.4f}")

    real = [gains[name] for name in lines]
    return C.report(
        [
            (
                "every intended modification is detectable in development",
                min(dev for dev, _ in real) > REPLICATION_GAIN,
            ),
            (
                "and every one is still supported in replication",
                min(rep for _, rep in real) > REPLICATION_GAIN,
            ),
            (
                "the group paths make the DIF item visible",
                gains["dif_items"][0] > 0.02 and gains["dif_items"][1] > 0.02,
            ),
            (
                "the decoy is supported in development",
                gains["decoy"][0] > max(dev for dev, _ in real),
            ),
            (
                "and buys nothing in replication",
                gains["decoy"][1] < REPLICATION_GAIN,
            ),
            (
                "the weak item is weak in both samples",
                max(weak_dev, weak_rep) < WEAK_LOADING,
            ),
            (
                "chasing development fit prefers the decoy model",
                dec_dev > ref_dev + REPLICATION_GAIN,
            ),
        ]
    )


def naive(development, replication):
    """Confirm the largest development modification is the one to throw away."""
    kept = [item for item in ITEMS if item != WEAK_ITEM]
    ref, dec = reference_syntax(), decoy_syntax()
    decoy_line = C.method_syntax("Transient", DEVELOPMENT_METHOD_ITEMS, ["F1", "F2", "Format"])
    candidates = {
        C.method_syntax("Format", METHOD_ITEMS, ["F1", "F2"]): "real",
        f"{STABLE_DIF} ~ {GROUP}": "real",
        decoy_line: "development only",
    }
    print("  adding one modification at a time to the baseline, by development gain:\n")
    ranked = []
    for line, kind in candidates.items():
        gain = (
            round(cfi(development, ref, kept) - cfi(development, without(ref, line), kept), 4)
            if line != decoy_line
            else modification_gain(development, dec, decoy_line, kept)
        )
        ranked.append((line, kind, gain))
    ranked.sort(key=lambda row: -row[2])
    for line, kind, gain in ranked:
        print(f"    {line:26s} +{gain:.4f}   {kind}")
    best = ranked[0]
    rep_gain = modification_gain(replication, dec, decoy_line, kept)
    print(f"\n  the largest development gain is {best[0]}, which is {best[1]}.")
    print(f"  in the replication sample it is worth {rep_gain:+.4f}.")

    return C.report(
        [
            (
                "the largest development modification is the false one",
                best[1] == "development only",
            ),
            ("it is worth nothing in replication", rep_gain < REPLICATION_GAIN),
            (
                "so ranking by development fit retains a path that does not exist",
                best[2] > max(gain for _, kind, gain in ranked if kind == "real"),
            ),
        ]
    )


def main():
    action = C.mode()
    development = simulate(np.random.default_rng(SEED), N_DEVELOPMENT, True)
    replication = simulate(np.random.default_rng(SEED + 1), N_REPLICATION, False)
    if action == "verify":
        return verify(development, replication)
    if action == "naive":
        return naive(development, replication)

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    development.to_csv(TASK.artifacts / "data.csv", sep="\t", index=False)
    replication.to_csv(TASK.artifacts / "replication.csv", sep="\t", index=False)
    write_codebook(TASK.artifacts / "codebook.md")
    write_memo(TASK.artifacts / "preliminary_analysis.md")
    data_sha = C.file_sha256(TASK.artifacts / "data.csv")
    C.write_json(TASK.truth, truth(data_sha, len(development)))
    C.write_json(TASK.definition, build_task_json(data_sha))
    print(
        f"\n{len(development):,} development and {len(replication):,} replication rows -> {TASK.artifacts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
