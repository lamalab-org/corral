#!/usr/bin/env python3
"""Generate Level 2 Task 08 artifacts and scoring metadata.

Generating model: two correlated HSNS factors (phi 0.28), with four problems
planted in both samples and one planted in the development sample alone.

    HSNS5 ~~ HSNS7      residual covariance 0.22        replicates
    HSNS9 -> F1         cross-loading 0.38              replicates
    HSNS3               differential item functioning   replicates
    HSNS6               loading 0.30, too weak to keep  replicates
    HSNS1 ~~ HSNS2      residual covariance 0.30        development only

The development-only pair is the largest single improvement to development fit
and is absent from the replication sample, so chasing fit alone gets it wrong.
Both replicating covariances cross the two factors; a pair inside one factor
would be absorbed by it and leave nothing to find.
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

# Both residual covariances cross the two factors. A pair inside one factor is
# absorbed by that factor and leaves almost nothing to find.
STABLE_RESIDUAL = ("HSNS5", "HSNS7")
STABLE_RESIDUAL_CORR = 0.22
DEVELOPMENT_ONLY_RESIDUAL = ("HSNS1", "HSNS2")
DEVELOPMENT_RESIDUAL_CORR = 0.30

CROSS_LOADING = ("HSNS9", "F1", 0.38)
STABLE_DIF = "HSNS3"
DIF_SHIFT = 0.60
WEAK_ITEM = "HSNS6"

# A modification counts as supported in a sample when removing it costs at least
# this much CFI there.
REPLICATION_GAIN = 0.010
WEAK_LOADING = 0.30

FINDINGS = {
    "local_dependence": [list(STABLE_RESIDUAL)],
    "cross_loadings": [CROSS_LOADING[0]],
    "dif_items": [STABLE_DIF],
    "poor_items": [WEAK_ITEM],
}
REPLICATION = {
    "local_dependence": "replicates",
    "cross_loadings": "replicates",
    "dif_items": "replicates",
    "poor_items": "replicates",
    "rejected_development_modifications": [list(DEVELOPMENT_ONLY_RESIDUAL)],
}

PROMPT = """You are reviewing a two-factor HSNS measurement model. A development sample shows several item-level signs of misfit, and an independently collected replication sample contains the same questionnaire and group variable.

Work out which apparent problems are real and what each one is. Consider local dependence between item pairs, items loading on a second factor, items answered differently by the two groups once the factors are accounted for, and items too weak to keep. Improving development fit is not evidence on its own: every candidate modification improves it, and the one that improves it most is not in the replication sample at all.

Return a final model, a compositional account of the findings, and for each kind of finding whether it holds up in the replication sample. The categories are not mutually exclusive. Also report any modification the development sample supports that you decided to reject."""

SUBMISSION_FORMAT = """A single JSON object:

{
  "model_syntax": "<complete two-factor model with group paths>",
  "findings": {
    "local_dependence": [["item_a", "item_b"]],
    "cross_loadings": ["item_c"],
    "dif_items": ["item_d"],
    "poor_items": ["item_e"]
  },
  "replication": {
    "local_dependence": "replicates|does_not_replicate",
    "cross_loadings": "replicates|does_not_replicate",
    "dif_items": "replicates|does_not_replicate",
    "poor_items": "replicates|does_not_replicate",
    "rejected_development_modifications": [["item_f", "item_g"]]
  }
}

The model may include residual covariances, cross-loadings and item paths from
`gender`, and must give `gender` a path to each factor. An item reported as a
poor item may be left out of the measurement model; no other item may be.

Each replication verdict is recomputed by removing that modification from your
model and refitting in the replication sample, so the verdicts have to match what
your own model does there."""


def baseline_syntax():
    """Two correlated factors and a group path to each, with nothing added."""
    return build_syntax(cross_loading=False, residual=False, dif=False, drop_weak=False)


def build_syntax(cross_loading=True, residual=True, dif=True, drop_weak=True, extra=()):
    """Assemble a model from the modifications it does and does not contain.

    cross_loading  let the cross-loading item load on F1 as well
    residual       free the stable residual covariance
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
    if residual:
        lines.append(f"{STABLE_RESIDUAL[0]} ~~ {STABLE_RESIDUAL[1]}")
    if dif:
        lines.append(f"{STABLE_DIF} ~ {GROUP}")
    return "\n".join(lines + list(extra))


def reference_syntax():
    """The four real modifications, and the weak item dropped."""
    return build_syntax()


def decoy_syntax():
    """The reference model with the development-only pair added."""
    return build_syntax(
        extra=(f"{DEVELOPMENT_ONLY_RESIDUAL[0]} ~~ {DEVELOPMENT_ONLY_RESIDUAL[1]}",)
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
        if item in STABLE_RESIDUAL:
            signal += np.sqrt(STABLE_RESIDUAL_CORR) * shared_stable
            explained += STABLE_RESIDUAL_CORR
        if development and item in DEVELOPMENT_ONLY_RESIDUAL:
            signal += np.sqrt(DEVELOPMENT_RESIDUAL_CORR) * shared_decoy
            explained += DEVELOPMENT_RESIDUAL_CORR
        y = signal + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        tau = np.asarray(C.THRESHOLDS[item], dtype=float)
        if item == STABLE_DIF:
            out[item] = np.where(
                gender == 2, C.categorize(y, tau - DIF_SHIFT), C.categorize(y, tau)
            )
        else:
            out[item] = C.categorize(y, tau)
    frame = pd.DataFrame(out)[ITEMS]
    frame[GROUP] = gender
    frame[ITEMS] = frame[ITEMS].mask(rng.random((n, len(ITEMS))) < MISSING_RATE, 0)
    prefix = "D" if development else "R"
    frame.insert(0, "participant_id", [f"{prefix}{i:05d}" for i in range(n)])
    return frame[["participant_id", *ITEMS, GROUP]]


def complete(frame, items=None):
    """Rows answering every item the model uses."""
    columns = items or ITEMS
    return frame[(frame[columns] != 0).all(axis=1)][columns + [GROUP]].astype(float)


def cfi(frame, syntax, items=None):
    """CFI of a model in one sample."""
    import semopy

    model = C.fit(syntax, complete(frame, items), (items or ITEMS) + [GROUP])
    return float(semopy.calc_stats(model)["CFI"].iloc[0])


def without(syntax, line):
    """The same model with one line taken out."""
    return "\n".join(ln for ln in syntax.splitlines() if ln.strip() != line)


def modification_gain(frame, syntax, line, items=None):
    """How much CFI a single modification is worth in one sample."""
    return round(cfi(frame, syntax, items) - cfi(frame, without(syntax, line), items), 4)


def write_codebook(path):
    path.write_text(
        "# Codebook - misfit replication\n\n"
        "The development and replication files are tab-separated and hold the same ten "
        "HSNS items. Ratings run from 1 (Disagree) to 5 (Agree); `0` denotes a missing "
        "response.\n\n"
        "| variable | description |\n|---|---|\n"
        "| `participant_id` | identifier, unique within each file |\n"
        f"| `{GROUP}` | 1 = Group 1, 2 = Group 2 |\n\n"
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
                "max_factors": 3,
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
            "stable_residual": {"pair": list(STABLE_RESIDUAL), "correlation": STABLE_RESIDUAL_CORR},
            "development_only_residual": {
                "pair": list(DEVELOPMENT_ONLY_RESIDUAL),
                "correlation": DEVELOPMENT_RESIDUAL_CORR,
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
                "local_dependence": [list(STABLE_RESIDUAL), list(DEVELOPMENT_ONLY_RESIDUAL)],
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
                "local_dependence": [list(STABLE_RESIDUAL), [CROSS_LOADING[0], "HSNS8"]],
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
        "local_dependence": f"{STABLE_RESIDUAL[0]} ~~ {STABLE_RESIDUAL[1]}",
        "dif_items": f"{STABLE_DIF} ~ {GROUP}",
    }
    decoy_line = f"{DEVELOPMENT_ONLY_RESIDUAL[0]} ~~ {DEVELOPMENT_ONLY_RESIDUAL[1]}"

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
    decoy_line = f"{DEVELOPMENT_ONLY_RESIDUAL[0]} ~~ {DEVELOPMENT_ONLY_RESIDUAL[1]}"
    candidates = {
        f"{STABLE_RESIDUAL[0]} ~~ {STABLE_RESIDUAL[1]}": "real",
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
