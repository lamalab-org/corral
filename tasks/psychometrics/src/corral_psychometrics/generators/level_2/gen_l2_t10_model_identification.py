#!/usr/bin/env python3
"""Generate Level 2 Task 10 artifacts and scoring metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corral_psychometrics import paths
from corral_psychometrics.generators import common as C

TASK_ID = "psy_l2_t10_model_identification"
UUID = "a7d3f4b1-2c86-4e19-9a52-6f8d0b3c7415"
SEED = 20261010
TASK = paths.task(__file__)

ITEMS = [f"Q{i:02d}" for i in range(1, 13)]
DATASETS = [f"dataset_{i:02d}" for i in range(1, 7)]
MODELS = [f"model_{letter}" for letter in "abcdef"]
MODEL_D_BLOCKS = [(0, 3), (3, 7), (7, 12)]
N = 2_400

# The same cut points every other task uses, borrowed for twelve items.
THRESHOLDS = dict(zip(ITEMS, list(C.THRESHOLDS.values())[: len(ITEMS)]))

MODEL_DESCRIPTIONS = {
    "model_a": "one general factor; all items have comparable loadings",
    "model_b": "two correlated factors; Q01-Q06 and Q07-Q12 form the subscales",
    "model_c": "a bifactor model; every item loads on a general factor and one subscale",
    "model_d": "three correlated factors; Q01-Q03, Q04-Q07 and Q08-Q12 form the subscales",
    "model_e": "two correlated factors with Q06 also loading on the second factor",
    "model_f": "one general factor with a response-method effect shared by Q03 and Q10",
}

# A candidate belongs in a dataset's answer when its BIC is within this of the
# best-fitting candidate. The agent is told the number. Two is the conventional
# point below which a BIC difference is not worth interpreting, and it is well
# under what one extra free parameter costs here, log(2400) = 7.8.
TIE_MARGIN = 2.0

# Inside one block of every candidate, so it cannot mimic a general factor.
DEPENDENT_PAIR = ("Q02", "Q03")
DEPENDENT_SHARE = 0.28
WEAK_ITEM = "Q08"
WEAK_LOADING = 0.26
SHIFTED_ITEM = "Q11"
DUPLICATED_ROWS = 320

COMPLICATIONS = {
    "dataset_01": "responses to Q01 go missing more often at the low end of the scale",
    "dataset_02": f"{DEPENDENT_PAIR[0]} and {DEPENDENT_PAIR[1]} share residual variance",
    "dataset_03": f"a subgroup answers {SHIFTED_ITEM} one threshold lower",
    "dataset_04": f"{DUPLICATED_ROWS} rows are delivered twice",
    "dataset_05": f"{WEAK_ITEM} is weak but still measures the trait",
    "dataset_06": "a quarter of respondents answer everything one category higher",
}


def factor_syntax(model):
    """The lavaan-style model a candidate stands for."""
    join = "+".join
    if model == "model_a":
        return f"G =~ {join(ITEMS)}"
    if model == "model_b":
        return f"F1 =~ {join(ITEMS[:6])}\nF2 =~ {join(ITEMS[6:])}\nF1 ~~ F2"
    if model == "model_c":
        blocks = "\n".join(f"S{i + 1} =~ {join(ITEMS[i * 4:(i + 1) * 4])}" for i in range(3))
        zeros = "\n".join(
            f"{a} ~~ 0*{b}"
            for a, b in [
                ("G", "S1"),
                ("G", "S2"),
                ("G", "S3"),
                ("S1", "S2"),
                ("S1", "S3"),
                ("S2", "S3"),
            ]
        )
        return f"G =~ {join(ITEMS)}\n{blocks}\n{zeros}"
    if model == "model_d":
        return "\n".join(
            f"F{i + 1} =~ {join(ITEMS[a:b])}" for i, (a, b) in enumerate(MODEL_D_BLOCKS)
        )
    if model == "model_e":
        return f"F1 =~ {join(ITEMS[:6])}\nF2 =~ {join(ITEMS[6:])}+Q06\nF1 ~~ F2"
    return f"G =~ {join(ITEMS)}\nM =~ Q03+Q10\nG ~~ 0*M"


def latent_scores(model, rng, n):
    """The unobserved score behind each item, under one candidate model."""
    columns = []
    if model == "model_a":
        eta = rng.normal(size=n)
        columns = [0.68 * eta for _ in ITEMS]
    elif model == "model_b":
        eta = rng.multivariate_normal([0, 0], [[1, 0.35], [0.35, 1]], n)
        columns = [0.72 * eta[:, 0 if i < 6 else 1] for i in range(12)]
    elif model == "model_c":
        general = rng.normal(size=n)
        specific = rng.normal(size=(n, 3))
        columns = [0.62 * general + 0.38 * specific[:, i // 4] for i in range(12)]
    elif model == "model_d":
        eta = rng.multivariate_normal(
            [0, 0, 0], [[1, 0.25, 0.10], [0.25, 1, 0.30], [0.10, 0.30, 1]], n
        )
        block = {i: b for b, (a, z) in enumerate(MODEL_D_BLOCKS) for i in range(a, z)}
        columns = [0.73 * eta[:, block[i]] for i in range(12)]
    elif model == "model_e":
        eta = rng.multivariate_normal([0, 0], [[1, 0.28], [0.28, 1]], n)
        columns = [
            0.70 * eta[:, 0] + (0.38 * eta[:, 1] if i == 5 else 0.0) if i < 6 else 0.70 * eta[:, 1]
            for i in range(12)
        ]
    else:
        eta = rng.normal(size=n)
        method = rng.normal(size=n)
        columns = [0.68 * eta + (0.42 * method if i in (2, 9) else 0.0) for i in range(12)]
    return np.column_stack(columns)


def responses(model, rng, n, complication=None):
    """Answers from one candidate model, with one data-quality problem on top."""
    signal = latent_scores(model, rng, n)
    if complication == "weak_item":
        index = ITEMS.index(WEAK_ITEM)
        shared = signal[:, index]
        scale = WEAK_LOADING / max(float(np.std(shared)), 1e-9)
        signal[:, index] = shared * scale
    if complication == "dependence":
        shared = rng.normal(size=n)
        for item in DEPENDENT_PAIR:
            signal[:, ITEMS.index(item)] += np.sqrt(DEPENDENT_SHARE) * shared

    values = {}
    for index, item in enumerate(ITEMS):
        column = signal[:, index]
        residual = np.sqrt(max(1.0 - float(np.var(column)), 0.05))
        values[item] = C.categorize(column + rng.normal(0, residual, n), THRESHOLDS[item])
    frame = pd.DataFrame(values)[ITEMS]

    if complication == "shift":
        subgroup = rng.random(n) < 0.42
        cut = np.asarray(THRESHOLDS[SHIFTED_ITEM]) - 0.55
        column = signal[:, ITEMS.index(SHIFTED_ITEM)]
        residual = np.sqrt(max(1.0 - float(np.var(column)), 0.05))
        shifted = C.categorize(column + rng.normal(0, residual, n), cut)
        frame.loc[subgroup, SHIFTED_ITEM] = shifted[subgroup]
    if complication == "response_style":
        subgroup = rng.random(n) < 0.25
        frame.loc[subgroup, ITEMS] = np.clip(frame.loc[subgroup, ITEMS] + 1, 1, 5)
    if complication == "missing":
        low = frame["Q01"] <= 2
        drop = (rng.random(n) < 0.35) & low.to_numpy()
        frame.loc[drop, "Q01"] = 0
        frame[ITEMS] = frame[ITEMS].mask(rng.random((n, len(ITEMS))) < 0.012, 0)
    if complication == "duplicates":
        picked = rng.choice(n, size=DUPLICATED_ROWS, replace=False)
        frame = pd.concat([frame, frame.iloc[picked]], ignore_index=True)
    return frame


COMPLICATION_KIND = {
    "dataset_01": "missing",
    "dataset_02": "dependence",
    "dataset_03": "shift",
    "dataset_04": "duplicates",
    "dataset_05": "weak_item",
    "dataset_06": "response_style",
}


def analysis_sample(frame):
    """Complete responses, with repeated rows collapsed."""
    clean = frame[ITEMS]
    clean = clean[(clean != 0).all(axis=1)]
    return clean[~clean.duplicated()].astype(float)


def bic_by_model(frame):
    """BIC of every candidate model on one sample. Lower is better."""
    import semopy

    X = analysis_sample(frame)
    out = {}
    for model in MODELS:
        try:
            fitted = semopy.Model(factor_syntax(model))
            fitted.fit(X)
            chi2 = float(semopy.calc_stats(fitted)["chi2"].iloc[0])
            out[model] = round(chi2 + len(fitted.param_vals) * np.log(len(X)), 2)
        except Exception:  # noqa: BLE001
            out[model] = float("inf")
    return out


def supported(frame, margin=TIE_MARGIN):
    """Candidates the sample cannot separate from the best-fitting one."""
    bic = bic_by_model(frame)
    best = min(bic.values())
    return sorted(model for model, value in bic.items() if value - best <= margin), bic


def assignment(rng):
    """Which model each dataset was drawn from. Stored only in the hidden truth."""
    return dict(zip(DATASETS, rng.permutation(MODELS)))


def build_frames(rng, mapping, with_complications=True):
    return {
        dataset: responses(
            mapping[dataset],
            rng,
            N,
            COMPLICATION_KIND[dataset] if with_complications else None,
        )
        for dataset in DATASETS
    }


PROMPT = """You have six anonymised response datasets and a catalogue of six candidate measurement models. Each dataset is an independent sample from one of the candidates, and each also carries a different data-quality problem.

Identify which candidate generated each dataset. Treat the data-quality problems as distortions of the evidence rather than as model identities: a dependent item pair, a weak item, a subgroup that answers one item differently, repeated rows, missing responses and a response-style shift can all disturb the summaries used to tell the models apart.

Fit each candidate to each dataset and compare them by BIC after removing rows with missing responses and collapsing exact duplicate response patterns. Report every candidate whose BIC is within 2 of the best-fitting one for that dataset; where that is a single model, report one, and where it is more, report them all. Do not break a tie arbitrarily."""

SUBMISSION_FORMAT = """A single JSON object:

{
  "assignments": {
    "<dataset_id>": ["<model_id>", ...]
  }
}

Include every dataset exactly once and use only identifiers from
`candidate_models.csv`. Report more than one model only when their BIC values
are within 2 of each other. No working is submitted."""


def write_codebook(path):
    path.write_text(
        "# Codebook - model identification\n\n"
        "Each `dataset_*.csv` is tab-separated and holds an independent anonymised "
        "sample answering the same twelve items. Responses run from 1 (low) to 5 "
        "(high); `0` denotes a missing response.\n\n"
        "| variable | meaning |\n|---|---|\n"
        "| `respondent_id` | row identifier |\n"
        + "\n".join(f"| `{item}` | ordered response, 1 to 5 |" for item in ITEMS)
        + "\n\n`candidate_models.csv` gives each model's description and lavaan-style "
        "syntax. The order of the catalogue carries no information.\n"
    )


def build_task_json(data_sha):
    return [
        {
            "id": TASK_ID,
            "name": "Which model generated each dataset?",
            "uuid": UUID,
            "keywords": ["psychometrics", "model identification", "robustness"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                # Exactly the files copied into the agent's workspace.
                "public_inputs": [
                    "candidate_models.csv",
                    "codebook.md",
                    "dataset_01.csv",
                    "dataset_02.csv",
                    "dataset_03.csv",
                    "dataset_04.csv",
                    "dataset_05.csv",
                    "dataset_06.csv",
                ],
                "datasets": [f"{name}.csv" for name in DATASETS],
                "candidate_models": "candidate_models.csv",
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_identification",
            "scoring_params": {
                "task_type": "model_identification",
                "data_dir": TASK.artifacts_relpath,
                "truth_path": TASK.truth_relpath,
                "datasets": DATASETS,
                "models": MODELS,
                "tie_margin": TIE_MARGIN,
            },
        }
    ]


def truth(data_sha, mapping, answers, measured):
    return {
        "task_id": TASK_ID,
        "scored": {"assignments": answers},
        "generative_parameters": {
            "dataset_to_model": mapping,
            "model_descriptions": MODEL_DESCRIPTIONS,
            "complications": COMPLICATIONS,
            "tie_margin": TIE_MARGIN,
            "bic_by_dataset": measured,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, N * len(DATASETS), data_sha),
    }


def candidate_submissions():
    answers = json.loads(TASK.truth.read_text())["scored"]["assignments"]
    catalogue_order = {dataset: [model] for dataset, model in zip(DATASETS, MODELS)}
    single = {dataset: value[:1] for dataset, value in answers.items()}
    tied = next((d for d, v in answers.items() if len(v) > 1), None)
    cases = {
        "correct": {"assignments": answers},
        "reads the catalogue in order": {"assignments": catalogue_order},
        "assigns everything to one model": {
            "assignments": {dataset: ["model_a"] for dataset in DATASETS}
        },
    }
    if tied:
        cases["breaks the tie arbitrarily"] = {"assignments": single}
    else:
        wrong = dict(answers)
        first, second = DATASETS[0], DATASETS[1]
        wrong[first], wrong[second] = answers[second], answers[first]
        cases["swaps two datasets"] = {"assignments": wrong}
    return cases


def verify(mapping, frames, clean_frames):
    """Confirm each dataset's own model is recoverable despite its complication."""
    print(f"hidden permutation: {', '.join(f'{d[-2:]}->{m[-1]}' for d, m in mapping.items())}")
    print(f"\n  {'dataset':11s} {'from':7s} {'complication':16s} {'answer by BIC':22s} recovered")
    answers, measured, recovered, unchanged = {}, {}, [], []
    for dataset in DATASETS:
        chosen, bic = supported(frames[dataset])
        answers[dataset] = chosen
        measured[dataset] = bic
        recovered.append(mapping[dataset] in chosen)
        clean_choice, _ = supported(clean_frames[dataset])
        unchanged.append(clean_choice == chosen)
        print(
            f"  {dataset:11s} {mapping[dataset][-1]:7s} {COMPLICATION_KIND[dataset]:16s} "
            f"{','.join(m[-1] for m in chosen):22s} {'yes' if recovered[-1] else 'NO'}"
        )
    spread = {
        dataset: round(sorted(measured[dataset].values())[1] - min(measured[dataset].values()), 1)
        for dataset in DATASETS
    }
    print(
        "\n  BIC gap to the runner-up: " + ", ".join(f"{d[-2:]}:{v:.0f}" for d, v in spread.items())
    )

    redraws = {}
    for offset in (91, 92):
        other = build_frames(np.random.default_rng(SEED + offset), mapping)
        for dataset in DATASETS:
            chosen, _ = supported(other[dataset])
            redraws.setdefault(dataset, chosen)
            if redraws[dataset] != chosen:
                redraws[dataset] = None
    print(
        "  same answers on two independent redraws: "
        + ", ".join(f"{d[-2:]}:{'yes' if redraws[d] == answers[d] else 'NO'}" for d in DATASETS)
    )

    return C.report(
        [
            ("every dataset's own model is in its answer", all(recovered)),
            (
                "the complication does not change any answer",
                all(unchanged),
            ),
            (
                "the mapping is not the catalogue order",
                [mapping[d] for d in DATASETS] != MODELS,
            ),
            (
                "at least one dataset is identified uniquely",
                any(len(v) == 1 for v in answers.values()),
            ),
            (
                "no dataset admits every candidate",
                max(len(v) for v in answers.values()) < len(MODELS),
            ),
            (
                "the answers are the same on independently redrawn samples",
                all(redraws[d] == answers[d] for d in DATASETS),
            ),
        ]
    )


def naive(mapping, frames, clean_frames):
    """Confirm picking the best-fitting model, without a parsimony penalty, is wrong."""
    del clean_frames
    import semopy

    print("  choosing the model with the smallest chi-square, ignoring how many")
    print("  parameters it spends:\n")
    print(f"  {'dataset':11s} {'correct':9s} {'best chi-square':16s} {'parameters':>10s}")
    wrong = []
    for dataset in DATASETS:
        chosen, _ = supported(frames[dataset])
        X = analysis_sample(frames[dataset])
        chi, npar = {}, {}
        for model in MODELS:
            try:
                fitted = semopy.Model(factor_syntax(model))
                fitted.fit(X)
                chi[model] = float(semopy.calc_stats(fitted)["chi2"].iloc[0])
                npar[model] = len(fitted.param_vals)
            except Exception:  # noqa: BLE001
                chi[model] = float("inf")
                npar[model] = 0
        best = min(chi, key=chi.get)
        if [best] != chosen:
            wrong.append((dataset, npar[best] - npar[chosen[0]]))
        print(f"  {dataset:11s} {','.join(m[-1] for m in chosen):9s} {best:16s} {npar[best]:10d}")
    catalogue = dict(zip(DATASETS, MODELS))
    in_order = [
        dataset for dataset in DATASETS if [catalogue[dataset]] != supported(frames[dataset])[0]
    ]
    print(
        f"\n  it disagrees with the parsimony-adjusted answer on {len(wrong)} of "
        f"{len(DATASETS)} datasets: {', '.join(d for d, _ in wrong) if wrong else 'none'}"
    )
    print(f"  reading the catalogue in order is wrong on {len(in_order)} of {len(DATASETS)}")

    return C.report(
        [
            ("fit alone prefers a model that is not the answer", len(wrong) > 0),
            (
                "and every time, one that spends more parameters",
                bool(wrong) and all(extra > 0 for _, extra in wrong),
            ),
            ("reading the catalogue in order is wrong too", len(in_order) > 0),
        ]
    )


def main():
    action = C.mode()
    mapping = assignment(np.random.default_rng(SEED))
    frames = build_frames(np.random.default_rng(SEED + 1), mapping)
    if action in ("verify", "naive"):
        clean = build_frames(np.random.default_rng(SEED + 1), mapping, with_complications=False)
        return (
            verify(mapping, frames, clean) if action == "verify" else naive(mapping, frames, clean)
        )

    TASK.artifacts.mkdir(parents=True, exist_ok=True)
    TASK.definition.parent.mkdir(parents=True, exist_ok=True)
    answers, measured = {}, {}
    for dataset in DATASETS:
        chosen, bic = supported(frames[dataset])
        answers[dataset], measured[dataset] = chosen, bic
        written = frames[dataset].copy()
        written.insert(0, "respondent_id", [f"{dataset}_{i:05d}" for i in range(len(written))])
        written.to_csv(TASK.artifacts / f"{dataset}.csv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"model_id": key, "description": value, "syntax": factor_syntax(key)}
            for key, value in MODEL_DESCRIPTIONS.items()
        ]
    ).to_csv(TASK.artifacts / "candidate_models.csv", sep="\t", index=False)
    write_codebook(TASK.artifacts / "codebook.md")
    data_sha = C.file_sha256(TASK.artifacts / "dataset_01.csv")
    C.write_json(TASK.truth, truth(data_sha, mapping, answers, measured))
    C.write_json(TASK.definition, build_task_json(data_sha))
    print(f"\n{len(DATASETS)} anonymised datasets -> {TASK.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
