"""Normalize saved SOAP/Ridge records by their contents, preserving chronology."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read

from .common import EvidenceError, UnsupportedEvidence, finite_array
from .formats import rename
from .verification import stored


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def stage(row):
    """Event labels are optional when the payload identifies the record's role."""
    name = row.get("stage", row.get("event", row.get("type")))
    aliases = {
        "hyperparameter_search": "model_selection",
        "cv_selected": "model_selection",
        "cv_candidate": "candidate",
        "pipelines_frozen": "freeze",
        "phase1_done_pipelines_frozen": "freeze",
        "final_frozen": "final_fit",
        "lc_frozen": "learning_curve_fit",
        "eval_final": "test_evaluation",
    }
    name = aliases.get(name, name)
    if name in {
        "train_labels",
        "model_selection",
        "candidate",
        "freeze",
        "final_fit",
        "learning_curve_fit",
        "learning_curve_fits",
        "test_labels",
        "test_evaluation",
    }:
        return name
    if "frozen" in row or "hashes" in row or "manifest" in row:
        return "freeze"
    if "pipelines" in row:
        return "learning_curve_fits"
    if "train_indices" in row or ("indices" in row and "n_train" in row):
        return "learning_curve_fit"
    if "independent" in row and any(k in row for k in ("n", "n_train", "path", "file")):
        return "learning_curve_evaluation"
    if "strained_pooled" in row or ("final" in row and "independent" in row["final"]):
        return "test_evaluation"
    if {"independent", "strained"} <= row.keys() or "independent_seed" in row:
        return "test_labels"
    if "seed_independent" in row and "seed_strained" in row:
        return "test_labels"
    if any(k in row for k in ("cv", "cv_scheme", "candidates")):
        return "model_selection"
    if any(k.startswith("cv_") for k in row) and ("soap" in row or "r_cut" in row):
        return "candidate"
    if any(k in row for k in ("file", "path")) and any(
        k in row for k in ("params", "soap", "alpha")
    ):
        return "final_fit"
    if "seed" in row and any(k in row for k in ("file", "path", "energies")):
        return "train_labels"
    return name


def parameters(record):
    result = rename(record, {"alpha": ("best_alpha",)})
    nested = result.pop("soap", {})
    require(isinstance(nested, dict), "SOAP settings must be an object")
    for key, value in nested.items():
        require(key not in result or result[key] == value, f"Conflicting SOAP {key}")
        result[key] = value
    if "preprocessing" in result:
        description = result["preprocessing"]
        if isinstance(description, str) and re.search(
            r"\bStandardScaler\b", description
        ):
            scale = True
        elif description in ("identity", "none", None):
            scale = False
        else:
            raise UnsupportedEvidence("Unsupported saved-pipeline preprocessing")
        require(
            "scale" not in result or result["scale"] == scale,
            "Conflicting preprocessing settings",
        )
        result["scale"] = scale
    return result


def timestamp(value):
    if isinstance(value, str):
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise UnsupportedEvidence(
                "Cannot interpret recorded calculation timestamp"
            ) from exc
        return instant.replace(tzinfo=instant.tzinfo or timezone.utc).timestamp()
    return float(finite_array(value, shape=()))


def normalize(rows, linked, log_path=None):
    """Group equivalent records without moving fits across the freeze boundary.

    A dataset without an explicit filename can be located by its complete saved
    label vector. Matching is confined to the log's directory and must be unique;
    filenames alone never establish dataset identity.
    """
    output = []
    for original in rows:
        row = rename(
            original,
            {
                "file": ("path",),
                "indices": ("train_indices",),
                "cv": ("cv_scheme",),
                "frozen": ("hashes",),
            },
        )
        row["stage"] = stage(original)
        if "time" in row:
            row["time"] = timestamp(row["time"])
        output.append(row)
    roles = [row["stage"] for row in output]
    candidates = [row for row in output if row["stage"] == "candidate"]
    selections = [row for row in output if row["stage"] == "model_selection"]
    if len(selections) == 1 and candidates:
        selection = selections[0]
        require(
            "train_labels" in roles
            and all(
                roles.index("train_labels") < i < roles.index("model_selection")
                for i, r in enumerate(output)
                if r["stage"] == "candidate"
            ),
            "Validation candidates must follow training labels and precede selection",
        )
        require(
            "file" not in selection and "candidates" not in selection,
            "Ambiguous inline and linked validation candidate records",
        )
        selection["candidates"] = candidates
    individual = [row for row in output if row["stage"] == "learning_curve_fit"]
    if individual:
        require(
            "learning_curve_fits" not in roles, "Ambiguous learning-curve fit records"
        )
        indices = [
            i for i, r in enumerate(output) if r["stage"] == "learning_curve_fit"
        ]
        require(
            "final_fit" in roles
            and "freeze" in roles
            and all(
                roles.index("final_fit") < i < roles.index("freeze") for i in indices
            ),
            "Training and all fits must precede freezing and test-label generation",
        )
        output[indices[-1]] = {
            **output[indices[-1]],
            "stage": "learning_curve_fits",
            "pipelines": individual,
        }

    freezes = [row for row in output if row["stage"] == "freeze"]
    if len(freezes) == 1 and "frozen" in freezes[0]:
        hashes = freezes[0]["frozen"]
        require(isinstance(hashes, dict), "Frozen pipeline digests must be an object")
        by_name = {}
        for filename, digest in hashes.items():
            name = Path(filename).name
            require(name not in by_name, "Ambiguous frozen pipeline filenames")
            by_name[name] = digest
        freezes[0]["frozen"] = by_name
        for row in output:
            fits = [row] if row["stage"] == "final_fit" else row.get("pipelines", [])
            for fit in fits:
                fit.update(
                    rename(
                        fit,
                        {
                            "file": ("path",),
                            "indices": ("train_indices",),
                            "n_train": ("n",),
                        },
                    )
                )
                if (
                    "file" in fit
                    and "sha256" not in fit
                    and Path(fit["file"]).name in by_name
                ):
                    fit["sha256"] = by_name[Path(fit["file"]).name]

    final = next((row for row in output if row["stage"] == "final_fit"), {})
    if final and "params" not in final and "candidate_index" not in final:
        final["params"] = parameters(final)
    shared = parameters(final.get("params", {}))
    for selection in selections:
        for candidate in selection.get("candidates", []):
            if (
                "scale" not in candidate
                and "preprocessing" not in candidate
                and "scale" in shared
            ):
                candidate["scale"] = shared["scale"]
    for row in output:
        if row["stage"] == "learning_curve_fits" and "scale" in shared:
            for fit in row["pipelines"]:
                fit.setdefault("scale", shared["scale"])

    def locate(record, labels):
        if "file" in record or labels is None or log_path is None:
            return
        expected = finite_array(labels, shape=(100,))
        paths = list(log_path.parent.glob("*.extxyz"))
        if len(paths) > 100:
            raise UnsupportedEvidence(
                "Too many unlinked datasets; retain explicit dataset links"
            )
        matches = []
        for path in paths:
            frames = read(linked(str(path)), index=":", format="extxyz")
            values = [stored(frame).get("energy") for frame in frames]
            if (
                len(values) == 100
                and all(v is not None for v in values)
                and np.allclose(values, expected, rtol=0, atol=1e-8)
            ):
                matches.append(str(path))
        if len(matches) != 1:
            raise UnsupportedEvidence(
                "Cannot uniquely link the recorded energy labels to a saved dataset"
            )
        record["file"] = matches[0]

    for row in output:
        if row["stage"] == "train_labels":
            locate(row, row.get("energies"))
        elif row["stage"] == "test_labels" and "independent_seed" in row:
            for name, energy_key in (
                ("independent", "E_indep"),
                ("strained", "E_strained"),
            ):
                record = row.setdefault(name, {"seed": row[f"{name}_seed"]})
                require(
                    record["seed"] == row[f"{name}_seed"],
                    "Conflicting test dataset seed",
                )
                locate(record, row.get(energy_key))
    return output


def selection_metric(record, candidates):
    """Respect the declared objective; infer an unambiguous logged metric."""
    declarations = []
    for source in (record, record.get("cv", {})):
        if isinstance(source, dict):
            declarations.extend(
                source[k]
                for k in ("selection_metric", "metric", "scoring")
                if k in source
            )
    found = set()
    for declaration in declarations:
        text = str(declaration).lower().replace("²", "2")
        names = {
            "mae": r"\bmae\b|mean_absolute_error",
            "rmse": r"\brmse\b|root_mean_squared_error",
            "r2": r"\br2\b|r_squared",
        }
        matches = {name for name, pattern in names.items() if re.search(pattern, text)}
        if len(matches) != 1:
            raise UnsupportedEvidence(
                f"Unsupported model selection metric: {declaration}"
            )
        found.update(matches)
    require(len(found) <= 1, "Conflicting model selection metrics")
    if found:
        return found.pop()
    common = {
        key
        for key in ("mae", "rmse", "r2")
        if all(f"cv_{key}" in row for row in candidates)
    }
    if len(common) == 1:
        return common.pop()
    # Historical records containing several metrics used RMSE for selection.
    if "rmse" in common:
        return "rmse"
    raise UnsupportedEvidence(
        "Cannot identify the training-validation selection metric"
    )
