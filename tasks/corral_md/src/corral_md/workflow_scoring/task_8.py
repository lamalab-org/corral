"""Artifact-only SOAP/Ridge checks; no training, descriptor calls, or model loading."""

from __future__ import annotations

import ast
import json
import pickletools
import re
from functools import lru_cache

import numpy as np
from ase.build import bulk

from .common import (
    EvidenceError,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
    scientific_screen,
)
from .regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
    transformed_features,
)

_NAMES = ("train", "id_test", "strained_test")
_STRUCTURE_ALIASES = {
    "train": ("train_structures", "training_structures"),
    "id_test": ("id_test_structures", "in_distribution_structures"),
    "strained_test": ("strained_test_structures", "strained_structures"),
}
_STRAINS = np.array([-0.02, -0.01, 0, 0.01, 0.02])


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _ids(values, count=None):
    _require(isinstance(values, list), "Row IDs must be a list")
    _require(
        all(isinstance(v, str) and v for v in values),
        "Row IDs must be nonempty strings",
    )
    _require(len(set(values)) == len(values), "Row IDs must be unique")
    if count is not None:
        _require(len(values) == count, f"Expected {count} row IDs")
    return values


def _metric_matches(claimed, actual):
    return all(
        result_close(claimed[key], actual[key], atol=1e-07)
        for key in ("mae", "rmse", "r2")
    )


def _preprocessing(model, features):
    return preprocessing_matches(features, {"preprocessing": "identity", **model})


def _supported(model):
    return model.get("preprocessing", "identity") in (
        "identity",
        "StandardScaler",
        "standard_scaler",
        "MinMaxScaler",
        "PCA",
    )


def _model_metadata(model, features, row_ids):
    _require(model["family"].lower() == "ridge", "Estimator family must be Ridge")
    _require(
        model["descriptor"]["family"].lower() == "soap",
        "Descriptor family must be SOAP",
    )
    _require(
        isinstance(model["descriptor"]["parameters"], dict)
        and bool(model["descriptor"]["parameters"]),
        "Record SOAP settings",
    )
    _require(
        set(_ids(model["training_row_ids"])) == set(row_ids),
        "Model fitting row IDs do not match its training partition",
    )
    coef = finite_array(model["coef"], ndim=1)
    _require(len(coef) > 0, "Model coefficients cannot be empty")
    if _supported(model):
        transformed = transformed_features(features, model)
        _require(
            len(coef) == transformed.shape[1],
            "Coefficient dimension differs from transformed features",
        )
    finite_array(model["intercept"], shape=())
    alpha = finite_array(model["alpha"], shape=()).item()
    _require(alpha >= 0, "Ridge alpha must be nonnegative")
    _require(
        type(model.get("fit_intercept", True)) is bool, "fit_intercept must be boolean"
    )
    for flag in ("with_mean", "with_std", "positive"):
        _require(type(model.get(flag, False)) is bool, f"{flag} must be boolean")
    return True


def _checkpoint(path, model):
    """Inspect data/serialization structure without constructing model objects."""
    if path.stat().st_size > 32 * 1024 * 1024:
        return None, "Checkpoint exceeds the static inspection limit"
    payload = path.read_bytes()
    try:
        saved = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        saved = None
    if saved is not None:
        _require(
            isinstance(saved, dict), "Pipeline checkpoint must contain fitted state"
        )
        for key in (
            "family",
            "descriptor",
            "preprocessing",
            "fit_intercept",
            "with_mean",
            "with_std",
            "positive",
            "whiten",
            "clip",
            "feature_range",
            "svd_solver",
        ):
            _require(
                saved.get(key) == model.get(key),
                "Checkpoint configuration differs from portable model",
            )
        for key in (
            "coef",
            "intercept",
            "alpha",
            "mean",
            "scale",
            "offset",
            "data_min",
            "data_max",
            "components",
            "explained_variance",
        ):
            if key in model or key in saved:
                _require(
                    close(saved[key], model[key]),
                    "Checkpoint fitted state differs from portable model",
                )
        return True
    # Disassembly never invokes GLOBAL/REDUCE or imports serialized classes.
    # Joblib's custom binary array blocks and compressed formats need a separate
    # adapter; do not pretend a magic header proves a usable estimator.
    try:
        strings, opcodes, stop = set(), set(), None
        for opcode, value, position in pickletools.genops(payload):
            opcodes.add(opcode.name)
            if isinstance(value, str):
                strings.update(value.replace("\n", " ").split())
            if opcode.name == "STOP":
                stop = position
    except ValueError:
        if payload.startswith((b"\x80", b"\x78", b"BZh", b"\x1f\x8b", b"\xfd7zXZ")):
            return (
                None,
                "Unsupported or incomplete serialized checkpoint; loadability unverified",
            )
        return False, "Checkpoint is not fitted JSON state or a supported pickle"
    _require(stop == len(payload) - 1, "Incomplete pickle or trailing checkpoint data")
    _require(
        "BUILD" in opcodes and bool(opcodes & {"GLOBAL", "STACK_GLOBAL"}),
        "Pickle contains no estimator object state",
    )
    _require(
        {"Ridge", "coef_", "intercept_"} <= strings,
        "Checkpoint lacks fitted Ridge state",
    )
    _require(
        any(s.lower() == "soap" for s in strings), "Checkpoint lacks SOAP configuration"
    )
    if model.get("preprocessing") == "StandardScaler":
        _require(
            {"StandardScaler", "mean_", "scale_"} <= strings,
            "Checkpoint lacks fitted preprocessing",
        )
    if model.get("preprocessing") in {"PCA", "MinMaxScaler"}:
        state = {
            "PCA": {"PCA", "components_", "mean_", "explained_variance_"},
            "MinMaxScaler": {
                "MinMaxScaler",
                "scale_",
                "min_",
                "data_min_",
                "data_max_",
            },
        }
        _require(
            state[model["preprocessing"]] <= strings,
            "Checkpoint lacks fitted preprocessing",
        )
    return True


def _inference_source(path):
    """Reject malformed and obvious stub entry points without running them."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    compile(
        tree, str(path), "exec"
    )  # Includes contextual errors (e.g. return outside a function).

    def implementation(statements):
        def stub_raise(node):
            if not isinstance(node, ast.Raise):
                return False
            # A normal CLI may exit with its main function's return value.
            exc = node.exc
            return not (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "SystemExit"
                and len(exc.args) == 1
                and isinstance(exc.args[0], ast.Call)
            )

        meaningful = [
            s
            for s in statements
            if not isinstance(s, ast.Pass)
            and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        return bool(meaningful) and not any(stub_raise(s) for s in meaningful)

    _require(
        implementation(tree.body),
        "Inference entry point is empty or raises immediately",
    )
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    main = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
    ]
    for guard in main:
        _require(
            implementation(guard.body), "Inference main entry point is a placeholder"
        )
    bodies = [node.body for node in functions + main] or [tree.body]
    return any(
        implementation(body)
        and any(
            isinstance(node, ast.Call | ast.MatMult)
            for statement in body
            if not isinstance(statement, ast.Raise)
            for node in ast.walk(statement)
        )
        for body in bodies
    )


def evaluate(e, r):
    """Check exported evidence and award ninety independently diagnosed points."""

    @lru_cache(None)
    def bundle():
        data = e.json("regression_data", "datasets", "dataset_index")
        _require(isinstance(data, dict), "Dataset index must be an object")
        all_ids, dataset_ids = [], []
        for name in _NAMES:
            all_ids += _ids(data[name]["row_ids"], 100)
            dataset_ids.append(data[name]["dataset_id"])
        _ids(all_ids, 300)
        _ids(dataset_ids, 3)
        return data

    @lru_cache(None)
    def dataset(name):
        data = bundle()[name]
        x = finite_array(data["features"], ndim=2)
        _require(x.shape[0] == 100, "Feature rows must match the dataset")
        y = finite_array(data["energy_eV"], shape=(100,))
        sigma = finite_array(data["sigma_A"], shape=(100,))
        strain = finite_array(data["strain"], shape=(100,))
        frames = e.trajectory(*_STRUCTURE_ALIASES[name])
        _require(len(frames) == 100, "Expected 100 structures in each dataset")
        frame_ids = _ids([a.info["row_id"] for a in frames], 100)
        _require(
            set(frame_ids) == set(data["row_ids"]), "Structure and array row IDs differ"
        )
        lookup = dict(zip(frame_ids, frames, strict=True))
        ordered = [lookup[key] for key in data["row_ids"]]
        for atoms, value in zip(ordered, y, strict=True):
            stored = atoms.info.get(
                "energy",
                getattr(getattr(atoms, "calc", None), "results", {}).get("energy"),
            )
            _require(
                stored is not None and close(stored, value),
                "Stored structure energy does not match its row label",
            )
        if name != "strained_test":
            indices = finite_array(data["generation_index"], shape=(100,))
            _require(
                np.array_equal(np.sort(indices), np.arange(100)),
                "Generation indices must be a permutation of 0..99",
            )
            _require(
                close(sigma, 0.01 + 0.001 * indices) and close(strain, np.zeros(100)),
                "Incorrect distortion schedule or zero-strain dataset",
            )
        else:
            _require(
                close(sigma, np.full(100, 0.05)),
                "Strained displacement sigma must be 0.05 A",
            )
            _require(
                all(
                    np.count_nonzero(np.isclose(strain, value, atol=1e-9, rtol=0)) == 20
                    for value in _STRAINS
                ),
                "Expected twenty structures at each required strain",
            )
        return data, x, y, sigma, strain, ordered

    @lru_cache(None)
    def geometry(name):
        _, _, _, sigma, strain, frames = dataset(name)
        ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        displacements = []
        for atoms, eps, sig in zip(frames, strain, sigma, strict=True):
            _require(
                len(atoms) == 64 and set(atoms.numbers) == {14} and np.all(atoms.pbc),
                "Expected fully periodic Si64",
            )
            cell = ideal.cell.array * (1 + eps)
            _require(
                close(atoms.cell.array, cell, rtol=0, atol=1e-5),
                "Incorrect strained cell",
            )
            positions = finite_array(atoms.positions, shape=(64, 3))
            delta = positions[:, None, :] - ideal.positions[None, :, :] * (1 + eps)
            length = cell[0, 0]
            delta -= np.rint(delta / length) * length
            mapping = np.argmin(np.linalg.norm(delta, axis=2), axis=1)
            _require(
                len(set(mapping)) == 64,
                "Distortion does not preserve distinct ideal diamond sites",
            )
            displacements.append(delta[np.arange(64), mapping] / sig)
        return np.asarray(displacements)

    def gaussian_compatibility():
        compatible = []
        for name in _NAMES:
            z = geometry(name)
            compatible.append(np.all(np.abs(z.mean(axis=(0, 1))) <= 0.08))
            compatible.append(0.85 <= np.sqrt(np.mean(z**2)) <= 1.15)
            compatible.append(0.02 <= np.mean(np.abs(z) > 2) <= 0.08)
            if name != "strained_test":
                groups = np.asarray(dataset(name)[0]["generation_index"]) // 25
            else:
                groups = np.argmin(abs(dataset(name)[4][:, None] - _STRAINS), axis=1)
            compatible.extend(
                0.75 <= np.sqrt(np.mean(z[groups == group] ** 2)) <= 1.25
                for group in np.unique(groups)
            )
        # Different labels alone must not disguise duplicated saved structures.
        fingerprints = []
        for name in _NAMES:
            for atoms in dataset(name)[5]:
                wrapped = np.mod(atoms.positions / atoms.cell.lengths(), 1)
                order = np.lexsort(wrapped.T[::-1])
                fingerprints.append(
                    (
                        tuple(np.round(atoms.cell.array.ravel(), 7)),
                        tuple(np.round(wrapped[order].ravel(), 7)),
                    )
                )
        if len(set(fingerprints)) != 300:
            return False
        return scientific_screen(
            all(compatible),
            "Saved displacement-distribution diagnostics are inconclusive",
        )

    def recorded_inputs():
        for name in _NAMES:
            dataset(name)
        s = e.settings
        _require(
            str(s["teacher_model"]).split("/")[-1] == "teacher.model",
            "Wrong recorded teacher name",
        )
        if "teacher_sha256" in s:
            _require(
                isinstance(s["teacher_sha256"], str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", s["teacher_sha256"]) is not None,
                "Invalid optional teacher SHA256 identity",
            )
        _require(
            s["energy_unit"] == "eV" and s["length_unit"] in ("Angstrom", "A", "Å"),
            "Wrong units",
        )
        seeds = []
        for name in _NAMES:
            config = s["generation"][name]
            _require(type(config["seed"]) is int, "RNG seeds must be integers")
            _require(
                all(
                    isinstance(config[key], str) and config[key].strip()
                    for key in ("library", "method")
                ),
                "Record RNG library and method",
            )
            seeds.append(config["seed"])
        return len(set(seeds)) == 3

    r.check(
        "training_and_id_geometry",
        6,
        lambda: all(
            geometry(name).shape == (100, 64, 3) for name in ("train", "id_test")
        ),
    )
    r.check(
        "strained_geometry", 6, lambda: geometry("strained_test").shape == (100, 64, 3)
    )
    r.check("distortion_distributions_and_independence", 4, gaussian_compatibility)
    r.check("aligned_labels_recorded_inputs_and_rng", 4, recorded_inputs)

    @lru_cache(None)
    def model():
        value = e.json("portable_model", "final_model", "pipeline_parameters")
        _model_metadata(value, dataset("train")[1], dataset("train")[0]["row_ids"])
        return value

    @lru_cache(None)
    def validation():
        doc = e.json("validation", "tuning_records", "cross_validation")
        method = doc.get("method", "recorded_splits")
        _require(
            isinstance(method, str) and bool(method.strip()),
            "Record the validation method",
        )
        if method != "recorded_splits":
            _require(
                doc["dataset"] in {"train", "training"},
                "Validation may use only the training dataset",
            )
            raw = finite_array(doc["selection_indices"], ndim=1)
            indices = raw.astype(int)
            _require(
                np.array_equal(raw, indices)
                and np.all((indices >= 0) & (indices < 100))
                and len(np.unique(indices)) == len(indices),
                "Validation selections must identify distinct training rows",
            )
            _require(
                isinstance(doc["calculations"], dict) and bool(doc["calculations"]),
                "Preserve numerical calculations for the chosen validation strategy",
            )
            return doc, []
        _require(
            isinstance(doc["records"], list) and doc["records"],
            "Training-only validation records are required",
        )
        rows = dataset("train")[0]["row_ids"]
        records = []
        for item in doc["records"]:
            train, test = partition(item["train_indices"], item["test_indices"], 100)
            x = finite_array(item.get("features", dataset("train")[1]), ndim=2)
            _require(
                x.shape[0] == 100,
                "Candidate features must align with all training row IDs",
            )
            _model_metadata(item["model"], x, [rows[i] for i in train])
            records.append((item, train, test, x))
        return doc, records

    def review_validation(doc):
        if doc.get("method", "recorded_splits") != "recorded_splits":
            raise UnsupportedEvidence(
                "The documented training-only validation strategy requires independent review"
            )

    def model_package():
        value = model()
        dimension = dataset("train")[1].shape[1]
        for name in _NAMES:
            x = dataset(name)[1]
            _require(
                x.shape[1] == dimension,
                "Final pipeline feature dimensions differ between datasets",
            )
        _require(
            _inference_source(e.artifact("inference_entrypoint", "inference_script")),
            "Inference entry point has no prediction implementation",
        )
        return _checkpoint(
            e.artifact("pipeline_checkpoint", "trained_pipeline", "estimator"), value
        )

    def partitions_and_budget():
        validation()
        return set(_ids(e.settings["training_label_ids"], 100)) == set(
            dataset("train")[0]["row_ids"]
        ) and bool(model())

    def preprocessing():
        cases = [(model(), dataset("train")[1])]
        doc, records = validation()
        cases += [(item["model"], x[train]) for item, train, _, x in records]
        outcomes = [_preprocessing(m, x) for m, x in cases]
        if any(value is False for value in outcomes):
            return False
        review_validation(doc)
        return None if any(value is None for value in outcomes) else True

    def fitted_coefficients():
        y = dataset("train")[2]
        doc, records = validation()
        cases = [(model(), dataset("train")[1], y)]
        cases += [(item["model"], x[train], y[train]) for item, train, _, x in records]
        if any(not _supported(m) for m, _, _ in cases):
            return None
        if not all(ridge_stationarity(x, labels, m) for m, x, labels in cases):
            return False
        review_validation(doc)
        return True

    def validation_results():
        doc, records = validation()
        _require(
            isinstance(doc["selection_criterion"], str)
            and doc["selection_criterion"].strip(),
            "Record the model selection criterion",
        )
        candidates = doc["candidates"]
        _require(
            isinstance(candidates, dict) and candidates, "Record candidate settings"
        )
        review_validation(doc)
        selected = []
        for item, _, test, x in records:
            _require(item["candidate_id"] in candidates, "Missing candidate settings")
            if not _supported(item["model"]):
                return None
            p = predict(x[test], item["model"])
            _require(
                result_close(item["predictions_eV"], p, atol=1e-07),
                "CV predictions do not follow saved coefficients",
            )
            _require(
                _metric_matches(item["metrics"], metrics(dataset("train")[2][test], p)),
                "Incorrect CV metrics",
            )
            if item["candidate_id"] == doc.get("selected_candidate_id"):
                selected.append(item["model"])
        if "final_selection" in doc:
            selection = doc["final_selection"]
            _require(
                isinstance(selection["method"], str) and selection["method"].strip(),
                "Record the final selection method",
            )
            raw = finite_array(selection["selection_indices"], ndim=1)
            indices = raw.astype(int)
            _require(
                np.array_equal(raw, indices)
                and np.all((indices >= 0) & (indices < 100))
                and len(np.unique(indices)) == len(indices),
                "Final selection indices must identify distinct training rows",
            )
            dataset_name = selection["dataset"]
            _require(
                isinstance(dataset_name, str)
                and dataset_name.strip().lower() in {"train", "training"},
                "Final selection may use only the training dataset",
            )
            _require(
                isinstance(selection["calculations"], dict)
                and bool(selection["calculations"]),
                "Preserve calculations supporting the final selection",
            )
            raise UnsupportedEvidence(
                "The documented training-only final selection requires independent review"
            )
        _require(selected, "Selected candidate has no validation evidence")
        final = model()
        return all(
            close(m["alpha"], final["alpha"])
            and m["descriptor"] == final["descriptor"]
            and m.get("preprocessing", "identity")
            == final.get("preprocessing", "identity")
            for m in selected
        )

    r.check(
        "portable_pipeline_and_dimensions",
        5,
        model_package,
        "Inspect checkpoint state and inference syntax without loading or executing them",
    )
    r.check("training_partitions_and_label_budget", 5, partitions_and_budget)
    r.check(
        "training_only_preprocessing",
        5,
        preprocessing,
        "Check each exported scaler against its own fitting rows",
    )
    r.check(
        "ridge_coefficient_consistency",
        6,
        fitted_coefficients,
        "Check normal-equation residuals without fitting a model",
    )
    r.check("validation_predictions_metrics_and_selection", 4, validation_results)

    @lru_cache(None)
    def test_prediction(name):
        geometry(name)
        value = model()
        if not _supported(value):
            raise UnsupportedEvidence(
                "Preprocessing requires independent prediction verification"
            )
        p = predict(dataset(name)[1], value)
        _require(
            result_close(p, dataset(name)[0]["predictions_eV"], atol=1e-07),
            "Saved predictions differ from exported coefficients",
        )
        return p

    def test_metrics(name, result):
        if not _supported(model()):
            return None
        return _metric_matches(result, metrics(dataset(name)[2], test_prediction(name)))

    def strain_metrics():
        if not _supported(model()):
            return None
        predictions = test_prediction("strained_test")
        labels, strain = dataset("strained_test")[2], dataset("strained_test")[4]
        claims = e.results["strain_groups"]
        _require(
            isinstance(claims, list) and len(claims) == 5,
            "Report all five strain groups",
        )
        for value in _STRAINS:
            matches = [
                row for row in claims if close(row["strain"], value, rtol=0, atol=1e-9)
            ]
            _require(len(matches) == 1, "Missing or duplicate strain-group metric")
            group = np.isclose(strain, value, rtol=0, atol=1e-9)
            _require(
                _metric_matches(matches[0], metrics(labels[group], predictions[group])),
                "Incorrect strain-group metrics",
            )
        return True

    r.check(
        "in_distribution_metrics",
        8,
        lambda: test_metrics("id_test", e.results["id_test"]),
    )
    r.check("strain_resolved_metrics", 10, strain_metrics)
    r.check(
        "pooled_strained_metrics",
        7,
        lambda: test_metrics("strained_test", e.results["strained_pooled"]),
    )

    @lru_cache(None)
    def curves():
        doc = e.json("learning_curve", "learning_curve_records")
        _require(
            isinstance(doc["records"], list) and len(doc["records"]) >= 2,
            "Multiple curve sizes are required",
        )
        rows = dataset("train")[0]["row_ids"]
        output = []
        for item in doc["records"]:
            raw = finite_array(item["train_indices"], ndim=1)
            indices = raw.astype(int)
            _require(
                np.array_equal(raw, indices)
                and np.all((indices >= 0) & (indices < 100))
                and len(set(indices)) == len(indices),
                "Curve training indices must be distinct rows of training set",
            )
            train_x = finite_array(
                item.get("train_features", dataset("train")[1]), ndim=2
            )
            test_x = finite_array(
                item.get("id_test_features", dataset("id_test")[1]), ndim=2
            )
            _require(
                train_x.shape[0] == 100 and test_x.shape == (100, train_x.shape[1]),
                "Curve features must align to train and ID rows",
            )
            _model_metadata(item["model"], train_x, [rows[i] for i in indices])
            output.append((item, indices, train_x, test_x))
        return output

    def curve_sizes():
        records = curves()
        _require(
            len({len(indices) for _, indices, _, _ in records}) >= 2,
            "Learning curve must change training sample size",
        )
        return True

    def curve_design():
        records = curves()
        generation = np.asarray(dataset("train")[0]["generation_index"])
        schedule = np.sort(generation)
        full_cdf = np.arange(1, len(schedule) + 1) / len(schedule)
        diagnostics = []
        for _, indices, _, _ in records:
            selected = np.sort(generation[indices])
            _require(
                len(indices) > 0,
                "Each learning-curve training subset must be nonempty",
            )
            # Compare the whole empirical CDF, not just endpoints or bins. The
            # DKW reference envelope is deliberately conservative: deterministic
            # designs need not reproduce a random sampler, and the finite
            # population correction is omitted. This only determines what the
            # adapter can certify automatically; exceeding it is not a failure.
            subset_cdf = np.searchsorted(selected, schedule, side="right") / len(
                selected
            )
            distance = float(np.max(np.abs(subset_cdf - full_cdf)))
            review_bound = float(np.sqrt(np.log(2 / 0.01) / (2 * len(selected))))
            diagnostics.append(
                f"n={len(selected)}: CDF distance={distance:.6g}, "
                f"review bound={review_bound:.6g}"
            )
            if distance > review_bound:
                raise UnsupportedEvidence(
                    "Subset distribution needs independent coverage review "
                    f"({diagnostics[-1]}); inspect saved selections and sampling "
                    "metadata for meaningful size comparisons. This is a "
                    "conservative adapter screen, not a required sampling method."
                )
        return True, "; ".join(diagnostics)

    def curve_models():
        outcomes = []
        for item, indices, x, _ in curves():
            value = item["model"]
            if not _supported(value):
                outcomes.append(None)
                continue
            outcomes.append(
                _preprocessing(value, x[indices])
                and ridge_stationarity(x[indices], dataset("train")[2][indices], value)
            )
        return (
            False
            if any(value is False for value in outcomes)
            else None
            if any(value is None for value in outcomes)
            else True
        )

    def curve_predictions():
        geometry("id_test")
        for item, _, _, x in curves():
            if not _supported(item["model"]):
                return None
            p = predict(x, item["model"])
            _require(
                result_close(item["predictions_eV"], p, atol=1e-07),
                "Learning-curve predictions do not match their saved model",
            )
            _require(
                _metric_matches(item["metrics"], metrics(dataset("id_test")[2], p)),
                "Incorrect learning-curve metrics",
            )
        return True

    def curve_reported_results():
        claims = e.results["learning_curve"]
        records = curves()
        _require(
            len(claims) == len(records),
            "Reported curve must include every saved record",
        )
        for claim, (item, indices, _, x) in zip(claims, records, strict=True):
            if not _supported(item["model"]):
                return None
            _require(
                claim["size"] == len(indices), "Incorrect reported curve sample size"
            )
            _require(
                _metric_matches(
                    claim, metrics(dataset("id_test")[2], predict(x, item["model"]))
                ),
                "Reported curve differs from numerical evidence",
            )
        return True

    r.check("learning_curve_sizes", 2, curve_sizes)
    r.check("learning_curve_distortion_coverage", 3, curve_design)
    r.check("learning_curve_training_only_models", 7, curve_models)
    r.check("learning_curve_predictions_and_metrics", 6, curve_predictions)
    r.check("reported_learning_curve", 2, curve_reported_results)
    r.unverified(
        "learning_curve_sampling_adequacy",
        "Saved row selections establish reproducibility. A conservative empirical-distribution screen routes ambiguous coverage to review without requiring endpoints or quartile counts; scientific sampling adequacy is not established by this screen alone.",
    )
    r.unverified(
        "pipeline_runtime",
        "Static checkpoint and source checks cannot prove loadability or successful structure-to-energy inference.",
    )
    r.unverified(
        "execution_and_teacher_provenance",
        "Recorded labels, RNG seeds, teacher SHA256 and SOAP features cannot prove actual teacher/RNG/descriptor execution or absence of relaxation and undisclosed extra labels.",
    )
    r.unverified(
        "model_choice_freezing",
        "Saved partitions and fitted state support training-only numerical checks; they do not prove choices were frozen before viewing test labels.",
    )
