"""Score silicon SOAP-Ridge calculations from one final recording.

Reconstruct every metric from portable fitted state; verify all teacher labels
and SOAP features independently. Saved pipelines are read as numerical data;
submitted code is never executed.
Recorded chronology checks the provided workflow, not unrecorded test access.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read

from .common import (
    EvaluationError,
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    finite_array,
    read_json,
)
from .formats import (
    artifact_reader,
    check_model_digest,
    check_units,
    dataset_records,
    event_record,
    records,
)
from .level1_trusted import PINNED_TEACHER_SHA256
from .pipeline_data import read_pipeline
from .pipeline_records import normalize as normalize_pipeline_records
from .pipeline_records import parameters as _pipeline_parameters
from .pipeline_records import selection_metric
from .pipeline_records import stage as _pipeline_stage
from .regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
)
from .verification import Plan, compare, file_ref, geometry, stored


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _dataset_generator(method):
    """Resolve explicit generator labels without evaluating submitted text."""
    generators = {
        "numpy.default_rng": np.random.default_rng,
        "numpy.default_rng(PCG64)": np.random.default_rng,
        "PCG64": np.random.default_rng,
        "numpy.random.RandomState": np.random.RandomState,
        "numpy.RandomState": np.random.RandomState,
        "RandomState": np.random.RandomState,
        "numpy.default_rng(seed).normal, one (64,3) draw per config in order": np.random.default_rng,
    }
    return generators.get(method) if isinstance(method, str) else None


def _matches_generated_dataset(
    name, frames, seed, method=None, *, restore_precision=False
):
    """Check the same seed, cell and displacement schedule for either log layout."""
    _require(type(seed) is int and seed >= 0, "Invalid generation seed")
    _require(len(frames) == 100, "Each dataset needs exactly 100 structures")
    constructor = _dataset_generator(method)
    # Full coordinates let us identify a supported generator by numerical replay,
    # without requiring a particular spelling of an optional prose description.
    # An explicit supported label still restricts the check to that generator.
    choices = (
        [constructor(seed)]
        if constructor is not None
        else [np.random.default_rng(seed), np.random.RandomState(seed)]
    )
    matching_generators = [True] * len(choices)
    replay = [[] for _ in choices]
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    for i, frame in enumerate(frames):
        strain = (-0.02 + 0.01 * (i // 20)) if name == "strained_test" else 0
        sigma = 0.05 if name == "strained_test" else 0.01 + 0.001 * i
        cell = ideal.cell.array * (1 + strain)
        _require(
            np.allclose(frame.cell.array, cell, rtol=0, atol=1e-7),
            "Incorrect silicon cell/strain schedule",
        )
        for index, rng in enumerate(choices):
            expected = ideal.positions * (1 + strain) + rng.normal(0, sigma, (64, 3))
            replay[index].append((cell, expected))
            delta = (frame.positions - expected) @ np.linalg.inv(cell)
            matching_generators[index] &= np.allclose(
                delta - np.rint(delta), 0, rtol=0, atol=1e-8
            )
    matched = any(matching_generators)
    if not matched and constructor is None and method is not None:
        raise UnsupportedEvidence(
            "Recorded coordinates do not reproduce the seed and displacement "
            "schedule with a supported generator; the recorded generator "
            "description is not recognized"
        )
    if matched and restore_precision:
        # Text trajectories commonly round coordinates to eight decimal places.
        # SOAP features near zero can amplify that rounding during standardization.
        # Replay only a generator already matched to every saved coordinate/cell;
        # retained energy labels and the bytes used for the fingerprint stay intact.
        chosen = matching_generators.index(True)
        for frame, (cell, positions) in zip(frames, replay[chosen], strict=True):
            frame.set_cell(cell)
            frame.positions = positions
    return matched


def _seeded_level1_records(raw):
    """Normalize level-1 rows, reconstructing coordinates only when absent.

    Level 1 asks for seeds, row identities, checkpoint digest and total energies,
    not serialized coordinates. A declared generator makes its unrelaxed task
    geometries reproducible, and retained coordinate hashes are checked too.
    Complete coordinates are preserved and independently replayed by read_log.
    """
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    header, *rows = [event_record(row) for row in raw]
    if header.get("event") != "start" or any("rows" in row for row in rows):
        return None
    rows = [row for row in rows if row.get("event") != "end"]
    if not rows or not all(row.get("event") in {"config", "sample"} for row in rows):
        return None
    _require(
        len(rows) == header.get("n_configs", 100) == 100,
        "Retain all 100 generated energies",
    )
    _require("model_sha256" in header, "Retain the checkpoint digest")
    seed = header.get("random_seed")
    _require(type(seed) is int and seed >= 0, "Retain a reproducible dataset seed")
    method = header.get("rng")
    constructor = _dataset_generator(method)
    retained_positions = all("positions_A" in row for row in rows)
    if not retained_positions and constructor is None:
        raise UnsupportedEvidence(
            "Coordinate-free dataset needs an identified reproducible generator"
        )
    for key, expected in (
        ("lattice_a_A", 5.43),
        ("supercell", [2, 2, 2]),
        ("n_atoms", 64),
    ):
        _require(
            key not in header or header[key] == expected,
            f"Incorrect silicon dataset {key}",
        )
    if "pbc" in header:
        _require(
            header["pbc"] is True or np.array_equal(header["pbc"], [True] * 3),
            "Incorrect silicon dataset pbc",
        )
    rng = None if retained_positions else constructor(seed)
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    expanded = []
    for i, row in enumerate(rows):
        _require(row.get("config_index", i) == i, "Retain dataset generation order")
        _require(
            row.get("random_seed", seed) == seed,
            "Dataset rows disagree on generation seed",
        )
        sigma = 0.01 + 0.001 * i
        _require(
            np.isclose(row.get("std_A", sigma), sigma),
            "Incorrect Gaussian displacement schedule",
        )
        if retained_positions:
            positions = finite_array(row["positions_A"], shape=(64, 3))
        else:
            positions = ideal.positions + rng.normal(0, sigma, (64, 3))
            if "positions_A" in row:
                _require(
                    np.allclose(positions, row["positions_A"], rtol=0, atol=1e-8),
                    "Recorded positions disagree with their generation seed",
                )
        if "positions_sha256" in row:
            _require(
                hashlib.sha256(positions.tobytes()).hexdigest()
                == row["positions_sha256"],
                "Recorded coordinate hash disagrees with its dataset coordinates",
            )
        cell = finite_array(row.get("cell_A", ideal.cell.array), shape=(3, 3))
        _require(
            np.allclose(ideal.cell.array, cell, rtol=0, atol=1e-8),
            "Incorrect silicon dataset cell",
        )
        expanded.append(
            {
                **row,
                "positions": positions.tolist(),
                "cell": cell.tolist(),
                "energy_eV": row["potential_energy_eV"],
            }
        )
    return [
        header,
        {
            "event": "dataset",
            "name": "train",
            "seed": seed,
            "rng": method,
            "rows": expanded,
        },
        {"event": "end"},
    ]


@artifact_reader
def read_log(evidence, level):
    path = evidence.artifact("run_log")
    log = _seeded_level1_records(records(path)) if level == 1 else None
    log = dataset_records(path, level) if log is None else log
    _require(len(log) >= 3, "Truncated calculation log")
    header = log[0]
    _require(header["event"] == "start", "Missing dataset metadata")
    check_model_digest(header, PINNED_TEACHER_SHA256)
    # Precision is not a required artifact field. Use float64 for the independent
    # reference calculation when no execution precision was recorded.
    header.setdefault("default_dtype", "float64")
    _require(
        header["default_dtype"] in {"float32", "float64"}, "Invalid teacher precision"
    )
    check_units(header, {"energy": "eV", "length": "Angstrom"})
    events = [row["event"] for row in log]
    if level == 1:
        _require(
            events == ["start", "dataset", "end"],
            "Level 1 needs only the completed training dataset",
        )
        candidates, frozen, tests = [], None, []
    else:
        _require(
            events.count("freeze") == 1,
            "Freeze every pipeline before generating test labels",
        )
        index = events.index("freeze")
        _require(
            index >= 3
            and events[:2] == ["start", "dataset"]
            and events[2:index] == ["candidate"] * (index - 2)
            and events[index:]
            == [
                "freeze",
                "dataset",
                "test_features",
                "dataset",
                "test_features",
                "end",
            ],
            "Incomplete calculations or test data accessed before freezing",
        )
        candidates, frozen = log[2:index], log[index]
        tests = log[index + 1 : index + 5]
    datasets = {}
    for name, record in zip(
        ("train", "id_test", "strained_test"), [log[1], *tests[::2]], strict=False
    ):
        _require(record["name"] == name, "Incorrect dataset order")
        seed = record["seed"]
        method = record.get("rng")
        identities = [
            row.get("row_id", row.get("index", index))
            for index, row in enumerate(record["rows"])
        ]
        _require(
            all(type(identity) in (str, int) for identity in identities)
            and len(set(identities)) == 100,
            "Retain unique row identities in generation order",
        )
        frames, labels = [], []
        for row in record["rows"]:
            positions = finite_array(row["positions"], shape=(64, 3))
            cell = finite_array(row["cell"], shape=(3, 3))
            frames.append(Atoms("Si64", positions=positions, cell=cell, pbc=True))
            labels.append(finite_array(row["energy_eV"], shape=()).item())
        if not _matches_generated_dataset(name, frames, seed, method):
            if method is None:
                raise UnsupportedEvidence(
                    "Cannot reproduce dataset displacements with the recorded seed; the random generator was not identified"
                )
            raise EvidenceError(
                "Structures do not reproduce the recorded independent Gaussian draws"
            )
        datasets[name] = {"frames": frames, "labels": np.array(labels), "seed": seed}
    if level == 2:
        seeds = [datasets[name]["seed"] for name in datasets]
        _require(
            len(set(seeds)) == 3 and frozen["test_seeds"] == seeds[1:],
            "Use distinct, frozen test seeds",
        )
        for name, row in zip(("id_test", "strained_test"), tests[1::2], strict=True):
            _require(row["name"] == name, "Misaligned test descriptors")
            datasets[name]["features"] = finite_array(row["features"], ndim=2)
    return header, datasets, candidates, frozen


def _metric_matches(claim, actual, *, optional=True):
    _require(isinstance(claim, dict), "Metrics must be an object")
    return all(
        np.allclose(
            finite_array(claim[key], shape=()),
            value,
            rtol=0 if key == "r2" else 0.001,
            atol=1e-4 if key == "r2" else 1e-6,
        )
        for key, value in actual.items()
        if not optional or key in claim
    )


def _check_model(model, config, x, y):
    kind = config.get("preprocessing", "StandardScaler")
    options = config.get("preprocessing_options", {})
    _require(
        model["preprocessing"] == kind
        and model["alpha"] == config["alpha"]
        and model["fit_intercept"] == config.get("fit_intercept", True)
        and model["positive"] == config.get("positive", False),
        "Fitted pipeline differs from its candidate",
    )
    if kind == "StandardScaler":
        _require(
            all(
                model[key] == options.get(key, True)
                for key in ("with_mean", "with_std")
            ),
            "Scaler settings changed after selection",
        )
    elif kind == "MinMaxScaler":
        _require(
            model["feature_range"] == list(options.get("feature_range", [0, 1]))
            and model["clip"] == options.get("clip", False),
            "Scaler settings changed after selection",
        )
    elif kind == "PCA":
        _require(
            model["whiten"] == options.get("whiten", False), "PCA whitening changed"
        )
        if type(options.get("n_components")) is int:
            _require(
                len(model["components"]) == options["n_components"],
                "PCA dimension changed",
            )
    _require(
        preprocessing_matches(x, model),
        "Preprocessing was not fitted on its own training rows",
    )
    _require(
        ridge_stationarity(x, y, model),
        "Coefficients do not solve the training-only Ridge problem",
    )


def reconstruct(datasets, candidates, frozen):
    y = datasets["train"]["labels"]
    criterion = frozen["selection_metric"]
    _require(criterion in {"mae", "rmse", "r2"}, "Unknown model selection criterion")
    scores, validation, splits = [], [], None
    for candidate in candidates:
        config = candidate["config"]
        soap = config["soap"]
        _require(
            soap["species"] in (["Si"], [14])
            and soap["periodic"] is True
            and soap["average"] in {"inner", "outer"}
            and not soap.get("sparse", False),
            "Use dense global periodic silicon SOAP descriptors",
        )
        x = finite_array(candidate["features"], ndim=2)
        _require(
            x.shape[0] == 100 and x.shape[1] > 0,
            "Descriptors must align with all training rows",
        )
        _require(
            candidate["folds"], "Training-only validation calculations are required"
        )
        current_splits, values = [], []
        for fold in candidate["folds"]:
            train, test = partition(fold["train_indices"], fold["test_indices"], 100)
            _require(
                len(train) > 0 and len(test) >= 2,
                "Empty training or insufficient validation rows",
            )
            current_splits.append((train.tolist(), test.tolist()))
            _check_model(fold["model"], config, x[train], y[train])
            actual = metrics(y[test], predict(x[test], fold["model"]))
            _require(
                _metric_matches(fold["metrics"], actual), "Incorrect validation metrics"
            )
            values.append(actual)
        if splits is None:
            splits = current_splits
        _require(
            current_splits == splits,
            "Compare candidates using the same training-only splits",
        )
        validation.append(values)
        scores.append(float(np.mean([value[criterion] for value in values])))
    selected = frozen["selected_candidate"]
    _require(
        type(selected) is int and 0 <= selected < len(candidates),
        "Invalid selected candidate",
    )
    optimum = max(scores) if criterion == "r2" else min(scores)
    _require(
        np.isclose(scores[selected], optimum, rtol=1e-8, atol=1e-10),
        "Final selection contradicts training validation",
    )
    candidate = candidates[selected]
    config, x = candidate["config"], np.array(candidate["features"])
    _check_model(frozen["final"], config, x, y)
    output = {"validation": validation}
    for name in ("id_test", "strained_test"):
        data = datasets[name]
        features = finite_array(data["features"], shape=(100, x.shape[1]))
        prediction = predict(features, frozen["final"])
        output["id_test" if name == "id_test" else "strained_pooled"] = metrics(
            data["labels"], prediction
        )
        if name == "strained_test":
            output["strain_groups"] = [
                {
                    "strain": float(strain),
                    **metrics(
                        data["labels"][20 * i : 20 * (i + 1)],
                        prediction[20 * i : 20 * (i + 1)],
                    ),
                }
                for i, strain in enumerate(np.arange(-2, 3) / 100)
            ]
    output["learning_curve"] = []
    for row in frozen["learning_curve"]:
        raw = finite_array(row["train_indices"], ndim=1)
        indices = raw.astype(int)
        _require(
            np.array_equal(raw, indices)
            and np.all((indices >= 0) & (indices < 100))
            and len(np.unique(indices)) == len(indices),
            "Invalid learning-curve training indices",
        )
        _require(len(indices) > 0, "Empty learning-curve subset")
        _check_model(row["model"], config, x[indices], y[indices])
        # A conservative CDF screen permits random and stratified designs.
        distance = np.max(
            np.abs(
                np.searchsorted(np.sort(indices), np.arange(100), side="right")
                / len(indices)
                - np.arange(1, 101) / 100
            )
        )
        _require(
            distance <= np.sqrt(np.log(200) / (2 * len(indices))),
            "Learning subsets do not span the training distortion distribution",
        )
        output["learning_curve"].append(
            {
                "size": len(indices),
                **metrics(
                    datasets["id_test"]["labels"],
                    predict(datasets["id_test"]["features"], row["model"]),
                ),
            }
        )
    _require(
        len({row["size"] for row in output["learning_curve"]}) >= 2,
        "Use multiple learning-curve sizes",
    )
    return output


def answers_match(claims, calculated):
    _require(
        all(
            _metric_matches(claims[key], calculated[key], optional=True)
            for key in ("id_test", "strained_pooled")
            if key in claims
        ),
        "Incorrect test MAE, RMSE, or R²",
    )
    for key, identity in (("strain_groups", "strain"), ("learning_curve", "size")):
        if key not in claims:
            continue
        _require(len(claims[key]) == len(calculated[key]), f"Report every {key} result")
        # Preserve learning-curve record order so repeated sizes remain unambiguous.
        for claim, actual in zip(claims[key], calculated[key], strict=True):
            _require(
                np.isclose(
                    finite_array(claim[identity], shape=()),
                    actual[identity],
                    rtol=0,
                    atol=1e-9,
                )
                and _metric_matches(
                    claim, {k: actual[k] for k in ("mae", "rmse", "r2")}
                ),
                f"Incorrect {key} result",
            )
    return True


def build_plan(evidence, header, datasets, candidates, frozen, challenge):
    plan = Plan(evidence, 8, challenge)
    for name, data in datasets.items():
        plan.frames(
            f"teacher_{name}",
            data["frames"],
            count=100,
            chunk_size=25,
            values=[{"energy": float(y)} for y in data["labels"]],
            properties=["energy"],
            parameters={
                "model": "teacher.model",
                "dispersion": False,
                "default_dtype": header["default_dtype"],
            },
        )
    feature_sets = [
        (
            f"candidate_{i}",
            datasets["train"]["frames"],
            row["features"],
            row["config"]["soap"],
        )
        for i, row in enumerate(candidates)
    ]
    if frozen is not None:
        soap = candidates[frozen["selected_candidate"]]["config"]["soap"]
        feature_sets += [
            (name, datasets[name]["frames"], datasets[name]["features"], soap)
            for name in ("id_test", "strained_test")
        ]
    for name, frames, features, soap in feature_sets:
        plan.add(
            f"soap_{name}",
            "soap",
            {"frames": [geometry(a) for a in frames]},
            {"features": np.asarray(features).tolist()},
            parameters=soap,
            tolerance={"features": [1e-6, 1e-8]},
        )
    return plan


def _soap_settings(parameters):
    return {
        "species": ["Si"],
        "periodic": True,
        "rbf": parameters.get("rbf", "gto"),
        "average": parameters.get("average", "inner"),
        "sparse": False,
        **{key: parameters[key] for key in ("r_cut", "n_max", "l_max", "sigma")},
    }


def _descriptor_key(parameters):
    return json.dumps(_soap_settings(parameters), sort_keys=True)


def _validation_splits(record, size):
    from sklearn.model_selection import KFold, RepeatedKFold

    specification = record["cv"]
    if isinstance(specification, dict):
        kind = specification.get("method", "KFold")
        arguments = {
            key: specification[key]
            for key in ("n_splits", "n_repeats", "random_state")
            if key in specification
        }
    else:
        _require(
            isinstance(specification, str),
            "Retain reproducible training-only validation settings",
        )
        match = re.search(r"\b(RepeatedKFold|KFold)\(([^)]*)\)", specification)
        _require(
            match is not None, "Retain reproducible training-only validation settings"
        )
        kind = match[1]
        arguments = {
            key: int(value)
            for key, value in re.findall(
                r"(n_splits|n_repeats|random_state)\s*=\s*(\d+)", match[2]
            )
        }
        positional = re.match(r"\s*(\d+)\s*(?:,|$)", match[2])
        if positional:
            count = int(positional[1])
            _require(
                "n_splits" not in arguments or arguments["n_splits"] == count,
                "Conflicting validation fold count",
            )
            arguments["n_splits"] = count
        # Human-readable records often spell out the same constructor values.
        for key, expression in (
            ("n_splits", r"\b(\d+)\s+splits\b"),
            ("n_repeats", r"\b(\d+)\s+repeats\b"),
        ):
            explicit = re.search(expression, match[2])
            if explicit:
                _require(
                    key not in arguments or arguments[key] == int(explicit[1]),
                    f"Conflicting validation setting: {key}",
                )
                arguments[key] = int(explicit[1])
    _require(2 <= arguments.get("n_splits", 0) <= size, "Invalid validation fold count")
    if kind == "RepeatedKFold":
        _require(
            1 <= arguments.get("n_repeats", 0) <= 100,
            "Invalid repeated-validation count",
        )
        splitter = RepeatedKFold(**arguments)
    else:
        arguments.pop("n_repeats", None)
        splitter = KFold(**arguments, shuffle="random_state" in arguments)
    return list(splitter.split(np.arange(size)))


def _cross_validation(x, y, candidate, splits):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scores, references, predictions = [], [], []
    for train, test in splits:
        training, testing = x[train], x[test]
        if candidate["scale"]:
            scaler = StandardScaler().fit(training)
            training, testing = scaler.transform(training), scaler.transform(testing)
        model = Ridge(alpha=candidate["alpha"]).fit(training, y[train])
        prediction = model.predict(testing)
        scores.append(metrics(y[test], prediction))
        references.append(y[test])
        predictions.append(prediction)
    return {
        "fold_mean": {
            key: float(np.mean([score[key] for score in scores]))
            for key in ("mae", "rmse", "r2")
        },
        "pooled": metrics(np.concatenate(references), np.concatenate(predictions)),
    }


def _check_fit_agreement(record, reference, linked):
    """Reject conflicting descriptions of the same retained fitted artifact."""
    _require(
        isinstance(record, dict) and {"file", "sha256"}.issubset(record),
        "Retain the identity of explicitly recorded pipeline fits",
    )
    _require(
        linked(record["file"]) == linked(reference["file"])
        and record["sha256"] == reference["sha256"],
        "Fit record and frozen manifest disagree on pipeline identity",
    )
    observed = _pipeline_parameters(record.get("params", record))
    expected = _pipeline_parameters(reference)
    for key in (
        "r_cut",
        "n_max",
        "l_max",
        "sigma",
        "average",
        "rbf",
        "scale",
        "alpha",
        "indices",
        "n_train",
    ):
        if key in observed and key in expected:
            actual, retained = np.asarray(observed[key]), np.asarray(expected[key])
            if key == "indices":
                actual, retained = np.sort(actual), np.sort(retained)
            _require(
                np.array_equal(actual, retained),
                f"Fit record and frozen manifest disagree on {key}",
            )


def _saved_pipeline_records(rows, linked, log_path=None):
    """Read separate fit records or a combined, explicitly frozen manifest.

    The manifest certifies the final and learning-curve fits at its position in
    the original event log. It does not add timestamps or infer unrecorded fits.
    """
    rows = normalize_pipeline_records(rows, linked, log_path)
    stages = [_pipeline_stage(row) for row in rows]
    required = (
        "train_labels",
        "model_selection",
        "freeze",
        "test_labels",
        "test_evaluation",
    )
    _require(
        all(stages.count(stage) == 1 for stage in required),
        "Retain unambiguous training, validation, freeze and test evidence",
    )
    order = [stages.index(stage) for stage in required]
    _require(
        order == sorted(order),
        "Training and all fits must precede freezing and test-label generation",
    )
    result = {_pipeline_stage(row): dict(row) for row in rows}
    freeze = result["freeze"]
    if "manifest" in freeze:
        for stage in ("final_fit", "learning_curve_fits"):
            _require(
                stages.count(stage) <= 1
                and (stage not in stages or order[1] < stages.index(stage) < order[2]),
                "Recorded fits must precede freezing and test-label generation",
            )
        manifest = read_json(linked(freeze["manifest"]))
        _require(
            isinstance(manifest, dict), "Frozen pipeline manifest must be an object"
        )
        final = manifest["final"]
        if "final" in freeze:
            _require(
                freeze["final"] == final,
                "Freeze record and manifest disagree on final fit",
            )
        curves = manifest["learning_curve"]
        _require(
            isinstance(curves, dict),
            "Retain learning-curve model records by training size",
        )
        normalized_curves = [
            {**row, "n_train": int(count)} for count, row in curves.items()
        ]
        if "final_fit" in result:
            _check_fit_agreement(result["final_fit"], final, linked)
        if "learning_curve_fits" in result:
            fits = result["learning_curve_fits"].get("pipelines")
            _require(
                isinstance(fits, list) and len(fits) == len(normalized_curves),
                "Fit records and frozen manifest disagree on learning-curve pipelines",
            )
            by_file = {linked(row["file"]): row for row in normalized_curves}
            observed_files = [linked(row["file"]) for row in fits]
            _require(
                len(set(observed_files)) == len(fits)
                and set(observed_files) == set(by_file),
                "Fit records and frozen manifest disagree on learning-curve pipelines",
            )
            for row, path in zip(fits, observed_files, strict=True):
                _check_fit_agreement(row, by_file[path], linked)
        explicit_hashes = freeze.get("frozen", {})
        _require(
            isinstance(explicit_hashes, dict),
            "Frozen pipeline digests must be an object",
        )
        frozen_hashes = {}
        for filename, digest in explicit_hashes.items():
            name = Path(filename).name
            _require(
                name not in frozen_hashes or frozen_hashes[name] == digest,
                "Conflicting frozen pipeline digests",
            )
            frozen_hashes[name] = digest
        for row in [final, *curves.values()]:
            name = Path(row["file"]).name
            _require(
                name not in frozen_hashes or frozen_hashes[name] == row["sha256"],
                "Freeze record and manifest disagree on frozen pipeline digest",
            )
            frozen_hashes[name] = row["sha256"]
        result["final_fit"] = {**final, "params": _pipeline_parameters(final)}
        result["learning_curve_fits"] = {"pipelines": normalized_curves}
        freeze["frozen"] = frozen_hashes
        if "frozen_at" in manifest and "time" in freeze:
            _require(
                manifest["frozen_at"] <= freeze["time"],
                "Pipeline manifest was frozen after its recorded freeze event",
            )
    else:
        fits = ("final_fit", "learning_curve_fits")
        _require(
            all(stages.count(stage) == 1 for stage in fits)
            and order[1] < stages.index(fits[0]) < stages.index(fits[1]) < order[2],
            "Training and all fits must precede freezing and test-label generation",
        )
    recorded_times = [row["time"] for row in rows if "time" in row]
    if recorded_times:
        times = finite_array(recorded_times, ndim=1)
        _require(
            np.all(np.diff(times) >= 0),
            "Calculation timestamps contradict recorded chronology",
        )
    tests = result["test_labels"]
    if "files" in tests:
        _require(
            len(tests["files"]) == 2, "Retain independent and strained test datasets"
        )
        for index, name in enumerate(("independent", "strained")):
            tests[name] = {"file": tests["files"][index], "seed": tests[f"seed_{name}"]}
    evaluation = result["test_evaluation"]
    if "final" in evaluation:
        evaluation = dict(evaluation["final"])
        result["test_evaluation"] = evaluation
    for name in ("independent", "strained_pooled"):
        if name in evaluation:
            evaluation[name] = {
                key.lower(): value for key, value in evaluation[name].items()
            }
    return result


def _evaluate_saved_pipelines(
    evidence, rows, verifier, rubric, calculated, verification, fingerprint
):
    """Rebuild descriptors and fits when the log links to saved numerical state."""
    paths = {}
    parsed = {}

    def linked(name):
        path = evidence._path(name)
        _require(path.is_file(), f"Missing saved calculation artifact: {name}")
        paths[str(path)] = file_ref(path)["sha256"]
        return path

    def parse():
        records = _saved_pipeline_records(rows, linked, evidence.artifact("run_log"))
        train, tests = records["train_labels"], records["test_labels"]
        model_digest = train.get("model_sha256")
        if model_digest is not None:
            _require(
                model_digest == PINNED_TEACHER_SHA256,
                "Training labels use a different teacher checkpoint",
            )
        datasets = {}
        seed_values = []
        for name, record in (
            ("train", train),
            ("id_test", tests["independent"]),
            ("strained_test", tests["strained"]),
        ):
            frames = read(linked(record["file"]), index=":", format="extxyz")
            seed_values.append(record["seed"])
            values = [stored(frame) for frame in frames]
            _require(
                all("energy" in value for value in values),
                "Retain the total energy of every saved structure",
            )
            _require(
                all(
                    np.array_equal(frame.numbers, np.full(64, 14)) and frame.pbc.all()
                    for frame in frames
                ),
                "Expected periodic Si64 structures",
            )
            datasets[name] = {
                "frames": frames,
                "labels": np.array([value["energy"] for value in values]),
                "seed": record["seed"],
                "rng": record.get("rng"),
            }
        _require(
            len(set(seed_values)) == 3, "Use independent training and test random seeds"
        )
        for name, dataset in datasets.items():
            finite_array(dataset["labels"], shape=(100,))
            _require(
                _matches_generated_dataset(
                    name,
                    dataset["frames"],
                    dataset["seed"],
                    dataset["rng"],
                    restore_precision=True,
                ),
                "Saved structures do not reproduce their recorded seed and displacement schedule",
            )
        selection, final = records["model_selection"], records["final_fit"]
        grid = (
            selection["candidates"]
            if "candidates" in selection
            else read_json(linked(selection["file"]))
        )
        if isinstance(grid, dict):
            if "cv" in grid and "cv" in selection:
                recorded = _validation_splits(selection, 100)
                retained = _validation_splits(grid, 100)
                _require(
                    len(recorded) == len(retained)
                    and all(
                        np.array_equal(observed, expected)
                        for split, reference in zip(recorded, retained, strict=True)
                        for observed, expected in zip(split, reference, strict=True)
                    ),
                    "Selection record and grid disagree on validation splits",
                )
            selection = {**selection, "cv": grid.get("cv", selection.get("cv"))}
            for key in ("selection_metric", "metric", "scoring"):
                if key in grid:
                    _require(
                        key not in selection or selection[key] == grid[key],
                        "Conflicting model selection metric declarations",
                    )
                    selection[key] = grid[key]
            grid = grid["results"]
        _require(
            isinstance(grid, list) and len(grid) >= 2,
            "Retain multiple training-validation candidates",
        )
        _require(
            len(grid) <= 10000,
            "Validation grid exceeds the supported calculation budget",
        )
        grid = [_pipeline_parameters(candidate) for candidate in grid]
        criterion = selection_metric(selection, grid)
        splits = _validation_splits(selection, 100)
        final_path = linked(final["file"])
        model, descriptor = read_pipeline(final_path)
        _require(
            paths[str(final_path)] == final["sha256"],
            "Final pipeline changed after fitting",
        )
        _require(
            all(
                final["params"].get(key, value) == value
                for key, value in descriptor.items()
            ),
            "Saved SOAP settings differ from the recorded final fit",
        )
        frozen = records["freeze"]["frozen"]
        _require(
            frozen.get(final_path.name) == paths[str(final_path)],
            "Final pipeline differs from its frozen digest",
        )
        curves = []
        for row in records["learning_curve_fits"]["pipelines"]:
            path = linked(row["file"])
            state, descriptor = read_pipeline(path)
            _require(
                paths[str(path)] == row["sha256"] == frozen.get(path.name),
                "Learning-curve pipeline differs from its frozen digest",
            )
            indices = finite_array(row["indices"], ndim=1)
            _require(
                np.array_equal(indices, indices.astype(int))
                and np.all((indices >= 0) & (indices < 100))
                and len(set(indices)) == len(indices) == row["n_train"],
                "Invalid learning-curve training subset",
            )
            _require(
                all(
                    _soap_settings(final["params"])[key] == value
                    for key, value in descriptor.items()
                ),
                "Learning-curve descriptor differs from its frozen settings",
            )
            curves.append({**row, "model": state, "train_indices": indices.astype(int)})
        _require(
            len({len(row["train_indices"]) for row in curves}) >= 2,
            "Retain multiple learning-curve training sizes",
        )
        parsed.update(
            records=records,
            datasets=datasets,
            grid=grid,
            splits=splits,
            final=final,
            model=model,
            curves=curves,
            criterion=criterion,
        )
        return True

    rubric.check("saved_datasets_and_frozen_pipelines", 20, parse)
    if rubric.failed or rubric.unverified_checks:
        return fingerprint
    fingerprint = hashlib.sha256(
        json.dumps(
            {"manifest": evidence.fingerprint(), "linked_files": paths}, sort_keys=True
        ).encode()
    ).hexdigest()
    features = {}

    def independently_calculate():
        if verifier is None:
            raise EvaluationError(
                "Task 8 pipeline reconstruction requires the configured independent calculation backend"
            )
        plan = Plan(evidence, 8, secrets.token_hex(32))
        for name, dataset in parsed["datasets"].items():
            plan.frames(
                f"teacher_{name}",
                dataset["frames"],
                count=100,
                chunk_size=25,
                values=[{"energy": float(y)} for y in dataset["labels"]],
                properties=["energy"],
                parameters={
                    "model": "teacher.model",
                    "dispersion": False,
                    "default_dtype": "float64",
                },
            )
        descriptors = {
            _descriptor_key(candidate): _soap_settings(candidate)
            for candidate in parsed["grid"]
        }
        descriptors.setdefault(
            _descriptor_key(parsed["final"]["params"]),
            _soap_settings(parsed["final"]["params"]),
        )
        identifiers = {}
        for index, (key, parameters) in enumerate(descriptors.items()):
            identifier = f"soap_train_{index}"
            plan.add(
                identifier,
                "soap",
                {
                    "frames": [
                        geometry(frame)
                        for frame in parsed["datasets"]["train"]["frames"]
                    ]
                },
                None,
                parameters=parameters,
            )
            identifiers[identifier] = ("train", key)
        key = _descriptor_key(parsed["final"]["params"])
        for name in ("id_test", "strained_test"):
            identifier = f"soap_{name}"
            plan.add(
                identifier,
                "soap",
                {
                    "frames": [
                        geometry(frame) for frame in parsed["datasets"][name]["frames"]
                    ]
                },
                None,
                parameters=descriptors[key],
            )
            identifiers[identifier] = (name, key)
        response = verifier._calculate(plan, fingerprint)
        _require(
            response.get("challenge") == plan.challenge
            and response.get("evidence_sha256") == fingerprint,
            "Independent response belongs to different evidence",
        )
        jobs = response.get("jobs", [])
        _require(
            len(jobs) == len(plan.jobs)
            and {job["id"] for job in jobs} == set(plan.expected),
            "Incomplete independent calculation response",
        )
        for job in jobs:
            if job["status"] != "complete":
                raise EvaluationError(
                    f"Independent {job['id']} failed: {job.get('detail')}"
                )
            if job["id"] in identifiers:
                features[identifiers[job["id"]]] = finite_array(
                    job["value"]["features"], ndim=2
                )
            else:
                expected = plan.expected[job["id"]]
                _require(
                    compare(expected["values"], job["value"], expected["tolerance"]),
                    f"Incorrect retained teacher labels: {job['id']}",
                )
        verification.update(
            status="passed",
            release_id=response.get("release_id"),
            teacher_configurations=300,
            soap_training_settings=len(descriptors),
        )
        return True

    rubric.check("independent_teacher_and_descriptors", 20, independently_calculate)

    def reconstruct():
        y = parsed["datasets"]["train"]["labels"]
        validation = []
        aggregations = {"fold_mean", "pooled"}
        for candidate in parsed["grid"]:
            score = _cross_validation(
                features["train", _descriptor_key(candidate)],
                y,
                candidate,
                parsed["splits"],
            )
            aggregations &= {
                method
                for method, values in score.items()
                if all(
                    np.isclose(
                        candidate[f"cv_{key}"], values[key], rtol=0.001, atol=1e-6
                    )
                    for key in ("mae", "rmse", "r2")
                    if f"cv_{key}" in candidate
                )
            }
            _require(
                aggregations,
                "Recorded training-only CV metrics disagree with independently reconstructed fits",
            )
            validation.append(score)
        final = parsed["final"]["params"]
        matches = [
            i
            for i, candidate in enumerate(parsed["grid"])
            if all(candidate[key] == final[key] for key in ("scale", "alpha"))
            and _descriptor_key(candidate) == _descriptor_key(final)
        ]
        _require(
            len(matches) == 1,
            "Final pipeline is not a unique recorded validation candidate",
        )
        criterion = parsed["criterion"]
        optimum = max if criterion == "r2" else min
        aggregations = {
            method
            for method in aggregations
            if np.isclose(
                validation[matches[0]][method][criterion],
                optimum(row[method][criterion] for row in validation),
                rtol=1e-5,
                atol=1e-7,
            )
        }
        _require(
            aggregations,
            "Final selection contradicts training-only validation",
        )
        aggregation = "pooled" if "pooled" in aggregations else "fold_mean"
        validation = [row[aggregation] for row in validation]
        calculated["validation_aggregation"] = aggregation
        calculated["selection_metric"] = criterion
        x = features["train", _descriptor_key(final)]

        def check_model(state, settings, indices):
            config = {
                "preprocessing": "StandardScaler" if settings["scale"] else "identity",
                "alpha": settings["alpha"],
            }
            _check_model(state, config, x[indices], y[indices])

        check_model(parsed["model"], final, np.arange(100))
        calculated["validation"] = validation
        calculated["training_count"] = 100
        for name in ("id_test", "strained_test"):
            prediction = predict(
                features[name, _descriptor_key(final)], parsed["model"]
            )
            labels = parsed["datasets"][name]["labels"]
            calculated["id_test" if name == "id_test" else "strained_pooled"] = metrics(
                labels, prediction
            )
            if name == "strained_test":
                calculated["strain_groups"] = [
                    {
                        "strain": float(strain),
                        **metrics(
                            labels[i * 20 : (i + 1) * 20],
                            prediction[i * 20 : (i + 1) * 20],
                        ),
                    }
                    for i, strain in enumerate(np.arange(-2, 3) / 100)
                ]
        calculated["learning_curve"] = []
        for row in parsed["curves"]:
            indices = row["train_indices"]
            distance = np.max(
                abs(
                    np.searchsorted(np.sort(indices), np.arange(100), side="right")
                    / len(indices)
                    - np.arange(1, 101) / 100
                )
            )
            _require(
                distance <= np.sqrt(np.log(200) / (2 * len(indices))),
                "Learning-curve subsets do not span the distortion distribution",
            )
            check_model(row["model"], row, indices)
            calculated["learning_curve"].append(
                {
                    "size": len(indices),
                    **metrics(
                        parsed["datasets"]["id_test"]["labels"],
                        predict(
                            features["id_test", _descriptor_key(final)], row["model"]
                        ),
                    ),
                }
            )
        return True

    rubric.check("reconstructed_validation_and_predictions", 40, reconstruct)

    def reported():
        record = parsed["records"]["test_evaluation"]
        _require(
            all(
                compare(
                    {key: record[name][key] for key in ("mae", "rmse", "r2")},
                    {key: calculated[target][key] for key in ("mae", "rmse", "r2")},
                    {key: [0.001, 1e-6] for key in ("mae", "rmse", "r2")},
                )
                for name, target in (
                    ("independent", "id_test"),
                    ("strained_pooled", "strained_pooled"),
                )
            ),
            "Recorded test metrics disagree with saved pipeline predictions",
        )
        return answers_match(evidence.results, calculated)

    rubric.check("reported_metrics", 20, reported)
    if any(file_ref(Path(path))["sha256"] != digest for path, digest in paths.items()):
        raise EvaluationError("Linked calculation artifacts changed during scoring")
    return fingerprint


def _evaluate_inline_pipelines(
    evidence, level, verifier, rubric, calculated, verification, fingerprint
):
    """Validate fitted states and descriptor arrays embedded in the recording."""
    try:
        header, datasets, candidates, frozen = read_log(evidence, level)
    except Exception as exc:
        rubric.error("calculation_log", 100, exc)
        return
    rubric.check("structures_labels_and_freeze_order", 20, True)
    calculated["training_count"] = 100
    if level == 2:

        def pipelines():
            calculated.update(reconstruct(datasets, candidates, frozen))
            return True

        rubric.check("training_only_pipelines_and_selection", 35, pipelines)
        rubric.check(
            "reported_metrics", 25, lambda: answers_match(evidence.results, calculated)
        )

    def independent():
        if verifier is None:
            raise UnsupportedEvidence(
                "Independent teacher and SOAP checks require a local or Modal backend"
            )
        plan = build_plan(
            evidence, header, datasets, candidates, frozen, secrets.token_hex(32)
        )
        try:
            response = verifier._calculate(plan, fingerprint)
        except Exception as exc:
            raise UnsupportedEvidence(
                f"Independent calculations unavailable: {exc}"
            ) from exc
        _require(
            response.get("evidence_sha256") == fingerprint
            and response.get("challenge") == plan.challenge,
            "Independent calculation returned a different evidence identity",
        )
        jobs = response.get("jobs", [])
        _require(
            len(jobs) == len(plan.jobs)
            and {job["id"] for job in jobs} == set(plan.expected),
            "Missing or duplicate independent calculation",
        )
        verification["release_id"] = response.get("release_id")
        for job in jobs:
            if job["status"] == "unsupported":
                raise UnsupportedEvidence(
                    job.get("detail", "Independent calculation unsupported")
                )
            _require(
                job["status"] == "complete",
                job.get("detail", "Independent calculation failed"),
            )
            expected = plan.expected[job["id"]]
            _require(
                compare(expected["values"], job["value"], expected["tolerance"]),
                f"Independent calculation disagrees: {job['id']}",
            )
        verification["status"] = "passed"
        return True

    rubric.check("independent_teacher_and_soap", 80 if level == 1 else 20, independent)


def evaluate_submission(submission, *, level, verifier=None):
    rubric = Rubric(8, binary=True, fail_fast=True)
    calculated, verification = {}, {}
    try:
        evidence = Evidence(submission, task_number=8, level=level)
        manifest_fingerprint = evidence.fingerprint()
        raw = records(evidence.artifact("run_log"))
    except Exception as exc:
        rubric.error("calculation_log", 100, exc)
        return {**rubric.as_dict(), "level": level, "calculated_results": calculated}
    fingerprint = manifest_fingerprint
    if (
        level == 2
        and isinstance(raw, list)
        and any(_pipeline_stage(row) == "train_labels" for row in raw)
    ):
        fingerprint = _evaluate_saved_pipelines(
            evidence, raw, verifier, rubric, calculated, verification, fingerprint
        )
    else:
        _evaluate_inline_pipelines(
            evidence, level, verifier, rubric, calculated, verification, fingerprint
        )
    if evidence.fingerprint() != manifest_fingerprint:
        raise EvaluationError(
            "Submission changed during grading; retry with frozen evidence"
        )
    identity = hashlib.sha256(
        f"task8.level_{level}\0{fingerprint}\0{verification.get('release_id')}".encode()
    ).hexdigest()
    return {
        **rubric.as_dict(),
        "level": level,
        "evidence_sha256": identity,
        "calculated_results": calculated,
        "verification": verification,
    }
