"""Score silicon SOAP-Ridge calculations from one final recording.

Reconstruct every metric from portable fitted state; verify all teacher labels
and SOAP features independently. No submitted code or pickle is executed.
Recorded chronology checks the provided workflow, not unrecorded test access.
"""

from __future__ import annotations

import hashlib
import json
import secrets

import numpy as np
from ase import Atoms
from ase.build import bulk

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    finite_array,
)
from .level1_trusted import PINNED_TEACHER_SHA256
from .regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
)
from .verification import Plan, compare, geometry


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_log(evidence, level):
    with evidence.artifact("run_log").open() as stream:
        records = [json.loads(line) for line in stream]
    _require(len(records) >= 3, "Truncated calculation log")
    header = records[0]
    _require(
        header["event"] == "start" and header["model_sha256"] == PINNED_TEACHER_SHA256,
        "Use the supplied teacher checkpoint",
    )
    _require(
        header["default_dtype"] in {"float32", "float64"}, "Invalid teacher precision"
    )
    _require(
        header["units"] == {"energy": "eV", "length": "Angstrom"}, "Incorrect log units"
    )
    events = [row["event"] for row in records]
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
        candidates, frozen = records[2:index], records[index]
        tests = records[index + 1 : index + 5]
    datasets = {}
    for name, record in zip(
        ("train", "id_test", "strained_test"), [records[1], *tests[::2]], strict=False
    ):
        _require(record["name"] == name, "Incorrect dataset order")
        seed = record["seed"]
        _require(
            type(seed) is int and seed >= 0 and record["rng"] == "numpy.default_rng",
            "Invalid generation seed or method",
        )
        _require(
            len(record["rows"]) == 100, "Each dataset needs exactly 100 structures"
        )
        rng = np.random.default_rng(seed)
        ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        frames, labels = [], []
        for i, row in enumerate(record["rows"]):
            strain = (
                np.repeat(np.arange(-2, 3) / 100, 20)[i]
                if name == "strained_test"
                else 0
            )
            sigma = 0.05 if name == "strained_test" else 0.01 + 0.001 * i
            cell = ideal.cell.array * (1 + strain)
            expected = ideal.positions * (1 + strain) + rng.normal(0, sigma, (64, 3))
            positions = finite_array(row["positions"], shape=(64, 3))
            saved_cell = finite_array(row["cell"], shape=(3, 3))
            _require(
                row["row_id"] == f"{name}:{i}"
                and np.allclose(saved_cell, cell, rtol=0, atol=1e-7),
                "Incorrect row identity or silicon cell",
            )
            delta = (positions - expected) @ np.linalg.inv(cell)
            _require(
                np.allclose(delta - np.rint(delta), 0, rtol=0, atol=1e-8),
                "Structures do not reproduce the prescribed independent Gaussian draws",
            )
            frames.append(Atoms("Si64", positions=positions, cell=cell, pbc=True))
            labels.append(finite_array(row["energy_eV"], shape=()).item())
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


def _metric_matches(claim, actual, *, optional=False):
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


def evaluate_submission(submission, *, level, verifier=None, review=None):
    rubric = Rubric(8, binary=True, fail_fast=True)
    calculated, verification = {}, {}
    try:
        evidence = Evidence(submission)
        fingerprint = evidence.fingerprint()
        header, datasets, candidates, frozen = read_log(evidence, level)
    except Exception as exc:
        rubric.check("calculation_log", 100, False, str(exc))
        return {**rubric.as_dict(), "level": level, "calculated_results": calculated}
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
    if evidence.fingerprint() != fingerprint:
        raise RuntimeError(
            "Submission changed during grading; retry with frozen evidence"
        )
    identity = hashlib.sha256(
        f"task8.level_{level}\0{fingerprint}\0{verification.get('release_id')}".encode()
    ).hexdigest()
    if review is not None:
        rubric.apply_review(review, identity)
    return {
        **rubric.as_dict(),
        "level": level,
        "evidence_sha256": identity,
        "calculated_results": calculated,
        "verification": verification,
    }
