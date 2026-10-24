"""Reconstruct metrics from saved models without importing their agent code."""

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from corral_md.scoring.common import EvaluationError, EvidenceError
from corral_md.scoring.pipeline_data import SavedArray, read_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location(
    "task8_fixtures", Path(__file__).with_name("test_score_task_8.py")
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
features, make_submission, read_log, scorer = (
    fixtures.features,
    fixtures.make_submission,
    fixtures.read_log,
    fixtures.scorer,
)


@pytest.fixture
def saved_runtime(tmp_path, monkeypatch):
    return fixtures.runtime.__wrapped__(tmp_path, monkeypatch)


class SoapRidge:
    def predict(self, _):
        raise AssertionError("Agent methods must never be called by the scorer")


def save_pipeline(path, x, y, parameters):
    model = SoapRidge()
    for key, value in parameters.items():
        setattr(model, key, value)
    model.scaler_ = None
    model.ridge_ = Ridge(alpha=parameters["alpha"]).fit(x, y)
    path.write_bytes(pickle.dumps(model))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_submission(root, model):
    manifest, _ = make_submission(root, model)
    rows = read_log(root)
    datasets = {}
    for row in rows:
        if row.get("event") != "dataset":
            continue
        frames = []
        for record in row["rows"]:
            frame = Atoms(
                "Si64", positions=record["positions"], cell=record["cell"], pbc=True
            )
            frame.calc = SinglePointCalculator(frame, energy=record["energy_eV"])
            frames.append(frame)
        path = root / f"{row['name']}.extxyz"
        write(path, frames, format="extxyz")
        datasets[row["name"]] = (
            row,
            frames,
            np.array([r["energy_eV"] for r in row["rows"]]),
        )
    parameters = {
        "r_cut": 4.5,
        "n_max": 4,
        "l_max": 4,
        "sigma": 0.5,
        "average": "inner",
        "scale": False,
    }
    train, frames, y = datasets["train"]
    x = features(frames, parameters)
    candidates = []
    for alpha in (0.01, 1.0):
        errors = []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(x):
            prediction = Ridge(alpha=alpha).fit(x[tr], y[tr]).predict(x[va])
            errors.extend(prediction - y[va])
        candidates.append(
            {
                **parameters,
                "alpha": alpha,
                "cv_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "cv_mae": float(np.mean(np.abs(errors))),
            }
        )
    best = min(candidates, key=lambda row: row["cv_rmse"])
    parameters["alpha"] = best["alpha"]
    (root / "grid.json").write_text(json.dumps(candidates))
    frozen = {}
    frozen["final.pkl"] = save_pipeline(root / "final.pkl", x, y, parameters)
    curves = []
    for count in (30, 100):
        indices = np.linspace(0, 99, count).astype(int)
        name = f"lc_{count}.pkl"
        digest = save_pipeline(root / name, x[indices], y[indices], parameters)
        frozen[name] = digest
        curves.append(
            {
                "file": name,
                "sha256": digest,
                "n_train": count,
                "indices": indices.tolist(),
                "scale": False,
                "alpha": parameters["alpha"],
            }
        )

    def measured(name):
        _, frames, labels = datasets[name]
        pred = (
            Ridge(alpha=parameters["alpha"])
            .fit(x, y)
            .predict(features(frames, parameters))
        )
        return {
            "mae": float(np.mean(abs(pred - labels))),
            "rmse": float(np.sqrt(np.mean((pred - labels) ** 2))),
            "r2": float(
                1 - np.sum((pred - labels) ** 2) / np.sum((labels - labels.mean()) ** 2)
            ),
        }

    log = [
        {"stage": "train_labels", "file": "train.extxyz", "seed": train["seed"]},
        {
            "stage": "model_selection",
            "file": "grid.json",
            "cv": {"method": "KFold", "n_splits": 5, "random_state": 0},
        },
        {
            "stage": "final_fit",
            "file": "final.pkl",
            "sha256": frozen["final.pkl"],
            "params": parameters,
        },
        {"stage": "learning_curve_fits", "pipelines": curves},
        {"stage": "freeze", "frozen": frozen},
        {
            "stage": "test_labels",
            **{
                key: {"file": f"{name}.extxyz", "seed": datasets[name][0]["seed"]}
                for key, name in (
                    ("independent", "id_test"),
                    ("strained", "strained_test"),
                )
            },
        },
        {
            "stage": "test_evaluation",
            "independent": measured("id_test"),
            "strained_pooled": measured("strained_test"),
        },
    ]
    (root / "run.jsonl").write_text("\n".join(map(json.dumps, log)))
    # The unchanged submission example lists only run_log. Sidecars linked from
    # that log must not require extra top-level artifact keys.
    manifest.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    return manifest


def test_saved_model_metrics_are_reconstructed(tmp_path, saved_runtime):
    path = stage_submission(tmp_path, saved_runtime)
    report = scorer().evaluate(path)
    assert report["score"] == 1, report
    result = report["calculated_results"]
    assert len(result["strain_groups"]) == 5
    assert [r["size"] for r in result["learning_curve"]] == [30, 100]
    assert result["id_test"]["rmse"] >= 0
    assert "pending_checks" not in report


@pytest.mark.parametrize("artifact", ["run.jsonl", "final.pkl"])
def test_saved_pipeline_scoring_rejects_evidence_changed_during_verification(
    tmp_path, saved_runtime, artifact
):
    path = stage_submission(tmp_path, saved_runtime)

    def mutate(plan, fingerprint):
        response = fixtures.transport(plan, fingerprint)
        with (tmp_path / artifact).open("ab") as stream:
            stream.write(b" ")
        return response

    with pytest.raises(EvaluationError, match="changed during"):
        scorer(backend=mutate).evaluate(path)


def test_saved_pipeline_scoring_rejects_removed_review_argument(
    tmp_path, saved_runtime
):
    path = stage_submission(tmp_path, saved_runtime)
    with pytest.raises(TypeError, match="review"):
        scorer().evaluate(path, review={"evidence_sha256": "wrong", "decisions": {}})


@pytest.mark.parametrize(
    "defect", ["metrics", "freeze", "cv", "geometry", "missing_pipeline", "teacher"]
)
def test_saved_pipeline_failures_are_scientific_failures(
    tmp_path, saved_runtime, defect
):
    path = stage_submission(tmp_path, saved_runtime)
    log = read_log(tmp_path)
    if defect == "metrics":
        log[-1]["independent"]["rmse"] += 1
    elif defect == "freeze":
        log[4]["frozen"]["final.pkl"] = "0" * 64
    elif defect == "cv":
        grid = json.loads((tmp_path / "grid.json").read_text())
        grid[0]["cv_rmse"] += 1
        (tmp_path / "grid.json").write_text(json.dumps(grid))
    elif defect == "geometry":
        log[0]["seed"] += 1
    elif defect == "teacher":
        log[0]["model_sha256"] = "0" * 64
    else:
        (tmp_path / "final.pkl").unlink()
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, log)))
    assert scorer().evaluate(path)["score"] == 0


def test_pipeline_reader_rejects_executable_pickle_globals(tmp_path):
    class Executable:
        def __reduce__(self):
            return eval, ("1+1",)

    path = tmp_path / "malicious.pkl"
    path.write_bytes(pickle.dumps(Executable()))
    with pytest.raises(EvidenceError, match="Non-numerical pickle global"):
        read_pipeline(path)


@pytest.mark.parametrize("protocol", [4, 5])
def test_saved_scaler_and_ridge_state_reconstruct_predictions(tmp_path, protocol):
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    model = SoapRidge()
    model.scaler_ = StandardScaler().fit(x)
    model.ridge_ = Ridge(alpha=0.1).fit(model.scaler_.transform(x), y)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model, protocol=protocol))
    state, _ = read_pipeline(path)
    from corral_md.scoring.regression import predict

    np.testing.assert_allclose(
        predict(x, state), model.ridge_.predict(model.scaler_.transform(x))
    )


def test_saved_array_shape_is_bounded_before_allocation():
    with pytest.raises(EvidenceError, match="shape or byte length"):
        SavedArray().__setstate__((1, (10**12,), np.dtype("f8"), False, b""))
