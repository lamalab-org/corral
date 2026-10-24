"""Exercise equivalent saved workflows, selection objectives and joblib state."""

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from corral_md.scoring.common import EvaluationError, EvidenceError
from corral_md.scoring.pipeline_data import read_pipeline
from corral_md.scoring.pipeline_records import selection_metric
from corral_md.scoring.regression import predict
from corral_md.scoring.task_8 import _matches_generated_dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location(
    "record_normalization_fixtures",
    Path(__file__).with_name("test_scoring_saved_pipelines.py"),
)
saved = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = saved
spec.loader.exec_module(saved)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return saved.fixtures.runtime.__wrapped__(tmp_path, monkeypatch)


def write_rows(root, rows):
    (root / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))


@pytest.mark.parametrize("names", ["arbitrary", "absent"])
def test_record_contents_identify_roles_without_prescribed_names(
    tmp_path, runtime, names
):
    submission = saved.stage_submission(tmp_path, runtime)
    rows = saved.read_log(tmp_path)
    for i, row in enumerate(rows):
        row.pop("stage")
        if names == "arbitrary":
            row["event"] = f"user-defined activity {i}"
    write_rows(tmp_path, rows)
    assert saved.scorer().evaluate(submission)["score"] == 1


@pytest.mark.parametrize("criterion", ["mae", "rmse", "r2"])
@pytest.mark.parametrize("declaration", [True, False])
def test_selection_uses_declared_or_single_logged_metric(
    tmp_path, runtime, criterion, declaration
):
    submission = saved.stage_submission(tmp_path, runtime)
    rows = saved.read_log(tmp_path)
    grid = json.loads((tmp_path / "grid.json").read_text())
    frames = read(tmp_path / "train.extxyz", index=":")
    y = np.array([a.get_potential_energy() for a in frames])
    for candidate in grid:
        x = saved.features(frames, candidate)
        actual, predictions = [], []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(x):
            actual.extend(y[va])
            predictions.extend(
                Ridge(alpha=candidate["alpha"]).fit(x[tr], y[tr]).predict(x[va])
            )
        measurements = {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
            "r2": r2_score(actual, predictions),
        }
        for key in ("mae", "rmse", "r2"):
            candidate.pop(f"cv_{key}", None)
        candidate[f"cv_{criterion}"] = measurements[criterion]
    if declaration:
        rows[1]["scoring"] = criterion.upper()
    (tmp_path / "grid.json").write_text(json.dumps(grid))
    write_rows(tmp_path, rows)
    report = saved.scorer().evaluate(submission)
    assert report["score"] == 1, report
    assert report["calculated_results"]["selection_metric"] == criterion
    # Declaring a supported objective must not suppress checks of its numbers.
    grid[0][f"cv_{criterion}"] += 10
    (tmp_path / "grid.json").write_text(json.dumps(grid))
    assert saved.scorer().evaluate(submission)["score"] == 0


class SOAPRidgeModel:
    def predict(self, _):
        raise AssertionError("Submitted model methods must never execute")


def split_records(root, model):
    """Observed run layout: inline candidates, individual fits and ISO times."""
    submission = saved.stage_submission(root, model)
    train, selection, final, curves, freeze, tests, evaluation = saved.read_log(root)
    grid = json.loads((root / selection["file"]).read_text())
    final_params = final["params"]
    hashes = {}
    for fit in [final, *curves["pipelines"]]:
        path = root / fit["file"]
        # Evaluator-owned synthetic objects only; production uses the restricted reader.
        old = pickle.loads(path.read_bytes())
        state = SOAPRidgeModel()
        state.soap_params = {
            k: final_params[k] for k in ("r_cut", "n_max", "l_max", "sigma")
        }
        state.sk_pipeline = Pipeline([("ridge", old.ridge_)])
        joblib.dump(state, path)
        hashes[fit["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()

    def labels(filename):
        return [a.get_potential_energy() for a in read(root / filename, index=":")]

    rows = [
        {
            "event": "train_labels",
            "seed": train["seed"],
            "energies": labels(train["file"]),
        }
    ]
    for candidate in grid:
        rows.append(
            {
                "event": "cv_candidate",
                "soap": {k: candidate[k] for k in state.soap_params},
                "best_alpha": candidate["alpha"],
                "cv_mae": candidate["cv_mae"],
            }
        )
    rows.append(
        {
            "event": "cv_selected",
            "cv_scheme": "KFold(5, shuffle=True, random_state=0)",
            "scoring": "MAE (eV/structure)",
        }
    )
    rows.append(
        {
            "event": "final_frozen",
            "path": final["file"],
            "soap": state.soap_params,
            "alpha": final_params["alpha"],
            "preprocessing": "identity",
        }
    )
    for fit in curves["pipelines"]:
        rows.append(
            {
                "event": "lc_frozen",
                "n": fit["n_train"],
                "train_indices": fit["indices"],
                "path": fit["file"],
                "alpha": fit["alpha"],
            }
        )
    rows.append({"event": "phase1_done_pipelines_frozen", "hashes": hashes})
    rows.append(
        {
            "event": "test_labels",
            "independent_seed": tests["independent"]["seed"],
            "strained_seed": tests["strained"]["seed"],
            "E_indep": labels(tests["independent"]["file"]),
            "E_strained": labels(tests["strained"]["file"]),
        }
    )
    rows.append({**evaluation, "event": "eval_final"})
    rows[-1].pop("stage")
    rows.append(
        {
            "event": "eval_lc",
            "path": curves["pipelines"][0]["file"],
            "n": 30,
            "independent": evaluation["independent"],
            "strained_pooled": evaluation["strained_pooled"],
        }
    )
    for i, row in enumerate(rows):
        row["time"] = f"2026-10-24T09:40:{i:02d}Z"
    write_rows(root, rows)
    return submission, rows


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "late_fit",
        "late_candidate",
        "hash",
        "timestamp",
        "metrics",
        "duplicate_selection",
        "duplicate_dataset",
    ],
)
def test_observed_layout_keeps_substantive_checks(tmp_path, runtime, defect):
    submission, rows = split_records(tmp_path, runtime)
    if defect == "late_fit":
        fit = rows.pop(next(i for i, r in enumerate(rows) if r["event"] == "lc_frozen"))
        rows.append(fit)
    elif defect == "late_candidate":
        rows.append(rows.pop(1))
    elif defect == "hash":
        next(r for r in rows if "hashes" in r)["hashes"]["final.pkl"] = "0" * 64
    elif defect == "timestamp":
        rows[-1]["time"] = "2026-10-23T09:40:00Z"
    elif defect == "metrics":
        next(r for r in rows if r["event"] == "eval_final")["independent"]["rmse"] += 1
    elif defect == "duplicate_selection":
        rows.insert(4, dict(rows[3]))
    elif defect == "duplicate_dataset":
        (tmp_path / "duplicate.extxyz").write_bytes(
            (tmp_path / "train.extxyz").read_bytes()
        )
    write_rows(tmp_path, rows)
    if defect == "duplicate_dataset":
        with pytest.raises(EvaluationError, match="uniquely link"):
            saved.scorer().evaluate(submission)
    else:
        report = saved.scorer().evaluate(submission)
        assert report["score"] == (1 if defect is None else 0), report


@pytest.mark.parametrize("protocol", [4, 5])
def test_joblib_pipeline_numerical_state_matches_predictions(tmp_path, protocol):
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    state = SOAPRidgeModel()
    state.soap_params = {"r_cut": 4.0, "n_max": 6, "l_max": 6, "sigma": 0.3}
    state.sk_pipeline = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=0.1))]
    ).fit(x, y)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(state, path, protocol=protocol)
    model, descriptor = read_pipeline(path)
    np.testing.assert_allclose(predict(x, model), state.sk_pipeline.predict(x))
    assert descriptor == state.soap_params


def test_joblib_reader_rejects_executable_reducers(tmp_path):
    class Executable:
        def __reduce__(self):
            return eval, ("1+1",)

    state = SOAPRidgeModel()
    state.sk_pipeline = Executable()
    path = tmp_path / "unsafe.joblib"
    joblib.dump(state, path)
    with pytest.raises(EvidenceError, match="Non-numerical pickle global"):
        read_pipeline(path)


def test_conflicting_metric_declarations_fail():
    with pytest.raises(EvidenceError, match="Conflicting model selection"):
        selection_metric({"scoring": "MAE", "cv": {"metric": "RMSE"}}, [])


@pytest.mark.parametrize("defect", [False, True])
def test_coordinate_precision_replay_requires_matching_retained_geometry(defect):
    rng = np.random.default_rng(12345)
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    exact, frames = [], []
    for i in range(100):
        positions = ideal.positions + rng.normal(0, 0.01 + 0.001 * i, (64, 3))
        exact.append(positions)
        atoms = ideal.copy()
        atoms.positions = np.round(positions, 8)
        frames.append(atoms)
    if defect:
        frames[7].positions[0, 0] += 0.001
    before = [a.positions.copy() for a in frames]
    assert _matches_generated_dataset(
        "train", frames, 12345, restore_precision=True
    ) == (not defect)
    for frame, positions in zip(frames, before if defect else exact, strict=True):
        np.testing.assert_array_equal(frame.positions, positions)
