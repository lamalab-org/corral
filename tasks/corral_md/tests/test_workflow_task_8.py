"""Task 8 scientific consistency checks with no inference or simulation execution."""

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1 import evaluate as evaluate_level1
from corral_md.workflow_scoring.regression import metrics, predict
from corral_md.workflow_scoring.task_8 import evaluate
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def _fit(x, y, row_ids, alpha=0.2):
    """Fixture construction only; production scoring never performs this fit."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < np.finfo(float).eps] = 1
    z = (x - mean) / scale
    intercept = float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
    return {
        "family": "Ridge",
        "descriptor": {
            "family": "SOAP",
            "parameters": {"r_cut": 5, "n_max": 3, "l_max": 2, "average": "outer"},
        },
        "preprocessing": "StandardScaler",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "alpha": alpha,
        "fit_intercept": True,
        "training_row_ids": list(row_ids),
    }


@pytest.fixture
def submission(tmp_path):
    base = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    datasets, artifacts, features, targets = {}, {}, {}, {}
    for seed, name in enumerate(("train", "id_test", "strained_test"), 101):
        rng = np.random.default_rng(seed)
        row_ids = [f"{name}-{i}" for i in range(100)]
        sigma = (
            0.01 + 0.001 * np.arange(100)
            if name != "strained_test"
            else np.full(100, 0.05)
        )
        strain = (
            np.zeros(100)
            if name != "strained_test"
            else np.repeat([-0.02, -0.01, 0, 0.01, 0.02], 20)
        )
        x = rng.normal(size=(100, 4)) + (0 if name == "train" else 3)
        y = -200 + x @ np.array([1.2, -0.4, 0.8, 0.1]) + rng.normal(scale=0.1, size=100)
        # Bad but honestly evaluated transfer must earn metric credit, even negative R2.
        if name != "train":
            y += 10
        frames = [
            {
                "symbols": ["Si"] * 64,
                "positions": (
                    base.positions * (1 + strain[i])
                    + rng.normal(scale=sigma[i], size=(64, 3))
                ).tolist(),
                "cell": (base.cell.array * (1 + strain[i])).tolist(),
                "pbc": [True] * 3,
                "energy": float(y[i]),
                "info": {"row_id": row_ids[i]},
            }
            for i in range(100)
        ]
        artifacts[f"{name}_structures"] = _write(tmp_path / f"{name}.json", frames)
        datasets[name] = {
            "dataset_id": name,
            "row_ids": row_ids,
            "features": x.tolist(),
            "energy_eV": y.tolist(),
            "sigma_A": sigma.tolist(),
            "strain": strain.tolist(),
            "generation_index": list(range(100)),
        }
        features[name], targets[name] = x, y
    train_rows = datasets["train"]["row_ids"]
    final = _fit(features["train"], targets["train"], train_rows)
    artifacts["portable_model"] = _write(tmp_path / "model.json", final)
    for name, dataset in datasets.items():
        dataset["predictions_eV"] = predict(features[name], final).tolist()
    artifacts["regression_data"] = _write(tmp_path / "datasets.json", datasets)
    records = []
    for fold in range(2):
        test = np.arange(fold, 100, 2)
        train = np.arange(1 - fold, 100, 2)
        model = _fit(
            features["train"][train],
            targets["train"][train],
            [train_rows[i] for i in train],
        )
        p = predict(features["train"][test], model)
        records.append(
            {
                "candidate_id": "ridge-a",
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "model": model,
                "predictions_eV": p.tolist(),
                "metrics": metrics(targets["train"][test], p),
            }
        )
    artifacts["validation"] = _write(
        tmp_path / "validation.json",
        {
            "records": records,
            "selected_candidate_id": "ridge-a",
            "selection_criterion": "Minimum mean held-out RMSE among documented candidates.",
            "candidates": {"ridge-a": {"alpha": 0.2}},
        },
    )
    curve, curve_results = [], []
    for n in (20, 40):
        indices = np.linspace(0, 99, n).astype(int)
        value = _fit(
            features["train"][indices],
            targets["train"][indices],
            [train_rows[i] for i in indices],
        )
        p = predict(features["id_test"], value)
        score = metrics(targets["id_test"], p)
        curve.append(
            {
                "train_indices": indices.tolist(),
                "model": value,
                "predictions_eV": p.tolist(),
                "metrics": score,
            }
        )
        curve_results.append({"size": n, **score})
    artifacts["learning_curve"] = _write(
        tmp_path / "curve.json",
        {
            "design": "Evenly spaced indices spanning the full sigma schedule.",
            "records": curve,
        },
    )
    script = tmp_path / "infer.py"
    script.write_text("""import json
from pathlib import Path
import numpy as np
from dscribe.descriptors import SOAP

def predict(structures):
    model = json.loads(Path(__file__).with_name("pipeline.json").read_text())
    soap = SOAP(species=["Si"], periodic=True, **model["descriptor"]["parameters"])
    x = np.atleast_2d(soap.create(structures))
    x = (x - model["mean"]) / model["scale"]
    return x @ np.asarray(model["coef"]) + model["intercept"]
""")
    checkpoint = tmp_path / "pipeline.json"
    _write(checkpoint, final)
    artifacts["pipeline_checkpoint"] = str(checkpoint)
    artifacts["inference_entrypoint"] = str(script)
    settings = {
        "teacher_model": "/models/teacher.model",
        "teacher_sha256": "a" * 64,
        "energy_unit": "eV",
        "length_unit": "Angstrom",
        "training_label_ids": train_rows,
        "generation": {
            name: {"seed": seed, "library": "numpy", "method": "Generator.normal"}
            for seed, name in enumerate(datasets, 101)
        },
    }
    strains = np.asarray(datasets["strained_test"]["strain"])
    p = np.asarray(datasets["strained_test"]["predictions_eV"])
    results = {
        "id_test": metrics(targets["id_test"], datasets["id_test"]["predictions_eV"]),
        "strained_pooled": metrics(targets["strained_test"], p),
        "strain_groups": [
            {
                "strain": strain,
                **metrics(
                    targets["strained_test"][strains == strain], p[strains == strain]
                ),
            }
            for strain in [-0.02, -0.01, 0, 0.01, 0.02]
        ],
        "learning_curve": curve_results,
    }
    report = tmp_path / "report.md"
    report.write_text(
        "Large transfer errors and negative R2 do not establish a universally poor descriptor. These values describe this dataset and model only."
    )
    manifest = {
        "artifacts": artifacts,
        "results": results,
        "settings": _write(tmp_path / "settings.json", settings),
        "scripts": [str(script)],
        "report": str(report),
    }
    return Path(_write(tmp_path / "manifest.json", manifest))


def _score(path):
    rubric = Rubric(8)
    evaluate(Evidence(path), rubric)
    return rubric


def _check(rubric, name):
    return next(check for check in rubric.checks if check["name"] == name)


def _edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    _write(path, data)


@pytest.mark.parametrize(
    "source",
    [
        "# implement inference later\n",
        "raise RuntimeError('broken inference')\n",
        "def predict(structures):\n    raise NotImplementedError\n",
        "def predict(:\n",
        "return 42\n",
        "if __name__ == '__main__':\n    raise RuntimeError('broken')\n",
    ],
)
def test_task8_broken_inference_loses_pipeline_credit(submission, source):
    (submission.parent / "infer.py").write_text(source)
    rubric = _score(submission)
    assert _check(rubric, "portable_pipeline_and_dimensions")["status"] == "failed"
    assert _check(rubric, "in_distribution_metrics")["status"] == "passed"


@pytest.mark.parametrize(
    "payload",
    [
        b"placeholder",
        b'"placeholder"',
        b"{}",
        pickle.dumps("placeholder"),
        b"\x80\x04",
        pickle.dumps("SOAP Ridge coef_ intercept_ StandardScaler mean_ scale_"),
    ],
)
def test_task8_placeholder_checkpoints_lose_pipeline_credit(submission, payload):
    (submission.parent / "pipeline.json").write_bytes(payload)
    assert _check(_score(submission), "portable_pipeline_and_dimensions")["earned"] == 0


def test_task8_checkpoint_must_match_exported_state(submission):
    _edit(submission.parent / "pipeline.json", lambda d: d.__setitem__("intercept", 0))
    assert (
        _check(_score(submission), "portable_pipeline_and_dimensions")["status"]
        == "failed"
    )


def test_task8_cli_exit_is_not_a_raise_only_stub(submission):
    path = submission.parent / "infer.py"
    path.write_text(
        path.read_text()
        + """
def main():
    import sys
    from ase.io import read
    print(predict(read(sys.argv[1], index=":")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
    )
    assert (
        _check(_score(submission), "portable_pipeline_and_dimensions")["status"]
        == "passed"
    )


def test_task8_pickle_metadata_is_inspected_without_loading(submission, monkeypatch):
    model = json.loads((submission.parent / "model.json").read_text())
    ridge = Ridge(alpha=model["alpha"])
    ridge.coef_, ridge.intercept_ = np.asarray(model["coef"]), model["intercept"]
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.asarray(model["mean"]), np.asarray(model["scale"])
    checkpoint = {
        "descriptor": model["descriptor"],
        "scaler": scaler,
        "estimator": ridge,
    }
    (submission.parent / "pipeline.json").write_bytes(pickle.dumps(checkpoint))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Checkpoint inspection must never deserialize")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    assert (
        _check(_score(submission), "portable_pipeline_and_dimensions")["status"]
        == "passed"
    )


def test_task8_learning_curve_needs_subsets_but_not_design_prose(submission):
    _edit(submission, lambda d: d.pop("report"))
    path = json.loads(submission.read_text())["artifacts"]["learning_curve"]
    _edit(Path(path), lambda d: d.pop("design"))
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_task8_consistent_artifacts_full_task_points_no_execution(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Scoring must not execute, fit, or deserialize")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(c["points"] for c in result.checks) == 90
    assert _check(result, "execution_and_teacher_provenance")["status"] == "unverified"
    assert (
        "does not prove independent RNG draws"
        in _check(result, "distortion_statistical_plausibility")["detail"]
    )
    assert json.loads(submission.read_text())["results"]["id_test"]["r2"] < 0


def test_level1_accepts_row_aligned_energy_table_without_duplicate_frame_energy(submission):
    manifest = json.loads(submission.read_text())
    path = Path(manifest["artifacts"]["train_structures"])
    frames = json.loads(path.read_text())
    for frame in frames:
        frame.pop("energy")
    _write(path, frames)

    rubric = Rubric(8, fail_fast=True)
    evaluate_level1(Evidence(submission), rubric, 8)
    assert _check(rubric, "training_structure_geometry")["status"] == "passed"

    frames[0]["energy"] = 0.0
    _write(path, frames)
    rubric = Rubric(8, fail_fast=True)
    evaluate_level1(Evidence(submission), rubric, 8)
    assert _check(rubric, "training_structure_geometry")["status"] == "failed"


def test_level1_accepts_ordered_frames_without_duplicate_row_ids(submission):
    manifest = json.loads(submission.read_text())
    path = Path(manifest["artifacts"]["train_structures"])
    frames = json.loads(path.read_text())
    for frame in frames:
        frame["info"].pop("row_id")
    _write(path, frames)

    rubric = Rubric(8, fail_fast=True)
    evaluate_level1(Evidence(submission), rubric, 8)
    assert _check(rubric, "training_structure_geometry")["status"] == "passed"

    frames[0], frames[1] = frames[1], frames[0]
    _write(path, frames)
    rubric = Rubric(8, fail_fast=True)
    evaluate_level1(Evidence(submission), rubric, 8)
    assert _check(rubric, "training_structure_geometry")["status"] == "failed"


@pytest.mark.parametrize(
    ("case", "failed"),
    [
        ("geometry", "strained_geometry"),
        ("label_order", "aligned_labels_recorded_inputs_and_rng"),
        ("metric", "in_distribution_metrics"),
        ("strain_metric", "strain_resolved_metrics"),
        ("pooled_metric", "pooled_strained_metrics"),
        ("leakage_scaler", "training_only_preprocessing"),
        ("overlap", "training_partitions_and_label_budget"),
        ("coefficient", "ridge_coefficient_consistency"),
        ("nonfinite", "portable_pipeline_and_dimensions"),
        ("training_budget", "training_partitions_and_label_budget"),
        ("curve_prefix", "learning_curve_distortion_coverage"),
        ("curve_prediction", "learning_curve_predictions_and_metrics"),
        ("curve_coef", "learning_curve_training_only_models"),
        ("curve_size", "reported_learning_curve"),
        ("same_seed", "aligned_labels_recorded_inputs_and_rng"),
        ("not_gaussian", "distortion_statistical_plausibility"),
        ("missing", "validation_predictions_metrics_and_selection"),
    ],
)
def test_task8_detects_inconsistent_or_missing_evidence(submission, case, failed):
    directory = submission.parent
    if case == "geometry":
        _edit(
            directory / "strained_test.json",
            lambda d: d[0]["cell"][0].__setitem__(0, 10.86),
        )
    elif case == "label_order":
        _edit(
            directory / "datasets.json", lambda d: d["id_test"]["energy_eV"].reverse()
        )
    elif case in {"metric", "strain_metric", "pooled_metric"}:

        def change(d):
            row = d["results"][
                {
                    "metric": "id_test",
                    "pooled_metric": "strained_pooled",
                    "strain_metric": "strain_groups",
                }[case]
            ]
            if isinstance(row, list):
                row = row[0]
            row["rmse"] = 0

        _edit(submission, change)
    elif case == "leakage_scaler":
        final = json.loads((directory / "model.json").read_text())
        _edit(
            directory / "validation.json",
            lambda d: d["records"][0]["model"].update(
                {"mean": final["mean"], "scale": final["scale"]}
            ),
        )
    elif case == "overlap":
        _edit(
            directory / "validation.json",
            lambda d: d["records"][0]["train_indices"].append(
                d["records"][0]["test_indices"][0]
            ),
        )
    elif case == "coefficient":
        _edit(directory / "model.json", lambda d: d["coef"].__setitem__(0, 100))
    elif case == "nonfinite":
        _edit(
            directory / "model.json", lambda d: d["coef"].__setitem__(0, float("nan"))
        )
    elif case == "training_budget":
        _edit(
            directory / "settings.json",
            lambda d: d["training_label_ids"].append("extra-teacher-label"),
        )
    elif case == "curve_prefix":

        def change(d):
            row = d["records"][0]
            row["train_indices"] = list(range(20))
            row["model"]["training_row_ids"] = [f"train-{i}" for i in range(20)]

        _edit(directory / "curve.json", change)
    elif case == "curve_prediction":
        _edit(
            directory / "curve.json",
            lambda d: d["records"][0]["predictions_eV"].__setitem__(0, 100),
        )
    elif case == "curve_coef":
        _edit(
            directory / "curve.json",
            lambda d: d["records"][0]["model"]["coef"].__setitem__(0, 100),
        )
    elif case == "curve_size":
        _edit(
            submission,
            lambda d: d["results"]["learning_curve"][0].__setitem__("size", 99),
        )
    elif case == "same_seed":
        _edit(
            directory / "settings.json",
            lambda d: d["generation"]["id_test"].__setitem__("seed", 101),
        )
    elif case == "not_gaussian":
        base = (
            bulk("Si", "diamond", a=5.43, cubic=True)
            .repeat((2, 2, 2))
            .positions.tolist()
        )
        _edit(
            directory / "train.json",
            lambda d: [row.__setitem__("positions", base) for row in d],
        )
    else:
        (directory / "validation.json").unlink()
    result = _score(submission)
    if case == "curve_prefix":
        assert _check(result, failed)["status"] == "unverified", result.checks
        assert result.score is None
    else:
        assert _check(result, failed)["status"] == "failed", result.checks
        assert result.score < 0.9


def test_task8_accepts_structure_frame_atom_and_dataset_row_permutations(submission):
    directory = submission.parent
    for name in ("train", "id_test", "strained_test"):

        def change_frames(frames):
            frames.reverse()
            for frame in frames:
                frame["positions"].reverse()

        _edit(directory / f"{name}.json", change_frames)

    # Reorder test array rows independently; alignment uses IDs, not trajectory index.
    def change_data(data):
        for name in ("id_test", "strained_test"):
            for value in data[name].values():
                if isinstance(value, list):
                    value.reverse()

    _edit(directory / "datasets.json", change_data)
    # Learning-curve predictions have the ID array order too.
    _edit(
        directory / "curve.json",
        lambda d: [row["predictions_eV"].reverse() for row in d["records"]],
    )
    assert _score(submission).score == pytest.approx(0.9)


def test_task8_unsupported_preprocessing_is_unverified_not_silently_identity(
    submission,
):
    _edit(
        submission.parent / "model.json",
        lambda d: d.__setitem__("preprocessing", "KernelPCA"),
    )
    result = _score(submission)
    assert _check(result, "training_only_preprocessing")["status"] == "unverified"
    assert _check(result, "in_distribution_metrics")["status"] == "unverified"
    assert _check(result, "ridge_coefficient_consistency")["status"] == "unverified"
    assert result.score is None


def test_task8_empty_submission_earns_no_task_points(tmp_path):
    path = Path(_write(tmp_path / "manifest.json", {"results": {}, "artifacts": {}}))
    result = _score(path)
    assert result.score == 0
    assert sum(check["points"] for check in result.checks) == 90


def _alternative_pipeline(
    submission, kind, *, positive=False, uneven_curve=False, first_subset=None
):
    """Construct complete independently fitted alternative evidence fixtures."""

    directory = submission.parent
    data = json.loads((directory / "datasets.json").read_text())
    x = np.asarray(data["train"]["features"])
    y = np.asarray(data["train"]["energy_eV"])
    rows = data["train"]["row_ids"]

    def fitted(indices):
        transformer = (
            MinMaxScaler(feature_range=(-1, 2))
            if kind == "MinMaxScaler"
            else PCA(n_components=2, whiten=True)
        )
        z = transformer.fit_transform(x[indices])
        ridge = Ridge(alpha=0.2, positive=positive, tol=1e-12).fit(z, y[indices])
        model = _fit(x[indices], y[indices], [rows[i] for i in indices])
        model.update(
            preprocessing=kind,
            coef=ridge.coef_.tolist(),
            intercept=float(ridge.intercept_),
            positive=positive,
        )
        model.pop("mean")
        model.pop("scale")
        if kind == "MinMaxScaler":
            model.update(
                data_min=transformer.data_min_.tolist(),
                data_max=transformer.data_max_.tolist(),
                scale=transformer.scale_.tolist(),
                offset=transformer.min_.tolist(),
                feature_range=[-1, 2],
            )
        else:
            model.update(
                mean=transformer.mean_.tolist(),
                components=transformer.components_.tolist(),
                explained_variance=transformer.explained_variance_.tolist(),
                whiten=True,
            )
        return model

    final = fitted(np.arange(100))
    for file in ("model.json", "pipeline.json"):
        _write(directory / file, final)
    for values in data.values():
        values["predictions_eV"] = predict(values["features"], final).tolist()
    _write(directory / "datasets.json", data)
    validation = json.loads((directory / "validation.json").read_text())
    for record in validation["records"]:
        record["model"] = fitted(record["train_indices"])
        indices = np.asarray(record["test_indices"])
        p = predict(x[indices], record["model"])
        record.update(predictions_eV=p.tolist(), metrics=metrics(y[indices], p))
    _write(directory / "validation.json", validation)
    curve = json.loads((directory / "curve.json").read_text())
    results = json.loads(submission.read_text())
    curve_results = []
    for index, record in enumerate(curve["records"]):
        if uneven_curve and index == 0:
            # Broad range with uneven counts and one empty quartile: [9,5,0,7].
            record["train_indices"] = [
                *range(0, 25, 3),
                *range(25, 50, 5),
                *range(75, 100, 4),
            ]
        if first_subset is not None and index == 0:
            record["train_indices"] = first_subset
        record["model"] = fitted(record["train_indices"])
        p = predict(data["id_test"]["features"], record["model"])
        record.update(
            predictions_eV=p.tolist(), metrics=metrics(data["id_test"]["energy_eV"], p)
        )
        curve_results.append(
            {"size": len(record["train_indices"]), **record["metrics"]}
        )
    _write(directory / "curve.json", curve)
    result = results["results"]
    result["learning_curve"] = curve_results
    for key, name in (("id_test", "id_test"), ("strained_pooled", "strained_test")):
        result[key] = metrics(data[name]["energy_eV"], data[name]["predictions_eV"])
    strain = np.asarray(data["strained_test"]["strain"])
    sy = np.asarray(data["strained_test"]["energy_eV"])
    sp = np.asarray(data["strained_test"]["predictions_eV"])
    result["strain_groups"] = [
        {"strain": float(s), **metrics(sy[strain == s], sp[strain == s])}
        for s in np.unique(strain)
    ]
    _write(submission, results)
    source = (directory / "infer.py").read_text()
    transform = (
        'x = x * model["scale"] + model["offset"]'
        if kind == "MinMaxScaler"
        else 'x = (x - model["mean"]) @ np.asarray(model["components"]).T / np.sqrt(model["explained_variance"])'
    )
    (directory / "infer.py").write_text(
        source.replace('x = (x - model["mean"]) / model["scale"]', transform)
    )


@pytest.mark.parametrize(
    ("kind", "positive"), [("MinMaxScaler", False), ("PCA", False), ("PCA", True)]
)
def test_complete_alternative_preprocessing_and_uneven_curve_earn_equal_credit(
    submission, kind, positive
):
    _alternative_pipeline(submission, kind, positive=positive, uneven_curve=True)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert _check(result, "learning_curve_sampling_adequacy")["status"] == "unverified"


@pytest.mark.parametrize("kind", ["MinMaxScaler", "PCA"])
def test_alternative_preprocessing_rejects_fitting_on_validation_holdout(
    submission, kind
):
    _alternative_pipeline(submission, kind)
    final = json.loads((submission.parent / "model.json").read_text())
    state_keys = (
        ("scale", "offset", "data_min", "data_max")
        if kind == "MinMaxScaler"
        else ("mean", "components", "explained_variance")
    )
    _edit(
        submission.parent / "validation.json",
        lambda d: d["records"][0]["model"].update({k: final[k] for k in state_keys}),
    )
    assert (
        _check(_score(submission), "training_only_preprocessing")["status"] == "failed"
    )


def test_missing_supported_pca_state_fails_instead_of_requesting_review(submission):
    _edit(
        submission.parent / "model.json",
        lambda d: d.__setitem__("preprocessing", "PCA"),
    )
    assert (
        _check(_score(submission), "training_only_preprocessing")["status"] == "failed"
    )


@pytest.mark.parametrize(
    "indices",
    [
        list(range(0, 100, 2)),
        list(range(1, 100, 2)),
        list(range(1, 99, 2)),
        list(range(2, 99, 3)),
        list(range(99)),
        list(range(1, 100)),
        list(range(1, 99)),
        np.random.default_rng(103).choice(100, size=20, replace=False).tolist(),
    ],
    ids=[
        "even",
        "odd",
        "both-endpoints-omitted",
        "stride-three",
        "near-complete-prefix",
        "near-complete-suffix",
        "near-complete-interior",
        "random",
    ],
)
def test_broad_alternative_subsets_earn_full_credit_without_required_endpoints(
    submission, indices
):
    _alternative_pipeline(submission, "PCA", first_subset=indices)
    _edit(
        submission.parent / "curve.json",
        lambda d: d.update(design={"method": "saved_training_row_selection"}),
    )
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    check = _check(result, "learning_curve_distortion_coverage")
    assert check["status"] == "passed"
    assert "CDF distance=" in check["detail"]


@pytest.mark.parametrize(
    "indices",
    [
        list(range(20)),
        list(range(0, 40, 2)),
        [*range(17), 35, 65, 99],
    ],
    ids=["narrow-prefix", "narrow-decimation", "skew-with-both-endpoints"],
)
def test_ambiguous_distribution_needs_review_but_retains_numeric_checks(
    submission, indices
):
    _alternative_pipeline(submission, "PCA", first_subset=indices)
    result = _score(submission)
    assert result.score is None
    assert (
        _check(result, "learning_curve_distortion_coverage")["status"] == "unverified"
    )
    for name in (
        "learning_curve_sizes",
        "learning_curve_training_only_models",
        "learning_curve_predictions_and_metrics",
        "reported_learning_curve",
    ):
        assert _check(result, name)["status"] == "passed"


@pytest.mark.parametrize("corruption", ["missing_indices", "test_row", "leaked_scaler"])
def test_broad_curve_design_still_requires_reproducible_training_only_evidence(
    submission, corruption
):
    _alternative_pipeline(
        submission, "MinMaxScaler", first_subset=list(range(1, 99, 2))
    )

    def change(data):
        record = data["records"][0]
        if corruption == "missing_indices":
            record.pop("train_indices")
        elif corruption == "test_row":
            record["model"]["training_row_ids"][0] = "id_test-0"
        else:
            # Fitted extrema from all rows leak the subset's held-out rows.
            final = json.loads((submission.parent / "model.json").read_text())
            record["model"].update(
                {k: final[k] for k in ("scale", "offset", "data_min", "data_max")}
            )

    _edit(submission.parent / "curve.json", change)
    result = _score(submission)
    assert result.score < 0.9
    assert _check(result, "learning_curve_training_only_models")["status"] == "failed"


@pytest.mark.parametrize("dataset", ["train", "id_test"])
def test_documented_final_selection_can_differ_from_cv_candidates_without_test_access(
    submission, dataset
):
    directory = submission.parent
    _edit(
        directory / "validation.json",
        lambda d: d.update(
            final_selection={
                "method": "interpolated_regularization",
                "dataset": dataset,
                "selection_indices": list(range(100)),
                "calculations": {
                    "alpha_bracket": [0.1, 0.3],
                    "interpolated_alpha": 0.2,
                },
            }
        ),
    )
    # The CV candidates differ from the independently fitted final model.
    # Refit their models and keep their numerical validation evidence consistent.
    data = json.loads((directory / "datasets.json").read_text())
    x, y = np.asarray(data["train"]["features"]), np.asarray(data["train"]["energy_eV"])
    validation = json.loads((directory / "validation.json").read_text())
    validation["candidates"] = {}
    validation.pop("selected_candidate_id")
    for index, item in enumerate(validation["records"]):
        alpha = (0.1, 0.3)[index]
        item["candidate_id"] = f"ridge-{alpha}"
        validation["candidates"][item["candidate_id"]] = {"alpha": alpha}
        train, test = (
            np.asarray(item["train_indices"]),
            np.asarray(item["test_indices"]),
        )
        item["model"] = _fit(
            x[train],
            y[train],
            [data["train"]["row_ids"][i] for i in train],
            alpha=alpha,
        )
        prediction = predict(x[test], item["model"])
        item.update(
            predictions_eV=prediction.tolist(), metrics=metrics(y[test], prediction)
        )
    _write(directory / "validation.json", validation)
    result = _score(submission)
    expected = "unverified" if dataset == "train" else "failed"
    assert (
        _check(result, "validation_predictions_metrics_and_selection")["status"]
        == expected
    )
    assert _check(result, "training_only_preprocessing")["status"] == "passed"
    assert _check(result, "ridge_coefficient_consistency")["status"] == "passed"
