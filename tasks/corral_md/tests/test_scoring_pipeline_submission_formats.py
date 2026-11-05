"""Evaluate event logs with frozen manifests and nested SOAP pipeline state."""

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest
from corral_md.scoring.common import EvaluationError, EvidenceError
from corral_md.scoring.pipeline_data import read_pipeline
from corral_md.scoring.task_8 import _validation_splits

spec = importlib.util.spec_from_file_location(
    "saved_pipeline_fixtures",
    Path(__file__).with_name("test_scoring_saved_pipelines.py"),
)
saved = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = saved
spec.loader.exec_module(saved)
fixtures, read_log, scorer, stage_submission = (
    saved.fixtures,
    saved.read_log,
    saved.scorer,
    saved.stage_submission,
)


class SoapRidgePipeline:
    def predict(self, _):
        raise AssertionError("Submitted methods must never run during evaluation")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return fixtures.runtime.__wrapped__(tmp_path, monkeypatch)


def manifest_submission(root, model):
    manifest = stage_submission(root, model)
    rows = read_log(root)
    train, selection, final, curves, _freeze, tests, evaluation = rows
    parameters = final.pop("params")
    descriptor = {
        key: parameters[key] for key in ("r_cut", "n_max", "l_max", "sigma", "average")
    }
    final.update(soap=descriptor, scale=parameters["scale"], alpha=parameters["alpha"])
    for record in [final, *curves["pipelines"]]:
        path = root / record["file"]
        # These are evaluator-owned synthetic fixtures, never submitted models.
        old = pickle.loads(path.read_bytes())
        state = SoapRidgePipeline()
        state.soap_params = descriptor
        state.ridge_, state.scaler_ = old.ridge_, old.scaler_
        path.write_bytes(pickle.dumps(state))
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = {
        "frozen_at": 2,
        "final": final,
        "learning_curve": {str(row["n_train"]): row for row in curves["pipelines"]},
    }
    (root / "frozen.json").write_text(json.dumps(frozen))
    grid_path = root / selection["file"]
    grid = json.loads(grid_path.read_text())
    nested = [
        {
            "soap": {key: candidate[key] for key in descriptor},
            **{key: candidate[key] for key in ("scale", "alpha", "cv_rmse")},
        }
        for candidate in grid
    ]
    grid_path.write_text(
        json.dumps(
            {
                "cv": "RepeatedKFold(5 splits x 1 repeats, random_state=0), metric RMSE (eV)",
                "results": nested,
            }
        )
    )
    selection.pop("cv")
    selection["stage"] = "hyperparameter_search"
    log = [
        train,
        selection,
        {"stage": "pipelines_frozen", "manifest": "frozen.json", "final": final},
        {
            "stage": "test_labels",
            "files": [tests[key]["file"] for key in ("independent", "strained")],
            **{
                f"seed_{key}": tests[key]["seed"] for key in ("independent", "strained")
            },
        },
        {
            "stage": "test_evaluation",
            "final": {
                name: {key.upper(): value for key, value in evaluation[name].items()}
                for name in ("independent", "strained_pooled")
            },
        },
    ]
    for i, row in enumerate(log):
        row["event"] = row.pop("stage")
        row["time"] = i
    (root / "run.jsonl").write_text("\n".join(map(json.dumps, log)))
    return manifest


def test_manifest_freeze_and_nested_numerical_state_are_independently_scored(
    tmp_path, runtime
):
    submission = manifest_submission(tmp_path, runtime)
    report = scorer().evaluate(submission)
    assert report["score"] == 1, report
    assert "pending_checks" not in report
    assert report["verification"]["teacher_configurations"] == 300
    assert len(report["calculated_results"]["strain_groups"]) == 5
    assert [row["size"] for row in report["calculated_results"]["learning_curve"]] == [
        30,
        100,
    ]


@pytest.mark.parametrize("misordered", [False, True])
def test_frozen_manifest_needs_no_repeated_metrics_but_preserves_chronology(
    tmp_path, runtime, misordered
):
    submission = manifest_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    rows = [row for row in rows if row["event"] != "test_evaluation"]
    if misordered:
        rows[2], rows[3] = rows[3], rows[2]
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer().evaluate(submission)
    assert report["score"] == (0 if misordered else 1), report
    if not misordered:
        assert "id_test" in report["calculated_results"]
        assert len(report["calculated_results"]["learning_curve"]) == 2


@pytest.mark.parametrize(
    "defect", ["chronology", "post_test_fit", "hash", "missing", "descriptor", "metric"]
)
def test_manifest_adapter_preserves_scientific_failure_checks(
    tmp_path, runtime, defect
):
    submission = manifest_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    if defect == "chronology":
        rows[2], rows[3] = rows[3], rows[2]
    elif defect == "post_test_fit":
        rows.append({"event": "final_fit", "file": "final.pkl"})
    elif defect == "hash":
        frozen["learning_curve"]["30"]["sha256"] = "0" * 64
    elif defect == "missing":
        (tmp_path / "lc_30.pkl").unlink()
    elif defect == "descriptor":
        frozen["final"]["soap"]["average"] = "outer"
        rows[2]["final"] = frozen["final"]
    else:
        rows[-1]["final"]["independent"]["RMSE"] += 1
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer().evaluate(submission)
    assert report["score"] == 0, report
    assert "pending_checks" not in report


def redundant_fit_records(rows, frozen):
    final = frozen["final"]
    fits = [
        {
            "event": "final_fit",
            "time": 1.25,
            "file": final["file"],
            "sha256": final["sha256"],
            "params": {
                **final["soap"],
                "scale": final["scale"],
                "alpha": final["alpha"],
            },
        },
        {
            "event": "learning_curve_fits",
            "time": 1.5,
            "pipelines": [
                {**row, "n_train": int(count)}
                for count, row in frozen["learning_curve"].items()
            ],
        },
    ]
    rows[2]["frozen"] = {
        row["file"]: row["sha256"]
        for row in [final, *frozen["learning_curve"].values()]
    }
    rows[2:2] = fits
    return fits


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "reordered_indices",
        "freeze_digest",
        "final_digest",
        "final_settings",
        "curve_digest",
        "curve_indices",
    ],
)
def test_redundant_fit_records_cannot_override_contradictory_evidence(
    tmp_path, runtime, defect
):
    submission = manifest_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    final, curves = redundant_fit_records(rows, frozen)
    if defect == "freeze_digest":
        rows[4]["frozen"]["final.pkl"] = "0" * 64
    elif defect == "final_digest":
        final["sha256"] = "0" * 64
    elif defect == "final_settings":
        final["params"]["alpha"] += 1
    elif defect == "curve_digest":
        curves["pipelines"][0]["sha256"] = "0" * 64
    elif defect == "curve_indices":
        curves["pipelines"][0]["indices"][0] = 1
    elif defect == "reordered_indices":
        curves["pipelines"][0]["indices"] = list(
            reversed(curves["pipelines"][0]["indices"])
        )
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer().evaluate(submission)
    assert report["score"] == (1 if defect in (None, "reordered_indices") else 0), (
        report
    )
    assert "pending_checks" not in report


@pytest.mark.parametrize("seed", [0, 17])
def test_redundant_validation_specs_are_compared_semantically(tmp_path, runtime, seed):
    submission = manifest_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    rows[1]["cv"] = {
        "method": "RepeatedKFold",
        "n_splits": 5,
        "n_repeats": 1,
        "random_state": seed,
    }
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer().evaluate(submission)
    assert report["score"] == (1 if seed == 0 else 0), report
    assert "pending_checks" not in report


def test_repeated_fold_prose_keeps_repetition_and_seed():
    import numpy as np
    from sklearn.model_selection import RepeatedKFold

    actual = _validation_splits(
        {"cv": "RepeatedKFold(5 splits x 3 repeats, random_state=17)"}, 100
    )
    expected = list(
        RepeatedKFold(n_splits=5, n_repeats=3, random_state=17).split(np.arange(100))
    )
    assert len(actual) == 15
    assert all(
        np.array_equal(a, b)
        for pair, reference in zip(actual, expected, strict=True)
        for a, b in zip(pair, reference, strict=True)
    )


def test_pipeline_wrapper_does_not_allow_executable_pickle_reducers(tmp_path):
    class Executable:
        def __reduce__(self):
            return eval, ("1+1",)

    state = SoapRidgePipeline()
    state.ridge_ = Executable()
    path = tmp_path / "pipeline.pkl"
    path.write_bytes(pickle.dumps(state))
    with pytest.raises(EvidenceError, match="Non-numerical pickle global"):
        read_pipeline(path)


def seeded_submission(root, model):
    import numpy as np

    submission, _ = fixtures.make_submission(root, model)
    header, dataset, *_ = read_log(root)
    rows = [
        {
            **header,
            "type": "metadata",
            "seed": dataset["seed"],
            "rng": "numpy.default_rng(seed).normal, one (64,3) draw per config in order",
        }
    ]
    rows[0].pop("event")
    for i, row in enumerate(dataset["rows"]):
        rows.append(
            {
                "type": "config",
                "config_index": i,
                "row_id": row["row_id"],
                "seed": dataset["seed"],
                "energy_eV": row["energy_eV"],
                "positions_sha256": hashlib.sha256(
                    np.asarray(row["positions"]).tobytes()
                ).hexdigest(),
            }
        )
    (root / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    submission.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    return submission


@pytest.mark.parametrize("pbc", [True, [True, True, True], [True, True, False]])
def test_seeded_level1_dataset_does_not_require_unrequested_coordinate_arrays(
    tmp_path, runtime, pbc
):
    submission = seeded_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    rows[0]["pbc"] = pbc
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer(level=1).evaluate(submission)
    assert report["score"] == (0 if pbc == [True, True, False] else 1), report
    if report["score"] == 1:
        assert report["verification"]["status"] == "passed"


def test_seeded_level1_dataset_needs_no_repeated_checkpoint_identity(tmp_path, runtime):
    submission = seeded_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    del rows[0]["model_sha256"]
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer(level=1).evaluate(submission)
    assert report["score"] == 1, report
    assert report["verification"]["status"] == "passed"


@pytest.mark.parametrize(
    "defect", ["energy", "hash", "order", "duplicate_id", "missing_row", "checkpoint"]
)
def test_seeded_level1_dataset_rejects_inconsistent_retained_evidence(
    tmp_path, runtime, defect
):
    submission = seeded_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    if defect == "energy":
        rows[1]["energy_eV"] += 1
    elif defect == "hash":
        rows[1]["positions_sha256"] = "0" * 64
    elif defect == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif defect == "duplicate_id":
        rows[2]["row_id"] = rows[1]["row_id"]
    elif defect == "missing_row":
        rows.pop()
    else:
        rows[0]["model_sha256"] = "0" * 64
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer(level=1).evaluate(submission)
    assert report["score"] == 0, report
    assert "pending_checks" not in report


DESCRIPTIVE_GENERATOR = (
    "numpy.default_rng(seed).normal(0, sigma, (64,3)) sequentially per config"
)


def coordinate_submission(root, model, method, layout="metadata_rows"):
    submission, _ = fixtures.make_submission(root, model, level=1)
    header, dataset, end = read_log(root)
    header.update(seed=dataset["seed"], rng=method)
    if layout == "dataset":
        dataset["rng"] = method
        rows = [header, dataset, end]
    else:
        rows = [
            {**header, **row, "event": "config", "config_index": i}
            for i, row in enumerate(dataset["rows"])
        ]
        if layout == "metadata_rows":
            rows = [header, *rows]
    (root / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return submission


@pytest.mark.parametrize("layout", ["metadata_rows", "dataset", "rows_only"])
@pytest.mark.parametrize(
    "method", [DESCRIPTIVE_GENERATOR, "NumPy Gaussian draws", None]
)
def test_retained_coordinates_verify_generator_without_required_label_spelling(
    tmp_path, runtime, method, layout
):
    submission = coordinate_submission(tmp_path, runtime, method, layout)
    report = scorer(level=1).evaluate(submission)
    assert report["score"] == 1, report
    assert report["verification"]["status"] == "passed"


@pytest.mark.parametrize(
    "defect", ["energy", "cell", "hash", "order", "known_generator"]
)
def test_retained_coordinates_still_reject_inconsistent_evidence(
    tmp_path, runtime, defect
):
    submission = coordinate_submission(tmp_path, runtime, DESCRIPTIVE_GENERATOR)
    rows = read_log(tmp_path)
    if defect == "energy":
        rows[1]["energy_eV"] += 1
    elif defect == "cell":
        rows[1]["cell"][0][0] += 1
    elif defect == "hash":
        rows[1]["positions_sha256"] = "0" * 64
    elif defect == "order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        # An explicit supported name remains a scientific claim to verify.
        rows[0]["rng"] = "numpy.random.RandomState"
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = scorer(level=1).evaluate(submission)
    assert report["score"] == 0, report
    assert "pending_checks" not in report


@pytest.mark.parametrize("defect", ["positions", "seed", "missing_positions"])
def test_descriptive_generator_needs_reproducible_retained_coordinates(
    tmp_path, runtime, defect
):
    submission = coordinate_submission(tmp_path, runtime, DESCRIPTIVE_GENERATOR)
    rows = read_log(tmp_path)
    if defect == "positions":
        rows[1]["positions"][0][0] += 0.01
    elif defect == "seed":
        for row in rows:
            row["seed"] += 1
    else:
        rows[1].pop("positions")
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(EvaluationError) as caught:
        scorer(level=1).evaluate(submission)
    report = caught.value.report
    assert "score" not in report
    assert report["unsupported_checks"] == ["calculation_log"]


def test_coordinate_free_description_still_needs_an_identified_generator(
    tmp_path, runtime
):
    submission = seeded_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    rows[0]["rng"] = DESCRIPTIVE_GENERATOR
    (tmp_path / "run.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(EvaluationError) as caught:
        scorer(level=1).evaluate(submission)
    report = caught.value.report
    assert "score" not in report
    assert "Coordinate-free" in report["checks"][0]["detail"]
