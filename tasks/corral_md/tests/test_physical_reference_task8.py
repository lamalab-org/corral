"""Apply the historical prediction threshold to both current pipeline formats."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from corral_md.scoring import task_8
from corral_md.scoring.physical_reference import REFERENCE_COMMIT

spec = importlib.util.spec_from_file_location(
    "physical_reference_saved_fixtures",
    Path(__file__).with_name("test_scoring_saved_pipelines.py"),
)
saved = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = saved
spec.loader.exec_module(saved)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return saved.fixtures.runtime.__wrapped__(tmp_path, monkeypatch)


def submission(root, model, recording_format):
    if recording_format == "saved":
        return saved.stage_submission(root, model)
    return saved.make_submission(root, model)[0]


@pytest.mark.parametrize("recording_format", ["inline", "saved"])
def test_prediction_reference_is_bound_to_reconstructed_heldout_data(
    tmp_path, runtime, recording_format
):
    report = saved.scorer().evaluate(submission(tmp_path, runtime, recording_format))
    assert report["score"] == 1, report
    reference = report["calculated_results"]["physical_reference"]
    assert reference == {
        "minimum_r2": 0.98,
        "source_commit": REFERENCE_COMMIT,
        "evaluation_basis": "reconstructed_independent_in_distribution_test",
    }
    assert sum(check["points"] for check in report["checks"]) == 100


@pytest.mark.parametrize("recording_format", ["inline", "saved"])
def test_honestly_reported_poor_model_fails_after_independent_verification(
    tmp_path, runtime, monkeypatch, recording_format
):
    # This nonlinear teacher is independently evaluated, but the exported
    # linear model cannot predict it well from the synthetic descriptor.
    monkeypatch.setattr(
        saved.fixtures,
        "energy",
        lambda atoms: float(
            -300 + np.sin(atoms.positions[:, 0].mean() * 1000) + np.var(atoms.positions)
        ),
    )
    report = saved.scorer().evaluate(submission(tmp_path, runtime, recording_format))
    assert report["score"] == 0, report
    assert report["calculated_results"]["id_test"]["r2"] < 0.98
    assert report["verification"]["status"] == "passed"
    assert report["checks"][-1]["name"] == "physical_reference_r2"
    assert report["checks"][-1]["status"] == "failed"


@pytest.mark.parametrize("recording_format", ["inline", "saved"])
@pytest.mark.parametrize("r2", [0.98, 0.979999, float("nan"), float("inf")])
def test_reconstructed_r2_boundary_and_nonfinite_values(
    tmp_path, runtime, monkeypatch, recording_format, r2
):
    path = submission(tmp_path, runtime, recording_format)
    path.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    if recording_format == "saved":
        rows = [
            row for row in saved.read_log(tmp_path) if row["stage"] != "test_evaluation"
        ]
        saved.fixtures.write_log(tmp_path, rows)
    original = task_8.metrics

    def reconstructed(reference, prediction):
        result = original(reference, prediction)
        if len(reference) == 100:
            result["r2"] = r2
        return result

    monkeypatch.setattr(task_8, "metrics", reconstructed)
    report = saved.scorer().evaluate(path)
    assert report["score"] == (1 if r2 == 0.98 else 0), report
    assert report["checks"][-1]["name"] == "physical_reference_r2"
    assert report["checks"][-1]["status"] == ("passed" if r2 == 0.98 else "failed")


def test_level1_has_no_prediction_quality_requirement(tmp_path, runtime):
    path, _ = saved.make_submission(tmp_path, runtime, level=1)
    report = saved.scorer(level=1).evaluate(path)
    assert report["score"] == 1, report
    assert "physical_reference" not in report["calculated_results"]
    assert all(check["name"] != "physical_reference_r2" for check in report["checks"])
    assert sum(check["points"] for check in report["checks"]) == 100
