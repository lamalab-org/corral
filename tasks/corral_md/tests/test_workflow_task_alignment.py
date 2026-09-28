"""Task requirements survive changes to optional reporting and method choices."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from corral_md.score import check_level2_workflow
from corral_md.workflow_scoring.common import Rubric
from corral_md.workflow_scoring.task_7 import _npt_fit


def _submission(number, root):
    spec = importlib.util.spec_from_file_location(
        "alignment_fixtures",
        Path(__file__).with_name("test_workflow_numerical_policy.py"),
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return fixtures._submission(number, root)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data))


def check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


def test_task9_full_credit_without_prescribed_diagnostics(tmp_path):
    path = _submission(9, tmp_path)
    grader = check_level2_workflow(9)
    assert grader(path) == 1
    manifest = read(path)
    for run in manifest["results"]["runs"].values():
        for key in list(run):
            if key != "production_temperature_mean_K":
                del run[key]
    del manifest["results"]["comparisons"]
    del manifest["results"]["validation"]["comparison"]
    write(path, manifest)
    models = read(tmp_path / "models.json")
    for key in ("minimum_gap_frames", "minimum_gap_time_fs"):
        del models["validation"]["time_aware"][key]
    write(tmp_path / "models.json", models)
    # The task chooses no fixed number of saved equilibration states. Boundary
    # states, full thermal logs and all 200 production frames are still present.
    for run in manifest["artifacts"]["runs"].values():
        frames = read(run["equilibration"])
        write(run["equilibration"], [frames[0], frames[-1]])
    report = grader.evaluate(path)
    assert report["score"] == 1, report
    assert report["pending_checks"] == []


@pytest.mark.parametrize(
    ("location", "key", "failed"),
    [
        (
            ("results", "runs", "main"),
            "equilibration_first_half_temperature_K",
            "kinetic_temperature_equilibration_and_logs",
        ),
        (
            ("results", "comparisons"),
            "independent_500_minus_800_rmse_eV",
            "quantified_temporal_and_temperature_transfer_comparisons",
        ),
        (
            ("results", "validation", "comparison"),
            "training_count_ratio",
            "comparable_or_quantified_validation_budgets",
        ),
    ],
)
def test_task9_optional_claims_still_must_be_correct(tmp_path, location, key, failed):
    path = _submission(9, tmp_path)
    manifest = read(path)
    record = manifest
    for name in location:
        record = record[name]
    record[key] += 1000
    write(path, manifest)
    report = check_level2_workflow(9).evaluate(path)
    assert check(report, failed)["status"] == "failed"
    del record[key]
    write(path, manifest)
    assert check_level2_workflow(9)(path) == 1


def test_task9_omitting_diagnostics_does_not_hide_bad_logs_or_required_results(
    tmp_path,
):
    path = _submission(9, tmp_path)
    manifest = read(path)
    for run in manifest["results"]["runs"].values():
        for key in list(run):
            if key != "production_temperature_mean_K":
                del run[key]
    write(path, manifest)
    log_path = manifest["artifacts"]["runs"]["main"]["log"]
    logs = read(log_path)
    logs[0]["temperature_K"] += 100
    write(log_path, logs)
    grader = check_level2_workflow(9)
    name = "kinetic_temperature_equilibration_and_logs"
    assert check(grader.evaluate(path), name)["status"] == "failed"
    logs[0]["temperature_K"] -= 100
    write(log_path, logs)
    del manifest["results"]["runs"]["main"]["production_temperature_mean_K"]
    write(path, manifest)
    assert check(grader.evaluate(path), name)["status"] == "failed"


def test_task1_equilibrium_summaries_are_optional_but_trace_and_window_are_not(
    tmp_path,
):
    path = _submission(1, tmp_path)
    manifest = read(path)
    analysis_path = tmp_path / manifest["artifacts"]["analysis"]
    analysis = read(analysis_path)
    validation = analysis["validation"]
    for key in list(validation):
        if "mean" in key or "half_difference" in key:
            del validation[key]
    write(analysis_path, analysis)
    grader = check_level2_workflow(1)
    assert grader(path) == 1
    validation["pressure_half_difference_bar"] = 9999
    write(analysis_path, analysis)
    assert check(grader.evaluate(path), "equilibrated_production")["status"] == "failed"
    del validation["pressure_half_difference_bar"]
    del validation["equilibrium_window_ps"]
    write(analysis_path, analysis)
    assert check(grader.evaluate(path), "equilibrated_production")["status"] == "failed"


def test_task7_initialization_temperature_is_open(tmp_path):
    path = _submission(7, tmp_path)
    frames = read(tmp_path / "h300.json")
    frames[0]["momenta"] = (np.array(frames[0]["momenta"]) * 2).tolist()
    write(tmp_path / "h300.json", frames)
    trace = read(tmp_path / "thermal_trace.json")
    first = trace[0]
    delta_kinetic = 3 * first["kinetic_energy_eV"]
    first["kinetic_energy_eV"] *= 4
    first["temperature_K"] *= 4
    for axis in ("sxx", "syy", "szz"):
        first[axis] += 2 * delta_kinetic / (3 * first["volume_A3"])
    write(tmp_path / "thermal_trace.json", trace)
    settings = read(tmp_path / "settings.json")
    settings["initial_temperature_K"] = first["temperature_K"]
    write(tmp_path / "settings.json", settings)
    report = check_level2_workflow(7).evaluate(path)
    assert report["score"] == 1, report


def test_task7_total_stress_convention_preserves_pressure_results(tmp_path):
    path = _submission(7, tmp_path)
    trace = read(tmp_path / "thermal_trace.json")
    for row in trace:
        kinetic_pressure = 2 * row["kinetic_energy_eV"] / (3 * row["volume_A3"])
        for axis in ("sxx", "syy", "szz"):
            row[axis] -= kinetic_pressure
    write(tmp_path / "thermal_trace.json", trace)
    settings = read(tmp_path / "settings.json")
    settings["stress_kind"] = "total"
    write(tmp_path / "settings.json", settings)
    report = check_level2_workflow(7).evaluate(path)
    assert report["score"] == 1, report


def test_task7_one_sided_local_derivative_is_reviewed_not_forbidden():
    rubric = Rubric(7)
    rubric.check(
        "local_npt_expansion_fit",
        6,
        lambda: _npt_fit(
            {"410": [410, 101], "420": [420, 102]},
            {"method": "linear", "stage_ids": ["410", "420"]},
            center=400,
            volume=100,
        ),
    )
    assert rubric.checks[0]["status"] == "unverified"


def test_task8_teacher_digest_is_optional(tmp_path):
    path = _submission(8, tmp_path)
    settings = read(tmp_path / "settings.json")
    del settings["teacher_sha256"]
    write(tmp_path / "settings.json", settings)
    assert check_level2_workflow(8)(path) == 1


def test_task8_distribution_screen_requests_review_without_changing_other_credit(
    tmp_path,
):
    path = _submission(8, tmp_path)
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2)).positions
    frames = read(tmp_path / "train.json")
    for frame in frames:
        frame["positions"] = (
            ideal + 1.2 * (np.asarray(frame["positions"]) - ideal)
        ).tolist()
    write(tmp_path / "train.json", frames)
    report = check_level2_workflow(8).evaluate(path)
    assert report["pending_checks"] == ["distortion_statistical_plausibility"]
    assert report["score_bounds"] == [0, 1]
    assert not [c for c in report["checks"] if c["status"] == "failed"], report


def alternative_validation(tmp_path):
    """Save a real GCV calculation without inventing explicit validation folds."""
    path = _submission(8, tmp_path)
    manifest = read(path)
    data = read(manifest["artifacts"]["regression_data"])["train"]
    model = read(tmp_path / "model.json")
    x = (np.array(data["features"]) - model["mean"]) / model["scale"]
    y = np.array(data["energy_eV"])
    prediction = x @ model["coef"] + model["intercept"]
    singular_values = np.linalg.svd(x, compute_uv=False)
    effective_dof = 1 + np.sum(
        singular_values**2 / (singular_values**2 + model["alpha"])
    )
    gcv = np.mean((y - prediction) ** 2) / (1 - effective_dof / len(y)) ** 2
    doc = {
        "method": "generalized_cross_validation",
        "dataset": "train",
        "selection_indices": list(range(100)),
        "selection_criterion": "predeclared_alpha_with_gcv_assessment",
        "candidates": {"fixed": model},
        "calculations": {
            "prediction_eV": prediction.tolist(),
            "effective_dof": float(effective_dof),
            "gcv_eV2": float(gcv),
        },
    }
    write(tmp_path / "validation.json", doc)
    return path, doc


def test_task8_alternative_validation_is_reviewed_without_losing_other_credit(tmp_path):
    path, _ = alternative_validation(tmp_path)
    grader = check_level2_workflow(8)
    report = grader.evaluate(path)
    assert report["status"] == "pending_review"
    assert not [c for c in report["checks"] if c["status"] == "failed"], report
    assert set(report["pending_checks"]) == {
        "training_only_preprocessing",
        "ridge_coefficient_consistency",
        "validation_predictions_metrics_and_selection",
    }
    assert report["score_bounds"] == [0, 1]
    assert check(report, "in_distribution_metrics")["status"] == "passed"
    assert check(report, "training_partitions_and_label_budget")["status"] == "passed"


@pytest.mark.parametrize(
    "mutation", ["test_data", "bad_indices", "no_calculations", "wrong_final_fit"]
)
def test_task8_alternative_validation_cannot_hide_contradictions(tmp_path, mutation):
    path, doc = alternative_validation(tmp_path)
    if mutation == "test_data":
        doc["dataset"] = "id_test"
    elif mutation == "bad_indices":
        doc["selection_indices"][-1] = 100
    elif mutation == "no_calculations":
        doc["calculations"] = {}
    else:
        model = read(tmp_path / "model.json")
        model["coef"][0] += 10
        write(tmp_path / "model.json", model)
    write(tmp_path / "validation.json", doc)
    report = check_level2_workflow(8).evaluate(path)
    name = (
        "ridge_coefficient_consistency"
        if mutation == "wrong_final_fit"
        else "training_partitions_and_label_budget"
    )
    if mutation == "wrong_final_fit":
        assert check(report, name)["status"] == "skipped", report
        assert any(item["status"] == "failed" for item in report["checks"])
    else:
        assert check(report, name)["status"] == "failed", report
    assert report["score"] == 0


def test_inconclusive_physical_screen_is_resolved_only_by_evaluator(tmp_path):
    path = _submission(1, tmp_path)
    manifest = read(path)
    artifacts = manifest["artifacts"]
    thermo_path = tmp_path / artifacts["thermo"]
    thermo = pd.read_csv(thermo_path)
    thermo.loc[thermo.time_ps >= 1000, "pressure_bar"] = 2000
    thermo.to_csv(thermo_path, index=False)
    log_path = tmp_path / artifacts["lammps_log"]
    log_path.write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{row.step} {row.temperature_K} {row.pressure_bar} {row.density_g_cm3}"
            for row in thermo.itertuples()
        )
        + "\n"
    )
    analysis_path = tmp_path / artifacts["analysis"]
    analysis = read(analysis_path)
    analysis["validation"]["pressure_mean_bar"] = 2000
    write(analysis_path, analysis)
    grader = check_level2_workflow(1)
    report = grader.evaluate(path)
    assert report["score"] is None
    assert report["pending_checks"] == ["equilibrated_production"], report
    assert not [c for c in report["checks"] if c["status"] == "failed"], report
    review = {
        "evidence_sha256": report["evidence_sha256"],
        "decisions": {
            "equilibrated_production": {
                "passed": False,
                "reviewer": "test evaluator",
                "reason": "Synthetic review rejects the saved pressure-control evidence.",
            }
        },
    }
    assert grader.score_submission(path, review=review) == 0
