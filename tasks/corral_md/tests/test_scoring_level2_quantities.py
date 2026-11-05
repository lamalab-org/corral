"""Numerical agreement and independent physical targets both gate Level 2."""

import json

import numpy as np
import pytest
from ase import units
from corral_md.scoring import check_level2_workflow
from corral_md.scoring.level2_quantities import Level2QuantityScorer
from corral_md.scoring.physical_reference import (
    ALUMINUM_EXPANSION_PER_K,
    reference_spectra,
)
from corral_md.scoring.task_6 import spectral_summaries


def write_log(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def answer(tmp_path, task, rows, results):
    write_log(tmp_path / "run.jsonl", rows)
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps({"artifacts": {"run_log": "run.jsonl"}, "results": results})
    )
    return path, check_level2_workflow(task, verification_backend="offline")


def phonons(tmp_path):
    _, frequency, power = reference_spectra()[0]
    summaries = spectral_summaries(frequency, power)
    return answer(
        tmp_path,
        6,
        [{"type": "vdos", "freq_THz": frequency.tolist(), "vdos": power.tolist()}],
        {
            "mean_frequency_THz": summaries["mean_frequency_THz"],
            "std_frequency_THz": summaries["std_frequency_THz"],
        },
    )


def expansion(tmp_path):
    rows = []
    for i, t in enumerate([300, 350, 400, 450, 500, 400, 300]):
        for delta in [-1, 1]:
            rows.append(
                {
                    "stage": i,
                    "phase": "heat" if i < 5 else "cool",
                    "part": "prod",
                    "T_target": t,
                    "T": t + delta,
                    "V": 1800 * (1 + 1e-4 * (t + delta - 300)),
                }
            )
        rows.append(
            {
                "stage": i,
                "phase": "heat",
                "part": "equil",
                "T_target": t,
                "T": 1000,
                "V": 1e6,
            }
        )
    return answer(tmp_path, 7, rows, {"beta_volume": 1e-4})


def heat_capacity(tmp_path, method="ols"):
    rows = []
    for i, t in enumerate([300, 400, 500, 600, 700, 800, 900, 300]):
        for delta in [-1, 1]:
            rows.append(
                {
                    "stage": i,
                    "phase": "production",
                    "temperature_K": t + delta,
                    "energy_eV_per_atom": -4
                    + 3 * units.kB * (t + delta)
                    + (10 if i > 4 else 0),
                }
            )
        rows.append(
            {
                "stage": i,
                "phase": "equilibration",
                "temperature_K": 1000,
                "energy_eV_per_atom": 1000,
            }
        )
    fit = {"method": method}
    if method == "wls":
        fit["weights"] = [1, 2, 3, 4, 5]
    return answer(
        tmp_path,
        10,
        rows,
        {"heat_capacity": {"eV_per_atom_K": 3 * units.kB}, "fit": fit},
    )


@pytest.mark.parametrize("task", [6, 7, 9, 10])
@pytest.mark.parametrize("backend", ["offline", "local", "modal"])
def test_default_policy_needs_no_model_backend(task, backend):
    assert isinstance(
        check_level2_workflow(task, verification_backend=backend), Level2QuantityScorer
    )


@pytest.mark.parametrize("make", [phonons, expansion, heat_capacity])
def test_final_numerical_logs_need_no_workflow_metadata(tmp_path, make):
    path, scorer = make(tmp_path)
    report = scorer.evaluate(path)
    assert report["score"] == 1
    assert report["status"] == "complete"
    assert report["computed"]
    assert scorer.evaluate_submission(path).score == 1
    assert "pending_checks" not in report


def test_vdos_unnormalized_amplitude_does_not_change_physical_similarity(tmp_path):
    path, scorer = phonons(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[0]["vdos"] = (np.asarray(rows[0]["vdos"]) * 123).tolist()
    write_log(log, rows)
    report = scorer.evaluate(path)
    assert report["score"] == 1, report


def test_self_consistent_vdos_with_wrong_frequency_scale_fails(tmp_path):
    path, scorer = phonons(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[0]["freq_THz"] = (np.asarray(rows[0]["freq_THz"]) * 3).tolist()
    write_log(log, rows)
    data = json.loads(path.read_text())
    for key in ("mean_frequency_THz", "std_frequency_THz"):
        data["results"][key] *= 3
    path.write_text(json.dumps(data))
    report = scorer.evaluate(path)
    assert report["score"] == 0
    assert "cosine similarity" in report["checks"][0]["detail"]


@pytest.mark.parametrize(
    ("make", "key"),
    [
        (phonons, "mean_frequency_THz"),
        (expansion, "beta_volume"),
        (heat_capacity, "heat_capacity"),
    ],
)
@pytest.mark.parametrize("defect", ["wrong", "missing", "nonfinite"])
def test_wrong_or_missing_final_answer_fails(tmp_path, make, key, defect):
    path, scorer = make(tmp_path)
    data = json.loads(path.read_text())
    if defect == "missing":
        del data["results"][key]
    else:
        value = 10 if defect == "wrong" else float("nan")
        data["results"][key] = (
            {"eV_per_atom_K": value} if key == "heat_capacity" else value
        )
    path.write_text(json.dumps(data))
    assert scorer(path) == 0


@pytest.mark.parametrize("method", ["ols", "wls", "theil_sen"])
def test_heat_capacity_uses_exactly_five_forward_stages(tmp_path, method):
    path, scorer = heat_capacity(tmp_path, method)
    assert scorer(path) == 1


@pytest.mark.parametrize(
    ("scale", "expected"), [(0.9, 1), (1.1, 1), (0.89, 0), (1.11, 0)]
)
def test_heat_capacity_checks_all_seven_stage_temperature_targets(
    tmp_path, scale, expected
):
    path, scorer = heat_capacity(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    # Alter the 900 K stage only: the first-five-stage Cv and its reported value
    # remain exactly correct even when this physical target is missed.
    for row in rows:
        if row["stage"] == 6 and row["phase"] == "production":
            row["temperature_K"] *= scale
    write_log(log, rows)
    assert scorer(path) == expected


def test_heat_capacity_requires_seven_forward_stages(tmp_path):
    path, scorer = heat_capacity(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    write_log(log, [row for row in rows if row["stage"] < 5])
    assert scorer(path) == 0


def test_heat_capacity_ignores_extra_closing_stage_temperature(tmp_path):
    path, scorer = heat_capacity(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row["stage"] == 7:
            row["temperature_K"] = 1000
    write_log(log, rows)
    assert scorer(path) == 1


@pytest.mark.parametrize(
    ("scale", "expected"), [(0.9, 1), (1.1, 1), (0.89, 0), (1.11, 0)]
)
def test_expansion_rejects_self_consistent_but_wrong_physical_coefficient(
    tmp_path, scale, expected
):
    path, scorer = expansion(tmp_path)
    beta = ALUMINUM_EXPANSION_PER_K * scale
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row["part"] == "prod":
            row["V"] = 1800 * (1 + beta * (row["T"] - 300))
    write_log(log, rows)
    data = json.loads(path.read_text())
    data["results"]["beta_volume"] = beta
    path.write_text(json.dumps(data))
    assert scorer(path) == expected


def test_expansion_report_cannot_use_agreement_tolerance_to_exceed_reference(tmp_path):
    path, scorer = expansion(tmp_path)
    beta = ALUMINUM_EXPANSION_PER_K * 1.1
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row["part"] == "prod":
            row["V"] = 1800 * (1 + beta * (row["T"] - 300))
    write_log(log, rows)
    data = json.loads(path.read_text())
    data["results"]["beta_volume"] = beta + 1e-10
    path.write_text(json.dumps(data))
    report = scorer.evaluate(path)
    assert report["score"] == 0
    assert "Reported volumetric" in report["checks"][0]["detail"]


def test_expansion_uses_all_heating_means_and_excludes_cooling(tmp_path):
    path, scorer = expansion(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row["phase"] == "cool":
            row["V"] *= 2
    write_log(log, rows)
    assert scorer(path) == 1
    for row in rows:
        if row["stage"] == 4 and row["part"] == "prod":
            row["V"] *= 2
    write_log(log, rows)
    assert scorer(path) == 0


def test_heat_capacity_derives_temperature_from_momenta(tmp_path):
    path, scorer = heat_capacity(tmp_path)
    log = tmp_path / "run.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        t = row.pop("temperature_K")
        row["momenta"] = np.full((108, 3), np.sqrt(26.9815385 * units.kB * t)).tolist()
    write_log(log, rows)
    assert scorer(path) == 1


def metric_values(y, p):
    residual = np.asarray(p) - y
    return {
        "MAE_eV": float(np.mean(abs(residual))),
        "RMSE_eV": float(np.sqrt(np.mean(residual**2))),
        "R2": float(1 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2)),
    }


def copper(tmp_path, *, refit=False, poor_time_aware=False):
    labels = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    runs = ("main", "independent_800", "independent_500", "independent_1100")
    links = {}
    for name in runs:
        log = tmp_path / f"{name}.jsonl"
        # No positions, atomic-number arrays, checkpoints, or calculator identity.
        write_log(
            log,
            [
                {"phase": "production", "time_fs": i * 20, "epot_eV": y}
                for i, y in enumerate(labels)
            ],
        )
        links[name] = log.name
    rows = [{"event": "start", "standardize": True}]
    if refit:
        for name in runs:
            rows.append(
                {"event": "features", "run": name, "values": labels[:, None].tolist()}
            )
    for name in ("randomized", "time_aware"):
        train, test = np.arange(4), np.arange(4, 6)
        if refit:
            # For standardized 1D x=y, Ridge shrinks slope by n/(n+alpha).
            predicted = labels[train].mean() + (
                labels[test] - labels[train].mean()
            ) * 4 / (4 + 1e-4)
            rows.append(
                {
                    "type": "validation",
                    "strategy": name,
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                    "selected_alpha": 1e-4,
                    "test_metrics": metric_values(labels[test], predicted),
                }
            )
        else:
            # The chronological holdout has a physical quality gate; the
            # randomized split remains an independently reported diagnostic.
            predicted = (
                np.array([-5.0, -5.0])
                if name == "randomized" or poor_time_aware
                else labels[test] + 0.1
            )
            rows.extend(
                [
                    {
                        "event": "fit",
                        "name": name,
                        "train_indices": train.tolist(),
                        "test_indices": test.tolist(),
                    },
                    {
                        "event": "evaluate",
                        "name": name,
                        "predictions": predicted.tolist(),
                        "metrics": metric_values(labels[test], predicted),
                    },
                ]
            )
    rows.append({"type": "final_model", "selected_alpha": 1e-4})
    for name in runs[1:]:
        if refit:
            predicted = labels.mean() + (labels - labels.mean()) * 6 / (6 + 1e-4)
            row = {"type": "independent_test", "run": name}
        else:
            predicted = labels + 0.5
            row = {"event": "evaluate", "name": name, "predictions": predicted.tolist()}
        rows.append({**row, "metrics": metric_values(labels, predicted)})
    write_log(tmp_path / "regression.jsonl", rows)
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps(
            {"artifacts": {"run_logs": links, "regression_log": "regression.jsonl"}}
        )
    )
    return path, check_level2_workflow(9, verification_backend="offline")


@pytest.mark.parametrize("refit", [False, True])
def test_prediction_metrics_recomputed_with_time_aware_quality_target(tmp_path, refit):
    path, scorer = copper(tmp_path, refit=refit)
    report = scorer.evaluate(path)
    assert report["score"] == 1, report
    if not refit:
        assert report["computed"]["metrics"]["randomized"]["r2"] < 0
    assert report["computed"]["metrics"]["time_aware"]["r2"] >= 0.98


def test_self_consistent_poor_time_aware_predictions_fail(tmp_path):
    path, scorer = copper(tmp_path, poor_time_aware=True)
    report = scorer.evaluate(path)
    assert report["score"] == 0
    assert "quality threshold 0.98" in report["checks"][0]["detail"]


@pytest.mark.parametrize(("r2", "expected"), [(0.98, 1), (0.9799, 0)])
def test_time_aware_quality_threshold_boundary(tmp_path, r2, expected):
    path, scorer = copper(tmp_path)
    log = tmp_path / "regression.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    labels = np.array([16.0, 32.0])
    offset = np.sqrt((1 - r2) * np.sum((labels - labels.mean()) ** 2) / len(labels))
    predictions = labels + offset
    for row in rows:
        if row.get("event") == "evaluate" and row.get("name") == "time_aware":
            row["predictions"] = predictions.tolist()
            row["metrics"] = metric_values(labels, predictions)
    write_log(log, rows)
    assert scorer(path) == expected


@pytest.mark.parametrize("train_indices", [[0, 1, 4], [0, 5]])
def test_time_aware_logged_predictions_require_chronological_holdout(
    tmp_path, train_indices
):
    path, scorer = copper(tmp_path)
    log = tmp_path / "regression.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row.get("event") == "fit" and row.get("name") == "time_aware":
            row["train_indices"] = train_indices
    write_log(log, rows)
    report = scorer.evaluate(path)
    assert report["score"] == 0
    assert "must precede" in report["checks"][0]["detail"]


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("defect", ["one_test_row", "constant_test_labels"])
def test_time_aware_r2_requires_multiple_nonconstant_test_labels(
    tmp_path, refit, defect
):
    path, scorer = copper(tmp_path, refit=refit)
    log = tmp_path / "regression.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if defect == "one_test_row":
        for row in rows:
            name = row.get("strategy", row.get("name"))
            if name == "time_aware":
                if "test_indices" in row:
                    row["test_indices"] = [4]
                if "predictions" in row:
                    row["predictions"] = [16.0]
                row.pop("test_metrics", None)
                row.pop("metrics", None)
    else:
        main_log = tmp_path / "main.jsonl"
        samples = [json.loads(line) for line in main_log.read_text().splitlines()]
        samples[-1]["epot_eV"] = samples[-2]["epot_eV"]
        write_log(main_log, samples)
        for row in rows:
            # Allow the randomized diagnostic to finish so the time-aware
            # variance gate, rather than a stale metric claim, decides the score.
            row.pop("test_metrics", None)
            row.pop("metrics", None)
            if row.get("name") == "time_aware" and "predictions" in row:
                row["predictions"] = [16.0, 16.0]
    write_log(log, rows)
    report = scorer.evaluate(path)
    assert report["score"] == 0
    assert (
        "at least two test labels with nonzero variance"
        in report["checks"][0]["detail"]
    )


@pytest.mark.parametrize("refit", [False, True])
def test_wrong_prediction_metric_fails(tmp_path, refit):
    path, scorer = copper(tmp_path, refit=refit)
    log = tmp_path / "regression.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[-1]["metrics"]["MAE_eV"] = 100
    write_log(log, rows)
    assert scorer(path) == 0


def test_missing_prediction_calculation_fails(tmp_path):
    path, scorer = copper(tmp_path)
    log = tmp_path / "regression.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    write_log(log, rows[:-1])
    assert scorer(path) == 0


def test_prediction_metrics_do_not_require_correlation_timing_metadata(tmp_path):
    path, scorer = copper(tmp_path)
    log = tmp_path / "main.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        del row["time_fs"]
    write_log(log, rows)
    report = scorer.evaluate(path)
    assert report["score"] == 1
    assert report["computed"]["correlation"]["status"] == "unavailable"
