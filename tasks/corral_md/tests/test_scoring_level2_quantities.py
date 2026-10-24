"""Final numerical answers are sufficient; wrong or missing answers still fail."""

import json

import numpy as np
import pytest
from ase import units
from corral_md.scoring import check_level2_workflow
from corral_md.scoring.level2_quantities import Level2QuantityScorer


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
    return answer(
        tmp_path,
        6,
        [{"type": "vdos", "freq_THz": [0, 1, 2, 3, 4], "vdos": [0, 1, 2, 1, 0]}],
        {"mean_frequency_THz": 2.0, "std_frequency_THz": np.sqrt(0.5)},
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


@pytest.mark.parametrize(
    "make,key",
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


def copper(tmp_path, *, refit=False):
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
            # For standardized 1D x=y, Ridge(alpha=1) shrinks slope by n/(n+1).
            predicted = (
                labels[train].mean() + (labels[test] - labels[train].mean()) * 4 / 5
            )
            rows.append(
                {
                    "type": "validation",
                    "strategy": name,
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                    "selected_alpha": 1.0,
                    "test_metrics": metric_values(labels[test], predicted),
                }
            )
        else:
            # Poor predictions / negative R2 are legitimate numerical outcomes.
            predicted = np.array([-5.0, -5.0])
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
    rows.append({"type": "final_model", "selected_alpha": 1.0})
    for name in runs[1:]:
        if refit:
            predicted = labels.mean() + (labels - labels.mean()) * 6 / 7
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
def test_prediction_metrics_recomputed_with_poor_quality_allowed(tmp_path, refit):
    path, scorer = copper(tmp_path, refit=refit)
    report = scorer.evaluate(path)
    assert report["score"] == 1, report
    if not refit:
        assert report["computed"]["metrics"]["randomized"]["r2"] < 0


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
