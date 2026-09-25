"""Consistent synthetic saved evidence and adversarial task-9 audit regressions."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1 import evaluate as evaluate_level1
from corral_md.workflow_scoring.regression import metrics, predict
from corral_md.workflow_scoring.task_9 import RUNS, evaluate


def write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def read(path):
    return json.loads(path.read_text())


def score(path):
    r = Rubric(9)
    evaluate(Evidence(path), r)
    return r


def check(r, name):
    return next(item for item in r.checks if item["name"] == name)


def test_level1_accepts_one_saved_equilibration_boundary_state(submission):
    manifest = read(submission)
    artifacts = manifest["artifacts"]["runs"]["main"]
    equilibration_path = Path(artifacts["equilibration"])
    boundary_path = Path(artifacts["boundary"])
    last = read(equilibration_path)[-1]
    last.pop("energy")
    last.pop("time_fs")
    write(equilibration_path, [last])
    boundary = copy.deepcopy(last)
    write(boundary_path, [boundary])
    settings_path = submission.parent / "settings.json"
    settings = read(settings_path)
    settings["runs"]["main"]["equilibration_steps"] = 300
    write(settings_path, settings)

    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    for name in (
        "fixed_cell_cu32_chronological_frames",
        "recorded_initialization_and_teacher",
        "equilibration_to_production_continuity",
    ):
        assert check(rubric, name)["status"] == "passed", rubric.checks

    changed = read(boundary_path)
    changed[0]["momenta"][0][0] += 1
    write(boundary_path, changed)
    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "equilibration_to_production_continuity")["status"] == "failed"


@pytest.mark.parametrize("link_asset", [False, True])
def test_level1_uses_mounted_cu32_without_a_local_copy(submission, link_asset):
    manifest = read(submission)
    if link_asset:
        manifest["artifacts"]["input_structure"] = (
            "/workspace/structures/cu/cu32.extxyz"
        )
    else:
        manifest["artifacts"].pop("input_structure")
    write(submission, manifest)

    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "fixed_cell_cu32_chronological_frames")["status"] == "passed"


def test_level1_accepts_sparse_raw_log_with_matching_samples(submission):
    manifest = read(submission)
    artifacts = manifest["artifacts"]["runs"]["main"]
    log_path = Path(artifacts["log"])
    log = read(log_path)
    production_log = log[4:]
    write(log_path, production_log[::20] + [production_log[-1]])

    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "thermal_log_and_production_temperature")["status"] == "passed"

    changed = read(log_path)
    changed[-1]["temperature_K"] += 100
    write(log_path, changed)
    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "thermal_log_and_production_temperature")["status"] == "failed"


def test_level1_accepts_ase_temperature_reporting_and_unit_labeled_frames(submission):
    manifest = read(submission)
    artifacts = manifest["artifacts"]["runs"]["main"]
    trajectory_path = Path(artifacts["trajectory"])
    frames = read(trajectory_path)
    for frame in frames:
        for key, alias in (
            ("positions", "positions_A"),
            ("cell", "cell_A"),
            ("momenta", "momenta_ase_units"),
            ("energy", "potential_energy_eV"),
            ("forces", "forces_eV_per_A"),
        ):
            frame[alias] = frame.pop(key)
        frame.pop("symbols")
        frame.pop("pbc")
    write(
        trajectory_path,
        {"symbols": ["Cu"] * 32, "pbc": [True] * 3, "frames": frames},
    )
    log_path = Path(artifacts["log"])
    production_log = read(log_path)[4:]
    sparse_log = production_log[::20] + [production_log[-1]]
    for row in sparse_log:
        row["temperature_K"] *= 93 / 96
    write(log_path, sparse_log)
    manifest["results"]["runs"]["main"]["production_temperature_mean_K"] *= 93 / 96
    write(submission, manifest)

    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "fixed_cell_cu32_chronological_frames")["status"] == "passed"
    assert check(rubric, "thermal_log_and_production_temperature")["status"] == "passed"

    conflicting = read(trajectory_path)
    conflicting["frames"][0]["positions"] = copy.deepcopy(
        conflicting["frames"][0]["positions_A"]
    )
    conflicting["frames"][0]["positions"][0][0] += 1
    write(trajectory_path, conflicting)
    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "fixed_cell_cu32_chronological_frames")["status"] == "failed"


def test_level1_accepts_unbiased_interpolated_correlation(submission):
    manifest = read(submission)
    correlation_path = Path(manifest["artifacts"]["correlation"])
    doc = read(correlation_path)
    lags = np.asarray(doc["lag_indices"])
    values = np.asarray(doc["values"]) * 200 / (200 - lags)
    doc["values"] = values.tolist()
    doc["estimator"] = (
        "mean-subtracted potential-energy autocorrelation, "
        "unbiased covariance at each lag"
    )
    doc["normalization"] = (
        "C(k)/C(0); C(k)=sum((E_i-mean(E))*(E_(i+k)-mean(E)))/(N-k)"
    )
    characteristic = doc["characteristic"]
    characteristic["method"] = "first 1/e crossing, linearly interpolated"
    crossing = int(np.flatnonzero(values <= characteristic["threshold"])[0])
    fraction = (characteristic["threshold"] - values[crossing - 1]) / (
        values[crossing] - values[crossing - 1]
    )
    characteristic["value"] = float((crossing - 1 + fraction) * 5)
    characteristic["bound"] = None
    write(correlation_path, doc)

    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "temporal_correlation_and_characteristic_time")["status"] == "passed"

    doc["characteristic"]["value"] += 20
    write(correlation_path, doc)
    rubric = Rubric(9, fail_fast=False)
    evaluate_level1(Evidence(submission), rubric, 9)
    assert check(rubric, "temporal_correlation_and_characteristic_time")["status"] == "failed"


def fit(x, y, alpha):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - y.mean()))
    return {
        "estimator": "Ridge",
        "positive": False,
        "coef": coef.tolist(),
        "intercept": float(y.mean()),
        "alpha": alpha,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "with_mean": True,
        "with_std": True,
    }


def temp(frame):
    p = np.array(frame["momenta"])
    return float(np.sum(p**2 / np.array(frame["masses"])[:, None]) / (93 * units.kB))


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    masses = atoms.get_masses()
    base = {
        "symbols": ["Cu"] * 32,
        "positions": atoms.positions.tolist(),
        "masses": masses.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": [True] * 3,
    }
    artifacts = {"input_structure": write(tmp_path / "input.json", [base]), "runs": {}}
    settings = {
        "teacher": {"identity": "MACE-MP-0"},
        "soap": {
            "species": ["Cu"],
            "periodic": True,
            "average": "inner",
            "r_cut": 4,
            "n_max": 3,
            "l_max": 2,
        },
        "runs": {},
    }
    results = {
        "runs": {},
        "metric_units": {"mae": "eV", "rmse": "eV", "r2": "dimensionless"},
    }
    data = {}
    for k, (run, target) in enumerate(zip(RUNS, (800, 800, 500, 1100), strict=False)):
        rng = np.random.default_rng(50 + k)
        t = np.arange(200)
        x = np.stack(
            [np.sin(t / (12 + k) + k), np.cos(t / 7.8 + k), np.sin(t / 31 + 2 * k)],
            axis=1,
        )
        x += 0.01 * rng.normal(size=x.shape)
        y = -100 + x @ np.array([0.8, -0.4, 0.15]) + 0.01 * np.cos(t / 3)
        data[run] = (x, y)
        p = rng.normal(size=(32, 3))
        p -= p.mean(axis=0)
        p *= np.sqrt(target * 93 * units.kB / np.sum(p**2 / masses[:, None]))
        eq, prod, log = [], [], []
        for i in range(4):
            frame = copy.deepcopy(base)
            frame.update(
                momenta=p.tolist(),
                time_fs=100.0 * i,
                energy=float(y[0]),
                forces=np.zeros((32, 3)).tolist(),
            )
            frame["positions"] = (
                atoms.positions + 0.01 * i * np.sin(np.arange(96).reshape(32, 3))
            ).tolist()
            eq.append(frame)
        for i in range(200):
            frame = copy.deepcopy(eq[-1])
            frame.update(
                momenta=(p * (1 + 0.03 * np.sin(i / 10))).tolist(),
                time_fs=300 + 5.0 * i,
                energy=float(y[i]),
            )
            frame["positions"] = (
                np.array(eq[-1]["positions"])
                + 0.02 * np.sin(i / 11) * np.cos(np.arange(96).reshape(32, 3))
            ).tolist()
            prod.append(frame)
        for frame in eq + prod:
            log.append(  # noqa: PERF401 - chronological fixture rows
                {
                    "time_fs": frame["time_fs"],
                    "temperature_K": temp(frame),
                    "potential_energy_eV": frame["energy"],
                }
            )
        prefix = tmp_path / run
        artifacts["runs"][run] = {
            "trajectory": write(prefix.with_suffix(".prod.json"), prod),
            "equilibration": write(prefix.with_suffix(".eq.json"), eq),
            "boundary": write(prefix.with_suffix(".boundary.json"), [eq[-1], eq[-1]]),
            "log": write(prefix.with_suffix(".log.json"), log),
            "descriptors": write(prefix.with_suffix(".x.json"), x.tolist()),
            "labels": write(prefix.with_suffix(".y.json"), y.tolist()),
        }
        settings["runs"][run] = {
            "target_temperature_K": target,
            "seed": 50 + k,
            "timestep_fs": 1,
            "integrator": "Langevin",
            "thermostat": "Langevin",
            "temperature_dof": 93,
            "remove_com": True,
        }
        eq_t, prod_t = (
            np.array([temp(f) for f in eq]),
            np.array([temp(f) for f in prod]),
        )
        results["runs"][run] = {
            "production_temperature_mean_K": float(prod_t.mean()),
            "production_temperature_std_K": float(prod_t.std()),
            "equilibration_first_half_temperature_K": float(eq_t[:2].mean()),
            "equilibration_second_half_temperature_K": float(eq_t[2:].mean()),
            "equilibration_temperature_drift_K_per_ps": float(
                np.polyfit(np.arange(4) * 0.1, eq_t, 1)[0]
            ),
        }
    x, y = data["main"]
    model_doc = {"validation": {}}
    results["validation"] = {}
    evaluation = {}
    rng = np.random.default_rng(987)
    random_test = np.sort(rng.choice(200, 40, replace=False))
    random_train = np.setdiff1d(np.arange(200), random_test)
    for name, train, test in [
        ("randomized", random_train, random_test),
        ("time_aware", np.arange(160), np.arange(160, 200)),
    ]:
        item = {
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "tuning": [],
            "selection_criterion": "minimum_mean_rmse",
        }
        scores = {}
        for alpha in (0.01, 1.0):
            rmse = []
            for inner_test in [train[:20], train[-20:]]:
                inner_train = np.setdiff1d(train, inner_test)
                model = fit(x[inner_train], y[inner_train], alpha)
                p = predict(x[inner_test], model)
                m = metrics(y[inner_test], p)
                rmse.append(m["rmse"])
                item["tuning"].append(
                    {
                        "train_indices": inner_train.tolist(),
                        "test_indices": inner_test.tolist(),
                        "model": model,
                        "predictions": p.tolist(),
                        "metrics": m,
                    }
                )
            scores[alpha] = np.mean(rmse)
        selected = min(scores, key=scores.get)
        item["model"] = fit(x[train], y[train], selected)
        p = predict(x[test], item["model"])
        item["predictions"] = p.tolist()
        if name == "randomized":
            item["seed"] = 987
        else:
            item.update(strategy="forward", minimum_gap_frames=0, minimum_gap_time_fs=5)
        model_doc["validation"][name] = item
        results["validation"][name] = metrics(y[test], p)
        evaluation[name] = (y[test], p)
    results["validation"]["comparison"] = {
        "training_counts": [160, 160],
        "evaluation_counts": [40, 40],
        "training_count_ratio": 1,
        "evaluation_count_ratio": 1,
    }
    final_model = fit(x, y, model_doc["validation"]["time_aware"]["model"]["alpha"])
    model_doc["final"] = {
        "train_indices": list(range(200)),
        "model": final_model,
        "selection_source": "time_aware",
    }
    artifacts["models"] = write(tmp_path / "models.json", model_doc)
    results["tests"] = {}
    for name in RUNS[1:]:
        x, y = data[name]
        p = predict(x, final_model)
        artifacts["runs"][name]["predictions"] = write(
            tmp_path / f"{name}.predictions.json", p.tolist()
        )
        results["tests"][name] = metrics(y, p)
        evaluation[name] = (y, p)
    y = data["main"][1]
    delta = y - y.mean()
    lags = np.arange(81)
    acf = np.array(
        [np.dot(delta[: 200 - k], delta[k:]) / np.dot(delta, delta) for k in lags]
    )
    hit = np.flatnonzero(acf <= np.exp(-1))
    characteristic = {
        "method": "first_crossing",
        "threshold": float(np.exp(-1)),
        "bound": "estimate" if len(hit) else "lower",
        "value": float(5 * (hit[0] if len(hit) else lags[-1])),
        "time_unit": "fs",
        "sampling_interval": 5,
        "observation_duration": 995,
    }
    artifacts["correlation"] = write(
        tmp_path / "correlation.json",
        {
            "estimator": "acf",
            "normalization": "biased",
            "lag_indices": lags.tolist(),
            "lag_times": (5 * lags).tolist(),
            "time_unit": "fs",
            "values": acf.tolist(),
            "characteristic": characteristic,
        },
    )
    stability = []
    for name, (y, p) in evaluation.items():
        for indices in [np.arange(len(y) // 2), np.arange(len(y) // 2, len(y))]:
            stability.append(  # noqa: PERF401 - explicit resampling fixture
                {
                    "evaluation": name,
                    "indices": indices.tolist(),
                    "metrics": metrics(y[indices], p[indices]),
                }
            )
    artifacts["stability"] = write(tmp_path / "stability.json", {"records": stability})
    rmses = {name: metrics(*arrays)["rmse"] for name, arrays in evaluation.items()}
    results["comparisons"] = {
        "randomized_minus_time_aware_rmse_eV": rmses["randomized"]
        - rmses["time_aware"],
        "independent_500_minus_800_rmse_eV": rmses["independent_500"]
        - rmses["independent_800"],
        "independent_1100_minus_800_rmse_eV": rmses["independent_1100"]
        - rmses["independent_800"],
        "shifted_temperature_differences_K": [
            results["runs"][name]["production_temperature_mean_K"]
            - results["runs"]["independent_800"]["production_temperature_mean_K"]
            for name in RUNS[2:]
        ],
    }
    script = tmp_path / "inference.py"
    script.write_text("raise RuntimeError('Never run submitted scripts')\n")
    artifacts["inference_entrypoint"] = str(script)
    checkpoint = tmp_path / "pipeline.pkl"
    checkpoint.write_bytes(b"opaque pipeline artifact: never deserialize")
    artifacts["pipeline_checkpoint"] = str(checkpoint)
    report = tmp_path / "report.md"
    report.write_text(
        "Synthetic artifact consistency fixture; no claims of true teacher or MD provenance."
    )
    path = tmp_path / "manifest.json"
    write(
        path,
        {
            "artifacts": artifacts,
            "results": results,
            "settings": write(tmp_path / "settings.json", settings),
            "report": str(report),
            "scripts": [str(script)],
        },
    )
    return path


def test_complete_evidence_is_scored_without_model_or_script_execution(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Execution or pickle attempted")

    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    result = score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(c["points"] for c in result.checks) == 90
    assert (
        check(result, "execution_and_information_access_provenance")["status"]
        == "unverified"
    )


@pytest.mark.parametrize(
    ("mutation", "failed"),
    [
        ("geometry", "fixed_cell_cu32_and_chronological_frames"),
        ("framecount", "fixed_cell_cu32_and_chronological_frames"),
        ("cell", "fixed_cell_cu32_and_chronological_frames"),
        ("boundary", "equilibration_to_production_boundaries"),
        ("com", "distinct_initializations_and_recorded_md_settings"),
        ("seed", "distinct_initializations_and_recorded_md_settings"),
        ("energy", "energy_descriptor_alignment_and_teacher_configuration"),
        ("acf", "temporal_dependence_curve_and_units"),
        ("acfunits", "temporal_dependence_curve_and_units"),
        ("tau", "characteristic_correlation_time_or_bound"),
        ("prediction", "independent_500_coefficient_predictions"),
        ("metrics", "independent_500_mae_rmse_r2"),
        ("split", "disjoint_randomized_and_time_aware_holdouts"),
        ("scaler", "training_only_outer_preprocessing_and_ridge_consistency"),
        ("tuning", "nested_tuning_protects_both_outer_holdouts"),
        ("finalindices", "final_main_only_affine_pipeline"),
    ],
)
def test_mutated_evidence_loses_corresponding_credit(submission, mutation, failed):
    directory = submission.parent
    if mutation in {"geometry", "framecount", "cell", "boundary", "com"}:
        stage = (
            "eq"
            if mutation in {"geometry", "com"}
            else "boundary"
            if mutation == "boundary"
            else "prod"
        )
        file = directory / f"main.{stage}.json"
        rows = read(file)
        if mutation == "geometry":
            rows[0]["positions"][2][0] += 0.5
        elif mutation == "com":
            rows[0]["momenta"][0][0] += 1
        elif mutation == "framecount":
            rows.pop()
        elif mutation == "cell":
            rows[-1]["cell"][0][0] += 0.2
        else:
            rows[1]["momenta"][0][0] += 0.2
        write(file, rows)
    elif mutation == "seed":
        file = directory / "settings.json"
        doc = read(file)
        doc["runs"]["main"]["seed"] = doc["runs"]["independent_800"]["seed"]
        write(file, doc)
    elif mutation in {"acf", "acfunits", "tau"}:
        file = directory / "correlation.json"
        doc = read(file)
        if mutation == "acf":
            doc["values"][4] += 0.1
        elif mutation == "acfunits":
            doc["time_unit"] = "ps"
        else:
            doc["characteristic"]["value"] *= 10
        write(file, doc)
    elif mutation in {"split", "scaler", "tuning", "finalindices"}:
        file = directory / "models.json"
        doc = read(file)
        if mutation == "split":
            item = doc["validation"]["randomized"]
            item["train_indices"][0] = item["test_indices"][0]
        elif mutation == "scaler":
            doc["validation"]["randomized"]["model"]["mean"][0] += 0.5
        elif mutation == "tuning":
            item = doc["validation"]["randomized"]
            item["tuning"][0]["train_indices"][0] = item["test_indices"][0]
        else:
            doc["final"]["train_indices"][-1] = 200
        write(file, doc)
    elif mutation == "metrics":
        doc = read(submission)
        doc["results"]["tests"]["independent_500"]["rmse"] += 1
        write(submission, doc)
    elif mutation == "stability":
        file = directory / "stability.json"
        doc = read(file)
        doc["records"][0]["metrics"]["r2"] += 0.5
        write(file, doc)
    else:
        file = directory / (
            "main.y.json"
            if mutation == "energy"
            else "independent_500.predictions.json"
        )
        values = read(file)
        values[4] += 1
        write(file, values)
    result = score(submission)
    assert check(result, failed)["status"] == "failed", result.checks
    assert sum(c["points"] for c in result.checks) == 90


@pytest.mark.parametrize("method", ["unbiased", "integrated", "ps"])
def test_alternative_correlation_conventions_are_supported(submission, method):
    file = submission.parent / "correlation.json"
    doc = read(file)
    if method == "unbiased":
        doc["normalization"] = "unbiased"
        lags = np.array(doc["lag_indices"])
        doc["values"] = (np.array(doc["values"]) * 200 / (200 - lags)).tolist()
        hit = np.flatnonzero(np.array(doc["values"]) <= np.exp(-1))
        doc["characteristic"]["value"] = float(5 * (hit[0] if len(hit) else lags[-1]))
        doc["characteristic"]["bound"] = "estimate" if len(hit) else "lower"
    elif method == "integrated":
        doc["characteristic"].update(
            method="integrated",
            cutoff_lag=10,
            convention="half_plus_sum",
            value=float(5 * (0.5 + sum(doc["values"][1:11]))),
            bound="estimate",
        )
    else:
        doc["time_unit"] = "ps"
        doc["lag_times"] = [v / 1000 for v in doc["lag_times"]]
        doc["characteristic"]["time_unit"] = "ps"
        for key in ("value", "sampling_interval", "observation_duration"):
            doc["characteristic"][key] /= 1000
    write(file, doc)
    result = score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_missing_evidence_preserves_independent_checks_and_budget(submission):
    (submission.parent / "independent_500.predictions.json").unlink()
    result = score(submission)
    assert (
        check(result, "fixed_cell_cu32_and_chronological_frames")["status"] == "passed"
    )
    assert (
        check(result, "independent_500_coefficient_predictions")["status"] == "failed"
    )
    assert sum(c["points"] for c in result.checks) == 90


def test_empty_submission_has_no_credit(tmp_path):
    path = tmp_path / "empty.json"
    write(path, {"artifacts": {}, "results": {}})
    result = score(path)
    assert result.score == 0
    assert sum(c["points"] for c in result.checks) == 90


def test_unsupported_temporal_estimator_is_unverified(submission):
    file = submission.parent / "correlation.json"
    doc = read(file)
    doc["estimator"] = "mutual_information"
    write(file, doc)
    result = score(submission)
    assert (
        check(result, "temporal_dependence_curve_and_units")["status"] == "unverified"
    )
    assert check(result, "independent_500_mae_rmse_r2")["status"] == "passed"


def test_cold_production_requires_target_review_without_failing_initialization(
    submission,
):
    directory = submission.parent
    doc = read(submission)
    for stage in ("prod",):
        path = directory / f"main.{stage}.json"
        frames = read(path)
        for f in frames:
            f["momenta"] = (np.array(f["momenta"]) * np.sqrt(1 / 800)).tolist()
        write(path, frames)
    t = np.array([temp(f) for f in frames])
    doc["results"]["runs"]["main"]["production_temperature_mean_K"] = float(t.mean())
    doc["results"]["runs"]["main"]["production_temperature_std_K"] = float(t.std())
    write(submission, doc)
    result = score(submission)
    assert (
        check(result, "distinct_initializations_and_recorded_md_settings")["status"]
        == "passed"
    )
    assert check(result, "production_temperature_targets")["status"] == "unverified"
    assert check(result, "independent_500_mae_rmse_r2")["status"] == "passed"


def test_declared_random_seed_must_reproduce_the_randomized_holdout(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["validation"]["randomized"]["seed"] += 1
    write(path, doc)
    assert (
        check(score(submission), "disjoint_randomized_and_time_aware_holdouts")[
            "status"
        ]
        == "failed"
    )


def test_inner_fold_scaler_cannot_be_fitted_on_outer_training_rows(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    outer = doc["validation"]["randomized"]
    outer["tuning"][0]["model"]["mean"] = outer["model"]["mean"]
    write(path, doc)
    assert (
        check(score(submission), "nested_tuning_protects_both_outer_holdouts")["status"]
        == "failed"
    )


def test_claimed_ridge_coefficients_are_checked_without_refitting(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["model"]["coef"][0] += 1
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


def test_all_outer_test_metrics_are_recomputed(submission):
    doc = read(submission)
    doc["results"]["validation"]["randomized"]["rmse"] += 1
    write(submission, doc)
    assert (
        check(score(submission), "reproduced_validation_predictions_and_metrics")[
            "status"
        ]
        == "failed"
    )


def test_no_crossing_reports_a_bound_instead_of_a_fabricated_estimate(submission):
    path = submission.parent / "correlation.json"
    doc = read(path)
    for key in ("lag_indices", "lag_times", "values"):
        doc[key] = doc[key][:2]
    doc["characteristic"].update(value=5, bound="lower")
    write(path, doc)
    assert (
        check(score(submission), "characteristic_correlation_time_or_bound")["status"]
        == "passed"
    )
    doc["characteristic"]["bound"] = "estimate"
    write(path, doc)
    assert (
        check(score(submission), "characteristic_correlation_time_or_bound")["status"]
        == "failed"
    )


@pytest.mark.parametrize("name", ["pipeline_checkpoint", "inference_entrypoint"])
def test_complete_pipeline_and_inference_artifacts_must_be_retained(submission, name):
    doc = read(submission)
    del doc["artifacts"][name]
    write(submission, doc)
    result = score(submission)
    assert check(result, "final_main_only_affine_pipeline")["status"] == "failed"
    assert sum(c["points"] for c in result.checks) == 90
    assert check(result, "independent_500_mae_rmse_r2")["status"] == "passed"


def test_empty_checkpoint_does_not_count_as_saved_pipeline(submission):
    (submission.parent / "pipeline.pkl").write_bytes(b"")
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


def test_non_ridge_model_cannot_claim_ridge_fit_credit(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["model"]["estimator"] = "Lasso"
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


def test_positive_ridge_rejects_negative_coefficients(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["model"]["positive"] = True
    write(path, doc)
    result = score(submission)
    assert check(result, "final_main_only_affine_pipeline")["status"] == "failed"
    assert check(result, "independent_500_mae_rmse_r2")["status"] == "passed"


def test_nonboolean_scaling_flag_is_rejected(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["model"]["with_mean"] = "false"
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


def test_false_fit_intercept_requires_zero_saved_intercept(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["model"]["fit_intercept"] = False
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


def refresh_predictions(submission, doc):
    directory = submission.parent
    manifest = read(submission)
    result = manifest["results"]
    rmses = {}
    x, y = (
        np.asarray(read(directory / "main.x.json")),
        np.asarray(read(directory / "main.y.json")),
    )
    for name, item in doc["validation"].items():
        test = np.asarray(item["test_indices"])
        p = predict(x[test], item["model"])
        item["predictions"] = p.tolist()
        result["validation"][name] = metrics(y[test], p)
        rmses[name] = metrics(y[test], p)["rmse"]
    for name in RUNS[1:]:
        tx, ty = read(directory / f"{name}.x.json"), read(directory / f"{name}.y.json")
        p = predict(tx, doc["final"]["model"])
        write(directory / f"{name}.predictions.json", p.tolist())
        result["tests"][name] = metrics(ty, p)
        rmses[name] = metrics(ty, p)["rmse"]
    result["comparisons"].update(
        randomized_minus_time_aware_rmse_eV=rmses["randomized"] - rmses["time_aware"],
        independent_500_minus_800_rmse_eV=rmses["independent_500"]
        - rmses["independent_800"],
        independent_1100_minus_800_rmse_eV=rmses["independent_1100"]
        - rmses["independent_800"],
    )
    write(directory / "models.json", doc)
    write(submission, manifest)


@pytest.mark.parametrize(
    "criterion", ["minimum_mean_mae", "largest_alpha_within_rmse_tolerance"]
)
def test_complete_alternative_selection_and_final_subset_receive_equal_credit(
    submission, criterion
):
    directory = submission.parent
    doc = read(directory / "models.json")
    x, y = (
        np.asarray(read(directory / "main.x.json")),
        np.asarray(read(directory / "main.y.json")),
    )
    for item in doc["validation"].values():
        # Adaptive search can allocate a different number of folds to candidates.
        item["tuning"].pop()
        item["selection_criterion"] = criterion
        metric = "mae" if criterion == "minimum_mean_mae" else "rmse"
        scores = {}
        for record in item["tuning"]:
            scores.setdefault(record["model"]["alpha"], []).append(
                record["metrics"][metric]
            )
        if criterion == "minimum_mean_mae":
            selected = min(scores, key=lambda alpha: np.mean(scores[alpha]))
        else:
            item["selection_tolerance_eV"] = 1.0
            selected = max(scores)
            assert selected != min(scores, key=lambda alpha: np.mean(scores[alpha]))
        train = np.asarray(item["train_indices"])
        item["model"] = fit(x[train], y[train], selected)
    # A final choice from main-only analysis can differ from both outer fits.
    indices = np.arange(0, 200, 2)
    doc["final"] = {
        "train_indices": indices.tolist(),
        "model": fit(x[indices], y[indices], 0.15),
        "selection_source": "main_trajectory_regularization_stability_analysis",
        "selection_dataset": "main",
        "selection_indices": list(range(200)),
        "training_selection": "Every second chronological main frame",
    }
    refresh_predictions(submission, doc)
    result = score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_declared_selection_rule_cannot_hide_inconsistent_alpha(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    item = doc["validation"]["randomized"]
    item["selection_criterion"] = "largest_alpha_within_rmse_tolerance"
    item["selection_tolerance_eV"] = 1.0
    assert item["model"]["alpha"] != max(f["model"]["alpha"] for f in item["tuning"])
    write(path, doc)
    result = score(submission)
    assert (
        check(result, "nested_tuning_protects_both_outer_holdouts")["status"]
        == "passed"
    )
    assert check(result, "selection_follows_declared_criterion")["status"] == "failed"


def test_custom_selection_is_reviewable_without_losing_holdout_credit(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["validation"]["randomized"]["selection_criterion"] = (
        "one_standard_error_mae_with_fold_size_weights"
    )
    write(path, doc)
    result = score(submission)
    assert result.score is None
    assert (
        check(result, "selection_follows_declared_criterion")["status"] == "unverified"
    )
    assert (
        check(result, "nested_tuning_protects_both_outer_holdouts")["status"]
        == "passed"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sampling", "custom_seeded_random_engine"), ("strategy", "rolling_origin")],
)
def test_alternative_split_conventions_are_reviewable(submission, field, value):
    path = submission.parent / "models.json"
    doc = read(path)
    which = "randomized" if field == "sampling" else "time_aware"
    doc["validation"][which][field] = value
    write(path, doc)
    result = score(submission)
    assert result.score is None
    assert (
        check(result, "disjoint_randomized_and_time_aware_holdouts")["status"]
        == "unverified"
    )
    assert (
        check(result, "nested_tuning_protects_both_outer_holdouts")["status"]
        == "passed"
    )


def test_final_selection_cannot_use_independent_trajectory(submission):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"]["selection_source"] = "independent_800"
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


@pytest.mark.parametrize("dataset", ["independent_800", "validation_pool", ["main"]])
def test_explicit_nonmain_final_selection_namespace_is_rejected(submission, dataset):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"].update(
        selection_source="custom_selection",
        selection_dataset=dataset,
        selection_indices=list(range(200)),
    )
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    ("source", "dataset", "expected"),
    [
        (" TIME_AWARE ", " MAIN_TRAJECTORY ", "passed"),
        (" independent_800 ", "main", "failed"),
    ],
)
def test_final_selection_identifiers_are_normalized(
    submission, source, dataset, expected
):
    path = submission.parent / "models.json"
    doc = read(path)
    doc["final"].update(selection_source=source, selection_dataset=dataset)
    write(path, doc)
    assert (
        check(score(submission), "final_main_only_affine_pipeline")["status"]
        == expected
    )


@pytest.mark.parametrize(
    ("convention", "expected"),
    [("pairwise_time_bins", "unverified"), ("uniform_samples", "failed")],
)
def test_irregular_sampling_uses_review_for_declared_alternative_lags(
    submission, convention, expected
):
    directory = submission.parent
    frames = read(directory / "main.prod.json")
    frames[10]["time_fs"] += 1
    write(directory / "main.prod.json", frames)
    logs = read(directory / "main.log.json")
    for row in logs:
        if row["time_fs"] == frames[10]["time_fs"] - 1:
            row["time_fs"] += 1
    write(directory / "main.log.json", logs)
    doc = read(directory / "correlation.json")
    doc["lag_convention"] = convention
    write(directory / "correlation.json", doc)
    result = score(submission)
    assert check(result, "temporal_dependence_curve_and_units")["status"] == expected
    assert (
        check(result, "characteristic_correlation_time_or_bound")["status"] == expected
    )
    assert check(result, "independent_800_mae_rmse_r2")["status"] == "passed"


@pytest.mark.parametrize(
    "method", ["generalized_cross_validation", "predeclared_fixed_alpha"]
)
def test_alternative_tuning_is_reviewable_and_outer_holdout_access_still_fails(
    submission, method
):
    path = submission.parent / "models.json"
    doc = read(path)
    item = doc["validation"]["randomized"]
    item.update(
        tuning_method=method,
        tuning={
            "dataset": "main",
            "selection_indices": list(item["train_indices"])
            if method != "predeclared_fixed_alpha"
            else [],
            "calculations": {"selected_alpha": item["model"]["alpha"]},
        },
    )
    write(path, doc)
    result = score(submission)
    assert (
        check(result, "nested_tuning_protects_both_outer_holdouts")["status"]
        == "unverified"
    )
    assert (
        check(result, "selection_follows_declared_criterion")["status"] == "unverified"
    )
    assert (
        check(result, "training_only_outer_preprocessing_and_ridge_consistency")[
            "status"
        ]
        == "passed"
    )
    item["tuning"]["selection_indices"].append(item["test_indices"][0])
    write(path, doc)
    assert (
        check(score(submission), "nested_tuning_protects_both_outer_holdouts")["status"]
        == "failed"
    )
