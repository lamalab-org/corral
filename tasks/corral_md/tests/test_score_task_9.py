"""Saved harmonic trajectories and real sklearn fits exercise the log scorer."""

import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from corral_md.scoring.common import EvidenceError
from corral_md.scoring.evaluator import (
    PendingReviewError,
    WorkflowScorer,
)
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.task_9 import RUNS, correlation
from corral_md.scoring.verification import ModalVerifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def _soap_stub(monkeypatch):
    """Only expensive descriptor generation is stubbed; fitting uses sklearn."""
    module = ModuleType("dscribe.descriptors")

    class SOAP:
        def __init__(self, **kwargs):
            pass

        def create(self, frames):
            return np.array([a.positions[:2].ravel() for a in frames])

    module.SOAP = SOAP
    monkeypatch.setitem(sys.modules, "dscribe", ModuleType("dscribe"))
    monkeypatch.setitem(sys.modules, "dscribe.descriptors", module)


def append_fixture_row(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def write_regression_fixture(
    path,
    run_logs,
    *,
    soap,
    alphas=(0.01, 0.1, 1.0),
    random_seed=0,
    train_size=120,
    test_size=40,
    gap=10,
    standardize=True,
):
    """Fit SOAP/Ridge, freeze, then evaluate; automatically retain all calculations.

    Use DScribe SOAP keyword arguments in ``soap``. Both outer holdouts have the
    same budgets. Three forward inner folds select alpha using training data
    only; the time-aware choice also fixes the final model before any holdout
    predictions are made. No independent trajectory is read before freezing.
    """
    from dscribe.descriptors import SOAP
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    if (
        any(type(v) is not int for v in (random_seed, train_size, test_size, gap))
        or train_size < 16
        or test_size < 2
        or gap < 0
        or train_size + test_size + gap > 200
    ):
        raise ValueError("Invalid validation seed, budgets or gap")
    alphas = np.asarray(alphas, dtype=float)
    if (
        alphas.ndim != 1
        or not len(alphas)
        or not np.all(np.isfinite(alphas))
        or np.any(alphas <= 0)
    ):
        raise ValueError("Supply positive finite Ridge alphas")
    descriptor = SOAP(**soap)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    append_fixture_row(
        path,
        {
            "event": "start",
            "soap": soap,
            "random_seed": random_seed,
            "train_size": train_size,
            "test_size": test_size,
            "gap": gap,
            "alphas": alphas.tolist(),
            "standardize": standardize,
        },
    )

    def dataset(name):
        rows = [
            json.loads(line) for line in Path(run_logs[name]).read_text().splitlines()
        ]
        start = next(i for i, row in enumerate(rows) if row.get("name") == "production")
        samples = [row for row in rows[start:] if row["event"] == "sample"]
        frames = [
            Atoms(
                numbers=rows[0]["numbers"],
                positions=row["positions_A"],
                cell=rows[0]["cell_A"],
                pbc=True,
            )
            for row in samples
        ]
        x = np.asarray(descriptor.create(frames))
        y = np.array([row["potential_energy_eV"] for row in samples])
        append_fixture_row(
            path, {"event": "features", "run": name, "values": x.tolist()}
        )
        return x, y

    def fit(x, y, indices, alpha):
        scaler = StandardScaler(with_mean=standardize, with_std=standardize).fit(
            x[indices]
        )
        z = scaler.transform(x[indices])
        model = Ridge(alpha=float(alpha)).fit(z, y[indices])
        return {
            "estimator": "Ridge",
            "alpha": float(alpha),
            "fit_intercept": True,
            "preprocessing": "StandardScaler" if standardize else "identity",
            "mean": scaler.mean_.tolist()
            if standardize
            else np.zeros(x.shape[1]).tolist(),
            "scale": scaler.scale_.tolist()
            if standardize
            else np.ones(x.shape[1]).tolist(),
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }

    def predict(x, model):
        return ((x - model["mean"]) / model["scale"]) @ model["coef"] + model[
            "intercept"
        ]

    x, y = dataset("main")
    permutation = np.random.default_rng(random_seed).permutation(200)
    splits = {
        "randomized": (
            np.sort(permutation[test_size : test_size + train_size]),
            np.sort(permutation[:test_size]),
        ),
        "time_aware": (
            np.arange(train_size),
            np.arange(train_size + gap, train_size + gap + test_size),
        ),
    }
    models = {}
    for name, (train, test) in splits.items():
        trials, scores = [], []
        for alpha in alphas:
            folds, errors = [], []
            for a, b in TimeSeriesSplit(n_splits=3).split(train):
                fit_indices, validation = train[a], train[b]
                model = fit(x, y, fit_indices, alpha)
                prediction = predict(x[validation], model)
                errors.append(float(np.mean((prediction - y[validation]) ** 2)))
                folds.append(
                    {
                        "train_indices": fit_indices.tolist(),
                        "test_indices": validation.tolist(),
                        "model": model,
                        "predictions": prediction.tolist(),
                    }
                )
            trials.append({"alpha": float(alpha), "folds": folds})
            scores.append(np.mean(errors))
        model = fit(x, y, train, alphas[np.argmin(scores)])
        models[name] = model
        append_fixture_row(
            path,
            {
                "event": "fit",
                "name": name,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "trials": trials,
                "model": model,
            },
        )
    final = fit(x, y, np.arange(200), models["time_aware"]["alpha"])
    append_fixture_row(path, {"event": "freeze", "model": final})
    for name, (_, test) in splits.items():
        append_fixture_row(
            path,
            {
                "event": "evaluate",
                "name": name,
                "predictions": predict(x[test], models[name]).tolist(),
            },
        )
    for name in ("independent_800", "independent_500", "independent_1100"):
        independent, _ = dataset(name)
        append_fixture_row(
            path,
            {
                "event": "evaluate",
                "name": name,
                "predictions": predict(independent, final).tolist(),
            },
        )
    append_fixture_row(path, {"event": "end"})


@pytest.fixture
def submission(tmp_path, _soap_stub):
    logs, temperatures, energies = {}, {}, {}
    for seed, (name, target) in zip((2, 4, 5, 6), RUNS.items(), strict=True):
        atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=target, rng=np.random.default_rng(seed)
        )
        Stationary(atoms, preserve_temperature=True)
        p = atoms.get_momenta()
        # Orthogonal equal-norm momentum directions give constant kinetic T.
        q = np.random.default_rng(seed + 100).normal(size=p.shape)
        q -= q.mean(axis=0)
        q -= p * np.sum(q * p) / np.sum(p**2)
        q *= np.linalg.norm(p) / np.linalg.norm(q)
        mass = atoms.get_masses()[0]
        temperature = np.sum(p**2) / (mass * 96 * units.kB)
        # Keep fixtures within the declared thermal screen, without using the scorer.
        assert abs(temperature - target) < 0.15 * target

        def sample(t, step, p=p, q=q, mass=mass, name=name, atoms=atoms):
            angle = 0.025 * t
            displacement = (
                (p * np.sin(angle) + q * (1 - np.cos(angle)))
                * units.fs
                / (mass * 0.025)
            )
            energy = -100 + 0.2 * np.sin(angle) + 0.04 * np.cos(0.037 * t)
            if name != "main":
                energy += 2  # Poor transfer is a valid measured result.
            return {
                "event": "sample",
                "step": step,
                "time_fs": t,
                "positions_A": (atoms.positions + displacement).tolist(),
                "momenta": (p * np.cos(angle) + q * np.sin(angle)).tolist(),
                "potential_energy_eV": energy,
                "forces_eV_A": np.zeros((32, 3)).tolist(),
            }

        records = [
            {
                "event": "start",
                "model_sha256": PINNED_TEACHER_SHA256,
                "random_seed": seed,
                "target_temperature_K": target,
                "numbers": atoms.numbers.tolist(),
                "masses_amu": atoms.get_masses().tolist(),
                "cell_A": atoms.cell.array.tolist(),
                "units": {
                    "time": "fs",
                    "position": "Angstrom",
                    "energy": "eV",
                    "momenta": "ASE",
                },
            }
        ]
        for stage, offset, steps, interval in (
            ("equilibration", 0, 1500, 100),
            ("production", 1500, 1990, 10),
        ):
            records.append(
                {
                    "event": "stage",
                    "name": stage,
                    "steps": steps,
                    "interval": interval,
                    "timestep_fs": 1,
                    "integrator": "Langevin",
                    "target_temperature_K": target,
                }
            )
            samples = [
                sample(offset + step, step) for step in range(0, steps + 1, interval)
            ]
            records.extend(samples)
            records.append({"event": "end", "name": stage, "steps": steps})
        path = tmp_path / f"{name}.jsonl"
        write_rows(path, records)
        logs[name] = path
        temperatures[name] = temperature
        energies[name] = np.array([row["potential_energy_eV"] for row in samples])
    write_regression_fixture(
        tmp_path / "regression.jsonl",
        logs,
        soap={
            "species": ["Cu"],
            "periodic": True,
            "average": "inner",
            "r_cut": 4,
            "n_max": 2,
            "l_max": 2,
        },
        alphas=[0.1, 1],
    )
    y = energies["main"] - energies["main"].mean()
    acf = np.array([y[: 200 - k] @ y[k:] / (y @ y) for k in range(51)])
    lag = np.flatnonzero(acf <= np.exp(-1))[0]
    measured = {}
    regression = rows(tmp_path / "regression.jsonl")
    splits = {
        row["name"]: row["test_indices"] for row in regression if row["event"] == "fit"
    }
    for row in regression:
        if row["event"] == "evaluate":
            name, p = row["name"], row["predictions"]
            y = energies["main"][splits[name]] if name in splits else energies[name]
            measured[name] = {
                "mae": mean_absolute_error(y, p),
                "rmse": np.sqrt(mean_squared_error(y, p)),
                "r2": r2_score(y, p),
            }
    answer = {
        "results": {
            "production_temperature_K": temperatures,
            "correlation": {"value_fs": int(lag) * 10, "bound": "estimate"},
            "metrics": measured,
        },
        "artifacts": {
            "run_logs": {k: p.name for k, p in logs.items()},
            "regression_log": "regression.jsonl",
        },
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(answer))
    return path


def response(plan, fingerprint):
    assert all("expected" not in job for job in plan.jobs)
    assert all("energy" not in frame for job in plan.jobs for frame in job["frames"])
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "jobs": [
            {
                "id": job["id"],
                "status": "complete",
                "value": plan.expected[job["id"]]["values"],
            }
            for job in plan.jobs
        ],
    }


def scorer(level=2, transport=response):
    return WorkflowScorer(9, level=level, verifier=ModalVerifier(transport=transport))


def test_recomputes_all_answers_and_accepts_poor_transfer(submission):
    report = scorer().evaluate(submission)
    assert report["score"] == 1, report
    assert report["computed"]["metrics"]["independent_500"]["r2"] < 0
    assert report["verification"]["checks"] == 8
    assert report["pending_checks"] == []


def test_log_only_answer_keeps_all_metrics_temperatures_and_correlation(submission):
    answer = read(submission)
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    report = scorer().evaluate(submission)
    assert report["score"] == 1, report
    assert len(report["computed"]["production_temperature_K"]) == 4
    assert "value_fs" in report["computed"]["correlation"]
    assert all(
        set(values) == {"mae", "rmse", "r2"}
        for values in report["computed"]["metrics"].values()
    )
    answer["results"] = {"metrics": {"independent_500": {"rmse": 999}}}
    submission.write_text(json.dumps(answer))
    assert scorer().evaluate(submission)["score"] == 0


def test_level1_only_needs_main_log(submission):
    answer = read(submission)
    answer["artifacts"] = {"run_logs": {"main": "main.jsonl"}}
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    for path in submission.parent.glob("*.jsonl"):
        if path.name != "main.jsonl":
            path.unlink()
    report = scorer(1).evaluate(submission)
    assert report["score"] == 1, report
    assert report["verification"]["checks"] == 1
    assert scorer(2).evaluate(submission)["score"] == 0


@pytest.mark.parametrize("key", ["temperature", "correlation", "mae", "rmse", "r2"])
def test_altered_answers_fail(submission, key):
    answer = read(submission)
    if key == "temperature":
        answer["results"]["production_temperature_K"]["main"] += 500
    elif key == "correlation":
        answer["results"]["correlation"]["value_fs"] *= 1000
    else:
        answer["results"]["metrics"]["independent_500"][key] += 20
    submission.write_text(json.dumps(answer))
    assert scorer().evaluate(submission)["score"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "time",
        "boundary",
        "model",
        "units",
        "seed",
        "cell",
        "cold",
        "nonfinite",
        "constant",
    ],
)
def test_invalid_run_logs_fail(submission, change):
    answer = read(submission)
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    path = submission.parent / "main.jsonl"
    records = rows(path)
    samples = [r for r in records if r["event"] == "sample"]
    if change == "truncated":
        records.pop()
    elif change == "time":
        samples[-1]["time_fs"] -= 20
    elif change == "boundary":
        samples[16]["momenta"][0][0] += 1
    elif change == "model":
        records[0]["model_sha256"] = "wrong"
    elif change == "units":
        records[0]["units"]["energy"] = "eV/atom"
    elif change == "seed":
        records[0]["random_seed"] += 1
    elif change == "cell":
        records[0]["cell_A"][0][0] += 1
    elif change == "cold":
        for row in samples[17:]:
            row["momenta"] = (np.array(row["momenta"]) / 10).tolist()
    elif change == "nonfinite":
        samples[-1]["forces_eV_A"][0][0] = float("nan")
    else:
        for row in samples:
            row["potential_energy_eV"] = -100
    write_rows(path, records)
    assert scorer().evaluate(submission)["score"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "leakage",
        "scale",
        "coefficients",
        "prediction",
        "budgets",
        "seed",
        "early_evaluation",
        "early_features",
        "late_fit",
        "missing",
        "selection",
    ],
)
def test_regression_checks_training_and_protected_evaluation(submission, change):
    answer = read(submission)
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    path = submission.parent / "regression.jsonl"
    records = rows(path)
    fits = [r for r in records if r["event"] == "fit"]
    evaluations = [r for r in records if r["event"] == "evaluate"]
    if change == "leakage":
        fits[0]["trials"][0]["folds"][0]["train_indices"][0] = fits[0]["test_indices"][
            0
        ]
    elif change == "scale":
        fits[0]["model"]["mean"][0] += 10
    elif change == "coefficients":
        fits[0]["model"]["coef"][0] += 10
    elif change == "prediction":
        evaluations[-1]["predictions"][0] += 10
    elif change == "budgets":
        fits[1]["test_indices"].pop()
    elif change == "seed":
        records[0]["random_seed"] += 1
    elif change == "early_evaluation":
        records.remove(evaluations[0])
        records.insert(2, evaluations[0])
    elif change == "early_features":
        row = next(r for r in records if r.get("run") == "independent_800")
        records.remove(row)
        records.insert(2, row)
    elif change == "late_fit":
        records.insert(-1, copy.deepcopy(fits[0]))
    elif change == "selection":
        next(r for r in records if r["event"] == "freeze")["model"]["alpha"] = 999
    else:
        records.remove(evaluations[-1])
    write_rows(path, records)
    assert scorer().evaluate(submission)["score"] == 0


def test_reused_initialization_is_rejected(submission):
    independent = submission.parent / "independent_800.jsonl"
    independent.write_bytes((submission.parent / "main.jsonl").read_bytes())
    assert scorer().evaluate(submission)["score"] == 0


def test_correlation_bounds_and_units():
    time = np.arange(200) * 10.0
    # A stable oscillation supports a lower bound at this short lag cutoff.
    run = {"time": time, "energy": np.sin(2 * np.pi * np.arange(200) / 100)}
    bound = {"value_fs": 100, "bound": "lower", "max_lag": 10}
    assert correlation(run, bound)["bound"] == "lower"
    run["energy"] = (-1.0) ** np.arange(200)
    assert correlation(run, {"value_fs": 10, "bound": "upper"})["bound"] == "upper"


def test_verifier_absence_is_pending_and_disagreement_fails(submission):
    offline = WorkflowScorer(9)
    assert offline.evaluate(submission)["score"] is None
    with pytest.raises(PendingReviewError):
        offline(submission)

    def wrong(plan, fingerprint):
        result = response(plan, fingerprint)
        result["jobs"][0]["value"] = {
            "frames": [{"energy": 1000, "forces": [[0] * 3] * 32}] * 3
        }
        return result

    assert scorer(transport=wrong).evaluate(submission)["score"] == 0


def test_unbiased_and_integrated_correlation_are_recomputed():
    rng = np.random.default_rng(3)
    energy = np.zeros(200)
    for i in range(1, 200):
        energy[i] = 0.7 * energy[i - 1] + rng.normal()
    y = energy - energy.mean()
    acf = np.array([y[: 200 - k] @ y[k:] / (y @ y) for k in range(51)])
    run = {"energy": energy, "time": np.arange(200) * 10.0}
    estimate = {
        "method": "integrated",
        "cutoff_lag": 8,
        "value_fs": 10 * (0.5 + acf[1:9].sum()),
        "bound": "estimate",
    }
    assert correlation(run, estimate)["value_fs"] == pytest.approx(estimate["value_fs"])
    acf *= 200 / (200 - np.arange(51))
    lag = int(np.flatnonzero(acf <= np.exp(-1))[0])
    estimate = {
        "normalization": "unbiased",
        "value_fs": 10 * lag,
        "bound": "estimate",
        "acf": acf.tolist(),
    }
    assert correlation(run, estimate)["value_fs"] == 10 * lag


@pytest.mark.parametrize("options", [{}, {"max_lag": 10, "bound": "lower"}])
def test_energy_drift_cannot_be_reported_as_a_correlation_estimate_or_bound(options):
    run = {"time": np.arange(200) * 10.0, "energy": np.linspace(0, 1000, 200)}
    with pytest.raises(EvidenceError, match="nonstationary"):
        correlation(run, options)


def test_correlation_must_be_stable_across_production_windows():
    energy = np.r_[
        np.sin(2 * np.pi * np.arange(100) / 80), np.sin(2 * np.pi * np.arange(100) / 10)
    ]
    run = {"time": np.arange(200) * 10.0, "energy": energy}
    with pytest.raises(EvidenceError, match="production-window sensitivity"):
        correlation(run)


@pytest.mark.parametrize("key", ["window_spread_fs", "energy_sampling"])
def test_optional_correlation_diagnostics_must_match_the_recorded_energy(key):
    energy = np.sin(2 * np.pi * np.arange(200) / 10)
    run = {"time": np.arange(200) * 10.0, "energy": energy}
    expected = correlation(run)
    correlation(run, {key: expected[key]})
    claim = {key: 1000}
    if key == "energy_sampling":
        claim[key] = {"energy_standard_error_eV": 1000}
    with pytest.raises(EvidenceError, match="Incorrect"):
        correlation(run, claim)


def test_optional_energy_sampling_claims_are_checked_for_each_run(submission):
    answer = json.loads(submission.read_text())
    answer["results"]["energy_sampling"] = {
        "independent_500": {"energy_standard_error_eV": 1000}
    }
    submission.write_text(json.dumps(answer))
    report = scorer().evaluate(submission)
    assert report["score"] == 0
    assert "Incorrect energy sampling" in report["checks"][0]["detail"]


def test_integrated_estimate_that_changes_between_halves_is_unresolved():
    # This record previously passed the cutoff test, despite unstable time windows.
    rng = np.random.default_rng(4)
    energy = np.zeros(200)
    for i in range(1, 200):
        energy[i] = 0.7 * energy[i - 1] + rng.normal()
    with pytest.raises(EvidenceError, match="production-window sensitivity"):
        correlation(
            {"time": np.arange(200) * 10.0, "energy": energy},
            {"method": "integrated", "cutoff_lag": 8},
        )


@pytest.mark.parametrize(("level", "run_name"), [(1, "main"), (2, "independent_500")])
def test_energy_stationarity_is_required_for_each_submitted_run(
    submission, level, run_name
):
    path = submission.parent / f"{run_name}.jsonl"
    records = rows(path)
    production = False
    for record in records:
        if record["event"] == "stage":
            production = record["name"] == "production"
        if production and record["event"] == "sample":
            record["potential_energy_eV"] += record["step"] * 1000
    write_rows(path, records)
    report = scorer(level).evaluate(submission)
    assert report["score"] == 0
    assert "nonstationary" in report["checks"][0]["detail"]


@pytest.mark.parametrize("change", ["soap", "unavailable", "identity", "missing"])
def test_independent_checks_cannot_silently_pass(submission, change):
    def changed(plan, fingerprint):
        result = response(plan, fingerprint)
        if change == "soap":
            job = next(j for j in result["jobs"] if j["id"].startswith("soap_"))
            job["value"] = {
                "features": (np.array(job["value"]["features"]) + 1).tolist()
            }
        elif change == "unavailable":
            result["jobs"][0]["status"] = "unsupported"
        elif change == "identity":
            result["challenge"] = "incorrect"
        else:
            result["jobs"].pop()
        return result

    report = scorer(transport=changed).evaluate(submission)
    assert report["score"] is None if change == "unavailable" else report["score"] == 0
