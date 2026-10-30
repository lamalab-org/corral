"""Saved harmonic trajectories and real sklearn fits exercise the log scorer."""

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from corral_md.scoring import check_level1_workflow, check_level2_workflow
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    for seed, (name, target) in zip(
        (2, 4, 5, 6),
        {
            "main": 800,
            "independent_800": 800,
            "independent_500": 500,
            "independent_1100": 1100,
        }.items(),
        strict=True,
    ):
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
            # Both components lie in the synthetic descriptor's position basis,
            # so a valid chronological Ridge fit clears the historical R2 floor.
            energy = -100 + 0.2 * np.sin(angle) + 0.04 * np.cos(angle)
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
                "velocity_rng": "numpy.default_rng",
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


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("representation", ["nested", "aliased"])
def test_copper_log_layout_does_not_change_scientific_score(
    submission, rewrite_md_recording, level, representation
):
    answer = json.loads(submission.read_text())
    if level == 1:
        answer = {
            "artifacts": {"run_logs": {"main": answer["artifacts"]["run_logs"]["main"]}}
        }
        submission.write_text(json.dumps(answer))
    for name in answer["artifacts"]["run_logs"].values():
        rewrite_md_recording(submission.parent / name, representation)
    factory = check_level1_workflow if level == 1 else check_level2_workflow
    report = factory(9).evaluate(submission)
    assert report["score"] == 1, report


def test_default_level2_final_quantity_policy_accepts_numerical_legacy_log(submission):
    from corral_md.scoring import check_level2_workflow

    report = check_level2_workflow(9, verification_backend="offline").evaluate(
        submission
    )
    assert report["score"] == 1, report
    assert report["scoring_policy"] == "level2-final-quantities-v2"
