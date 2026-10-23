"""Reconstruct copper correlation and regression answers from calculation logs.

No submitted code or serialized estimators are executed. Small independent MACE
and SOAP checks establish numerical labels/features without rerunning dynamics.
Thermal and correlation-resolution screens are explicit policy; calibration on
validated production runs remains necessary before freezing benchmark cutoffs.
"""

from __future__ import annotations

import json
import secrets

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)
from .dynamics import matches_pinned_cu32, sampled_motion
from .level1_trusted import PINNED_TEACHER_SHA256
from .regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
)
from .sampling import mean_standard_error
from .verification import Plan, compare, geometry

RUNS = {
    "main": 800,
    "independent_800": 800,
    "independent_500": 500,
    "independent_1100": 1100,
}
STRATEGIES = ("randomized", "time_aware")


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _number(value):
    _require(
        type(value) in (int, float) and np.isfinite(value), "Expected a finite number"
    )
    return float(value)


def _records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def read_run(path, target):
    rows = _records(path)
    header = rows[0]
    _require(
        header["event"] == "start" and header["model_sha256"] == PINNED_TEACHER_SHA256,
        "Use the supplied MACE-MP-0 model",
    )
    _require(
        header["units"]
        == {"time": "fs", "position": "Angstrom", "energy": "eV", "momenta": "ASE"},
        "Incorrect run-log units",
    )
    _require(
        header["numbers"] == [29] * 32 and header["target_temperature_K"] == target,
        "Incorrect copper system or target temperature",
    )
    seed = header["random_seed"]
    _require(type(seed) is int, "Record an integer velocity seed")
    masses = finite_array(header["masses_amu"], shape=(32,))
    _require(
        close(masses, np.full(32, 63.546), rtol=0, atol=1e-3), "Incorrect copper masses"
    )
    cell = finite_array(header["cell_A"], shape=(3, 3))
    stages, current = [], None
    for row in rows[1:]:
        if row["event"] == "stage":
            _require(
                current is None
                and len(stages) < 2
                and row["name"] == ("equilibration" if not stages else "production"),
                "Incorrect stage order",
            )
            _require(
                all(type(row[k]) is int and row[k] > 0 for k in ("steps", "interval"))
                and row["steps"] % row["interval"] == 0,
                "Invalid sampling schedule",
            )
            _require(
                0 < _number(row["timestep_fs"]) <= 2, "Use a timestep of at most 2 fs"
            )
            _require(
                row["target_temperature_K"] == target
                and row["integrator"]
                in {
                    "Langevin",
                    "NVTBerendsen",
                    "NoseHooverChainNVT",
                    "Bussi",
                    "Andersen",
                },
                "Use fixed-cell thermostatted dynamics at the target temperature",
            )
            current = {**row, "samples": []}
        elif row["event"] == "sample":
            _require(current is not None, "Sample outside a stage")
            current["samples"].append(row)
        elif row["event"] == "end":
            _require(
                current is not None
                and row["name"] == current["name"]
                and row["steps"] == current["steps"],
                "Incomplete stage",
            )
            stages.append(current)
            current = None
        else:
            raise EvidenceError("Unexpected trajectory log event")
    _require(
        current is None and len(stages) == 2, "Truncated equilibration/production log"
    )
    offset = 0.0
    for stage in stages:
        samples = stage["samples"]
        n = len(samples)
        _require(
            n == stage["steps"] // stage["interval"] + 1
            and (n == 200 if stage["name"] == "production" else n >= 16),
            "Retain >=16 equilibration samples and exactly 200 production frames",
        )
        steps = np.arange(n) * stage["interval"]
        _require(
            all(type(row["step"]) is int for row in samples)
            and np.array_equal([row["step"] for row in samples], steps),
            "Missing or reordered samples",
        )
        time = finite_array([row["time_fs"] for row in samples], shape=(n,))
        _require(
            close(time, offset + steps * stage["timestep_fs"], rtol=0, atol=1e-6),
            "Samples must be chronological and uniformly spaced in fs",
        )
        position = finite_array(
            [row["positions_A"] for row in samples], shape=(n, 32, 3)
        )
        momenta = finite_array([row["momenta"] for row in samples], shape=(n, 32, 3))
        energy = finite_array(
            [row["potential_energy_eV"] for row in samples], shape=(n,)
        )
        forces = finite_array([row["forces_eV_A"] for row in samples], shape=(n, 32, 3))
        frames = [
            Atoms("Cu32", positions=p, cell=cell, pbc=True, masses=masses, momenta=m)
            for p, m in zip(position, momenta, strict=True)
        ]
        temperature = np.sum(momenta**2 / masses[None, :, None], axis=(1, 2)) / (
            96 * units.kB
        )
        stage.update(
            time=time,
            frames=frames,
            energy=energy,
            forces=forces,
            temperature=temperature,
        )
        offset = time[-1]
    eq, prod = stages
    initial = eq["frames"][0]
    _require(matches_pinned_cu32(initial), "Initialize from the supplied Cu32 cell")
    expected = initial.copy()
    MaxwellBoltzmannDistribution(
        expected, temperature_K=target, rng=np.random.default_rng(seed)
    )
    Stationary(expected, preserve_temperature=True)
    _require(
        close(initial.get_momenta(), expected.get_momenta(), rtol=0, atol=1e-7),
        "Initial momenta do not reproduce the recorded seed and COM removal",
    )
    for key in (
        "positions_A",
        "momenta",
        "potential_energy_eV",
        "forces_eV_A",
        "time_fs",
    ):
        _require(
            close(eq["samples"][-1][key], prod["samples"][0][key], rtol=0, atol=1e-7),
            f"Equilibration-to-production discontinuity: {key}",
        )
    _require(
        sampled_motion(prod["frames"], prod["time"]),
        "Production frames do not show resolved motion",
    )
    tail = eq["temperature"][len(eq["temperature"]) // 2 :]
    blocks = np.array([part.mean() for part in np.array_split(prod["temperature"], 4)])
    _require(
        abs(tail.mean() - target) <= 0.15 * target
        and np.max(np.abs(blocks - target)) <= 0.2 * target
        and np.ptp(blocks) <= 0.2 * target,
        "Insufficient thermal settling or unstable production temperature",
    )
    return {
        "seed": seed,
        "initial_momenta": initial.get_momenta(),
        "energy_sampling": energy_stationarity(prod["energy"]),
        **prod,
    }


def energy_stationarity(energy):
    """Reject drift and changing fluctuation scales before interpreting an ACF.

    This is a coarse stationarity screen, not a thermodynamic equilibrium test.
    Relative limits allow physical energy fluctuations without a model reference.
    """
    energy = finite_array(energy, ndim=1)
    _require(len(energy) >= 32, "Insufficient energy samples for correlation analysis")
    sigma = float(energy.std(ddof=1))
    _require(sigma > 1e-10, "Constant energy has no defined normalized autocorrelation")
    coordinate = np.linspace(0, 1, len(energy))
    drift = float(np.polyfit(coordinate, energy - energy.mean(), 1)[0])
    blocks = np.array_split(energy, 4)
    means = np.array([part.mean() for part in blocks])
    deviations = np.array([part.std(ddof=1) for part in blocks])
    _require(
        abs(drift) <= 2 * sigma
        and np.ptp(means) <= 2 * sigma
        and deviations.min() >= 0.25 * deviations.max(),
        "Production energy is nonstationary: drift or changing fluctuations invalidate correlation",
    )
    return {
        "mean_energy_eV": float(energy.mean()),
        "energy_standard_error_eV": mean_standard_error(energy),
        "energy_std_eV": sigma,
        "fitted_energy_change_eV": drift,
    }


def correlation(run, reported=None):
    """Recompute a uniform-sampling ACF and a resolved time or censoring bound."""
    if reported is None:
        reported = {}
    _require(isinstance(reported, dict), "Correlation must be an object")
    sampling = energy_stationarity(run["energy"])
    y = run["energy"] - run["energy"].mean()
    variance = y @ y
    _require(
        variance > 1e-20, "Constant energy has no defined normalized autocorrelation"
    )
    dt = float(run["time"][1] - run["time"][0])
    cutoff = reported.get("max_lag", 50)
    _require(
        type(cutoff) is int and 10 <= cutoff <= 50,
        "Use 10-50 consecutive lags so each ACF lag has adequate observations",
    )
    acf = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) + cutoff] / variance
    normalization = reported.get("normalization", "biased")
    _require(
        normalization in {"biased", "unbiased"},
        "Use biased or unbiased ACF normalization",
    )
    if normalization == "unbiased":
        acf *= len(y) / (len(y) - np.arange(cutoff + 1))
    method = reported.get("method", "first_crossing")
    if method == "first_crossing":
        threshold = _number(reported.get("threshold", float(np.exp(-1))))
        _require(
            0 < threshold < 1, "ACF crossing threshold must lie between zero and one"
        )
        crossed = np.flatnonzero(acf <= threshold)
        lag = int(crossed[0]) if len(crossed) else cutoff
        bound = "upper" if lag == 1 else "estimate" if len(crossed) else "lower"
        value = lag * dt
    elif method == "integrated":
        lag = reported["cutoff_lag"]
        _require(
            type(lag) is int and 2 <= lag <= cutoff // 2, "Invalid integration cutoff"
        )
        value = dt * (0.5 + acf[1 : lag + 1].sum())
        alternatives = [
            dt * (0.5 + acf[1 : k + 1].sum()) for k in (max(1, lag // 2), 2 * lag)
        ]
        _require(
            value > 0
            and max(abs(v - value) for v in alternatives) <= 0.25 * value
            and run["time"][-1] - run["time"][0] >= 10 * value,
            "Integrated correlation time is unresolved; use a supported crossing bound",
        )
        bound = "estimate"
    else:
        raise EvidenceError("Use first_crossing or integrated correlation")
    # Keep the original estimator, then independently repeat it on chronological
    # windows. The spread is a sensitivity diagnostic, not a confidence interval.
    estimates = []
    for series in (
        run["energy"][: len(y) // 2],
        run["energy"][len(y) // 2 :],
        run["energy"][len(y) // 4 :],
    ):
        centered = series - np.mean(series)
        norm = centered @ centered
        _require(
            norm > 1e-20, "Correlation is unresolved in a constant production window"
        )
        local_acf = (
            np.correlate(centered, centered, mode="full")[
                len(series) - 1 : len(series) + cutoff
            ]
            / norm
        )
        if normalization == "unbiased":
            local_acf *= len(series) / (len(series) - np.arange(cutoff + 1))
        if method == "integrated":
            estimate = dt * (0.5 + local_acf[1 : lag + 1].sum())
            _require(
                estimate > 0,
                "Integrated correlation is unresolved across production windows",
            )
        else:
            crossed = np.flatnonzero(local_acf <= threshold)
            if bound == "lower":
                _require(
                    not len(crossed),
                    "Correlation lower bound is not supported across production windows",
                )
                estimate = cutoff * dt
            else:
                _require(
                    len(crossed) > 0,
                    "Correlation crossing is unresolved across production windows",
                )
                estimate = int(crossed[0]) * dt
                if bound == "upper":
                    _require(
                        estimate == dt,
                        "Correlation upper bound is not supported across production windows",
                    )
        estimates.append(float(estimate))
    spread = max(abs(estimate - value) for estimate in estimates)
    if bound == "estimate":
        _require(
            np.ptp(run["time"]) >= 10 * value and spread <= max(dt, 0.5 * value),
            "Correlation estimate is unresolved: insufficient duration or production-window sensitivity",
        )
    expected = {
        "value_fs": value,
        "bound": bound,
        "acf": acf.tolist(),
        "sampling_interval_fs": dt,
        "observation_duration_fs": float(np.ptp(run["time"])),
        "window_estimates_fs": estimates,
        "window_spread_fs": float(spread),
        "energy_sampling": sampling,
    }
    _require(
        ("bound" not in reported or reported["bound"] == bound)
        and (
            "value_fs" not in reported
            or result_close(reported["value_fs"], value, atol=1e-6)
        ),
        f"Correlation answer differs from recomputed {bound}: {value:g} fs",
    )
    for key in (
        "acf",
        "sampling_interval_fs",
        "observation_duration_fs",
        "window_estimates_fs",
        "window_spread_fs",
    ):
        if key in reported:
            _require(
                result_close(reported[key], expected[key], atol=1e-6),
                f"Incorrect correlation {key}",
            )
    check_energy_claims(reported.get("energy_sampling", {}), sampling)
    return expected


def check_energy_claims(claims, expected):
    _require(isinstance(claims, dict), "Energy sampling diagnostics must be an object")
    for key, value in expected.items():
        if key in claims:
            _require(
                result_close(claims[key], value, atol=1e-6),
                f"Incorrect energy sampling {key}",
            )


def regression(path, runs):
    """Rebuild predictions and audit nested fits, budgets and freeze ordering."""
    rows = _records(path)
    config = rows[0]
    _require(
        config["event"] == "start" and rows[-1] == {"event": "end"},
        "Truncated regression log",
    )
    soap = config["soap"]
    _require(
        soap["species"] == ["Cu"]
        and soap["periodic"] is True
        and soap["average"] in {"inner", "outer"},
        "Use periodic global copper SOAP",
    )
    seed, ntrain, ntest, gap = (
        config[k] for k in ("random_seed", "train_size", "test_size", "gap")
    )
    _require(
        all(type(v) is int for v in (seed, ntrain, ntest, gap))
        and ntrain >= 16
        and ntest >= 2
        and gap >= 0
        and ntrain + ntest + gap <= 200,
        "Invalid validation seed or sample budgets",
    )
    alphas = finite_array(config["alphas"], ndim=1)
    _require(
        np.all(alphas > 0) and len(np.unique(alphas)) == len(alphas),
        "Invalid Ridge alpha grid",
    )
    _require(type(config["standardize"]) is bool, "standardize must be boolean")
    permutation = np.random.default_rng(seed).permutation(200)
    splits = {
        "randomized": (
            np.sort(permutation[ntest : ntest + ntrain]),
            np.sort(permutation[:ntest]),
        ),
        "time_aware": (
            np.arange(ntrain),
            np.arange(ntrain + gap, ntrain + gap + ntest),
        ),
    }
    features, models, outputs = {}, {}, {}
    final = None

    def checked_model(model, indices, alpha):
        x, y = features["main"][indices], runs["main"]["energy"][indices]
        _require(
            model["estimator"] == "Ridge"
            and model["fit_intercept"] is True
            and model.get("positive", False) is False
            and _number(model["alpha"]) == alpha
            and model["preprocessing"]
            == ("StandardScaler" if config["standardize"] else "identity"),
            "Model differs from the declared Ridge calculation",
        )
        _require(
            preprocessing_matches(x, model) and ridge_stationarity(x, y, model),
            "Preprocessing or Ridge coefficients do not match training-only data",
        )

    for row in rows[1:-1]:
        event = row["event"]
        if event == "features":
            name = row["run"]
            _require(
                name in runs
                and name not in features
                and (name == "main" or final is not None),
                "Independent evaluation data accessed before final freeze, or repeated features",
            )
            x = finite_array(row["values"], ndim=2)
            _require(
                x.shape[0] == 200
                and (not features or x.shape[1] == features["main"].shape[1]),
                "Descriptor dimensions or frame alignment differ",
            )
            features[name] = x
        elif event == "fit":
            name = row["name"]
            _require(
                final is None and name in STRATEGIES and name not in models,
                "Repeated fit or fitting after final freeze",
            )
            train, test = partition(row["train_indices"], row["test_indices"], 200)
            _require(
                np.array_equal(train, splits[name][0])
                and np.array_equal(test, splits[name][1]),
                "Randomized and time-aware splits must use the declared comparable budgets",
            )
            _require(
                len(row["trials"]) == len(alphas),
                "Missing model-selection calculations",
            )
            scores, fold_layout = [], None
            for trial, alpha in zip(row["trials"], alphas, strict=True):
                _require(
                    _number(trial["alpha"]) == alpha and len(trial["folds"]) >= 2,
                    "Incomplete alpha trial",
                )
                errors, layout, held_out = [], [], set()
                for fold in trial["folds"]:
                    a, b = partition(fold["train_indices"], fold["test_indices"], 200)
                    _require(
                        len(a) >= 2
                        and len(b) >= 2
                        and np.all(np.isin(a, train))
                        and np.all(np.isin(b, train))
                        and a.max() < b.min()
                        and not held_out.intersection(b.tolist()),
                        "Tuning leaks holdouts or reuses validation rows",
                    )
                    held_out.update(b.tolist())
                    layout.append((a.tolist(), b.tolist()))
                    checked_model(fold["model"], a, alpha)
                    p = predict(features["main"][b], fold["model"])
                    _require(
                        close(p, fold["predictions"], atol=1e-7),
                        "Incorrect inner-fold predictions",
                    )
                    errors.append(np.mean((p - runs["main"]["energy"][b]) ** 2))
                _require(
                    fold_layout is None or layout == fold_layout,
                    "Alpha candidates used different folds",
                )
                fold_layout = layout
                scores.append(np.mean(errors))
            chosen = alphas[np.argmin(scores)]
            checked_model(row["model"], train, chosen)
            models[name] = row["model"]
        elif event == "freeze":
            _require(
                final is None and set(models) == set(STRATEGIES),
                "Freeze after both validation fits",
            )
            final = row["model"]
            checked_model(final, np.arange(200), models["time_aware"]["alpha"])
        elif event == "evaluate":
            name = row["name"]
            _require(
                final is not None and name not in outputs,
                "Evaluation precedes freeze or is duplicated",
            )
            if name in STRATEGIES:
                test = splits[name][1]
                x, y, model = (
                    features["main"][test],
                    runs["main"]["energy"][test],
                    models[name],
                )
            else:
                _require(
                    name in RUNS and name != "main", "Unexpected evaluation dataset"
                )
                x, y, model = features[name], runs[name]["energy"], final
            p = predict(x, model)
            _require(
                close(p, row["predictions"], atol=1e-7), f"Incorrect {name} predictions"
            )
            outputs[name] = metrics(y, p)
        else:
            raise EvidenceError("Unexpected regression event")
    _require(
        set(outputs) == {*STRATEGIES, *tuple(RUNS)[1:]} and set(features) == set(RUNS),
        "Missing validation or independent trajectory evaluation",
    )
    return outputs, features, soap


def build_plan(evidence, runs, features, soap, challenge):
    plan = Plan(evidence, 9, challenge)
    for name, run in runs.items():
        values = [
            {"energy": float(y), "forces": f.tolist()}
            for y, f in zip(run["energy"], run["forces"], strict=True)
        ]
        plan.frames(
            f"teacher_{name}",
            run["frames"],
            count=3,
            values=values,
            parameters={
                "model": "teacher.model",
                "default_dtype": "float64",
                "dispersion": False,
            },
        )
        if features:
            indices = plan.sample(200, 3, f"soap_{name}")
            plan.add(
                f"soap_{name}",
                "soap",
                {
                    "frames": [geometry(run["frames"][i]) for i in indices],
                    "sample_indices": indices,
                },
                {"features": features[name][indices].tolist()},
                parameters=soap,
                tolerance={"features": [1e-6, 1e-8]},
            )
    return plan


def evaluate_submission(submission, *, level, verifier=None, review=None):
    rubric = Rubric(9, binary=True, fail_fast=True)
    computed, runs, features, soap, verification = {}, {}, {}, {}, {}
    try:
        evidence = Evidence(submission)
        fingerprint = evidence.fingerprint()
    except Exception as exc:
        rubric.check("calculation_logs", 100, False, str(exc))
        return {**rubric.as_dict(), "level": level, "computed": computed}

    def trajectories():
        for name in RUNS if level == 2 else ("main",):
            runs[name] = read_run(evidence.artifact(f"run_logs.{name}"), RUNS[name])
        _require(
            len({run["seed"] for run in runs.values()}) == len(runs),
            "Velocity seeds must be distinct",
        )
        initial = [run["initial_momenta"] for run in runs.values()]
        _require(
            all(not close(a, b) for i, a in enumerate(initial) for b in initial[:i]),
            "Independent trajectories reuse initial momenta",
        )
        computed["production_temperature_K"] = {
            name: float(run["temperature"].mean()) for name, run in runs.items()
        }
        computed["energy_sampling"] = {
            name: run["energy_sampling"] for name, run in runs.items()
        }
        energy_claims = evidence.results.get("energy_sampling", {})
        _require(isinstance(energy_claims, dict), "Energy sampling must be an object")
        for name, value in computed["energy_sampling"].items():
            check_energy_claims(energy_claims.get(name, {}), value)
        claims = evidence.results.get("production_temperature_K", {})
        _require(isinstance(claims, dict), "Production temperatures must be an object")
        for name, value in computed["production_temperature_K"].items():
            if name not in claims:
                continue
            _require(
                result_close(claims[name], value, atol=1),
                f"Incorrect production temperature for {name}",
            )
        return True

    rubric.check("recorded_copper_trajectories", 30, trajectories)

    def temporal():
        claims = evidence.results.get("correlation", {})
        _require(isinstance(claims, dict), "Correlation must be an object")
        computed["correlation"] = correlation(runs["main"], claims)
        return True

    rubric.check("energy_correlation", 50 if level == 1 else 20, temporal)

    if level == 2:

        def predictions():
            output, x, parameters = regression(
                evidence.artifact("regression_log"), runs
            )
            features.update(x)
            soap.update(parameters)
            computed["metrics"] = output
            metrics_claims = evidence.results.get("metrics", {})
            _require(isinstance(metrics_claims, dict), "Metrics must be an object")
            for name, values in output.items():
                claims = metrics_claims.get(name, {})
                _require(
                    isinstance(claims, dict), f"Metrics for {name} must be an object"
                )
                for key, value in values.items():
                    if key not in claims:
                        continue
                    _require(
                        result_close(claims[key], value, atol=1e-7),
                        f"Incorrect {name} {key}: recomputed {value:g}",
                    )
            return True

        rubric.check("predictions_and_metrics", 30, predictions)

    def independent():
        if verifier is None:
            raise UnsupportedEvidence(
                "Small independent teacher/SOAP checks require a local or Modal backend"
            )
        plan = build_plan(evidence, runs, features, soap, secrets.token_hex(32))
        try:
            response = verifier._calculate(plan, fingerprint)
        except Exception as exc:
            raise UnsupportedEvidence(
                f"Independent calculations unavailable: {exc}"
            ) from exc
        _require(
            response.get("evidence_sha256") == fingerprint
            and response.get("challenge") == plan.challenge,
            "Independent response has a different evidence identity",
        )
        jobs = response.get("jobs", [])
        _require(
            len(jobs) == len(plan.jobs)
            and {j.get("id") for j in jobs} == set(plan.expected),
            "Missing or mismatched independent checks",
        )
        verification.update(release_id=response.get("release_id"), checks=len(jobs))
        unavailable = []
        for job in jobs:
            if job.get("status") == "unsupported":
                unavailable.append(job.get("detail", job["id"]))
                continue
            expected = plan.expected[job["id"]]
            _require(
                job.get("status") == "complete"
                and compare(
                    expected["values"], job.get("value"), expected["tolerance"]
                ),
                f"Independent calculation disagrees: {job['id']}",
            )
        if unavailable:
            raise UnsupportedEvidence("; ".join(unavailable))
        return True

    rubric.check("independent_model_calculation", 20, independent)
    if evidence.fingerprint() != fingerprint:
        raise RuntimeError("Submission changed during grading; retry with frozen logs")
    if review is not None:
        rubric.apply_review(review, fingerprint)
    return {
        **rubric.as_dict(),
        "level": level,
        "mode": "saved_logs",
        "computed": computed,
        "evidence_sha256": fingerprint,
        "verification": verification,
    }
