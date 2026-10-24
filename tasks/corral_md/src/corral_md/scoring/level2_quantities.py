"""Level-2 final numerical quantities from saved logs, without workflow gates.

This policy checks log-derived answers, not MD provenance, convergence, model
identity, fitting history, or independent accuracy of the saved trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from ase import Atoms, units

from corral.evaluation import SubmissionScore

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    finite_array,
    result_close,
)
from .final_quantities import correlation_values, require, scalar, value
from .formats import records
from .regression import metrics, predict

VERSION = "level2-final-quantities-v1"
RUNS = ("main", "independent_800", "independent_500", "independent_1100")
STRATEGIES = ("randomized", "time_aware")


def kind(row):
    return str(row.get("event", row.get("type", row.get("kind", "")))).lower()


def log_rows(path):
    """Flatten scalar and staged logs while retaining numerical context."""
    data = records(path)
    header, rows = {}, []

    def visit(batch, context):
        require(isinstance(batch, list), "Log samples must be a list")
        context = dict(context)
        for row in batch:
            require(isinstance(row, Mapping), "Log records must be objects")
            label = kind(row)
            if label in {"header", "start", "setup", "meta", "metadata"}:
                header.update(row)
                context.update(row)
            elif label in {"stage", "stage_start"}:
                context = {**header, **row}
            elif label not in {
                "end",
                "finish",
                "stage_end",
                "restart",
                "checkpoint",
                "progress",
                "note",
                "info",
                "warning",
            }:
                inherited = {
                    k: v
                    for k, v in context.items()
                    if k not in {"event", "type", "kind"}
                }
                rows.append({**inherited, **row})

    if isinstance(data, list):
        visit(data, {})
    else:
        require(isinstance(data, Mapping), "Log must contain numerical records")
        header.update(
            data.get("metadata", data.get("header", data.get("settings", {})))
        )
        stages = data.get("stages", [data])
        require(isinstance(stages, list), "Stages must be a list")
        for stage in stages:
            batch = next(
                (
                    stage[k]
                    for k in ("samples", "frames", "records", "rows")
                    if k in stage
                ),
                None,
            )
            if batch is None:
                rows.append({**header, **stage})
            else:
                context = {
                    **header,
                    **{
                        k: v
                        for k, v in stage.items()
                        if k not in {"samples", "frames", "records", "rows"}
                    },
                }
                visit(batch, context)
    return header, rows


def production_rows(rows):
    result = []
    for row in rows:
        labels = [str(row.get(k, "")).lower() for k in ("part", "phase", "name")]
        labels.append(kind(row))
        if any(s in {"production", "prod", "nve"} for s in labels):
            result.append(row)
        elif any(
            s in {"equilibration", "equil", "eq", "initial", "summary", "vdos"}
            for s in labels
        ):
            continue
        else:
            cutoff = value(
                row, "equilibration_steps", "n_equil_steps", "equil_steps", "n_equil"
            )
            step = value(row, "local_step", "stage_step", "step")
            if cutoff is not None and step is not None:
                if float(step) > float(cutoff):
                    result.append(row)
            else:
                raise UnsupportedEvidence(
                    "Cannot identify the production samples in this log"
                )
    require(bool(result), "No production samples")
    return result


def compare(actual, expected, name, atol=1e-8):
    require(
        result_close(actual, expected, atol=atol),
        f"Incorrect {name}; recomputed {expected}",
    )


def required_result(results, name):
    require(name in results, f"Missing requested numerical result: {name}")
    return results[name]


def vdos(evidence):
    from .task_6 import spectral_summaries

    header, rows = log_rows(evidence.artifact("run_log"))
    spectra = [
        r
        for r in rows
        if value(r, "frequency_THz", "frequencies_THz", "freq_THz") is not None
    ]
    if spectra:
        require(len(spectra) == 1, "Expected one final VDOS spectrum")
        row = spectra[0]
        frequency = value(row, "frequency_THz", "frequencies_THz", "freq_THz")
        power = value(row, "vdos", "density", "power", "spectrum")
        require(power is not None, "Missing numerical VDOS weights")
        source = "logged_spectrum"
    else:
        from scipy.signal import welch

        velocity_rows = [
            r
            for r in rows
            if value(r, "velocities", "velocity", "v", "momenta", "momenta_ase_units")
            is not None
            and (
                kind(r) in {"vel", "velocity"}
                or any(
                    str(r.get(k, "")).lower() in {"production", "prod"}
                    for k in ("part", "phase", "name")
                )
            )
        ]
        require(
            len(velocity_rows) >= 3, "Missing VDOS spectrum or production velocities"
        )
        velocities = []
        for row in velocity_rows:
            velocity = value(row, "velocities", "velocity", "v")
            if velocity is None:
                momentum = finite_array(
                    value(row, "momenta", "momenta_ase_units"), ndim=2
                )
                masses = finite_array(
                    value(
                        row,
                        "masses_amu",
                        "masses",
                        default=[26.9815385] * len(momentum),
                    ),
                    shape=(len(momentum),),
                )
                require(np.all(masses > 0), "Atomic masses must be positive")
                velocity = momentum / masses[:, None]
            velocities.append(velocity)
        velocity = finite_array(velocities, ndim=3)
        require(velocity.shape[-1] == 3, "Velocity samples need three components")
        times = finite_array([scalar(r, "time_fs") for r in velocity_rows], ndim=1)
        dt = uniform_spacing(times)
        settings = next((r for r in rows if kind(r) == "vdos"), header)
        segment = int(settings.get("nperseg", min(len(times), int(np.ceil(4000 / dt)))))
        overlap = int(settings.get("noverlap", segment // 2))
        require(
            3 <= segment <= len(times) and 0 <= overlap < segment,
            "Invalid Welch segment or overlap",
        )
        frequency, power = welch(
            velocity,
            fs=1000 / dt,
            window="hann",
            nperseg=segment,
            noverlap=overlap,
            detrend="constant",
            scaling="density",
            axis=0,
        )
        power = power.sum(axis=(1, 2))
        source = "logged_velocities_hann_welch"
    computed = spectral_summaries(frequency, power)
    for name in ("mean_frequency_THz", "std_frequency_THz"):
        compare(
            required_result(evidence.results, name), computed[name], name, atol=1e-6
        )
    return {**computed, "reference_basis": source}


def target(row):
    return value(row, "target_temperature_K", "target_K", "T_target", "target_T_K")


def stages_from_rows(rows):
    groups = {}
    for row in production_rows(rows):
        stage = value(row, "stage_index", "stage", "index")
        if stage is None or str(stage).lower() in {"production", "prod"}:
            stage = (row.get("branch", row.get("phase", "")), target(row))
        require(stage is not None, "Cannot associate production samples with a stage")
        groups.setdefault(stage, []).append(row)
    return list(groups.items())


def temperature(row, *, atoms=108):
    momenta = value(row, "momenta", "momenta_ase_units")
    if momenta is not None:
        p = finite_array(momenta, shape=(atoms, 3))
        mass = finite_array(
            value(row, "masses_amu", "masses", default=[26.9815385] * atoms),
            shape=(atoms,),
        )
        require(np.all(mass > 0), "Atomic masses must be positive")
        return float(np.sum(p**2 / mass[:, None]) / (3 * atoms * units.kB))
    return scalar(row, "temperature_K", "Temperature", "T_K", "T")


def expansion(evidence):
    _, rows = log_rows(evidence.artifact("run_log"))
    stages, cooling, previous = [], False, -np.inf
    for index, batch in stages_from_rows(rows):
        if str(batch[0].get("ensemble", "NPT")).upper() == "NVT":
            continue
        t_target = target(batch[0])
        require(t_target is not None, "Expansion stage needs its nominal temperature")
        t_target = float(t_target)
        branch = str(batch[0].get("branch", batch[0].get("phase", ""))).lower()
        cooling = cooling or t_target < previous
        heating = branch in {"heat", "heating"} or (
            branch not in {"cool", "cooling"} and not cooling
        )
        previous = t_target
        volumes = []
        for row in batch:
            volume = value(row, "volume_A3", "volume", "Volume", "V")
            if volume is None:
                volume = abs(
                    np.linalg.det(
                        finite_array(value(row, "cell_A", "cell"), shape=(3, 3))
                    )
                )
            volumes.append(float(finite_array(volume, shape=())))
        require(np.all(np.asarray(volumes) > 0), "Volumes must be positive")
        stages.append(
            {
                "stage": index,
                "target_temperature_K": t_target,
                "heating": heating,
                "temperature_K": float(np.mean([temperature(r) for r in batch])),
                "volume_A3": float(np.mean(volumes)),
                "sample_count": len(batch),
            }
        )
    heating = [
        s for s in stages if s["heating"] and 300 <= s["target_temperature_K"] <= 500
    ]
    require(
        {300, 400, 500}.issubset({s["target_temperature_K"] for s in heating}),
        "Thermal expansion needs heating samples at 300, 400 and 500 K",
    )
    reference = next(
        s["volume_A3"] for s in heating if s["target_temperature_K"] == 300
    )
    slope, intercept = linear_fit(
        [s["temperature_K"] for s in heating], [s["volume_A3"] for s in heating]
    )
    beta = slope / reference
    compare(required_result(evidence.results, "beta_volume"), beta, "beta_volume")
    return {
        "beta_volume": beta,
        "volume_slope_A3_per_K": slope,
        "volume_intercept_A3": intercept,
        "reference_volume_A3": reference,
        "stages": stages,
        "reference_basis": "logged_production_samples",
    }


def linear_fit(x, y, options=None):
    x, y = finite_array(x, ndim=1), finite_array(y, ndim=1)
    require(
        x.shape == y.shape and len(x) >= 2 and np.ptp(x) > 1e-8,
        "Fit requires distinct measured temperatures",
    )
    options = options or {}
    require(isinstance(options, Mapping), "Fit settings must be an object")
    method = options.get("method", "ols")
    require(method in {"ols", "wls", "theil_sen"}, "Unknown fit method")
    if method == "theil_sen":
        slope = float(
            np.median(
                [
                    (y[j] - y[i]) / (x[j] - x[i])
                    for i in range(len(x))
                    for j in range(i + 1, len(x))
                    if x[j] != x[i]
                ]
            )
        )
        return slope, float(np.median(y - slope * x))
    weights = (
        finite_array(options.get("weights"), shape=x.shape)
        if method == "wls"
        else np.ones_like(x)
    )
    require(np.all(weights > 0), "Fit weights must be positive")
    weights /= weights.max()
    xm, ym = np.average(x, weights=weights), np.average(y, weights=weights)
    variance = np.sum(weights * (x - xm) ** 2)
    require(variance > 1e-12, "Fit does not resolve a temperature slope")
    slope = float(np.sum(weights * (x - xm) * (y - ym)) / variance)
    return slope, float(ym - slope * xm)


def total_energy_per_atom(row):
    direct = value(
        row,
        "energy_eV_per_atom",
        "etot_per_atom_eV",
        "Etot_per_atom_eV",
        "total_energy_eV_per_atom",
    )
    if direct is not None:
        return float(finite_array(direct, shape=()))
    total = value(row, "total_energy_eV", "etot_eV", "Etot_eV", "Etot")
    if total is None:
        kinetic = value(row, "kinetic_energy_eV", "ekin_eV", "Ekin_eV", "Ekin")
        if kinetic is None:
            # Task 10 defines temperature with 3N degrees of freedom. This also
            # handles logs that retain momenta and potential energy only.
            kinetic = 1.5 * 108 * units.kB * temperature(row)
        total = scalar(
            row, "potential_energy_eV", "epot_eV", "Epot_eV", "Epot"
        ) + float(finite_array(kinetic, shape=()))
    return float(finite_array(total, shape=())) / 108


def heat_capacity(evidence):
    _, rows = log_rows(evidence.artifact("run_log"))
    groups = stages_from_rows(rows)
    # Explicit indices have task-defined meaning, regardless of JSON row ordering.
    if all(type(index) is int for index, _ in groups):
        groups.sort(key=lambda pair: pair[0])
        require(
            [i for i, _ in groups[:5]] == list(range(5)),
            "Heat capacity needs forward stages 0 through 4",
        )
    require(len(groups) >= 5, "Heat capacity needs five forward temperature stages")
    stages = [
        {
            "stage": index,
            "temperature_K": float(np.mean([temperature(r) for r in batch])),
            "energy_eV_per_atom": float(
                np.mean([total_energy_per_atom(r) for r in batch])
            ),
            "sample_count": len(batch),
        }
        for index, batch in groups
    ]
    slope, intercept = linear_fit(
        [s["temperature_K"] for s in stages[:5]],
        [s["energy_eV_per_atom"] for s in stages[:5]],
        evidence.results.get("fit"),
    )
    claim = required_result(evidence.results, "heat_capacity")
    require(isinstance(claim, Mapping), "Heat capacity must be an object")
    compare(
        required_result(claim, "eV_per_atom_K"), slope, "heat_capacity.eV_per_atom_K"
    )
    return {
        "heat_capacity": {
            "eV_per_atom_K": slope,
            "kB_per_atom": slope / units.kB,
            "intercept_eV_per_atom": intercept,
        },
        "stages": stages,
        "reference_basis": "logged_production_samples",
    }


def uniform_spacing(times):
    spacing = np.diff(times)
    require(
        len(spacing) > 0 and np.all(spacing > 0) and np.allclose(spacing, spacing[0]),
        "Expected chronological uniformly spaced samples",
    )
    return float(spacing[0])


def strategy(name):
    return {
        "random_split": "randomized",
        "randomized_split": "randomized",
        "time_aware_split": "time_aware",
        "time-aware": "time_aware",
    }.get(name, name)


def indices(values, size):
    a = finite_array(values, ndim=1)
    require(
        np.all(a == np.floor(a))
        and np.all((a >= 0) & (a < size))
        and len(np.unique(a)) == len(a),
        "Invalid prediction row indices",
    )
    return a.astype(int)


def compare_metrics(claim, expected, name):
    require(isinstance(claim, Mapping), "Prediction metrics must be an object")
    for key, aliases in {
        "mae": ("mae", "MAE_eV", "mae_eV"),
        "rmse": ("rmse", "RMSE_eV", "rmse_eV"),
        "r2": ("r2", "R2", "R_squared"),
    }.items():
        actual = value(claim, *aliases)
        if actual is not None:
            compare(actual, expected[key], f"{name}.{key}", atol=1e-6)


def soap_features(rows, config):
    from dscribe.descriptors import SOAP

    parameters = {
        k: config[k]
        for k in (
            "r_cut",
            "n_max",
            "l_max",
            "sigma",
            "average",
            "rbf",
            "periodic",
            "species",
            "compression",
            "weighting",
        )
        if k in config
    }
    require(
        all(k in parameters for k in ("r_cut", "n_max", "l_max")),
        "Missing SOAP numerical settings",
    )
    parameters.setdefault("species", ["Cu"])
    parameters.setdefault("periodic", True)
    parameters.setdefault("average", "inner")
    frames = []
    for row in rows:
        pos = finite_array(value(row, "positions_A", "positions"), ndim=2)
        require(pos.shape[1] == 3, "Positions need three components")
        numbers = value(row, "numbers", "atomic_numbers", default=[29] * len(pos))
        frames.append(
            Atoms(
                numbers=numbers,
                positions=pos,
                cell=finite_array(value(row, "cell_A", "cell"), shape=(3, 3)),
                pbc=row.get("pbc", True),
            )
        )
    return finite_array(SOAP(**parameters).create(frames), ndim=2)


def fitted_predictions(x, y, test, specification, *, standardize=True):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if "coef" in specification:
        return predict(test, specification)
    alpha = scalar(specification, "selected_alpha", "alpha")
    require(alpha >= 0, "Ridge regularization must be nonnegative")
    scaler = StandardScaler(with_mean=standardize, with_std=standardize).fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return model.predict(scaler.transform(test))


def copper_metrics(evidence):
    runs, energies = {}, {}
    for name in RUNS:
        _, rows = log_rows(evidence.artifact(f"run_logs.{name}"))
        runs[name] = production_rows(rows)
        energies[name] = finite_array(
            [
                scalar(r, "potential_energy_eV", "energy_eV", "epot_eV", "Epot_eV")
                for r in runs[name]
            ],
            ndim=1,
        )
    _, rows = log_rows(evidence.artifact("regression_log"))
    raw = records(evidence.artifact("regression_log"))
    config = raw[0] if isinstance(raw, list) else raw.get("metadata", {})
    descriptor = next(
        (r for r in rows if kind(r) in {"descriptor", "soap"}), config.get("soap", {})
    )
    features = {
        r["run"]: finite_array(r["values"], ndim=2)
        for r in rows
        if kind(r) == "features"
    }
    fits = {
        strategy(r.get("strategy", r.get("name"))): r
        for r in rows
        if kind(r) in {"fit", "validation"}
    }
    final = next((r for r in rows if kind(r) in {"freeze", "final_model"}), None)
    evaluations = {
        strategy(r.get("strategy", r.get("name", r.get("run")))): r
        for r in rows
        if kind(r) in {"evaluate", "validation", "independent_test"}
    }

    def x(name):
        if name not in features:
            features[name] = soap_features(runs[name], descriptor)
        require(
            features[name].shape[0] == len(energies[name]),
            "Features and energy samples differ in length",
        )
        return features[name]

    outputs, prediction_sources = {}, {}
    for name in (*STRATEGIES, *RUNS[1:]):
        require(name in evaluations, f"Missing {name} prediction calculation")
        event = evaluations[name]
        if name in STRATEGIES:
            require(name in fits, f"Missing {name} validation partition")
            fit = fits[name]
            test = indices(fit.get("test_indices"), len(energies["main"]))
            y = energies["main"][test]
        else:
            y = energies[name]
        predictions = value(event, "predictions", "predictions_eV", "y_pred")
        if predictions is not None:
            predictions = finite_array(predictions, shape=y.shape)
            prediction_sources[name] = "logged_predictions"
        else:
            require(final is not None, "Missing final Ridge calculation settings")
            if name in STRATEGIES:
                train = indices(fit.get("train_indices"), len(energies["main"]))
                require(
                    not np.intersect1d(train, test).size,
                    "Training and validation rows overlap",
                )
                specification = fit.get("model", fit)
                predictions = fitted_predictions(
                    x("main")[train],
                    energies["main"][train],
                    x("main")[test],
                    specification,
                    standardize=config.get("standardize", True),
                )
            else:
                predictions = fitted_predictions(
                    x("main"),
                    energies["main"],
                    x(name),
                    final.get("model", final),
                    standardize=config.get("standardize", True),
                )
            prediction_sources[name] = "recomputed_soap_ridge"
        outputs[name] = metrics(y, predictions)
        compare_metrics(
            event.get("test_metrics", event.get("metrics", {})), outputs[name], name
        )
    for name, claim in evidence.results.get("metrics", {}).items():
        require(strategy(name) in outputs, f"Unknown prediction metric group {name}")
        compare_metrics(claim, outputs[strategy(name)], name)
    # Correlation diagnostics do not block prediction-metric scoring.
    try:
        times = finite_array([scalar(r, "time_fs") for r in runs["main"]], ndim=1)
        correlation = correlation_values(energies["main"], uniform_spacing(times))
    except (EvidenceError, UnsupportedEvidence) as exc:
        correlation = {"status": "unavailable", "detail": str(exc)}
    return {
        "metrics": outputs,
        "correlation": correlation,
        "prediction_sources": prediction_sources,
        "reference_basis": "logged_labels_and_predictions_or_recomputed_soap_ridge",
    }


class Level2QuantityScorer:
    """Default minimal policy for level-2 tasks 6, 7, 9 and 10."""

    def __init__(self, task_number):
        if type(task_number) is not int or task_number not in (6, 7, 9, 10):
            raise ValueError("No level-2 final-quantity policy for this task")
        self.task_number, self.level = task_number, 2

    def evaluate(self, submission):
        rubric, computed, evidence = Rubric(self.task_number, binary=True), {}, None

        def calculate():
            nonlocal evidence
            evidence = Evidence(submission, task_number=self.task_number, level=2)
            function = {6: vdos, 7: expansion, 9: copper_metrics, 10: heat_capacity}[
                self.task_number
            ]
            computed.update(function(evidence))
            return True

        rubric.check("final_quantities", 100, calculate)
        return {
            **rubric.as_dict(),
            "level": 2,
            "mode": "final_quantities",
            "scoring_policy": VERSION,
            "computed": computed,
            "evidence_sha256": evidence.fingerprint() if evidence else None,
        }

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        feedback = (
            "; ".join(c["detail"] for c in report["checks"] if c["status"] == "failed")
            or None
        )
        return SubmissionScore(
            score=report["score"],
            feedback=feedback,
            metadata={"workflow_evaluation": report},
        )

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    __call__ = score_submission
