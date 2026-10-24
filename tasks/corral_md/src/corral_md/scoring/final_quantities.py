"""Level-1 scoring of final observables, without workflow/evidence-shape gates.

Only task 5 has a fixed-model reference. For the stochastic tasks, values are
recomputed from the submitted production data and compared with any reported
answers. This does not independently authenticate a trajectory or its model.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

import numpy as np
from ase import units

from corral.evaluation import SubmissionScore

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    finite_array,
    result_close,
)
from .formats import records

VERSION = "level1-final-quantities-v1"


def require(condition, detail):
    if not condition:
        raise EvidenceError(detail)


def value(row, *names, default=None):
    present = [name for name in names if name in row]
    if not present:
        return default
    answer = row[present[0]]
    for name in present[1:]:
        require(
            result_close(answer, row[name]), f"Conflicting numerical fields: {present}"
        )
    return answer


def scalar(row, *names):
    answer = value(row, *names)
    if answer is None:
        raise UnsupportedEvidence(f"Cannot identify the numerical quantity {names[0]}")
    return float(finite_array(answer, shape=()))


def read_samples(path):
    """Read flat or staged scalar logs; atomic arrays are never required."""
    data = records(path)
    if isinstance(data, Mapping) and ("phase" in data or "step" in data):
        data = [data]
    header, samples, summaries = {}, [], []

    def visit(items, inherited):
        require(isinstance(items, list), "Numerical samples must be a list")
        context = dict(inherited)
        for item in items:
            require(
                isinstance(item, Mapping), "Each numerical record must be an object"
            )
            label = str(
                item.get("event", item.get("type", item.get("kind", "")))
            ).lower()
            if label in {"start", "header", "meta", "metadata", "stage", "stage_start"}:
                context.update(item)
                header.update(item)
            elif label == "summary":
                summaries.append(dict(item))
            elif label not in {
                "end",
                "finish",
                "complete",
                "stage_end",
                "restart",
                "checkpoint",
                "progress",
                "info",
                "note",
                "warning",
            }:
                samples.append({**context, **item})

    if isinstance(data, list):
        visit(data, {})
    else:
        header.update(
            data.get("header", data.get("metadata", data.get("settings", {})))
        )
        stages = data.get("stages", [data])
        require(isinstance(stages, list), "Stages must be a list")
        for stage in stages:
            context = {
                **header,
                **{
                    k: v
                    for k, v in stage.items()
                    if k not in {"samples", "frames", "records", "rows"}
                },
            }
            batch = next(
                (
                    stage[k]
                    for k in ("samples", "frames", "records", "rows")
                    if k in stage
                ),
                None,
            )
            if batch is None:
                raise UnsupportedEvidence("Cannot identify numerical samples")
            visit(batch, context)
    production = []
    for row in samples:
        phase = row.get("phase", row.get("name", row.get("stage")))
        if phase in {"production", "prod"}:
            production.append(row)
        elif phase in {"equilibration", "equil", "eq"}:
            continue
        else:
            equil = value(
                row, "equilibration_steps", "n_equil_steps", "equil_steps", "n_equil"
            )
            if equil is None or "step" not in row:
                raise UnsupportedEvidence(
                    "Cannot distinguish production from equilibration samples"
                )
            if scalar(row, "step") > float(equil):
                production.append(row)
    require(bool(production), "No production samples")
    return header, production, summaries


def compare_claims(claims, expected, aliases):
    require(isinstance(claims, Mapping), "Reported numerical results must be an object")
    for key, names in aliases.items():
        for name in (key, *names):
            if name in claims:
                require(
                    result_close(claims[name], expected[key], atol=1e-6),
                    f"Incorrect {name}; recomputed {expected[key]}",
                )


def aluminum(evidence, task):
    header, rows, summaries = read_samples(evidence.artifact("run_log"))
    for row in rows:
        # Derive a missing scalar only when the log retained its raw inputs.
        # Atom-wise fields are optional, never an additional acceptance gate.
        momenta = value(row, "momenta", "momenta_ase_units")
        if momenta is not None and (
            value(row, "temperature_K", "Temperature", "T_K") is None
            or (
                task == 10
                and any(
                    value(row, *keys) is None
                    for keys in (("Tx_K", "Tx"), ("Ty_K", "Ty"), ("Tz_K", "Tz"))
                )
            )
        ):
            p = finite_array(momenta, shape=(108, 3))
            masses = finite_array(
                value(row, "masses_amu", "masses", default=[26.9815385] * 108),
                shape=(108,),
            )
            dof = float(value(row, "dof", default=324))
            require(
                np.all(masses > 0) and dof > 0, "Invalid kinetic-energy normalization"
            )
            directions = np.sum(p**2 / masses[:, None], axis=0) / ((dof / 3) * units.kB)
            row.setdefault("temperature_K", float(directions.mean()))
            row.setdefault(
                "kinetic_energy_eV", float(np.sum(p**2 / masses[:, None]) / 2)
            )
            for key, number in zip(("Tx_K", "Ty_K", "Tz_K"), directions, strict=True):
                row.setdefault(key, float(number))
        if task == 10:
            temperature = float(
                np.mean(
                    [
                        scalar(row, *keys)
                        for keys in (("Tx_K", "Tx"), ("Ty_K", "Ty"), ("Tz_K", "Tz"))
                    ]
                )
            )
            reported = value(row, "temperature_K", "Temperature", "T_K")
            if reported is not None:
                require(
                    result_close(reported, temperature, atol=1e-6),
                    "Overall temperature disagrees with directional temperatures",
                )
            row["temperature_K"] = temperature
        if task == 7 and value(row, "volume_A3", "volume", "Volume") is None:
            cell = value(row, "cell_A", "cell")
            if cell is not None:
                row["volume_A3"] = abs(
                    float(np.linalg.det(finite_array(cell, shape=(3, 3))))
                )
    temperature = [scalar(r, "temperature_K", "Temperature", "T_K") for r in rows]
    result = {"sample_count": len(rows), "temperature_K": float(np.mean(temperature))}
    if task == 7:
        result["volume_A3"] = float(
            np.mean([scalar(r, "volume_A3", "volume", "Volume") for r in rows])
        )
        pressure = []
        for row in rows:
            if "pressure_GPa" in row:
                pressure.append(scalar(row, "pressure_GPa"))
            elif "pressure_bar" in row or "Pressure_bar" in row:
                pressure.append(scalar(row, "pressure_bar", "Pressure_bar") * 1e-4)
            else:
                raise UnsupportedEvidence(
                    "Pressure needs an identifiable unit (bar or GPa)"
                )
        result["pressure_GPa"] = float(np.mean(pressure))
        require(result["volume_A3"] > 0, "Volume must be positive")
        require(
            abs(result["temperature_K"] - 300) <= 15,
            "Mean temperature misses the 300 K target",
        )
        # Retain the original final-mean pressure allowance, without using
        # effective-sample counts or convergence tests as scoring gates.
        require(
            abs(result["pressure_GPa"] - 1.01325e-4) <= 0.05,
            "Mean pressure misses the NPT target",
        )
        aliases = {
            "volume_A3": ("mean_V_A3",),
            "temperature_K": ("mean_T_K",),
            "pressure_GPa": (),
        }
    else:
        energies, directional = [], []
        for row in rows:
            direct = value(
                row,
                "energy_eV_per_atom",
                "etot_per_atom_eV",
                "Etot_per_atom_eV",
                "total_energy_eV_per_atom",
            )
            if direct is None:
                total = value(row, "etot_eV", "Etot_eV", "total_energy_eV")
                if total is None:
                    total = scalar(
                        row, "potential_energy_eV", "epot_eV", "Epot_eV"
                    ) + scalar(row, "kinetic_energy_eV", "ekin_eV", "Ekin_eV")
                direct = float(finite_array(total, shape=())) / 108
            energies.append(float(finite_array(direct, shape=())))
            directional.append(
                [
                    scalar(row, *keys)
                    for keys in (("Tx_K", "Tx"), ("Ty_K", "Ty"), ("Tz_K", "Tz"))
                ]
            )
        result.update(
            energy_eV_per_atom=float(np.mean(energies)),
            directional_temperature_K=np.mean(directional, axis=0).tolist(),
        )
        require(
            abs(result["temperature_K"] - 300) <= 30
            and np.max(np.abs(np.asarray(result["directional_temperature_K"]) - 300))
            <= 45,
            "Mean overall/directional temperatures miss the 300 K target",
        )
        aliases = {
            "energy_eV_per_atom": ("Etot_per_atom_eV_mean",),
            "temperature_K": ("Temperature_mean",),
            "directional_temperature_K": (),
        }
        for summary in summaries:
            for axis, mean in zip(
                ("Tx_mean", "Ty_mean", "Tz_mean"),
                result["directional_temperature_K"],
                strict=True,
            ):
                if axis in summary:
                    require(result_close(summary[axis], mean), f"Incorrect {axis}")
    claims = [evidence.results, evidence.manifest.get("summary", {}), *summaries]
    for claim in claims:
        compare_claims(claim, result, aliases)
        if task == 7 and "mean_P_bar" in claim:
            require(
                result_close(claim["mean_P_bar"], result["pressure_GPa"] * 1e4),
                "Incorrect mean_P_bar",
            )
        for stage in claim.get("stages", []):
            compare_claims(stage, result, aliases)
    return {"stages": [result], "reference_basis": "logged_production_samples"}


def phonons(evidence, verifier):
    from .task_5 import (
        ENERGY_FACTOR,
        REFERENCE_CELL,
        STRAINS,
        build_plan,
        mode_energies,
        raw_hessian,
    )

    log = evidence.json("calculation_log")
    require(isinstance(log, Mapping), "Phonon log must be an object")
    rows = next(
        (
            log[k]
            for k in ("calculations", "results", "rows", "records", "data")
            if isinstance(log.get(k), list)
        ),
        None,
    )
    if rows is None:
        raise UnsupportedEvidence("Cannot identify the phonon result rows")
    require(len(rows) == 5, "Expected five strain/phonon-energy pairs")
    pairs, maxima = [], []
    for row in rows:
        strain = scalar(row, "isotropic_strain", "strain_fraction", "strain")
        modes = value(
            row,
            "mode_energies_eV",
            "gamma_phonon_energies_eV",
            "gamma_frequencies_eV",
            "gamma_mode_energies_eV",
            "frequencies_eV",
        )
        maximum = value(
            row,
            "max_phonon_energy_eV",
            "highest_positive_energy_eV",
            "maximum_positive_energy_eV",
        )
        if modes is not None:
            modes = finite_array(modes, ndim=1)
            computed = float(modes.max())
            if maximum is not None:
                require(
                    result_close(maximum, computed),
                    "Reported maximum disagrees with the logged phonon energies",
                )
            maximum = computed
        if maximum is None:
            # Old logs may retain derivatives rather than the final spectrum.
            method = row.get("method", "retained_hessian")
            method = {
                "analytical": "analytic_hessian",
                "analytical_hessian": "analytic_hessian",
            }.get(method, method)
            if not (
                "hessian_eV_A2" in row
                or {"forces_plus_eV_A", "forces_minus_eV_A", "displacement_A"}
                <= row.keys()
            ):
                raise UnsupportedEvidence(
                    "Cannot identify a phonon energy or its numerical derivatives"
                )
            matrix = raw_hessian({**row, "method": method})
            eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / (2 * 28.0855))
            maximum = float(ENERGY_FACTOR * np.sqrt(max(0.0, eigenvalues[-1])))
        maximum = float(finite_array(maximum, shape=()))
        require(maximum > 0, "Highest positive phonon energy must be positive")
        pairs.append(strain)
        maxima.append(maximum)
    order = np.argsort(pairs)
    require(
        np.allclose(np.asarray(pairs)[order], STRAINS, rtol=0, atol=1e-10),
        "Missing or duplicated prescribed strain",
    )
    maxima = np.asarray(maxima)[order]
    if "highest_positive_energies_eV" in evidence.results:
        require(
            result_close(evidence.results["highest_positive_energies_eV"], maxima),
            "Incorrect highest_positive_energies_eV",
        )
    if verifier is None:
        raise UnsupportedEvidence(
            "Phonon reference values require a local or Modal calculation backend"
        )
    canonical = []
    for strain, maximum in zip(STRAINS, maxima, strict=True):
        cell = REFERENCE_CELL * (1 + strain)
        canonical.append(
            {
                "isotropic_strain": float(strain),
                "uniaxial_strain": 0.0,
                "mode_energies_eV": [float(maximum)],
                "structure": {
                    "numbers": [14, 14],
                    "pbc": [True] * 3,
                    "masses": [28.0855] * 2,
                    "cell": cell.tolist(),
                    "positions": (
                        np.asarray([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
                    ).tolist(),
                },
            }
        )
    fingerprint = evidence.fingerprint()
    plan = build_plan(
        evidence, {"calculations": canonical}, [None] * 5, secrets.token_hex(32)
    )
    try:
        response = verifier._calculate(plan, fingerprint)
    except Exception as exc:
        raise UnsupportedEvidence(f"Phonon reference unavailable: {exc}") from exc
    require(
        response.get("evidence_sha256") == fingerprint
        and response.get("challenge") == plan.challenge,
        "Reference identity mismatch",
    )
    jobs = response.get("jobs", [])
    if len(jobs) != 1 or jobs[0].get("status") != "complete":
        raise UnsupportedEvidence("Phonon reference calculation did not complete")
    frames = jobs[0].get("value", {}).get("frames", [])
    require(len(frames) == 5, "Incomplete phonon reference")
    reference = [
        float(
            mode_energies(
                finite_array(frame["hessian"], shape=(6, 6)), [28.0855] * 2
            ).max()
        )
        for frame in frames
    ]
    require(
        np.allclose(maxima, reference, rtol=0.01, atol=1e-5),
        "Highest positive phonon energies disagree with the reference",
    )
    return {
        "strain_pairs": [[float(s), 0.0] for s in STRAINS],
        "highest_positive_energies_eV": maxima.tolist(),
        "reference_energies_eV": reference,
        "reference_release": response.get("release_id"),
    }


def optional_artifact(evidence, role):
    """Read alternate evidence only when it is explicitly submitted."""
    if role in evidence._artifacts:
        return evidence.artifact(role)
    return None


def correlation_values(
    energy, dt, *, normalization="unbiased", cutoff=None, window_factor=5.0
):
    require(
        len(energy) >= 3 and dt > 0,
        "Correlation needs at least three samples and positive sampling time",
    )
    centered = energy - energy.mean()
    require(
        float(centered @ centered) > 1e-20,
        "Constant energy has no defined normalized autocorrelation",
    )
    spectrum = np.fft.rfft(centered, 2 * len(energy))
    acf = np.fft.irfft(spectrum * spectrum.conj())[: len(energy)]
    require(
        normalization in {"biased", "unbiased"}, "Unknown autocorrelation normalization"
    )
    if normalization == "unbiased":
        acf /= np.arange(len(energy), 0, -1)
    acf /= acf[0]
    crossing = np.flatnonzero(acf <= np.exp(-1))
    lag = int(crossing[0]) if len(crossing) else len(energy) - 1
    bound = "lower" if not len(crossing) else "upper" if lag == 1 else "estimate"
    integral = 0.5
    require(
        cutoff is None or (type(cutoff) is int and 1 <= cutoff < len(energy)),
        "Invalid correlation integration cutoff",
    )
    require(
        np.isfinite(window_factor) and window_factor > 0,
        "Invalid integration window factor",
    )
    for index in range(1, len(energy)):
        integral += acf[index]
        if (cutoff is not None and index == cutoff) or (
            cutoff is None and index >= window_factor * integral
        ):
            break
    require(integral > 0, "Integrated correlation time must be positive")
    return {
        "first_crossing_fs": float(lag * dt),
        "crossing_bound": bound,
        "integrated_fs": float(integral * dt),
        "integration_cutoff_lag": index,
        "sampling_interval_fs": float(dt),
        "normalization": normalization,
    }


def copper(evidence):
    log = evidence.artifact("run_logs.main")
    header, rows, _ = read_samples(log)
    settings_path = optional_artifact(evidence, "settings")
    settings = records(settings_path) if settings_path else {}
    require(isinstance(settings, Mapping), "MD settings must be an object")
    options = {**settings, **header}
    analysis_path = optional_artifact(evidence, "analysis")
    analysis = records(analysis_path) if analysis_path else {}
    require(isinstance(analysis, Mapping), "Correlation analysis must be an object")
    energy_path = optional_artifact(evidence, "energy_series")
    if energy_path:
        with energy_path.open("rb") as stream:
            energy = finite_array(np.load(stream, allow_pickle=False), ndim=1)
        dt = value(options, "energy_sampling_interval_fs")
        if dt is None and energy_path.name == "prod_pe_every_step.npy":
            dt = value(options, "timestep_fs", "dt_fs")
        if dt is None:
            raise UnsupportedEvidence("Energy series needs its sampling interval")
        dt = float(finite_array(dt, shape=()))
    else:
        energy = finite_array(
            [
                scalar(r, "potential_energy_eV", "energy_eV", "epot_eV", "Epot_eV")
                for r in rows
            ],
            ndim=1,
        )
        times = finite_array(
            [
                scalar(r, "time_fs") if "time_fs" in r else 1000 * scalar(r, "time_ps")
                for r in rows
            ],
            ndim=1,
        )
        spacing = np.diff(times)
        require(
            len(spacing) > 0
            and np.all(spacing > 0)
            and np.allclose(spacing, spacing[0]),
            "Correlation needs chronological uniformly spaced energies",
        )
        dt = float(spacing[0])
    claim = evidence.results.get("correlation", {})
    require(isinstance(claim, Mapping), "Correlation result must be an object")
    computed = correlation_values(
        energy,
        dt,
        normalization=claim.get("normalization", "unbiased"),
        cutoff=claim.get("cutoff_lag"),
        window_factor=claim.get("window_factor", 5.0),
    )
    if "value_fs" in claim:
        method = claim.get("method", "first_crossing")
        require(
            method in {"first_crossing", "integrated"}, "Unknown correlation estimator"
        )
        expected = computed[
            "integrated_fs" if method == "integrated" else "first_crossing_fs"
        ]
        require(
            result_close(claim["value_fs"], expected, atol=1e-6),
            "Correlation answer differs from the saved energy series",
        )
        if method == "first_crossing" and "bound" in claim:
            require(
                claim["bound"] == computed["crossing_bound"],
                "Incorrect correlation bound",
            )
    if energy_path:
        for key, expected in (
            ("full_step_tau_int_fs", computed["integrated_fs"]),
            ("full_step_1e_fs", computed["first_crossing_fs"]),
        ):
            if key in analysis:
                require(
                    result_close(analysis[key], expected, atol=1e-6), f"Incorrect {key}"
                )
    return {
        "correlation": computed,
        "energy_sample_count": len(energy),
        "reference_basis": "logged_production_energies",
    }


class FinalQuantityScorer:
    """Default level-1 policy for tasks 5, 7, 9 and 10."""

    def __init__(self, task_number, *, verifier=None):
        if task_number not in (5, 7, 9, 10):
            raise ValueError("No final-quantity policy for this task")
        self.task_number, self.level, self.verifier = task_number, 1, verifier

    def evaluate(self, submission):
        rubric, computed = Rubric(self.task_number, binary=True), {}
        evidence = None

        def calculate():
            nonlocal evidence
            evidence = Evidence(submission, task_number=self.task_number, level=1)
            if self.task_number == 5:
                computed.update(phonons(evidence, self.verifier))
            elif self.task_number == 9:
                computed.update(copper(evidence))
            else:
                computed.update(aluminum(evidence, self.task_number))
            return True

        rubric.check("final_quantities", 100, calculate)
        return {
            **rubric.as_dict(),
            "level": 1,
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
