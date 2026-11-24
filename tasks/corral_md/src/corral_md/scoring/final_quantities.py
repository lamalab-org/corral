"""Level-1 scoring of final observables, without workflow/evidence-shape gates.

Only task 5 has a fixed-model reference. The stochastic tasks recompute values
from production data; tasks 7 and 10 read only the documented sample fields.
This does not independently authenticate a trajectory or its model.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

import numpy as np

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
ALUMINUM_VERSION = "level1-final-quantities-v2"


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


def read_samples(path, *, sample_fields=None):
    """Read scalar logs, optionally selecting only documented sample fields."""
    data = records(path)
    if isinstance(data, Mapping):
        if sample_fields is not None:
            if data.get("type") == "sample" or not any(
                key in data
                for key in ("stages", "samples", "frames", "records", "rows")
            ):
                data = [data]
        elif "phase" in data or "step" in data:
            data = [data]
    header, samples, summaries = {}, [], []

    def visit(items, inherited):
        require(isinstance(items, list), "Numerical samples must be a list")
        context = dict(inherited)
        for item in items:
            require(
                isinstance(item, Mapping), "Each numerical record must be an object"
            )
            if sample_fields is not None:
                if item.get("type") == "sample":
                    # Metadata and aliases must never supply measured quantities.
                    samples.append(
                        {
                            key: item[key]
                            for key in ("phase", *sample_fields)
                            if key in item
                        }
                    )
                continue
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
        if sample_fields is None:
            header.update(
                data.get("header", data.get("metadata", data.get("settings", {})))
            )
        stages = data.get("stages", [data])
        require(isinstance(stages, list), "Stages must be a list")
        for stage in stages:
            require(isinstance(stage, Mapping), "Each stage must be an object")
            context = {}
            if sample_fields is None:
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
        if sample_fields is not None:
            require(
                row.get("phase") in ("equilibration", "production"),
                "Sample phase must be 'equilibration' or 'production'",
            )
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


def aluminum(evidence, task):
    fields = (
        ("temperature_K", "volume_A3", "pressure_GPa")
        if task == 7
        else ("Tx_K", "Ty_K", "Tz_K", "energy_eV_per_atom")
    )
    _header, rows, _summaries = read_samples(
        evidence.artifact("run_log"), sample_fields=fields
    )
    values = {}
    for field in fields:
        require(
            all(field in row for row in rows),
            f"Missing required production sample field: {field}",
        )
        values[field] = finite_array([row[field] for row in rows], ndim=1)
    result = {"sample_count": len(rows)}
    if task == 7:
        result.update({field: float(np.mean(values[field])) for field in fields})
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
    else:
        directional = np.column_stack([values[field] for field in fields[:3]])
        result.update(
            temperature_K=float(np.mean(directional)),
            energy_eV_per_atom=float(np.mean(values["energy_eV_per_atom"])),
            directional_temperature_K=np.mean(directional, axis=0).tolist(),
        )
        require(
            abs(result["temperature_K"] - 300) <= 30
            and np.max(np.abs(np.asarray(result["directional_temperature_K"]) - 300))
            <= 45,
            "Mean overall/directional temperatures miss the 300 K target",
        )
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
            if not (
                "hessian_eV_A2" in row
                or {"forces_plus_eV_A", "forces_minus_eV_A", "displacement_A"}
                <= row.keys()
            ):
                raise UnsupportedEvidence(
                    "Cannot identify a phonon energy or its numerical derivatives"
                )
            matrix = raw_hessian(row)
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
        rubric, computed = Rubric(self.task_number), {}
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
            "scoring_policy": (
                ALUMINUM_VERSION if self.task_number in (7, 10) else VERSION
            ),
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
