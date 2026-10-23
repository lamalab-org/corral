"""Task 3: reconstruct silver fitting and bulk results from calculation records.

Only the isolated calculation backend loads checkpoints. The submission contains
raw records and compact answers; it does not supply predictions to the grader.
"""

from __future__ import annotations

import hashlib
import secrets
from functools import lru_cache

import numpy as np
from ase import units
from ase.build import bulk

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)
from .verification import Plan, compare, file_ref, geometry, stored


def artifact_roles(level):
    return ("dataset", "run_log") + (
        ("trained_checkpoint", "md_trajectory") if level == 2 else ()
    )


def dataset(e):
    frames = e.trajectory("dataset")
    if len(frames) != 45 or any(
        len(a) != 2 or list(a.symbols) != ["Ag", "Ag"] or np.any(a.pbc) for a in frames
    ):
        raise EvidenceError("Expected 45 isolated, nonperiodic Ag2 structures")
    distances = [
        np.linalg.norm(finite_array(a.positions)[1] - a.positions[0]) for a in frames
    ]
    order = np.argsort(distances)
    if not close(
        np.array(distances)[order], np.linspace(0.6, 5, 45), rtol=0, atol=1e-5
    ):
        raise EvidenceError("Dimer separations must cover 0.6..5.0 Angstrom")
    return [frames[i] for i in order]


def labels(frames):
    rows = [stored(a) for a in frames]
    return (
        finite_array([row["energy"] for row in rows], shape=(len(frames),)),
        finite_array(
            [row["forces"] for row in rows], shape=(len(frames), len(frames[0]), 3)
        ),
    )


def fitting_errors(predictions, reference):
    energy, forces = reference
    de = (finite_array([r["energy"] for r in predictions], shape=(45,)) - energy) / 2
    force_errors = (
        finite_array([r["forces"] for r in predictions], shape=(45, 2, 3)) - forces
    )
    values = {
        "energy_rmse_per_atom_eV": float(np.sqrt(np.mean(de**2))),
        "force_rmse_eV_A": float(np.sqrt(np.mean(force_errors**2))),
    }
    finite_array(list(values.values()))
    return values


def bulk_values(e, md):
    frames = e.trajectory("md_trajectory")
    if len(frames) < 3:
        raise EvidenceError(
            "Retain bulk frames throughout the run, including both endpoints"
        )
    initial = frames[0]
    for a in frames:
        if (
            len(a) != 108
            or set(a.symbols) != {"Ag"}
            or not np.all(a.pbc)
            or not close(a.cell @ a.cell.T, np.eye(3) * 12.27**2, atol=1e-4, rtol=0)
            or not close(a.cell.array, initial.cell.array, rtol=0, atol=1e-6)
            or not close(a.get_masses(), np.full(108, 107.8682), rtol=0, atol=1e-4)
        ):
            raise EvidenceError(
                "Expected a fixed periodic Ag108 cell with the natural Ag mass"
            )
        finite_array(a.positions, shape=(108, 3))
    positions = initial.get_scaled_positions(wrap=False)
    reference = bulk("Ag", "fcc", a=4.09, cubic=True).repeat((3, 3, 3)).positions
    delta = (positions - positions[0])[:, None, :] * 12.27 - reference[None, :, :]
    delta -= np.rint(delta / 12.27) * 12.27
    distances = np.linalg.norm(delta, axis=2)
    if (
        np.max(distances.min(axis=1)) > 1e-5
        or len(np.unique(distances.argmin(axis=1))) != 108
    ):
        raise EvidenceError("Initial bulk structure is not the requested FCC crystal")
    times = finite_array([a.info["time_fs"] for a in frames])
    dt = float(finite_array(md["timestep_fs"], shape=()))
    steps = times / dt
    if (
        dt <= 0
        or not close(times[[0, -1]], [0, 5000], rtol=0, atol=1e-5)
        or np.any(np.diff(times) <= 0)
        or not close(steps, np.rint(steps), rtol=0, atol=1e-5)
        or not np.allclose(np.diff(times), np.diff(times)[0], rtol=0, atol=1e-5)
    ):
        raise EvidenceError(
            "Bulk frames must regularly sample the complete 0..5000 fs run"
        )
    if np.max(np.diff(times)) > 50 + 1e-5:
        raise UnsupportedEvidence(
            "Bulk sampling is too sparse to assess stability automatically; retain frames at least every 50 fs"
        )
    dof = md["temperature_dof"]
    if dof not in (321, 324):
        raise EvidenceError("temperature_dof must be 321 or 324")
    momenta = finite_array(
        [a.arrays["momenta"] for a in frames], shape=(len(frames), 108, 3)
    )
    temperature = np.sum(
        momenta**2 / initial.get_masses()[None, :, None], axis=(1, 2)
    ) / (dof * units.kB)
    if (
        md["target_temperature_K"] != 300
        or md["initial_temperature_K"] != 300
        or type(md["random_seed"]) is not int
        or md["velocity_initializations"] != 1
        or md["remove_com_once"] is not True
        or not isinstance(md["thermostat"], str)
        or not md["thermostat"].strip()
        or not np.allclose(momenta[0].sum(axis=0), 0, atol=1e-7)
        or abs(temperature[0] - 300) > 6 * 300 * np.sqrt(2 / 321)
    ):
        raise EvidenceError(
            "Bulk initialization does not match the requested 300 K protocol"
        )
    energy, forces = labels(frames)
    scaled = np.array([a.get_scaled_positions(wrap=False) for a in frames])
    # Remove uniform translation; unwrap successive displacements across the cell.
    displacement = np.diff(scaled, axis=0)
    displacement = (displacement - np.rint(displacement)) @ initial.cell.array
    displacement -= displacement.mean(axis=1, keepdims=True)
    final = displacement.sum(axis=0)
    minimum = []
    for positions in scaled:
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm((delta - np.rint(delta)) @ initial.cell.array, axis=2)
        np.fill_diagonal(distance, np.inf)
        minimum.append(float(distance.min()))
    values = {
        "temperature_mean_K": float(temperature.mean()),
        "temperature_std_K": float(temperature.std()),
        "potential_energy_range_eV": float(np.ptp(energy)),
        "force_max_eV_A": float(np.linalg.norm(forces, axis=2).max()),
        "final_displacement_rms_A": float(np.sqrt(np.mean(np.sum(final**2, axis=1)))),
        "min_pair_distance_A": min(minimum),
    }
    finite_array(list(values.values()))
    return frames, values, displacement, temperature, np.array(minimum)


def build_plan(e, level, challenge):
    """Small independent calculations; no training or MD is rerun."""
    p = Plan(e, 3, challenge)
    frames = dataset(e)
    log = e.json("run_log")
    dtype = log["default_dtype"]

    def add(name, selected, model, dispersion=False, **parameters):
        p.add(
            name,
            "mace",
            {
                "frames": [geometry(a) for a in selected],
                "properties": ["energy", "forces"],
                "parameter_digest": name in {"student_before", "student_after"},
            },
            {},
            parameters={
                "model": model,
                "default_dtype": dtype,
                "dispersion": dispersion,
                **parameters,
            },
        )

    add("teacher", frames, "teacher.model", dispersion=True)
    if dtype == "float64":
        p.jobs[-1].update(
            operation="reference",
            reference={"name": "task3_teacher_dimers"},
            fallback_operation="mace",
        )
    if level == 2:
        checkpoint = p.checkpoint(e.artifact("trained_checkpoint"))
        add("student_before", frames, "student.model")
        add("student_after", frames, "submitted", **checkpoint)
        trajectory = e.trajectory("md_trajectory")
        indices = sorted(
            {0, len(trajectory) - 1, *p.sample(len(trajectory), 3, "bulk")}
        )
        add("bulk", [trajectory[i] for i in indices], "submitted", **checkpoint)
        p.expected["bulk"]["indices"] = indices
    return p


def evaluate_submission(submission, *, level, verifier=None, review=None):
    r = Rubric(3, binary=True, fail_fast=True)
    calculated = {}
    verification = {}
    try:
        e = Evidence(submission)
        log = e.json("run_log")
        for role in artifact_roles(level):
            if e.artifact(role).stat().st_size == 0:
                raise EvidenceError(f"Empty {role}")
        fingerprint = e.fingerprint()
    except Exception as exc:
        r.check("calculation_records", 100, False, str(exc))
        return {**r.as_dict(), "level": level, "calculated_results": calculated}

    def settings():
        return (
            log["teacher_model"] == "teacher.model"
            and log["dispersion"] is True
            and log["default_dtype"] in {"float32", "float64"}
            and log["energy_unit"] == "eV"
            and log["force_unit"] == "eV/Angstrom"
        )

    r.check("reference_settings", 10, settings)
    r.check(
        "reference_dataset", 20 if level == 2 else 40, lambda: bool(labels(dataset(e)))
    )

    @lru_cache(None)
    def calculation_plan():
        return build_plan(e, level, secrets.token_hex(32))

    @lru_cache(None)
    def independent(name):
        if verifier is None:
            raise UnsupportedEvidence(
                "Independent model calculation requires a local or Modal backend"
            )
        plan = calculation_plan()
        job = next(job for job in plan.jobs if job["id"] == name)
        try:
            response = verifier._calculate(plan.subset([job]), fingerprint)
        except Exception as exc:
            raise UnsupportedEvidence(
                f"Independent calculation unavailable: {exc}"
            ) from exc
        if (
            response.get("evidence_sha256") != fingerprint
            or response.get("challenge") != plan.challenge
        ):
            raise EvidenceError(
                "Independent calculation returned a different evidence identity"
            )
        jobs = response.get("jobs", [])
        if len(jobs) != 1 or jobs[0].get("id") != name:
            raise EvidenceError(
                "Independent calculation returned missing or mismatched jobs"
            )
        result = jobs[0]
        verification[name] = {
            "status": result.get("status"),
            "release_id": response.get("release_id"),
            "sample_indices": plan.expected[name]["indices"],
        }
        if result.get("status") == "unsupported":
            raise UnsupportedEvidence(
                result.get("detail", "Unsupported model calculation")
            )
        if result.get("status") != "complete":
            raise EvidenceError(result.get("detail", "Model could not be evaluated"))
        return result["value"], plan

    def teacher():
        value, _ = independent("teacher")
        return compare(
            {
                "frames": [
                    {k: stored(a)[k] for k in ("energy", "forces")} for a in dataset(e)
                ]
            },
            value,
            {"energy": [1e-5, 1e-3], "forces": [1e-5, 1e-3]},
        )

    if not r.failed:
        r.check("independent_model_calculation", 20 if level == 2 else 50, teacher)

    if level == 2 and not r.failed:

        def training():
            record = log["training"]
            if (
                log["student_model"] != "student.model"
                or type(record["steps"]) is not int
                or record["steps"] < 1
            ):
                return False
            if (
                record["checkpoint_sha256"]
                != file_ref(e.artifact("trained_checkpoint"))["sha256"]
            ):
                return False
            before = independent("student_before")[0]["parameter_sha256"]
            after = independent("student_after")[0]["parameter_sha256"]
            return (
                before != after
                and before == record["initial_parameters_sha256"]
                and after == record["final_parameters_sha256"]
            )

        r.check("optimization_and_checkpoint", 15, training)

        def fitting():
            reference = labels(dataset(e))
            calculated["fitting"] = {
                phase: fitting_errors(
                    independent(f"student_{phase}")[0]["frames"], reference
                )
                for phase in ("before", "after")
            }
            claims = e.results.get("fitting", {})
            if not isinstance(claims, dict) or any(
                not isinstance(value, dict) for value in claims.values()
            ):
                return False
            return all(
                result_close(value, claims[phase][key], atol=1e-6)
                for phase, metrics in calculated["fitting"].items()
                for key, value in metrics.items()
                if key in claims.get(phase, {})
            )

        r.check("fitting_errors", 10, fitting)

        @lru_cache(None)
        def bulk_data():
            return bulk_values(e, log["md"])

        def bulk_model():
            frames = bulk_data()[0]
            if (
                log["md"]["checkpoint_sha256"]
                != file_ref(e.artifact("trained_checkpoint"))["sha256"]
            ):
                return False
            value, plan = independent("bulk")
            return compare(
                {
                    "frames": [
                        {k: stored(frames[i])[k] for k in ("energy", "forces")}
                        for i in plan.expected["bulk"]["indices"]
                    ]
                },
                value,
                {"energy": [1e-5, 1e-3], "forces": [1e-5, 1e-3]},
            )

        r.check("bulk_protocol_and_model", 10, bulk_model)

        def stability():
            _, values, displacement, temperatures, minimum = bulk_data()
            calculated["bulk"] = values
            claims = e.results.get("bulk", {})
            if not isinstance(claims, dict):
                raise EvidenceError("Optional bulk diagnostics must be an object")
            if not all(
                result_close(value, claims[key], atol=1e-5)
                for key, value in values.items()
                if key in claims
            ):
                return False
            # These screens detect unusable integration/sampling, not failure to
            # remain crystalline or failure to stay near the thermostat target.
            if minimum.min() <= 0 or np.any(temperatures <= 0):
                raise EvidenceError(
                    "Overlapping atoms or nonpositive kinetic temperature"
                )
            if np.max(np.linalg.norm(displacement, axis=2)) >= minimum.min() / 2:
                raise UnsupportedEvidence(
                    "Trajectory sampling cannot resolve the physical evolution; retain more frequent frames"
                )
            if temperatures.max() > 10 * 300 or values["force_max_eV_A"] > 1e4:
                raise UnsupportedEvidence(
                    "Possible numerical integration failure; confirm timestep convergence before interpreting transfer"
                )
            return True

        r.check("bulk_statistics_and_stability", 15, stability)

    remaining = 100 - sum(c["points"] for c in r.checks)
    if remaining:
        r.check(
            "remaining_evidence",
            remaining,
            False,
            "Not evaluated after an earlier failure",
        )
    if e.fingerprint() != fingerprint:
        raise RuntimeError(
            "Submission changed during grading; retry with frozen evidence"
        )
    releases = {
        item["release_id"] for item in verification.values() if item["release_id"]
    }
    if len(releases) > 1:
        raise RuntimeError("Model verification release changed during grading")
    identity = hashlib.sha256(
        f"silver.level_{level}\0{fingerprint}\0{sorted(releases)}".encode()
    ).hexdigest()
    if review is not None:
        r.apply_review(review, identity)
    return {
        **r.as_dict(),
        "mode": "independent_calculations" if verification else "artifact_only",
        "level": level,
        "evidence_sha256": identity,
        "calculated_results": calculated,
        "verification": verification,
    }
