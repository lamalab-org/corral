"""Level 1 Task 3: independently verify the silver dimer reference dataset."""

from __future__ import annotations

import hashlib
import secrets
from functools import lru_cache

import numpy as np

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
)
from .formats import calculation_log
from .verification import Plan, compare, geometry, stored


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
    if any("energy" not in row or "forces" not in row for row in rows):
        raise EvidenceError(
            "Retain total energies and complete atomic forces for every dimer"
        )
    return (
        finite_array([row["energy"] for row in rows], shape=(len(frames),)),
        finite_array(
            [row["forces"] for row in rows], shape=(len(frames), len(frames[0]), 3)
        ),
    )


def calculation_settings(e):
    """Read optional precision from the dataset or an explicitly linked legacy log."""
    metadata = calculation_log(e.trajectory("dataset")[0].info)
    legacy = calculation_log(e.json("run_log")) if "run_log" in e._artifacts else {}
    for key in (
        "teacher_model",
        "dispersion",
        "default_dtype",
        "energy_unit",
        "force_unit",
    ):
        if key in metadata and key in legacy and metadata[key] != legacy[key]:
            raise EvidenceError(f"Dataset and calculation log disagree on {key}")
    return {**legacy, **metadata}


def build_plan(e, challenge):
    """Check every teacher label; no training or MD is rerun."""
    plan = Plan(e, 3, challenge)
    dtype = calculation_settings(e).get("default_dtype", "float64")
    plan.add(
        "teacher",
        "mace",
        {
            "frames": [geometry(a) for a in dataset(e)],
            "properties": ["energy", "forces"],
        },
        {},
        parameters={
            "model": "teacher.model",
            "default_dtype": dtype,
            "dispersion": True,
        },
    )
    if dtype == "float64":
        plan.jobs[-1].update(
            operation="reference",
            reference={"name": "task3_teacher_dimers"},
            fallback_operation="mace",
        )
    return plan


def evaluate_submission(submission, *, level, verifier=None):
    if level != 1:
        raise ValueError("Task 3 Level 2 uses Task3TemperatureScorer")
    r = Rubric(3, binary=True, fail_fast=True)
    calculated = {}
    verification = {}
    try:
        e = Evidence(submission, task_number=3, level=level)
        log = calculation_settings(e)
        fingerprint = e.fingerprint()
    except Exception as exc:
        r.error("calculation_records", 100, exc)
        return {**r.as_dict(), "level": level, "calculated_results": calculated}

    def settings():
        return (
            ("teacher_model" not in log or log["teacher_model"] == "teacher.model")
            and ("dispersion" not in log or log["dispersion"] is True)
            and (
                "default_dtype" not in log
                or log["default_dtype"] in {"float32", "float64"}
            )
            and ("energy_unit" not in log or log["energy_unit"] == "eV")
            and ("force_unit" not in log or log["force_unit"] == "eV/Angstrom")
        )

    r.check("reference_settings", 10, settings)
    r.check("reference_dataset", 40, lambda: bool(labels(dataset(e))))

    @lru_cache(None)
    def calculation_plan():
        return build_plan(e, secrets.token_hex(32))

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
        r.check("independent_model_calculation", 50, teacher)

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
    return {
        **r.as_dict(),
        "mode": "independent_calculations" if verification else "artifact_only",
        "level": level,
        "evidence_sha256": identity,
        "calculated_results": calculated,
        "verification": verification,
    }
