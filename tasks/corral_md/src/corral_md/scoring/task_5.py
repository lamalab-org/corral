"""Reconstruct strained-silicon phonons and fits from one calculation log.

Only raw derivatives/forces, their structures, and calculator settings are
retained. Numerical screens are evaluator policy, pending calibration on real
runs. Independent model checks establish accuracy, not execution provenance.
"""

from __future__ import annotations

import hashlib
import secrets
from itertools import product

import numpy as np

from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)
from .verification import Plan, compare

# hbar * sqrt((eV / Angstrom**2) / atomic-mass-unit), in eV.
ENERGY_FACTOR = (1.054571817e-34 / 1.602176634e-19) * np.sqrt(
    1.602176634e-19 / (1e-20 * 1.66053906892e-27)
)
STRAINS = np.arange(-2, 3) / 100
REFERENCE_CELL = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))
TRANSLATIONS = np.tile(np.eye(3), (2, 1))
PROJECTION = np.eye(6) - TRANSLATIONS @ TRANSLATIONS.T / 2
HESSIAN_TOLERANCE = {"hessian": [0.01, 0.02]}


def _require(condition, detail):
    if not condition:
        raise EvidenceError(detail)


def strain_pairs(level):
    return np.asarray(list(product(STRAINS, [0.0] if level == 1 else STRAINS)))


def read_log(e, level):
    """Validate the complete grid and the actual unrelaxed calculation inputs."""
    log = e.json("calculation_log")
    _require(
        log["model"] == "teacher.model"
        and log["default_dtype"] in {"float32", "float64"}
        and log["dispersion"] is False,
        "Use the supplied MACE-MP-0 teacher.model without dispersion",
    )
    rows = log["calculations"]
    expected = strain_pairs(level)
    _require(
        isinstance(rows, list) and len(rows) == len(expected), "Incomplete strain grid"
    )
    pairs = finite_array(
        [[row["isotropic_strain"], row["uniaxial_strain"]] for row in rows],
        shape=expected.shape,
    )
    _require(
        close(sorted(pairs.tolist()), expected, rtol=0, atol=1e-10),
        "Each prescribed strain pair must occur exactly once",
    )
    for row, (iso, z) in zip(rows, pairs, strict=True):
        frame = row["structure"]
        cell = REFERENCE_CELL * (1 + iso) @ np.diag([1, 1, 1 + z])
        _require(
            frame["numbers"] == [14, 14]
            and frame["pbc"] == [True, True, True]
            and close(frame["cell"], cell, rtol=0, atol=1e-6),
            "Expected the periodic Si2 primitive cell with Cartesian z strain",
        )
        positions = finite_array(frame["positions"], shape=(2, 3))
        delta = (positions[1] - positions[0]) @ np.linalg.inv(cell)
        _require(
            any(
                np.linalg.norm(
                    (delta - sign * 0.25 - np.rint(delta - sign * 0.25)) @ cell
                )
                < 1e-6
                for sign in (-1, 1)
            ),
            "Atomic positions must remain unrelaxed (wrapping and translation are allowed)",
        )
        _require(
            close(frame["masses"], [28.0855, 28.0855], rtol=0, atol=0.001),
            "Use natural isotope-averaged silicon masses",
        )
    return log


def raw_hessian(row):
    """Read an energy Hessian or differentiate six Cartesian force pairs."""
    if row["method"] == "analytic_hessian":
        return finite_array(row["hessian_eV_A2"], shape=(6, 6))
    if row["method"] != "central_difference":
        raise EvidenceError(
            "Use analytic_hessian or central_difference calculation logs"
        )
    step = float(finite_array(row["displacement_A"], shape=()))
    _require(step > 0, "Displacement must be positive")
    plus = finite_array(row["forces_plus_eV_A"], shape=(6, 2, 3))
    minus = finite_array(row["forces_minus_eV_A"], shape=(6, 2, 3))
    return -(plus - minus).reshape(6, 6).T / (2 * step)


def mode_energies(hessian, masses):
    """Symmetrize, project translations, mass-weight, and keep signed modes."""
    scale = max(float(np.max(np.abs(hessian))), 1e-12)
    _require(
        np.max(np.abs(hessian - hessian.T)) <= 0.02 + 0.01 * scale
        and np.max(np.abs(hessian @ TRANSLATIONS)) <= 0.02 + 0.01 * scale,
        "Raw derivatives have excessive symmetry or acoustic-sum-rule error",
    )
    matrix = PROJECTION @ ((hessian + hessian.T) / 2) @ PROJECTION
    weights = np.repeat(masses, 3)
    eigenvalues = np.linalg.eigvalsh(matrix / np.sqrt(np.outer(weights, weights)))
    energies = ENERGY_FACTOR * np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    # Translation roundoff must not be mistaken for a physical positive mode.
    energies[np.abs(energies) < 1e-7] = 0
    _require(np.any(energies > 0), "No resolved positive Gamma-point mode")
    return energies


def reconstruct(log, level):
    rows = log["calculations"]
    hessians, modes = [], []
    for row in rows:
        hessian = raw_hessian(row)
        energies = mode_energies(hessian, row["structure"]["masses"])
        # Endpoints and the reference probe displacement sensitivity without
        # doubling every calculation; the log must retain the half-step forces.
        iso, z = np.round([row["isotropic_strain"], row["uniaxial_strain"]], 2)
        selected = (abs(iso) == 0.02 and (level == 1 or abs(z) == 0.02)) or (
            iso == z == 0
        )
        if row["method"] == "central_difference" and (selected or "sensitivity" in row):
            finer = {**row["sensitivity"], "method": "central_difference"}
            _require(
                close(finer["displacement_A"], row["displacement_A"] / 2, rtol=1e-8),
                "Sensitivity forces must use half the original displacement",
            )
            alternative = raw_hessian(finer)
            _require(
                compare(
                    {"hessian": hessian.tolist()},
                    {"hessian": alternative.tolist()},
                    HESSIAN_TOLERANCE,
                ),
                "Force derivatives are not converged with displacement size",
            )
            alternate_modes = mode_energies(alternative, row["structure"]["masses"])
            _require(
                close(energies.max(), alternate_modes.max(), rtol=0.005, atol=1e-6),
                "Selected phonon energy is sensitive to displacement size",
            )
        hessians.append(hessian)
        modes.append(energies)
    return hessians, np.asarray(modes)


def build_plan(e, log, hessians, challenge):
    """Use cached fixed-input references, with an isolated calculator fallback."""
    rows = log["calculations"]
    plan = Plan(e, 5, challenge)
    parameters = {key: log[key] for key in ("model", "default_dtype", "dispersion")}
    inputs = {
        "frames": [
            {
                key: row["structure"][key]
                for key in ("numbers", "positions", "cell", "pbc", "masses")
            }
            for row in rows
        ],
        "properties": ["hessian"],
        "sample_indices": list(range(len(rows))),
    }
    operation = "mace"
    if parameters["default_dtype"] == "float64":
        operation = "reference"
        inputs.update(
            reference={
                "name": "task5_strained_hessians",
                "strain_pairs": [
                    [row["isotropic_strain"], row["uniaxial_strain"]] for row in rows
                ],
            },
            fallback_operation="mace",
        )
    plan.add(
        "silicon_hessians",
        operation,
        inputs,
        {"frames": [{"hessian": matrix.tolist()} for matrix in hessians]},
        parameters=parameters,
        tolerance=HESSIAN_TOLERANCE,
    )
    return plan


def evaluate_submission(submission, *, level, verifier=None, review=None):
    r = Rubric(5, binary=True, fail_fast=True)
    calculated, verification = {}, {}
    try:
        e = Evidence(submission)
        fingerprint = e.fingerprint()
        log = read_log(e, level)
    except Exception as exc:
        r.check("calculation_log", 100, False, str(exc))
        return {**r.as_dict(), "level": level, "calculated_results": calculated}
    r.check("strain_grid_structures_and_settings", 20, True)
    hessians = []

    def phonons():
        matrices, modes = reconstruct(log, level)
        hessians.extend(matrices)
        calculated["strain_pairs"] = [
            [row["isotropic_strain"], row["uniaxial_strain"]]
            for row in log["calculations"]
        ]
        calculated["signed_mode_energies_eV"] = modes.tolist()
        calculated["highest_positive_energies_eV"] = modes.max(axis=1).tolist()
        return True

    r.check("phonon_reconstruction_and_sensitivity", 25, phonons)
    r.check(
        "highest_positive_energies",
        35 if level == 1 else 15,
        lambda: (
            ("highest_positive_energies_eV" not in e.results)
            or result_close(
                calculated["highest_positive_energies_eV"],
                e.results["highest_positive_energies_eV"],
                atol=1e-7,
            )
        ),
    )

    if level == 2:

        def fit():
            design = np.column_stack([np.ones(25), calculated["strain_pairs"]])
            y = np.asarray(calculated["highest_positive_energies_eV"])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            calculated["fit"] = {
                "intercept_eV": float(coefficients[0]),
                "isotropic_coefficient_eV_per_strain": float(coefficients[1]),
                "uniaxial_coefficient_eV_per_strain": float(coefficients[2]),
                "residuals_eV": (y - design @ coefficients).tolist(),
            }
            return all(
                result_close(value, e.results["fit"][key], atol=1e-7)
                for key, value in calculated["fit"].items()
                if key
                in {
                    "isotropic_coefficient_eV_per_strain",
                    "uniaxial_coefficient_eV_per_strain",
                }
                or key in e.results["fit"]
            )

        r.check("unweighted_fit_and_residuals", 20, fit)

    def independent():
        if verifier is None:
            raise UnsupportedEvidence(
                "Independent Hessian checks require a local or Modal backend"
            )
        plan = build_plan(e, log, hessians, secrets.token_hex(32))
        try:
            response = verifier._calculate(plan, fingerprint)
        except Exception as exc:
            raise UnsupportedEvidence(
                f"Independent calculation unavailable: {exc}"
            ) from exc
        _require(
            response.get("evidence_sha256") == fingerprint
            and response.get("challenge") == plan.challenge,
            "Independent calculation returned a different evidence identity",
        )
        jobs = response.get("jobs", [])
        _require(
            len(jobs) == 1 and jobs[0].get("id") == "silicon_hessians",
            "Missing or mismatched Hessian check",
        )
        job = jobs[0]
        verification.update(
            status=job.get("status"), release_id=response.get("release_id")
        )
        if job.get("status") == "unsupported":
            raise UnsupportedEvidence(
                job.get("detail", "Hessian calculation unsupported")
            )
        _require(
            job.get("status") == "complete",
            job.get("detail", "Hessian calculation failed"),
        )
        return compare(
            plan.expected["silicon_hessians"]["values"], job["value"], HESSIAN_TOLERANCE
        )

    r.check("independent_hessian_accuracy", 20, independent)
    if e.fingerprint() != fingerprint:
        raise RuntimeError(
            "Submission changed during grading; retry with frozen evidence"
        )
    identity = hashlib.sha256(
        f"task5.level_{level}\0{fingerprint}\0{verification.get('release_id')}".encode()
    ).hexdigest()
    if review is not None:
        r.apply_review(review, identity)
    return {
        **r.as_dict(),
        "level": level,
        "mode": "independent_calculations" if verification else "artifact_only",
        "evidence_sha256": identity,
        "calculated_results": calculated,
        "verification": verification,
    }
