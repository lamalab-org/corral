"""Check saved strained-silicon phonon calculations without evaluating a potential."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from .common import (
    RESULT_RTOL,
    EvidenceError,
    UnsupportedEvidence,
    finite_array,
    result_close,
)
from .common import close as _shared_close

if TYPE_CHECKING:
    from .common import Evidence, Rubric


# hbar * sqrt((eV / Angstrom**2) / atomic-mass-unit), in eV.
# SI 2019 definitions for eV and hbar; CODATA 2022 atomic mass unit.
ENERGY_FACTOR = (1.054571817e-34 / 1.602176634e-19) * np.sqrt(
    1.602176634e-19 / (1e-20 * 1.66053906892e-27)
)


class UnsupportedMethod(UnsupportedEvidence):
    """The numerical evidence uses a method this checker cannot reconstruct."""


def close(actual, expected, rtol=1e-5, atol=1e-7):
    return _shared_close(actual, expected, rtol=rtol, atol=atol)


def _array(e, value, shape=None):
    if isinstance(value, dict):
        value = e.array(value["artifact"])
    return finite_array(value, shape=shape)


def _order(record):
    axes = {"x": 0, "y": 1, "z": 2}
    order = record["dof_order"]
    if not isinstance(order, list) or len(order) != 6:
        raise EvidenceError("dof_order must enumerate six Cartesian degrees of freedom")
    ids = []
    for atom, supplied_axis in order:
        axis = axes.get(supplied_axis, supplied_axis)
        if (
            type(atom) is not int
            or type(axis) is not int
            or atom not in (0, 1)
            or axis not in (0, 1, 2)
        ):
            raise EvidenceError("Invalid atom/axis in dof_order")
        ids.append(3 * atom + axis)
    if sorted(ids) != list(range(6)):
        raise EvidenceError("dof_order must be a permutation")
    return np.asarray(ids)


def _records(e, key):
    value = e.json(key)
    value = value.get("calculations") if isinstance(value, dict) else value
    if not isinstance(value, list) or not value:
        raise EvidenceError("Expected a nonempty calculation record list")
    ids = [record["id"] for record in value]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise EvidenceError("Calculation IDs must be unique nonempty strings")
    return value


def _geometry(frame, expected_cell):
    if (
        len(frame) != 2
        or frame.get_chemical_symbols() != ["Si", "Si"]
        or not np.all(frame.pbc)
        or not close(frame.cell.array, expected_cell, rtol=0, atol=1e-6)
    ):
        return False
    positions = finite_array(frame.positions, shape=(2, 3))
    # Relative coordinates admit a global translation, periodic wrapping, and
    # either atom ordering without permitting internal atomic relaxation.
    fractional_delta = (positions[1] - positions[0]) @ np.linalg.inv(expected_cell)
    for sign in (-1, 1):
        difference = fractional_delta - sign * 0.25
        difference -= np.rint(difference)
        if np.linalg.norm(difference @ expected_cell) < 1e-6:
            return True
    return False


def _reconstruct(e, record):
    order = _order(record)
    method = record["method"]
    if method == "central_difference":
        plus = _array(e, record["plus_displacements_A"])
        minus = _array(e, record["minus_displacements_A"], plus.shape)
        if plus.ndim != 3 or plus.shape[1:] != (2, 3) or len(plus) < 6:
            raise EvidenceError(
                "Central differences need at least six [2,3] displacement pairs"
            )
        if not close(plus, -minus, rtol=1e-7, atol=1e-12):
            raise EvidenceError(
                "Displacement pairs must be centered on the unrelaxed structure"
            )
        fplus = _array(e, record["plus_forces_eV_A"], plus.shape)
        fminus = _array(e, record["minus_forces_eV_A"], plus.shape)
        displacements = (plus - minus).reshape(len(plus), 6)[:, order]
        forces = (fplus - fminus).reshape(len(plus), 6)[:, order]
        if np.any(np.linalg.norm(displacements, axis=1) <= 0):
            raise EvidenceError("Displacement pairs must have nonzero separation")
        raw_transpose, _, rank, _ = np.linalg.lstsq(displacements, -forces, rcond=None)
        if rank != 6:
            raise EvidenceError(
                "Saved displacements do not span all six degrees of freedom"
            )
        matrix = raw_transpose.T
    elif method == "finite_difference":
        displacements = _array(e, record["displacements_A"])
        if displacements.ndim != 3 or displacements.shape[1:] != (2, 3):
            raise EvidenceError("Finite differences require [n,2,3] displacements")
        forces = _array(e, record["forces_eV_A"], displacements.shape)
        weights = _array(e, record["derivative_weights_A_inv"], (6, len(forces)))
        coordinates = displacements.reshape(-1, 6)[:, order]
        if not close(weights.sum(axis=1), np.zeros(6), atol=1e-9) or not close(
            weights @ coordinates, np.eye(6), atol=1e-9
        ):
            raise EvidenceError(
                "Derivative weights must cancel constants and differentiate all six coordinates"
            )
        matrix = -(weights @ forces.reshape(-1, 6)[:, order]).T
    elif method == "analytic_hessian":
        matrix = _array(e, record["raw_derivatives_eV_A2"], (6, 6)).copy()
        if record["derivative_kind"] == "force_jacobian":
            matrix *= -1
        elif record["derivative_kind"] != "energy_hessian":
            raise UnsupportedMethod("Unknown analytical derivative convention")
    else:
        raise UnsupportedMethod(
            f"Cannot reconstruct the saved numerical method {method!r}"
        )
    symmetry = record["symmetrization"]
    if symmetry == "average_transpose":
        matrix = (matrix + matrix.T) / 2
    elif symmetry != "none":
        raise UnsupportedMethod(f"Unsupported symmetrization: {symmetry!r}")
    acoustic = record["acoustic_sum_rule"]
    if acoustic == "projection":
        translations = np.tile(np.eye(3), (2, 1))[order]
        projection = np.eye(6) - translations @ translations.T / 2
        matrix = projection @ matrix @ projection
    elif acoustic != "none":
        raise UnsupportedMethod(f"Unsupported acoustic sum rule: {acoustic!r}")
    if not close(matrix, matrix.T, rtol=1e-5, atol=1e-7):
        raise EvidenceError("Processed force constants are not symmetric")
    return matrix, order


def _modes(e, record, masses):
    matrix = _array(e, record["force_constants_eV_A2"], (6, 6))
    order = _order(record)
    if not close(matrix, matrix.T, atol=1e-7):
        raise EvidenceError("Saved processed force constants must be symmetric")
    try:
        reconstructed, _ = _reconstruct(e, record)
    except UnsupportedMethod:
        # The separate reconstruction check remains pending review. Matrix
        # diagonalization and the explicitly required OLS can still be checked.
        pass
    else:
        if not result_close(matrix, reconstructed, atol=1e-07):
            raise EvidenceError(
                "Saved force constants disagree with saved forces/derivatives"
            )
    weights = np.repeat(masses, 3)[order]
    dynamical = matrix / np.sqrt(weights[:, None] * weights[None, :])
    eigenvalues = np.linalg.eigvalsh((dynamical + dynamical.T) / 2)
    real = ENERGY_FACTOR * np.sqrt(np.maximum(eigenvalues, 0))
    imaginary = ENERGY_FACTOR * np.sqrt(np.maximum(-eigenvalues, 0))
    positive = real[(eigenvalues > 0) & (imaginary == 0)]
    if not len(positive):
        raise EvidenceError("No strictly positive real Gamma-point phonon energy")
    return real, imaginary, float(np.max(positive))


def _saved_modes_match(e, record, actual):
    real = _array(e, record["mode_real_eV"], (6,))
    imaginary = _array(e, record["mode_imaginary_eV"], (6,))
    if np.any(real < 0) or np.any(imaginary < 0):
        return False
    # Compare unordered complex modes through a tiny exhaustive matching problem.
    expected = np.stack(actual[:2], axis=1)
    observed = np.stack((real, imaginary), axis=1)
    compatible = np.all(
        (
            np.abs(expected[:, None] - observed[None])
            <= 1e-7
            + RESULT_RTOL
            * np.maximum(np.abs(expected[:, None]), np.abs(observed[None]))
        ),
        axis=2,
    )

    def match(row, used):
        return row == 6 or any(
            match(row + 1, used | {col})
            for col in range(6)
            if col not in used and compatible[row, col]
        )

    return match(0, set())


def evaluate(e: Evidence, r: Rubric) -> None:
    reference_cell = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))

    @lru_cache(None)
    def records():
        return _records(e, "strain_calculations")

    @lru_cache(None)
    def frames():
        return e.trajectory("strained_structures")

    @lru_cache(None)
    def reference_valid():
        reference = e.trajectory("reference_structure")
        return len(reference) == 1 and _geometry(reference[0], reference_cell)

    @lru_cache(None)
    def grid_valid():
        rows = records()
        if len(rows) != 25 or len(frames()) != 25:
            return False
        indices = [row["structure_index"] for row in rows]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(
            range(25)
        ):
            return False
        actual = finite_array(
            [[row["isotropic_strain"], row["uniaxial_strain"]] for row in rows],
            shape=(25, 2),
        )
        expected = np.asarray(list(product(np.arange(-2, 3) / 100, repeat=2)))
        return close(np.asarray(sorted(actual.tolist())), expected, rtol=0, atol=1e-10)

    @lru_cache(None)
    def strained_geometry_valid():
        if not grid_valid():
            return False
        for record in records():
            cell = reference_cell * (1 + float(record["isotropic_strain"]))
            cell = cell @ np.diag([1, 1, 1 + float(record["uniaxial_strain"])])
            if not _geometry(frames()[record["structure_index"]], cell):
                return False
        return True

    @lru_cache(None)
    def masses():
        declared = finite_array(e.settings["masses_amu"], shape=(2,))
        if not np.all(np.abs(declared - 28.0855) <= 0.001):
            raise EvidenceError(
                "The task requires natural isotope-averaged silicon masses"
            )
        for frame in frames() + e.trajectory("reference_structure"):
            if not close(frame.get_masses(), declared, rtol=0, atol=1e-5):
                raise EvidenceError("Recorded and saved structure masses disagree")
        return declared

    def prerequisite():
        if not reference_valid() or not strained_geometry_valid():
            raise EvidenceError(
                "Valid unrelaxed reference and all 25 Cartesian strains are prerequisites"
            )
        return masses()

    def supported(condition):
        def check():
            try:
                return condition()
            except UnsupportedMethod as exc:
                return None, str(exc)

        return check

    r.check("reference_primitive_structure", 5, reference_valid)
    r.check("complete_strain_grid_and_mapping", 5, grid_valid)
    r.check("cartesian_strains_without_relaxation", 7, strained_geometry_valid)
    r.check("natural_silicon_masses", 3, lambda: bool(len(masses())))

    def conventions():
        model = str(e.settings["model"]).lower().replace("_", "-")
        return (
            "mace-mp-0" in model
            and e.settings["energy_unit"] == "eV"
            and e.settings["force_unit"] == "eV/Angstrom"
            and e.settings["force_constant_unit"] == "eV/Angstrom^2"
            and all(
                len(_order(row)) == 6
                and isinstance(row["symmetrization"], str)
                and isinstance(row["acoustic_sum_rule"], str)
                for row in records()
            )
        )

    def force_constants():
        prerequisite()
        return all(
            result_close(
                _reconstruct(e, row)[0],
                _array(e, row["force_constants_eV_A2"], (6, 6)),
                atol=1e-07,
            )
            for row in records()
        )

    @lru_cache(None)
    def reconstructed():
        weights = prerequisite()
        return {row["id"]: _modes(e, row, weights) for row in records()}

    r.check(
        "recorded_model_units_and_conventions",
        3,
        conventions,
        "Checks declared identity and units, not the identity of an executed calculator",
    )
    r.check(
        "force_constants_reconstructed_from_raw_evidence",
        18,
        supported(force_constants),
    )
    r.check(
        "all_mode_energies_and_imaginary_modes",
        10,
        supported(
            lambda: all(
                _saved_modes_match(e, row, reconstructed()[row["id"]])
                for row in records()
            )
        ),
    )
    r.check(
        "highest_positive_real_mode",
        10,
        supported(
            lambda: all(
                result_close(
                    reconstructed()[row["id"]][2], row["highest_positive_energy_eV"]
                )
                for row in records()
            )
        ),
    )

    @lru_cache(None)
    def fit():
        row_by_id = {row["id"]: row for row in records()}
        claimed = e.result("fit")
        ids = claimed["calculation_ids"]
        if len(ids) != 25 or len(set(ids)) != 25 or set(ids) != set(row_by_id):
            raise EvidenceError("Fit must include every strain exactly once")
        matrix = np.asarray(
            [
                [1, row_by_id[i]["isotropic_strain"], row_by_id[i]["uniaxial_strain"]]
                for i in ids
            ]
        )
        y = np.asarray([reconstructed()[i][2] for i in ids])
        parameters = np.linalg.lstsq(matrix, y, rcond=None)[0]
        return parameters, y - matrix @ parameters

    r.check(
        "unweighted_ols_strain_coefficients",
        15,
        supported(
            lambda: result_close(
                fit()[0][1:],
                [
                    e.result("fit")["isotropic_coefficient_eV_per_strain"],
                    e.result("fit")["uniaxial_coefficient_eV_per_strain"],
                ],
            )
        ),
    )
    r.check(
        "unweighted_ols_intercept",
        6,
        supported(lambda: result_close(fit()[0][0], e.result("fit")["intercept_eV"])),
    )
    r.check(
        "all_25_fit_residuals",
        8,
        supported(lambda: result_close(fit()[1], e.result("fit")["residuals_eV"])),
    )

    r.unverified(
        "execution_provenance",
        "Saved arrays establish consistency, not the actual model used, force evaluations, or the absence of intermediate relaxation.",
    )
