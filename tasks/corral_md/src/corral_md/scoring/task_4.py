"""Task 4: reconstruct palladium phonons from the final force-calculation record.

Force calculations and reciprocal-space postprocessing are separate. Grading
reads data only; an isolated verifier checks a small sample with the pinned
model. No submitted scripts, reports, settings documents, or BZ samples are used.
"""

from __future__ import annotations

import itertools
import json
from functools import lru_cache

import numpy as np
from ase import Atoms, units
from corral_md.scoring.common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.verification import Plan, apply_verification

STANDARD = 3.89 / 2 * (np.ones((3, 3)) - np.eye(3))
NODES = np.array(
    [
        [0, 0, 0],
        [0.5, 0, 0.5],
        [0.625, 0.25, 0.625],
        [0.5, 0.5, 0.5],
        [0, 0, 0],
        [0.375, 0.375, 0.75],
    ]
)
ENERGY_FACTOR = units._hbar * 1e10 / np.sqrt(units._e * units._amu)


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_record(e):
    with np.load(e.artifact("force_log"), allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    record["metadata"] = json.loads(str(record["metadata"].item()))
    return record


def geometry(record):
    positions = finite_array(record["positions_A"], shape=(13, 125, 3))
    cell = finite_array(record["cell_A"], shape=(3, 3)) / 5
    require(
        close(cell @ cell.T, STANDARD @ STANDARD.T, rtol=0, atol=1e-6)
        and np.array_equal(record["numbers"], np.full(125, 46))
        and np.array_equal(record["pbc"], [True, True, True])
        and close(record["masses_amu"], np.full(125, 106.42), rtol=0, atol=1e-4),
        "Expected the fixed periodic FCC Pd125 cell, a=3.89 A, with Pd masses",
    )
    fractional = (positions[0] - positions[0, 0]) @ np.linalg.inv(cell)
    sites = np.rint(fractional).astype(int)
    require(
        close(fractional, sites, rtol=0, atol=1e-6)
        and len(set(map(tuple, sites % 5))) == 125,
        "Atom positions must cover the 5x5x5 primitive lattice exactly once",
    )
    # Fourier phases require centered translations, not ASE's unsigned repeat indices.
    sites = (sites + 2) % 5 - 2
    lookup = {tuple(site): i for i, site in enumerate(sites)}
    opposite = np.array([lookup[tuple(-site)] for site in sites])
    displacement = finite_array(record["displacements_A"], shape=(13, 3))
    delta = float(displacement[1, 0])
    require(delta > 0, "Displacement magnitude must be positive")
    expected = np.vstack(
        [
            np.zeros((1, 3)),
            delta * np.eye(3),
            -delta * np.eye(3),
            delta / 2 * np.eye(3),
            -delta / 2 * np.eye(3),
        ]
    )
    require(
        close(displacement, expected, rtol=0, atol=1e-10),
        "Retain central displacements at delta and delta/2 in the documented order",
    )
    error = positions - positions[0]
    error[:, 0] -= displacement
    scaled = error @ np.linalg.inv(5 * cell)
    require(
        np.max(np.abs((scaled - np.rint(scaled)) @ (5 * cell))) < 1e-6,
        "A recorded frame moved atoms other than the prescribed displacement",
    )
    return cell, sites, opposite, delta


def cubic_average(phi, cell, sites):
    orientation = np.linalg.solve(STANDARD, cell)
    lookup = {tuple(site): i for i, site in enumerate(sites)}
    average = np.zeros_like(phi)
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            cubic = np.eye(3)[list(permutation)] * np.asarray(signs)[:, None]
            rotation = orientation.T @ cubic @ orientation
            lattice = np.rint(cell @ rotation.T @ np.linalg.inv(cell)).astype(int)
            destinations = [
                lookup[tuple(site)] for site in (sites @ lattice + 2) % 5 - 2
            ]
            average[destinations] += rotation @ phi @ rotation.T
    return average / 48


def prepare(record):
    cell, sites, opposite, delta = geometry(record)
    finite_array(record["energies_eV"], shape=(13,))
    forces = finite_array(record["forces_eV_A"], shape=(13, 125, 3))
    coarse = -((forces[1:4] - forces[4:7]) / (2 * delta)).transpose(1, 2, 0)
    fine = -((forces[7:10] - forces[10:13]) / delta).transpose(1, 2, 0)
    raw = finite_array(record["force_constants_eV_A2"], shape=(125, 3, 3))
    metadata = record["metadata"]
    require(
        metadata["checkpoint_sha256"] == PINNED_TEACHER_SHA256
        and metadata["default_dtype"] in {"float32", "float64"}
        and metadata["dispersion"] is False,
        "Force record must identify the supplied checkpoint without dispersion",
    )
    if metadata["method"] == "analytical":
        matrix = finite_array(record["hessian_eV_A2"], shape=(375, 375))
        require(
            close(matrix, matrix.T, atol=1e-7), "Analytical Hessian is not symmetric"
        )
        expected = matrix.reshape(125, 3, 125, 3)[:, :, 0, :]
    elif metadata["method"] == "central_difference":
        expected = fine
    else:
        raise UnsupportedEvidence("Unsupported derivative method in the force record")
    require(
        close(raw, expected, atol=1e-7),
        "Retained force constants disagree with the recorded derivative",
    )
    # Preserve raw diagnostics before applying small symmetry/acoustic corrections.
    cubic = cubic_average(raw, cell, sites)
    phi = (cubic + cubic[opposite].swapaxes(1, 2)) / 2
    phi[0] -= phi.sum(axis=0)
    diagnostics = {
        "pair_symmetry_residual_eV_A2": float(
            np.max(abs(raw - raw[opposite].swapaxes(1, 2)))
        ),
        "acoustic_residual_eV_A2": float(np.max(abs(raw.sum(axis=0)))),
        "cubic_symmetry_residual_eV_A2": float(np.max(abs(raw - cubic))),
        "displacement_sensitivity_eV_A2": float(np.max(abs(fine - coarse))),
    }
    scale = max(float(np.max(abs(raw))), 1e-8)
    accuracy = max(*diagnostics.values(), float(np.max(abs(raw - fine))))
    if accuracy > 1e-4 + 0.01 * scale:
        raise UnsupportedEvidence(
            "Force constants need better displacement/derivative accuracy or symmetry/acoustic checks"
        )
    q = grid([8, 8, 8], [0.5, 0.5, 0.5])
    # Bound negative eigenvalues attributable to derivative/correction errors.
    error = max(
        np.max(np.linalg.norm(dynamical(change, sites, q), ord=2, axis=(1, 2)))
        for change in (fine - coarse, raw - fine, raw - phi)
    )
    tolerance = max(1e-7, float(ENERGY_FACTOR * np.sqrt(error)))
    if tolerance > 5e-4:
        raise UnsupportedEvidence(
            "Numerical mode uncertainty exceeds 0.5 meV; improve force accuracy"
        )
    diagnostics["imaginary_tolerance_eV"] = tolerance
    diagnostics["gamma_energies_eV"] = energies(phi, sites, [[0, 0, 0]])[0].tolist()
    return phi, sites, diagnostics


def dynamical(phi, sites, qpoints):
    phase = np.exp(-2j * np.pi * np.asarray(qpoints) @ sites.T)
    matrix = np.einsum("qr,rba->qba", phase, phi) / 106.42
    return (matrix + matrix.conj().swapaxes(1, 2)) / 2


def energies(phi, sites, qpoints):
    qpoints = finite_array(qpoints, ndim=2)
    require(qpoints.shape[1] == 3, "Expected three reciprocal coordinates per point")
    eigenvalues = np.concatenate(
        [
            np.linalg.eigvalsh(dynamical(phi, sites, qpoints[start : start + 1024]))
            for start in range(0, len(qpoints), 1024)
        ]
    )
    return np.sign(eigenvalues) * np.sqrt(abs(eigenvalues)) * ENERGY_FACTOR


def grid(mesh, shift=(0, 0, 0)):
    mesh = finite_array(mesh, shape=(3,))
    require(
        np.all(mesh == np.rint(mesh)) and np.all((mesh >= 2) & (mesh <= 64)),
        "DOS mesh must contain three integers between 2 and 64",
    )
    shift = finite_array(shift, shape=(3,))
    require(np.all((shift >= 0) & (shift < 1)), "Grid shifts must be in [0, 1)")
    return (np.indices(mesh.astype(int)).reshape(3, -1).T + shift) / mesh


def summary(values, tolerance):
    return {
        "real_mode_energy_eV_per_atom": float(
            np.maximum(values, 0).sum(axis=1).mean() / 2
        ),
        "imaginary_weight": float((values < 0).sum(axis=1).mean()),
        "resolved_imaginary_weight": float((values < -tolerance).sum(axis=1).mean()),
    }


def refined_spectrum(phi, sites, tolerance):
    previous = None
    previous_values = None
    for size in (12, 20, 32, 48):
        values = energies(phi, sites, grid([size] * 3, [0.5] * 3))
        stats = summary(values, tolerance)
        if previous is not None:
            error = abs(
                stats["real_mode_energy_eV_per_atom"]
                - previous["real_mode_energy_eV_per_atom"]
            )
            weight_error = abs(
                stats["resolved_imaginary_weight"]
                - previous["resolved_imaginary_weight"]
            )
            spectral_error = spectral_distance(values, previous_values)
            if error <= 1e-5 and weight_error <= 0.01 and spectral_error <= 0.01:
                return values, {
                    **stats,
                    "mesh": [size] * 3,
                    "zpe_sampling_error_eV": error,
                    "imaginary_weight_sampling_error": weight_error,
                    "dos_sampling_error": spectral_error,
                }
        previous = stats
        previous_values = values
    raise UnsupportedEvidence(
        "Reciprocal-space refinement did not resolve the mode integral"
    )


def spectral_distance(first, second):
    """Compare cumulative mode weights, independently of the DOS representation."""
    points = np.linspace(
        min(first.min(), second.min()), max(first.max(), second.max()), 101
    )
    return float(
        np.max(
            abs(
                np.searchsorted(np.sort(first.ravel()), points, side="right")
                / first.size
                - np.searchsorted(np.sort(second.ravel()), points, side="right")
                / second.size
            )
        )
    )


def answer(e, name):
    value = e.results.get(name)
    return value if isinstance(value, dict) else e.json(name)


def array(e, value, **kwargs):
    return finite_array(e.array(value) if isinstance(value, str) else value, **kwargs)


def check_bands(e, phi, sites):
    band = answer(e, "bands")
    q = array(e, band["qpoints"], ndim=2)
    indices = finite_array(band["node_indices"], shape=(6,))
    require(np.all(indices == np.rint(indices)), "Band node indices must be integers")
    indices = indices.astype(int)
    require(
        indices[0] == 0 and indices[-1] == len(q) - 1 and np.all(np.diff(indices) > 0),
        "Band nodes must span the complete Gamma-X-U-L-Gamma-K path",
    )
    require(
        close(q[indices], NODES, atol=1e-8, rtol=0), "Incorrect band path coordinates"
    )
    for left, right in itertools.pairwise(indices):
        delta = q[right] - q[left]
        segment = q[left : right + 1] - q[left]
        t = segment @ delta / (delta @ delta)
        require(
            np.all(np.diff(t) > 0)
            and close(segment, t[:, None] * delta, rtol=0, atol=1e-8),
            "Band points must follow the specified straight path segments",
        )
    return result_close(
        array(e, band["energies_eV"]), energies(phi, sites, q), atol=1e-7
    )


def check_dos(e, phi, sites, reference):
    doc = answer(e, "dos")
    values = energies(
        phi, sites, grid(doc["mesh"], doc.get("shift", [0, 0, 0]))
    ).ravel()
    weight = 3 / len(values)
    kind = doc["representation"]
    if kind == "sticks":
        reported = array(e, doc["energies_eV"], shape=values.shape)
        weights = array(e, doc["weights"], shape=values.shape)
        require(
            result_close(np.sort(reported), np.sort(values), atol=1e-7)
            and close(weights, np.full(len(values), weight), atol=1e-10),
            "DOS sticks must retain every signed mode with total weight three",
        )
    elif kind == "histogram":
        edges = array(e, doc["bin_edges_eV"], ndim=1)
        require(
            len(edges) >= 2
            and np.all(np.diff(edges) > 0)
            and edges[0] <= values.min()
            and edges[-1] >= values.max(),
            "DOS bins must cover all signed energies",
        )
        density = array(e, doc["density_per_eV"], shape=(len(edges) - 1,))
        expected = np.histogram(values, edges)[0] * weight / np.diff(edges)
        require(
            np.all(density >= 0)
            and result_close(density, expected, atol=1e-7)
            and close(np.sum(density * np.diff(edges)), 3),
            "DOS histogram shape or three-mode normalization disagrees",
        )
    elif kind == "gaussian":
        axis = array(e, doc["energy_grid_eV"], ndim=1)
        sigma = float(finite_array(doc["sigma_eV"], shape=()))
        require(
            len(axis) >= 2
            and np.all(np.diff(axis) > 0)
            and sigma > 0
            and axis[0] <= values.min() - 4 * sigma
            and axis[-1] >= values.max() + 4 * sigma
            and np.max(np.diff(axis)) <= sigma / 2,
            "Gaussian DOS must resolve its width and cover four-sigma tails",
        )
        expected = np.zeros(len(axis))
        for start in range(0, len(values), 256):
            expected += np.exp(
                -0.5 * ((axis[:, None] - values[start : start + 256]) / sigma) ** 2
            ).sum(axis=1)
        expected *= weight / (sigma * np.sqrt(2 * np.pi))
        density = array(e, doc["density_per_eV"], shape=axis.shape)
        require(
            np.all(density >= 0)
            and result_close(density, expected, atol=1e-7)
            and abs(np.trapezoid(density, axis) - 3) <= 0.003,
            "Gaussian DOS shape or three-mode normalization disagrees",
        )
    else:
        raise UnsupportedEvidence(
            "Supported DOS representations: sticks, histogram, gaussian"
        )
    require(
        spectral_distance(values, reference) <= 0.03,
        "DOS sampling is too coarse to resolve the integrated spectrum",
    )
    return values.reshape(-1, 3)


def build_plan(e, challenge):
    record = read_record(e)
    geometry(record)
    plan = Plan(e, 4, challenge)
    frames = [
        Atoms(
            numbers=record["numbers"],
            positions=positions,
            cell=record["cell_A"],
            pbc=record["pbc"],
            masses=record["masses_amu"],
        )
        for positions in record["positions_A"]
    ]
    parameters = {
        "model": "teacher.model",
        "dispersion": False,
        "default_dtype": record["metadata"]["default_dtype"],
    }
    targets = ["independent_model_calculation"]
    plan.frames(
        "palladium_forces",
        frames,
        count=4,
        parameters=parameters,
        targets=targets,
        properties=["energy", "forces"],
        values=[
            {"energy": float(energy), "forces": forces.tolist()}
            for energy, forces in zip(
                record["energies_eV"], record["forces_eV_A"], strict=True
            )
        ],
        tolerance={"energy": [1e-5, 1e-3], "forces": [1e-5, 1e-4]},
    )
    if record["metadata"]["method"] == "analytical":
        plan.frames(
            "palladium_hessian",
            frames[:1],
            parameters=parameters,
            targets=targets,
            properties=["hessian"],
            values=[{"hessian": record["hessian_eV_A2"].tolist()}],
        )
    return plan


def evaluate_submission(submission, *, level=2, verifier=None, review=None):
    if type(level) is not int or level not in (1, 2):
        raise ValueError("level must be 1 or 2")
    r = Rubric(4, binary=True, fail_fast=True)
    calculated = {}
    try:
        e = Evidence(submission)
        fingerprint = e.fingerprint()
    except (OSError, ValueError, TypeError) as exc:
        r.check("readable_submission", 100, False, str(exc))
        return {**r.as_dict(), "level": level, "calculated_results": calculated}

    @lru_cache(None)
    def prepared():
        return prepare(read_record(e))

    def constants():
        _, _, diagnostics = prepared()
        calculated.update(diagnostics)
        # Diagnostics come from the archive; optional agent summaries must agree.
        claims = e.results.get("diagnostics", {})
        require(isinstance(claims, dict), "diagnostics must be an object")
        return all(
            result_close(claims[key], value, atol=1e-7)
            for key, value in diagnostics.items()
            if key in claims
        )

    ready = r.check("usable_force_constants", 80 if level == 1 else 40, constants)
    if level == 2:

        @lru_cache(None)
        def spectrum():
            phi, sites, diagnostics = prepared()
            values, stats = refined_spectrum(
                phi, sites, diagnostics["imaginary_tolerance_eV"]
            )
            calculated.update(stats)
            return values

        def bands():
            phi, sites, _ = prepared()
            q = np.vstack(
                [
                    np.linspace(a, b, 50, endpoint=False)
                    for a, b in itertools.pairwise(NODES)
                ]
                + [NODES[-1:]]
            )
            calculated["bands"] = {
                "qpoints": q.tolist(),
                "energies_eV": energies(phi, sites, q).tolist(),
                "node_indices": [0, 50, 100, 150, 200, 250],
            }
            if "bands" in e.results or "bands" in e.manifest.get("artifacts", {}):
                return check_bands(e, phi, sites)
            return True

        def dos():
            phi, sites, diagnostics = prepared()
            values = spectrum()
            if "dos" in e.results or "dos" in e.manifest.get("artifacts", {}):
                values = check_dos(e, phi, sites, values)
                calculated["submitted_mesh_results"] = summary(
                    values, diagnostics["imaginary_tolerance_eV"]
                )
            reported = summary(values, diagnostics["imaginary_tolerance_eV"])
            calculated["dos_summary"] = reported
            for key in ("imaginary_weight", "resolved_imaginary_weight"):
                if key in e.results:
                    require(
                        result_close(e.results[key], reported[key], atol=1e-7),
                        f"Reported {key} disagrees with the retained force constants",
                    )
            resolved = calculated["resolved_imaginary_weight"]
            require(
                abs(reported["resolved_imaginary_weight"] - resolved) <= 0.02,
                "Submitted mesh misses resolved imaginary-mode weight",
            )
            unstable = bool(
                np.any(spectrum() < -diagnostics["imaginary_tolerance_eV"])
                or np.any(values < -diagnostics["imaginary_tolerance_eV"])
            )
            # Instability anywhere on the required path also rules out a stable ZPE.
            band_q = np.vstack(
                [np.linspace(a, b, 51) for a, b in itertools.pairwise(NODES)]
            )
            unstable |= bool(
                np.any(
                    energies(phi, sites, band_q)
                    < -diagnostics["imaginary_tolerance_eV"]
                )
            )
            calculated["zpe_eV_per_atom"] = (
                None if unstable else calculated["real_mode_energy_eV_per_atom"]
            )
            calculated["spectrum_status"] = (
                "unstable"
                if unstable
                else "numerical_residuals"
                if np.any(spectrum() < 0) or np.any(values < 0)
                else "stable"
            )
            if unstable:
                require(
                    "zpe_eV_per_atom" in e.results
                    and e.results["zpe_eV_per_atom"] is None,
                    "Resolved instability has no harmonic ZPE; report null",
                )
                if "real_mode_energy_eV_per_atom" in e.results:
                    require(
                        result_close(
                            e.results["real_mode_energy_eV_per_atom"],
                            calculated["real_mode_energy_eV_per_atom"],
                            atol=1e-5,
                        ),
                        "Real-mode-only energy disagrees with the refined integral",
                    )
                return True
            return result_close(
                e.results["zpe_eV_per_atom"],
                calculated["zpe_eV_per_atom"],
                atol=max(1e-5, 2 * calculated["zpe_sampling_error_eV"]),
            )

        r.check("bands", 20, bands if ready else None)
        r.check("dos_zpe_and_imaginary_modes", 20, dos if ready else None)
    r.unverified(
        "independent_model_calculation",
        "Saved forces require a small independent check with the pinned model",
        points=20,
    )
    verification = None
    if ready and not r.failed and verifier is not None:
        if e.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during grading; retry with frozen evidence"
            )
        verification = verifier.evaluate(e, 4)
        if verification["evidence_sha256"] != fingerprint:
            raise RuntimeError(
                "Submission changed during grading; retry with frozen evidence"
            )
        apply_verification(r, verification)
    if e.fingerprint() != fingerprint:
        raise RuntimeError(
            "Submission changed during grading; retry with frozen evidence"
        )
    if review is not None:
        r.apply_review(review, fingerprint)
    return {
        **r.as_dict(),
        "level": level,
        "evidence_sha256": fingerprint,
        "calculated_results": calculated,
        "verification": verification,
    }
