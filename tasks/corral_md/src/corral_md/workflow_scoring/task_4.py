"""Reconstruct palladium phonon observables from saved numerical evidence only."""

from __future__ import annotations

import itertools
import re
from functools import lru_cache

import numpy as np
from ase import units

from .common import (
    Evidence,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
)

ENERGY_FACTOR = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
STANDARD = 3.89 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
NODES = np.array(
    [
        [0.0, 0, 0],
        [0.5, 0, 0.5],
        [0.625, 0.25, 0.625],
        [0.5, 0.5, 0.5],
        [0, 0, 0],
        [0.375, 0.375, 0.75],
    ]
)


def _close(a, b, atol=1e-7):
    return close(a, b, rtol=1e-4, atol=atol)


def _integers(value, shape):
    a = finite_array(value, shape=shape)
    if not np.all(a == np.rint(a)):
        raise ValueError("Expected integer indexing")
    return a.astype(int)


def _stored(frame, key):
    result = getattr(getattr(frame, "calc", None), "results", {})
    value = frame.arrays.get(key, frame.info.get(key, result.get(key)))
    return finite_array(value, shape=() if key == "energy" else (125, 3))


def _values(e, data):
    value = data.get("values_eV_A2")
    if value is None:
        value = e.array(data["values_artifact"])
    value = finite_array(value, shape=(125, 3, 3))
    layout = data["layout"]
    if layout == "ASE_C_N":
        value = value.swapaxes(1, 2)
    elif layout != "response_source":
        raise UnsupportedMethod("Unsupported force-constant axis convention")
    return value


def _translations(data):
    cells = _integers(data["cell_translations"], (125, 3))
    if np.max(np.abs(cells)) > 2 or len(set(map(tuple, cells))) != 125:
        raise ValueError("Translations must cover the centered 5x5x5 primitive grid")
    return cells


def _opposites(cells):
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    return np.array([lookup[tuple(-cell)] for cell in cells])


def _basis(data):
    cell = finite_array(data["primitive_cell_A"], shape=(3, 3))
    transform = _integers(data.get("primitive_to_standard", np.eye(3)), (3, 3))
    if abs(round(np.linalg.det(transform))) != 1:
        raise ValueError("Primitive basis change must be unimodular")
    expected = transform @ STANDARD
    if not np.allclose(cell @ cell.T, expected @ expected.T, atol=1e-6, rtol=0):
        raise ValueError("Primitive cell is not the specified FCC a=3.89 Angstrom")
    return cell, transform


def _geometry(frames, data):
    cell, _ = _basis(data)
    cells = _translations(data)
    indices = _integers(data["atom_indices"], (125,))
    if sorted(indices.tolist()) != list(range(125)):
        raise ValueError("atom_indices must be a permutation")
    reference_index = data["reference_frame"]
    source = data["source_atom_index"]
    if type(reference_index) is not int or not 0 <= reference_index < len(frames):
        raise ValueError("Invalid reference frame")
    if type(source) is not int or not 0 <= source < 125:
        raise ValueError("Invalid displaced atom index")
    reference = frames[reference_index]
    mass = float(data["mass_amu"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be finite and positive")
    for frame in frames:
        if (
            len(frame) != 125
            or set(frame.get_chemical_symbols()) != {"Pd"}
            or not np.all(frame.pbc)
            or not np.allclose(frame.cell.array, 5 * cell, atol=1e-6, rtol=0)
            or not np.allclose(frame.get_masses(), mass, atol=1e-7, rtol=0)
        ):
            raise ValueError(
                "Raw frames must retain the specified cell, species, and masses"
            )
        finite_array(frame.positions, shape=(125, 3))
    fractional = (
        reference.positions[indices] - reference.positions[source]
    ) @ np.linalg.inv(cell)
    error = fractional - cells
    error -= 5 * np.rint(error / 5)
    if np.max(np.abs(error)) > 1e-6:
        raise ValueError("Atom/cell map disagrees with the undisplaced structure")
    if indices[np.where(np.all(cells == 0, axis=1))[0][0]] != source:
        raise ValueError("Origin translation must map to the displaced atom")
    return cell, cells, indices, reference, source


class UnsupportedMethod(UnsupportedEvidence):
    pass


def _reconstruct(e, frames, data):
    cell, cells, indices, reference, source = _geometry(frames, data)
    derivative = data["derivative"]
    method = derivative["method"]
    if method == "finite_difference":
        frame_ids = _integers(
            derivative["frame_indices"], (len(derivative["frame_indices"]),)
        )
        if (
            len(frame_ids) < 4
            or np.any(frame_ids < 0)
            or np.any(frame_ids >= len(frames))
        ):
            raise ValueError(
                "A rank-three derivative with constant cancellation needs at least four frames"
            )
        u = finite_array(derivative["displacements_A"], shape=(len(frame_ids), 3))
        w = finite_array(
            derivative["derivative_weights_A_inv"], shape=(3, len(frame_ids))
        )
        if not _close(w @ u, np.eye(3)) or not _close(w.sum(axis=1), np.zeros(3)):
            raise ValueError(
                "Derivative weights do not reproduce linear displacements and cancel constant forces"
            )
        forces = []
        for frame_id, displacement in zip(frame_ids, u, strict=False):
            frame = frames[frame_id]
            delta = frame.positions - reference.positions
            delta[source] -= displacement
            scaled = delta @ np.linalg.inv(5 * cell)
            scaled -= np.rint(scaled)
            if np.max(np.abs(scaled @ (5 * cell))) > 1e-6:
                raise ValueError(
                    "Displaced structure does not match its recorded displacement"
                )
            force = _stored(frame, "forces").copy()
            correction = derivative.get("force_correction", "none")
            if correction == "frederiksen":
                force[source] -= force.sum(axis=0)
            elif correction != "none":
                raise UnsupportedMethod(
                    "Unsupported force correction; force derivative is unverified"
                )
            forces.append(force[indices])
        phi = -np.einsum("aj,jrb->rba", w, np.asarray(forces))
    elif method == "analytical":
        if not str(derivative.get("description", "")).strip():
            raise ValueError("Analytical derivative description is required")
        hessian = finite_array(
            e.array(derivative.get("hessian_artifact", "raw_hessian")), shape=(375, 375)
        )
        if not _close(hessian, hessian.T):
            raise ValueError("Saved analytical energy Hessian is not symmetric")
        phi = hessian.reshape(125, 3, 125, 3)[indices, :, source, :].copy()
    else:
        raise UnsupportedMethod(
            "Unsupported derivative adapter; numerical reconstruction is unverified"
        )
    raw = phi.copy()
    origin = np.where(np.all(cells == 0, axis=1))[0][0]
    for operation in data.get("corrections", []):
        if operation == "pair_symmetry":
            phi = (phi + phi[_opposites(cells)].swapaxes(1, 2)) / 2
        elif operation == "acoustic_sum_rule":
            phi[origin] -= phi.sum(axis=0)
        elif isinstance(operation, dict) and set(operation) == {"cutoff_A"}:
            cutoff = float(operation["cutoff_A"])
            if not np.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("Cutoff must be finite and positive")
            phi[np.linalg.norm(cells @ cell, axis=1) > cutoff] = 0
        else:
            raise UnsupportedMethod(
                "Unsupported force-constant correction; transformation is unverified"
            )
    return raw, phi


def _energies(data, phi, qpoints):
    q = finite_array(qpoints, ndim=2)
    if q.shape[1] != 3:
        raise ValueError("Reciprocal coordinates must have three components")
    phase = np.exp(-2j * np.pi * q @ _translations(data).T)
    dynamical = np.einsum("qr,rba->qba", phase, phi) / float(data["mass_amu"])
    # The Hermitian part is explicit, not an implicit eigvalsh triangle choice.
    # Anti-Hermitian residuals are separately reported by the real-space diagnostic.
    dynamical = (dynamical + dynamical.conj().swapaxes(1, 2)) / 2
    squared = np.linalg.eigvalsh(dynamical)
    return np.sign(squared) * np.sqrt(np.abs(squared)) * ENERGY_FACTOR


def _full_grid(q, integration):
    mesh = _integers(integration["mesh"], (3,))
    shift = finite_array(integration.get("shift", [0, 0, 0]), shape=(3,))
    if np.any(mesh <= 0) or len(q) != int(np.prod(mesh)):
        raise ValueError("Grid dimensions do not match qpoints")
    fractional = np.asarray(q) * mesh - shift
    integers = np.rint(fractional)
    if not np.allclose(fractional, integers, atol=1e-8, rtol=0):
        raise ValueError("qpoints are not on the recorded reciprocal grid")
    if len(set(map(tuple, (integers.astype(int) % mesh)))) != len(q):
        raise ValueError("Full reciprocal grid has duplicate or missing points")


def _samples(data, phi, samples):
    q = finite_array(samples["qpoints"], ndim=2)
    if q.shape[1] != 3:
        raise ValueError("Expected three-component qpoints")
    w = finite_array(samples["weights"], shape=(len(q),))
    energy = finite_array(samples["energies_eV"], shape=(len(q), 3))
    if (
        np.any(w <= 0)
        or not _close(w.sum(), 1)
        or not result_close(energy, _energies(data, phi, q), atol=1e-7)
    ):
        raise ValueError(
            "Sample energies or weights disagree with the supplied force constants"
        )
    integration = samples["integration"]
    if integration["kind"] == "uniform_grid":
        _full_grid(q, integration)
        if not _close(w, np.full(len(q), 1 / len(q))):
            raise ValueError("Full uniform grid needs uniform weights")
    elif integration["kind"] == "symmetry_reduced":
        full_q = finite_array(samples["full_grid_qpoints"], ndim=2)
        if full_q.shape[1] != 3:
            raise ValueError("Expected three-component full-grid qpoints")
        _full_grid(full_q, integration)
        index = _integers(samples["representative_index"], (len(full_q),))
        if np.any(index < 0) or np.any(index >= len(q)):
            raise ValueError("Invalid representative mapping")
        if not result_close(_energies(data, phi, full_q), energy[index], atol=1e-7):
            raise ValueError(
                "Symmetry representatives do not reproduce full-grid mode energies"
            )
        if not _close(w, np.bincount(index, minlength=len(q)) / len(full_q)):
            raise ValueError("Irreducible weights disagree with multiplicities")
    else:
        raise UnsupportedMethod(
            "Unsupported quadrature; Brillouin-zone average is unverified"
        )
    return q, w, energy


def _summary(w, energy, tolerance=0):
    return (
        float(np.sum(w[:, None] * np.maximum(energy, 0)) / 2),
        float(np.sum(w[:, None] * (energy < 0))),
        float(np.sum(w[:, None] * (energy < -tolerance))),
    )


def evaluate(e: Evidence, r: Rubric, *, level: int = 2) -> None:
    if level not in (1, 2):
        raise ValueError("level must be 1 or 2")

    level1_points = (
        {
            "fcc_supercell_and_mapping": 15,
            "recorded_model_and_settings": 10,
            "raw_energy_force_records": 15,
            "derivative_reconstruction": 20,
            "force_constant_transformations": 20,
            "symmetry_and_acoustic_diagnostics": 10,
        }
        if level == 1
        else {}
    )

    def shared_check(name, points, condition, detail=""):
        return r.check(name, level1_points.get(name, points), condition, detail)

    @lru_cache(None)
    def data():
        return e.json("force_constants")

    @lru_cache(None)
    def frames():
        return e.trajectory("structures", "raw_structures")

    @lru_cache(None)
    def geometry():
        return _geometry(frames(), data())

    @lru_cache(None)
    def phi():
        geometry()
        return _values(e, data())

    def supported(call):
        try:
            return call()
        except UnsupportedMethod as exc:
            return None, str(exc)

    def identities():
        s = e.settings
        shared = (
            isinstance(s.get("checkpoint"), str)
            and bool(s["checkpoint"].strip())
            and isinstance(s.get("force_constant_method"), str)
            and bool(s["force_constant_method"].strip())
        )
        if level == 1:
            return shared
        return (
            shared
            and isinstance(s.get("checkpoint_sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", s["checkpoint_sha256"]) is not None
            and isinstance(s.get("software_versions"), dict)
            and bool(s["software_versions"])
        )

    shared_check(
        "fcc_supercell_and_mapping",
        10,
        lambda: bool(geometry()),
        "Fixed primitive FCC a=3.89 A, 125 Pd atoms, cell/atom map and recorded masses",
    )
    shared_check(
        "recorded_model_and_settings",
        5,
        identities,
        (
            "Recorded checkpoint identity and numerical method"
            if level == 1
            else "Recorded checkpoint identity and digest, software versions, numerical method"
        ),
    )

    def raw_records():
        geometry()
        return all(
            _stored(a, "energy").shape == () and _stored(a, "forces").shape == (125, 3)
            for a in frames()
        )

    @lru_cache(None)
    def reconstructed():
        return _reconstruct(e, frames(), data())

    records_ok = shared_check(
        "raw_energy_force_records",
        10,
        raw_records,
        "Complete stored force and energy arrays; no calculator evaluation",
    )
    shared_check(
        "derivative_reconstruction",
        12,
        lambda: supported(lambda: reconstructed()[0].shape == (125, 3, 3)),
        "Displacements and derivative weights/Hessian agree with the recorded geometry",
    )
    shared_check(
        "force_constant_transformations",
        12,
        lambda: supported(lambda: result_close(reconstructed()[1], phi(), atol=1e-7)),
        "Recorded corrections reproduce the supplied final force constants",
    )
    # An unsupported adapter is distinct from evidence that contradicts a
    # supported calculation. The latter cannot support downstream observables.
    force_evidence_ok = records_ok and r.checks[-1]["status"] != "failed"

    def diagnostics():
        values = phi()
        cells = _translations(data())
        claimed = e.results["validation"]
        norm = claimed.get("residual_norm", "max")
        norms = {
            "max": lambda value: np.max(np.abs(value)),
            "rms": lambda value: np.sqrt(np.mean(value**2)),
            "frobenius": lambda value: np.linalg.norm(value.ravel()),
        }
        if norm not in norms:
            raise UnsupportedMethod(f"Unsupported residual norm {norm!r}")
        residual = norms[norm]
        return (
            result_close(
                claimed["pair_symmetry_residual_eV_A2"],
                residual(values - values[_opposites(cells)].swapaxes(1, 2)),
                atol=1e-07,
            )
            and result_close(
                claimed["acoustic_residual_eV_A2"],
                residual(values.sum(axis=0)),
                atol=1e-07,
            )
            and (
                "gamma_energies_eV" not in claimed
                or result_close(
                    claimed["gamma_energies_eV"],
                    _energies(data(), values, [[0, 0, 0]])[0],
                    atol=1e-07,
                )
            )
        )

    if level == 1:
        shared_check(
            "symmetry_and_acoustic_diagnostics",
            10,
            lambda: force_evidence_ok and diagnostics(),
            "Recomputed residual norms and signed Gamma modes; no hidden stability threshold",
        )
        r.unverified(
            "execution_provenance",
            "Self-produced records cannot establish checkpoint execution or authentic force evaluations.",
        )
        return

    def bands():
        band = e.json("bands")
        q = finite_array(band["qpoints"], ndim=2)
        if q.shape[1] != 3:
            return False
        nodes = _integers(band["node_indices"], (6,))
        if nodes[0] != 0 or nodes[-1] != len(q) - 1 or np.any(np.diff(nodes) <= 0):
            return False
        _, transform = _basis(data())
        qstandard = q @ np.linalg.inv(transform).T
        if not np.allclose(qstandard[nodes], NODES, atol=1e-8, rtol=0):
            return False
        for left, right in itertools.pairwise(nodes):
            delta = qstandard[right] - qstandard[left]
            segment = qstandard[left : right + 1] - qstandard[left]
            t = segment @ delta / (delta @ delta)
            if np.any(np.diff(t) <= 0) or not np.allclose(
                segment, t[:, None] * delta, atol=1e-8, rtol=0
            ):
                return False
        return result_close(
            band["energies_eV"], _energies(data(), phi(), q), atol=1e-07
        )

    @lru_cache(None)
    def samples():
        return _samples(data(), phi(), e.json("bz_samples"))

    def dos():
        _, w, energy = samples()
        values, weights = energy.ravel(), np.repeat(w, 3)
        doc = e.json("dos")
        representation = doc["representation"]
        if representation == "sticks":
            reported = finite_array(doc["energies_eV"], shape=values.shape)
            reported_weights = finite_array(doc["weights"], shape=weights.shape)
            order = np.lexsort((weights, values))
            other = np.lexsort((reported_weights, reported))
            return (
                result_close(values[order], reported[other], atol=1e-07)
                and result_close(weights[order], reported_weights[other], atol=1e-07)
                and result_close(reported_weights.sum(), 3, atol=1e-07)
            )
        if representation == "histogram":
            grid = finite_array(doc["bin_edges_eV"], ndim=1)
            if (
                len(grid) < 2
                or np.any(np.diff(grid) <= 0)
                or grid[0] > values.min()
                or grid[-1] < values.max()
            ):
                return False
            density = finite_array(doc["density_per_eV"], shape=(len(grid) - 1,))
            expected = np.histogram(values, grid, weights=weights)[0] / np.diff(grid)
            return result_close(density, expected, atol=1e-07) and result_close(
                np.sum(density * np.diff(grid)), 3, atol=1e-07
            )
        if representation == "gaussian":
            grid = finite_array(doc["energy_grid_eV"], ndim=1)
            sigma = float(doc["sigma_eV"])
            if (
                len(grid) < 2
                or np.any(np.diff(grid) <= 0)
                or not np.isfinite(sigma)
                or sigma <= 0
                or grid[0] > values.min()
                or grid[-1] < values.max()
            ):
                return False
            density = finite_array(doc["density_per_eV"], shape=grid.shape)
            # Stream over modes to avoid an N_modes x N_grid allocation.
            expected = np.zeros(len(grid))
            normalization = doc.get("normalization", "none")
            if normalization not in {"none", "global", "per_mode"}:
                raise UnsupportedMethod("Unsupported DOS normalization")
            for value, weight in zip(values, weights, strict=False):
                gaussian = np.exp(-0.5 * ((grid - value) / sigma) ** 2) / (
                    sigma * np.sqrt(2 * np.pi)
                )
                if normalization == "per_mode":
                    gaussian /= np.trapezoid(gaussian, grid)
                expected += weight * gaussian
            if normalization == "global":
                expected *= 3 / np.trapezoid(expected, grid)
            area = np.trapezoid(density, grid)
            area_ok = (
                abs(area - 3) <= 0.03
                if normalization == "none"
                else result_close(area, 3, atol=1e-07)
            )
            return result_close(density, expected, atol=1e-07) and area_ok
        raise UnsupportedMethod(
            "Unsupported DOS adapter; spectral shape and normalization are unverified"
        )

    def zpe():
        _, w, energy = samples()
        tolerance = float(e.results.get("imaginary_tolerance_eV", 0))
        if not np.isfinite(tolerance) or tolerance < 0:
            return False
        expected, imaginary, resolved = _summary(w, energy, tolerance)
        kind = e.results["zpe_kind"]
        if imaginary == 0:
            label = kind == "harmonic_zpe"
        else:
            label = kind == "real_modes_only" or (
                kind == "residual_corrected_zpe" and tolerance > 0 and resolved == 0
            )
        return (
            label
            and result_close(e.results["zpe_eV_per_atom"], expected, atol=1e-07)
            and result_close(e.results["imaginary_weight"], imaginary, atol=1e-07)
            and result_close(
                e.results["resolved_imaginary_weight"], resolved, atol=1e-07
            )
        )

    r.check(
        "band_path_and_signed_energies",
        15,
        lambda: force_evidence_ok and bands(),
        "Independent Fourier transform and eigenspectrum along Gamma-X-U-L-Gamma-K",
    )
    r.check(
        "dos_from_signed_bz_samples",
        8,
        lambda: force_evidence_ok and supported(dos),
        "Full Brillouin-zone quadrature, DOS shape, and normalization to three modes",
    )
    r.check(
        "zpe_and_imaginary_mode_accounting",
        8,
        lambda: force_evidence_ok and supported(zpe),
        "Weighted energy, imaginary weight, and honest stable/residual/real-mode-only label",
    )

    r.check(
        "symmetry_and_acoustic_diagnostics",
        10,
        lambda: force_evidence_ok and diagnostics(),
        "Recomputed residual norms and signed Gamma modes; no hidden stability threshold",
    )
    r.unverified(
        "execution_provenance",
        "Self-produced records do not establish checkpoint execution, authentic forces, or authentic analytical derivatives.",
    )
