"""Versioned, evaluator-owned numerical references for fixed MD task inputs.

The reference document is generated once while publishing a release.  It only
contains calculations whose model, calculator settings, and atomic geometries
are fixed by the task.  Submission-dependent checkpoints and trajectories are
never included.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from io import BytesIO
from itertools import product
from pathlib import Path

import modal
import numpy as np

SCHEMA = 1
TASK3_SEPARATIONS = np.round(np.arange(0.6, 5.01, 0.1), 8)
TASK5_STRAINS = np.asarray(list(product(np.arange(-2, 3) / 100, repeat=2)))

TASK3_TEACHER = {
    "model": "teacher.model",
    "default_dtype": "float64",
    "dispersion": True,
}
TASK3_STUDENT = {
    "model": "student.model",
    "default_dtype": "float64",
    "dispersion": False,
}
TASK5_TEACHER = {
    "model": "teacher.model",
    "default_dtype": "float64",
    "dispersion": False,
}


class GroundTruthUnavailable(RuntimeError):
    """The selected release does not contain a compatible reference."""


def _frame(numbers, positions, cell, pbc):
    return {
        "numbers": list(numbers),
        "positions": np.asarray(positions, dtype=float).tolist(),
        "cell": np.asarray(cell, dtype=float).tolist(),
        "pbc": list(pbc),
    }


def _task3_frames():
    return [
        _frame(
            [47, 47],
            [[0, 0, 0], [float(separation), 0, 0]],
            np.zeros((3, 3)),
            [False, False, False],
        )
        for separation in TASK3_SEPARATIONS
    ]


def task5_cell(isotropic_strain, uniaxial_strain):
    cell = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))
    cell *= 1 + float(isotropic_strain)
    return cell @ np.diag([1, 1, 1 + float(uniaxial_strain)])


def _task5_frames():
    frames = []
    for isotropic, uniaxial in TASK5_STRAINS:
        cell = task5_cell(isotropic, uniaxial)
        frames.append(
            _frame(
                [14, 14],
                np.asarray([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell,
                cell,
                [True, True, True],
            )
        )
    return frames


def calculation_jobs():
    """Return the complete fixed-input calculation set for one release."""
    dimer_frames = _task3_frames()
    return [
        {
            "id": "task3_teacher",
            "operation": "mace",
            "frames": dimer_frames,
            "properties": ["energy", "forces"],
            "parameters": TASK3_TEACHER,
        },
        {
            "id": "task3_student",
            "operation": "mace",
            "frames": dimer_frames,
            "properties": ["energy", "forces"],
            "parameters": TASK3_STUDENT,
        },
        {
            "id": "task5_teacher",
            "operation": "mace",
            "frames": _task5_frames(),
            "properties": ["hessian"],
            "parameters": TASK5_TEACHER,
        },
    ]


def _finite(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(
            f"Ground-truth output has shape {result.shape}, expected {shape}"
        )
    return result


def build_document(response, release_id, assets):
    """Validate trusted worker output and produce the immutable reference document."""
    returned = response.get("jobs", [])
    if not isinstance(returned, list) or {item.get("id") for item in returned} != {
        "task3_teacher",
        "task3_student",
        "task5_teacher",
    }:
        raise ValueError("Ground-truth calculation response is incomplete")
    jobs = {item["id"]: item for item in returned}
    if any(item.get("status") != "complete" for item in jobs.values()):
        raise RuntimeError(f"Ground-truth calculation failed: {jobs}")

    def dimers(name, parameters):
        rows = jobs[name]["value"]["frames"]
        if len(rows) != len(TASK3_SEPARATIONS):
            raise ValueError("Ground-truth dimer grid has the wrong length")
        energies, radial_forces = [], []
        for row in rows:
            energies.append(float(_finite(row["energy"], ()).item()))
            forces = _finite(row["forces"], (2, 3))
            scale = max(1.0, float(np.max(np.abs(forces))))
            if not np.allclose(
                forces.sum(axis=0), 0, rtol=0, atol=1e-5 * scale
            ) or not np.allclose(forces[:, 1:], 0, rtol=0, atol=1e-5 * scale):
                raise ValueError("Canonical dimer forces are not radial and balanced")
            radial_forces.append(float(forces[0, 0]))
        return {
            "parameters": parameters,
            "energy_eV": energies,
            "force_on_atom_0_along_bond_eV_A": radial_forces,
        }

    hessian_rows = jobs["task5_teacher"]["value"]["frames"]
    if len(hessian_rows) != len(TASK5_STRAINS):
        raise ValueError("Ground-truth strain grid has the wrong length")
    hessians = []
    for row in hessian_rows:
        hessian = _finite(row["hessian"], (6, 6))
        if not np.allclose(hessian, hessian.T, rtol=1e-4, atol=1e-6):
            raise ValueError("Canonical Task 5 Hessian is not symmetric")
        hessians.append(hessian.tolist())
    return {
        "schema": SCHEMA,
        "release_id": release_id,
        "models": {
            name: assets["models"][name]["sha256"]
            for name in ("teacher.model", "student.model")
        },
        "task_3": {
            "separations_A": TASK3_SEPARATIONS.tolist(),
            "teacher": dimers("task3_teacher", TASK3_TEACHER),
            "student": dimers("task3_student", TASK3_STUDENT),
        },
        "task_5": {
            "strain_pairs": TASK5_STRAINS.tolist(),
            "teacher": {
                "parameters": TASK5_TEACHER,
                "hessians_eV_A2": hessians,
            },
        },
    }


def validate_document(document, release_id, assets):
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or document.get("release_id") != release_id
        or document.get("models")
        != {
            name: assets["models"][name]["sha256"]
            for name in ("teacher.model", "student.model")
        }
    ):
        raise GroundTruthUnavailable("Release ground truth is missing or incompatible")
    task3 = document.get("task_3", {})
    task5 = document.get("task_5", {})
    if not np.array_equal(
        _finite(task3.get("separations_A"), (45,)), TASK3_SEPARATIONS
    ):
        raise GroundTruthUnavailable("Task 3 ground-truth grid differs")
    for name, parameters in (
        ("teacher", TASK3_TEACHER),
        ("student", TASK3_STUDENT),
    ):
        if task3.get(name, {}).get("parameters") != parameters:
            raise GroundTruthUnavailable("Task 3 ground-truth settings differ")
        _finite(task3.get(name, {}).get("energy_eV"), (45,))
        _finite(
            task3.get(name, {}).get("force_on_atom_0_along_bond_eV_A"),
            (45,),
        )
    if (
        not np.array_equal(_finite(task5.get("strain_pairs"), (25, 2)), TASK5_STRAINS)
        or task5.get("teacher", {}).get("parameters") != TASK5_TEACHER
    ):
        raise GroundTruthUnavailable("Task 5 ground-truth inputs differ")
    _finite(task5.get("teacher", {}).get("hessians_eV_A2"), (25, 6, 6))
    return document


def postprocess_hessian(hessian, specification):
    """Apply the Task 5 transformations in standard atom/Cartesian order."""
    result = _finite(hessian, (6, 6)).copy()
    symmetry = specification.get("symmetrization", "none")
    if symmetry == "average_transpose":
        result = (result + result.T) / 2
    elif symmetry != "none":
        raise ValueError(f"Unsupported reference symmetrization: {symmetry!r}")
    acoustic = specification.get("acoustic_sum_rule", "none")
    if acoustic == "projection":
        translations = np.tile(np.eye(3), (2, 1))
        projection = np.eye(6) - translations @ translations.T / 2
        result = projection @ result @ projection
    elif acoustic != "none":
        raise ValueError(f"Unsupported reference acoustic sum rule: {acoustic!r}")
    return result


def _task3_result(job, document, variant):
    reference = document["task_3"]
    values = reference[variant]
    if job.get("parameters", {}) != values["parameters"]:
        raise GroundTruthUnavailable(
            "Task 3 calculator settings differ from the reference"
        )
    if job.get("properties") != ["energy", "forces"]:
        raise ValueError("Task 3 reference requires energy and forces")
    separations = _finite(reference["separations_A"], (45,))
    energies = _finite(values["energy_eV"], (45,))
    radial = _finite(values["force_on_atom_0_along_bond_eV_A"], (45,))
    output = []
    for record in job.get("frames", []):
        numbers = record.get("numbers")
        if numbers != [47, 47] or any(record.get("pbc", [])):
            raise ValueError("Task 3 reference expects a nonperiodic Ag2 dimer")
        positions = _finite(record.get("positions"), (2, 3))
        bond = positions[1] - positions[0]
        distance = float(np.linalg.norm(bond))
        index = int(np.argmin(np.abs(separations - distance)))
        if abs(separations[index] - distance) > 1e-5 or distance <= 0:
            raise ValueError("Task 3 dimer does not lie on the reference grid")
        force = radial[index] * bond / distance
        output.append(
            {
                "energy": float(energies[index]),
                "forces": [force.tolist(), (-force).tolist()],
            }
        )
    if len(output) != 45:
        raise ValueError("Task 3 reference expects all 45 dimers")
    return {"frames": output}


def _task5_result(job, document):
    reference = document["task_5"]
    values = reference["teacher"]
    if job.get("parameters", {}) != values["parameters"]:
        raise GroundTruthUnavailable(
            "Task 5 calculator settings differ from the reference"
        )
    if job.get("properties") != ["hessian"]:
        raise ValueError("Task 5 reference requires Hessians")
    pairs = _finite(reference["strain_pairs"], (25, 2))
    hessians = _finite(values["hessians_eV_A2"], (25, 6, 6))
    frames = job.get("frames", [])
    specifications = job.get("postprocess", [])
    requested = job.get("reference", {}).get("strain_pairs", [])
    if not (len(frames) == len(specifications) == len(requested) == 25):
        raise ValueError("Task 5 reference expects all 25 strain records")
    output = []
    for record, specification, requested_pair in zip(
        frames, specifications, requested, strict=True
    ):
        pair = _finite(requested_pair, (2,))
        distances = np.max(np.abs(pairs - pair), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > 1e-10:
            raise ValueError("Task 5 strain does not lie on the reference grid")
        cell = task5_cell(*pair)
        if record.get("numbers") != [14, 14] or not all(record.get("pbc", [])):
            raise ValueError("Task 5 reference expects periodic Si2")
        actual_cell = _finite(record.get("cell"), (3, 3))
        if not np.allclose(actual_cell, cell, rtol=0, atol=1e-6):
            raise ValueError("Task 5 cell differs from its reference strain")
        positions = _finite(record.get("positions"), (2, 3))
        fractional = (positions[1] - positions[0]) @ np.linalg.inv(cell)
        fractional -= np.rint(fractional)
        if np.linalg.norm((fractional - 0.25) @ cell) < 1e-6:
            order = np.arange(6)
        elif np.linalg.norm((fractional + 0.25) @ cell) < 1e-6:
            order = np.asarray([3, 4, 5, 0, 1, 2])
        else:
            raise ValueError("Task 5 basis differs from the fixed reference")
        hessian = hessians[index][np.ix_(order, order)]
        output.append({"hessian": postprocess_hessian(hessian, specification).tolist()})
    return {"frames": output}


def resolve_reference(job, document, release_id, assets):
    """Resolve one evaluator-created reference job without running a calculator."""
    validate_document(document, release_id, assets)
    reference = job.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("Reference job is missing its reference selector")
    name = reference.get("name")
    if name == "task3_teacher_dimers":
        return _task3_result(job, document, "teacher")
    if name == "task3_student_dimers":
        return _task3_result(job, document, "student")
    if name == "task5_strained_hessians":
        return _task5_result(job, document)
    raise ValueError(f"Unknown ground-truth reference: {name!r}")


def _payload_ref(payload):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def publish_ground_truth(app_name, release_id, volume, assets):
    """Calculate and publish fixed references once during release deployment."""
    destination = f"/corral/releases/{release_id}/ground_truth.json"
    try:
        existing = b"".join(volume.read_file(destination))
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    else:
        document = json.loads(existing)
        validate_document(document, release_id, assets)
        return {"path": destination, **_payload_ref(existing)}

    identifier = "ground-truth-" + release_id
    request = {
        "schema": 1,
        "task_number": 0,
        "jobs": calculation_jobs(),
        "evidence_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
        "challenge": identifier,
    }
    with tempfile.TemporaryDirectory(prefix="corral-md-ground-truth-") as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(request, allow_nan=False))
        payload = path.read_bytes()
        remote = f"/corral/verifications/{identifier}/input/request.json"
        with volume.batch_upload(force=True) as upload:
            upload.put_file(BytesIO(payload), remote)
        response = modal.Function.from_name(app_name, "verify_calculations").remote(
            identifier,
            release_id,
            {"request.json": _payload_ref(payload)},
        )
    document = build_document(response, release_id, assets)
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(payload), destination)
    return {"path": destination, **_payload_ref(payload)}


__all__ = [
    "TASK3_STUDENT",
    "TASK3_TEACHER",
    "TASK5_TEACHER",
    "GroundTruthUnavailable",
    "build_document",
    "calculation_jobs",
    "postprocess_hessian",
    "publish_ground_truth",
    "resolve_reference",
    "task5_cell",
    "validate_document",
]
