"""Evaluator-owned Modal recalculation and task-10 execution verification.

Only geometry, declared calculator parameters and isolated checkpoint bytes go
to calculation sandboxes. Expected answers remain with this evaluator. Sampling
is selected after the evidence is frozen and is recorded in the report.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .common import EvidenceError, UnsupportedEvidence, finite_array

TASK3_TEACHER_REFERENCE = {
    "model": "teacher.model",
    "default_dtype": "float64",
    "dispersion": True,
}
TASK3_STUDENT_REFERENCE = {
    "model": "student.model",
    "default_dtype": "float64",
    "dispersion": False,
}
TASK5_REFERENCE = {
    "model": "teacher.model",
    "default_dtype": "float64",
    "dispersion": False,
}


def file_ref(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": Path(path).stat().st_size}


def geometry(atoms):
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "masses": atoms.get_masses().tolist(),
    }


def stored(atoms):
    values = dict(getattr(getattr(atoms, "calc", None), "results", {}))
    for key, aliases in {
        "energy": ("energy", "potential_energy_eV"),
        "forces": ("forces",),
    }.items():
        for alias in aliases:
            if alias in atoms.info:
                values[key] = atoms.info[alias]
            if alias in atoms.arrays:
                values[key] = atoms.arrays[alias]
    return {
        k: finite_array(v).tolist()
        for k, v in values.items()
        if k in {"energy", "forces", "stress"}
    }


class Plan:
    def __init__(self, evidence, task_number, challenge):
        self.evidence = evidence
        self.task_number = task_number
        self.challenge = challenge
        self.jobs, self.expected, self.files, self.notes = [], {}, {}, []

    def sample(self, size, count, label):
        if size < 1:
            raise EvidenceError("Cannot sample empty evidence")
        # A fresh evaluator nonce prevents an agent from choosing which frames get checked.
        seed = int.from_bytes(
            hashlib.sha256((self.challenge + label).encode()).digest()[:8], "big"
        )
        return sorted(
            np.random.default_rng(seed)
            .choice(size, min(size, count), replace=False)
            .tolist()
        )

    def add(
        self,
        name,
        operation,
        inputs,
        expected,
        *,
        parameters=None,
        targets=(),
        tolerance=None,
    ):
        if name in self.expected:
            raise ValueError("Duplicate verification check")
        job = {
            "id": name,
            "operation": operation,
            **inputs,
            "parameters": parameters or {},
        }
        json.dumps(job, allow_nan=False)
        self.jobs.append(job)
        self.expected[name] = {
            "values": expected,
            "targets": list(targets),
            "tolerance": tolerance
            or {
                "energy": [0, 0.01],
                "forces": [0, 0.02],
                "stress": [0.01, 1e-4],
                "hessian": [0.01, 0.02],
            },
            "indices": inputs.get("sample_indices", []),
        }

    def frames(
        self,
        name,
        frames,
        *,
        count=3,
        parameters=None,
        targets=(),
        operation="mace",
        values=None,
        tolerance=None,
        properties=None,
        reference=None,
        postprocess=None,
        fallback_operation=None,
    ):
        indices = self.sample(len(frames), count, name)
        expected = (
            [stored(frames[i]) for i in indices]
            if values is None
            else [values[i] for i in indices]
        )
        properties = properties or sorted(
            set.intersection(*(set(row) for row in expected))
        )
        if not properties:
            self.notes.append(
                {
                    "id": name,
                    "status": "unverified",
                    "detail": "No saved energies or forces to compare",
                }
            )
            return
        expected = [{key: row[key] for key in properties} for row in expected]
        inputs = {
            "frames": [geometry(frames[i]) for i in indices],
            "properties": properties,
            "sample_indices": indices,
        }
        if reference is not None:
            inputs["reference"] = reference
        if postprocess is not None:
            inputs["postprocess"] = [postprocess[i] for i in indices]
        if fallback_operation is not None:
            inputs["fallback_operation"] = fallback_operation
        self.add(
            name,
            operation,
            inputs,
            {"frames": expected},
            parameters=parameters,
            targets=targets,
            tolerance=tolerance,
        )

    def checkpoint(self, path):
        path = Path(path)
        name = "checkpoint-" + file_ref(path)["sha256"] + path.suffix
        self.files[name] = path
        return {"checkpoint": name, "checkpoint_sha256": file_ref(path)["sha256"]}

    def attempt(self, name, function, targets=()):
        try:
            function()
        except UnsupportedEvidence as exc:
            self.notes.append(
                {
                    "id": name,
                    "status": "unverified",
                    "detail": str(exc),
                    "targets": list(targets),
                }
            )
        except (KeyError, IndexError, ValueError, TypeError, OSError) as exc:
            self.notes.append(
                {
                    "id": name,
                    "status": "unverified",
                    "detail": f"Cannot prepare check: {exc}",
                    "targets": list(targets),
                }
            )


def calculator_settings(e, *, model="teacher.model", dispersion=False, role=None):
    config = (
        e.settings.get(f"{role}_model_settings", {})
        if role
        else e.settings.get("model_settings", e.settings.get("calculator", {}))
    )
    if not isinstance(config, dict):
        raise UnsupportedEvidence("Calculator settings need review")
    allowed = {
        "default_dtype",
        "dtype",
        "precision",
        "device",
        "dispersion",
        "model",
        "model_path",
        "checkpoint",
    }
    if set(config) - allowed:
        raise UnsupportedEvidence(
            "Additional calculator settings need a verification adapter"
        )
    return {
        "model": model,
        "default_dtype": config.get(
            "default_dtype", config.get("dtype", config.get("precision", "float64"))
        ),
        "dispersion": config.get("dispersion", dispersion),
    }


def build_plan(e, task_number, challenge):
    p = Plan(e, task_number, challenge)

    def model(**kw):
        return calculator_settings(e, **kw)

    if task_number == 1:

        def silicon():
            frames = e.trajectory("heating_end_state")
            values = None
            if "heating_end_energy_forces" in e._artifacts:
                data = e.json("heating_end_energy_forces")
                values = [{"energy": data["energy_eV"], "forces": data["forces_eV_A"]}]
            p.frames(
                "sw_heating_state",
                frames,
                operation="sw",
                values=values,
                targets=["thermal_protocol_and_log"],
                tolerance={"energy": [0, 0.05], "forces": [0, 0.05]},
            )

        p.attempt("sw_heating_state", silicon)
    elif task_number == 2:
        p.notes.append(
            {
                "id": "model_calculations",
                "status": "not_applicable",
                "detail": "The paired task-2 plan requires log/state consistency, not a new calculator check",
            }
        )
    elif task_number == 3:
        fitting = [
            "dataset_energy_force_labels",
            "before_after_predictions",
            "aggregate_fitting_errors",
            "separation_resolved_errors",
        ]

        def silver():
            frames = e.trajectory("dataset", "labeled_dataset", "training_dataset")
            frames = sorted(
                frames, key=lambda a: np.linalg.norm(a.positions[1] - a.positions[0])
            )
            teacher_parameters = {**model(role="teacher"), "dispersion": True}
            p.frames(
                "teacher_dimer_labels",
                frames,
                count=len(frames),
                parameters=teacher_parameters,
                targets=fitting,
                tolerance={"energy": [1e-5, 1e-3], "forces": [1e-5, 1e-3]},
                operation=(
                    "reference"
                    if teacher_parameters == TASK3_TEACHER_REFERENCE
                    else "mace"
                ),
                reference=(
                    {"name": "task3_teacher_dimers"}
                    if teacher_parameters == TASK3_TEACHER_REFERENCE
                    else None
                ),
                fallback_operation=(
                    "mace" if teacher_parameters == TASK3_TEACHER_REFERENCE else None
                ),
            )
            checkpoint = p.checkpoint(
                e.artifact("trained_checkpoint", "checkpoint", "trained_model")
            )
            for which in ("before", "after"):
                data = e.json(f"predictions_{which}", f"{which}_predictions")
                order = np.argsort(data["separation_A"])
                values = [
                    {"energy": data["energy_eV"][i], "forces": data["forces_eV_A"][i]}
                    for i in order
                ]
                parameters = model(model="student.model", role="student")
                if which == "after":
                    parameters.update(model="submitted", **checkpoint)
                p.frames(
                    f"student_{which}",
                    frames,
                    count=len(frames),
                    values=values,
                    parameters=parameters,
                    targets=fitting[1:]
                    + (["checkpoint_container_and_digest"] if which == "after" else []),
                    operation=(
                        "reference"
                        if which == "before" and parameters == TASK3_STUDENT_REFERENCE
                        else "mace"
                    ),
                    reference=(
                        {"name": "task3_student_dimers"}
                        if which == "before" and parameters == TASK3_STUDENT_REFERENCE
                        else None
                    ),
                    fallback_operation=(
                        "mace"
                        if which == "before" and parameters == TASK3_STUDENT_REFERENCE
                        else None
                    ),
                )
            p.frames(
                "trained_bulk_model",
                e.trajectory("md_trajectory", "bulk_trajectory", "trajectory"),
                parameters={**model(role="md"), "model": "submitted", **checkpoint},
                targets=[
                    "checkpoint_container_and_digest",
                    "energy_and_force_assessment",
                ],
            )

        p.attempt("silver_model_checks", silver, fitting)
    elif task_number == 4:
        targets = [
            "raw_energy_force_records",
            "derivative_reconstruction",
            "force_constant_transformations",
            "band_path_and_signed_energies",
            "dos_from_signed_bz_samples",
            "zpe_and_imaginary_mode_accounting",
        ]
        p.attempt(
            "palladium_forces",
            lambda: p.frames(
                "palladium_forces",
                e.trajectory("structures", "raw_structures"),
                count=2000,
                parameters=model(),
                targets=targets,
            ),
            targets,
        )

        def palladium_hessian():
            data = e.json("force_constants")
            derivative = data["derivative"]
            if derivative["method"] == "analytical":
                frame = e.trajectory("structures", "raw_structures")[
                    data["reference_frame"]
                ]
                hessian = e.array(derivative.get("hessian_artifact", "raw_hessian"))
                p.frames(
                    "palladium_hessian",
                    [frame],
                    properties=["hessian"],
                    values=[{"hessian": hessian.tolist()}],
                    parameters=model(),
                    targets=targets[1:],
                )

        p.attempt("palladium_hessian", palladium_hessian, targets[1:])
    elif task_number == 5:
        targets = [
            "force_constants_reconstructed_from_raw_evidence",
            "all_mode_energies_and_imaginary_modes",
            "highest_positive_real_mode",
            "unweighted_ols_strain_coefficients",
            "unweighted_ols_intercept",
            "all_25_fit_residuals",
        ]

        def strains():
            from .task_5 import _array, _order, _records

            frames = e.trajectory("strained_structures")
            records = _records(e, "strain_calculations")
            parameters = model()
            if parameters == TASK5_REFERENCE:
                expected, selected, pairs, transformations = [], [], [], []
                for row in records:
                    base = frames[row["structure_index"]]
                    order = _order(row)
                    matrix = _array(e, row["force_constants_eV_A2"], (6, 6))
                    standard = np.empty((6, 6))
                    standard[np.ix_(order, order)] = matrix
                    selected.append(base)
                    expected.append({"hessian": standard.tolist()})
                    pairs.append([row["isotropic_strain"], row["uniaxial_strain"]])
                    transformations.append(
                        {
                            "symmetrization": row["symmetrization"],
                            "acoustic_sum_rule": row["acoustic_sum_rule"],
                        }
                    )
                p.add(
                    "silicon_reference_hessians",
                    "reference",
                    {
                        "frames": [geometry(frame) for frame in selected],
                        "properties": ["hessian"],
                        "sample_indices": list(range(len(selected))),
                        "reference": {
                            "name": "task5_strained_hessians",
                            "strain_pairs": pairs,
                        },
                        "postprocess": transformations,
                        "fallback_operation": "mace",
                    },
                    {"frames": expected},
                    parameters=parameters,
                    targets=targets,
                    tolerance={"hessian": [0.01, 0.02]},
                )
                return
            for row in records:
                base = frames[row["structure_index"]]
                if row["method"] == "analytic_hessian":
                    hessian = _array(e, row["raw_derivatives_eV_A2"], (6, 6))
                    if row["derivative_kind"] == "force_jacobian":
                        hessian = -hessian
                    order = _order(row)
                    expected = np.empty((6, 6))
                    expected[np.ix_(order, order)] = hessian
                    p.frames(
                        f"strain_{row['id']}",
                        [base],
                        properties=["hessian"],
                        values=[{"hessian": expected.tolist()}],
                        parameters=parameters,
                        targets=targets,
                    )
                    continue
                if row["method"] == "central_difference":
                    displacements = np.concatenate(
                        [
                            _array(e, row[k])
                            for k in ("plus_displacements_A", "minus_displacements_A")
                        ]
                    )
                    forces = np.concatenate(
                        [
                            _array(e, row[k])
                            for k in ("plus_forces_eV_A", "minus_forces_eV_A")
                        ]
                    )
                elif row["method"] == "finite_difference":
                    displacements, forces = (
                        _array(e, row["displacements_A"]),
                        _array(e, row["forces_eV_A"]),
                    )
                else:
                    raise UnsupportedEvidence(
                        "Derivative method needs a calculator adapter"
                    )
                displaced = []
                for delta in displacements:
                    atoms = base.copy()
                    atoms.positions += delta
                    displaced.append(atoms)
                p.frames(
                    f"strain_{row['id']}",
                    displaced,
                    count=len(displaced),
                    values=[{"forces": f.tolist()} for f in forces],
                    parameters=parameters,
                    targets=targets,
                )

        p.attempt("silicon_force_checks", strains, targets)
    elif task_number == 6:
        for name, aliases in (
            ("equilibration", ("equilibration_trajectory", "nvt_trajectory")),
            ("production", ("nve_trajectory", "production_trajectory")),
        ):
            targets = ["energy_trace_and_drift"]
            p.attempt(
                name,
                lambda name=name, aliases=aliases, targets=targets: p.frames(
                    name, e.trajectory(*aliases), parameters=model(), targets=targets
                ),
                targets,
            )
    elif task_number == 7:

        def expansion():
            table = e.table("thermal_trace", "raw_trace")
            for sid in e._artifacts["trajectories"]:
                frames = e.trajectory(f"trajectories.{sid}")
                p.frames(
                    f"energies_{sid}",
                    frames,
                    parameters=model(),
                    targets=[
                        "measured_npt_summaries_and_drift",
                        "fixed_cell_nvt_evidence",
                    ],
                )
                values = []
                for atoms in frames:
                    time = atoms.info.get(
                        "time_fs", 1000 * atoms.info.get("time_ps", 0)
                    )
                    rows = table[
                        (table.stage == sid)
                        & np.isclose(table.time_fs, time, rtol=0, atol=1e-6)
                    ]
                    if len(rows) != 1:
                        raise EvidenceError(
                            "Stress rows must align with trajectory times"
                        )
                    stress = np.array(
                        [
                            rows.iloc[0][k]
                            for k in ("sxx", "syy", "szz", "syz", "sxz", "sxy")
                        ]
                    )
                    factor = {
                        "eV/Angstrom^3": 1,
                        "GPa": 1 / 160.21766208,
                        "bar": 1e-4 / 160.21766208,
                    }[e.settings["stress_unit"]]
                    stress *= factor * (
                        1 if e.settings["stress_sign"] == "tensile_positive" else -1
                    )
                    if e.settings["stress_kind"] in {"total", "including_kinetic"}:
                        momenta = atoms.get_momenta()
                        kinetic = (
                            -np.einsum(
                                "ni,nj,n->ij", momenta, momenta, 1 / atoms.get_masses()
                            )
                            / atoms.get_volume()
                        )
                        stress -= kinetic.flat[[0, 4, 8, 5, 2, 1]]
                    elif e.settings["stress_kind"] != "potential_only":
                        raise UnsupportedEvidence("Unknown stress convention")
                    values.append({"stress": stress.tolist()})
                p.frames(
                    f"stress_{sid}",
                    frames,
                    values=values,
                    parameters=model(),
                    targets=[
                        "raw_stress_and_kinetic_pressure",
                        "local_pressure_derivatives",
                        "local_route_comparison",
                        "bulk_modulus_stability_diagnostic",
                    ],
                )

        p.attempt(
            "expansion_calculations", expansion, ["raw_stress_and_kinetic_pressure"]
        )
    elif task_number in {8, 9}:
        p.attempt(
            "regression_calculations",
            lambda: _regression_plan(p),
            ["portable_pipeline_and_dimensions"]
            if task_number == 8
            else ["final_main_only_affine_pipeline"],
        )
    elif task_number == 10:

        def heat():
            frames = e.trajectory("production_trajectory", "trajectory")
            for sid in dict.fromkeys(a.info.get("stage") for a in frames):
                p.frames(
                    f"energy_force_{sid}",
                    [a for a in frames if a.info.get("stage") == sid],
                    parameters=model(),
                    targets=[
                        "thermal_trace_from_saved_momenta_and_energy",
                        "total_energy_per_atom_means",
                        "measured_temperature_heat_capacity_fit",
                        "heat_capacity_units_and_kB_conversion",
                        "high_temperature_energy_residuals",
                        "return_minus_initial_energy",
                    ],
                )

        p.attempt("heat_capacity_calculations", heat, ["total_energy_per_atom_means"])
    return p


def _regression_plan(p):
    from .regression import predict

    e, number = p.evidence, p.task_number
    if number == 8:
        from .task_8 import _STRUCTURE_ALIASES

        data = e.json("regression_data", "datasets", "dataset_index")
        portable = e.json("portable_model", "final_model", "pipeline_parameters")
        soap = portable["descriptor"]["parameters"]
        datasets = []
        for name, aliases in _STRUCTURE_ALIASES.items():
            frames = e.trajectory(*aliases)
            lookup = {a.info["row_id"]: a for a in frames}
            ordered = [lookup[key] for key in data[name]["row_ids"]]
            datasets.append(
                (
                    name,
                    ordered,
                    np.asarray(data[name]["features"]),
                    np.asarray(data[name]["energy_eV"]),
                )
            )
        target = "aligned_labels_recorded_inputs_and_rng"
        pipeline_target = "portable_pipeline_and_dimensions"
    else:
        names = ("main", "independent_800", "independent_500", "independent_1100")
        portable = e.json("models", "pipeline")["final"]["model"]
        soap = e.settings["soap"]
        datasets = [
            (
                name,
                e.trajectory(f"runs.{name}.trajectory"),
                e.array(f"runs.{name}.descriptors"),
                e.array(f"runs.{name}.labels"),
            )
            for name in names
        ]
        target = "energy_descriptor_alignment_and_teacher_configuration"
        pipeline_target = "final_main_only_affine_pipeline"
    all_x = []
    for name, frames, features, labels in datasets:
        p.frames(
            f"teacher_{name}",
            frames,
            count=5 if number == 8 else 3,
            values=[
                {
                    "energy": float(v),
                    **(
                        {"forces": stored(a)["forces"]} if "forces" in stored(a) else {}
                    ),
                }
                for a, v in zip(frames, labels, strict=True)
            ],
            parameters=calculator_settings(e),
            targets=[target],
        )
        indices = p.sample(len(frames), 5 if number == 8 else 3, f"soap_{name}")
        p.add(
            f"soap_{name}",
            "soap",
            {
                "frames": [geometry(frames[i]) for i in indices],
                "sample_indices": indices,
            },
            {"features": np.asarray(features)[indices].tolist()},
            parameters=soap,
            targets=[target, pipeline_target],
            tolerance={"features": [1e-6, 1e-8]},
        )
        all_x.extend(np.asarray(features).tolist())
    checkpoint = e.artifact("pipeline_checkpoint", "trained_pipeline", "estimator")
    if checkpoint.suffix == ".json":
        # Portable JSON is already loaded and evaluated independently by the current scorer.
        saved = json.loads(checkpoint.read_text())
        actual = predict(all_x, saved)
        if not np.allclose(actual, predict(all_x, portable), rtol=1e-4, atol=1e-6):
            p.notes.append(
                {
                    "id": "pipeline_inference",
                    "status": "failed",
                    "detail": "Portable checkpoints disagree",
                    "targets": [pipeline_target],
                }
            )
        else:
            p.notes.append(
                {
                    "id": "pipeline_inference",
                    "status": "passed",
                    "detail": "Portable JSON pipeline predictions reproduced",
                }
            )
    else:
        p.add(
            "pipeline_inference",
            "pipeline",
            {"features": all_x},
            {"predictions": predict(all_x, portable).tolist()},
            parameters=p.checkpoint(checkpoint),
            targets=[pipeline_target],
            tolerance={"predictions": [1e-4, 1e-6]},
        )


def compare(expected, actual, tolerances):
    """Compare only evaluator-held expectations, rejecting missing and nonfinite output."""

    def visit(wanted, got, key=""):
        if isinstance(wanted, dict):
            return isinstance(got, dict) and all(
                k in got and visit(v, got[k], k) for k, v in wanted.items()
            )
        if key == "frames":
            return (
                isinstance(got, list)
                and len(wanted) == len(got)
                and all(visit(a, b) for a, b in zip(wanted, got, strict=True))
            )
        a, b = finite_array(wanted), finite_array(got)
        rtol, atol = tolerances[key]
        return a.shape == b.shape and bool(
            np.all(np.abs(a - b) <= atol + rtol * np.abs(b))
        )

    try:
        return visit(expected, actual)
    except (KeyError, ValueError, TypeError):
        return False


class ModalVerifier:
    def __init__(
        self,
        *,
        release_id=None,
        app_name=None,
        volume_name=None,
        run_id=None,
        action_id=None,
        require_provenance=False,
        transport=None,
    ):
        self.release_id, self.app_name, self.volume_name = (
            release_id,
            app_name,
            volume_name,
        )
        self.run_id, self.action_id = run_id, action_id
        self.require_provenance = require_provenance
        self.transport = transport

    def _configuration(self):
        from corral_md.modal_workspace import (
            configured_app_name,
            configured_release_id,
            configured_volume_name,
        )

        release = self.release_id or configured_release_id()
        if not release:
            raise RuntimeError(
                "No verifier release selected; deploy modal_app/release.py"
            )
        return (
            release,
            self.app_name or configured_app_name(release),
            self.volume_name or configured_volume_name(release),
        )

    def _calculate(self, plan, fingerprint):
        if self.transport:
            return self.transport(plan, fingerprint)
        import modal

        release, app, volume_name = self._configuration()
        volume = modal.Volume.from_name(volume_name)
        identifier = uuid.uuid4().hex
        request = {
            "schema": 1,
            "task_number": plan.task_number,
            "jobs": plan.jobs,
            "evidence_sha256": fingerprint,
            "challenge": plan.challenge,
        }
        with tempfile.TemporaryDirectory(prefix="corral-md-verify-") as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps(request, allow_nan=False))
            files = {"request.json": path, **plan.files}
            refs = {name: file_ref(source) for name, source in files.items()}
            with volume.batch_upload() as upload:
                for name, source in files.items():
                    upload.put_file(
                        source, f"/corral/verifications/{identifier}/input/{name}"
                    )
            response = modal.Function.from_name(app, "verify_calculations").remote(
                identifier, release, refs
            )
        if (
            response.get("verification_id") != identifier
            or response.get("release_id") != release
            or response.get("input_sha256")
            != hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()
        ):
            raise RuntimeError(
                "Verification response does not match submitted bytes/release"
            )
        return response

    def evaluate(self, evidence, task_number):
        fingerprint = evidence.fingerprint()
        challenge = secrets.token_hex(32)
        plan = build_plan(evidence, task_number, challenge)
        checks, remote = list(plan.notes), None
        if plan.jobs:
            try:
                remote = self._calculate(plan, fingerprint)
                if (
                    remote.get("evidence_sha256") != fingerprint
                    or remote.get("challenge") != challenge
                ):
                    raise RuntimeError(
                        "Verification response is bound to different evidence"
                    )
                jobs = remote.get("jobs", [])
                if len(jobs) != len(plan.jobs) or {j["id"] for j in jobs} != set(
                    plan.expected
                ):
                    raise RuntimeError(
                        "Verification response has missing or duplicate jobs"
                    )
                for job in jobs:
                    spec = plan.expected[job["id"]]
                    status = "unverified"
                    if job.get("status") == "complete":
                        status = (
                            "passed"
                            if compare(
                                spec["values"], job.get("value"), spec["tolerance"]
                            )
                            else "failed"
                        )
                    checks.append(
                        {
                            "id": job["id"],
                            "status": status,
                            "targets": spec["targets"],
                            "sample_indices": spec["indices"],
                            "detail": job.get(
                                "detail",
                                "Compared with independently calculated values",
                            ),
                        }
                    )
            except Exception as exc:
                checks.extend(
                    {
                        "id": job["id"],
                        "status": "unverified",
                        "targets": plan.expected[job["id"]]["targets"],
                        "detail": f"Verification unavailable: {type(exc).__name__}: {exc}",
                    }
                    for job in plan.jobs
                )
        if task_number == 10:
            checks.append(self.provenance(evidence))
        if evidence.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during independent verification; retry with frozen evidence"
            )
        backend = {
            k: remote[k]
            for k in ("verification_id", "release_id", "model_sha256")
            if remote and k in remote
        }
        if remote and "cache" in remote:
            backend["cache"] = remote["cache"]
        return {
            "mode": "modal",
            "evidence_sha256": fingerprint,
            "challenge": challenge,
            "checks": checks,
            "backend": backend,
        }

    def provenance(self, evidence):
        targets = [
            "recorded_nvt_model_and_initialization",
            "eight_stage_cycle_and_cumulative_time",
            "boundary_position_momentum_continuity",
            "equilibration_thermal_diagnostics",
        ]
        record = {
            "id": "trusted_md_execution",
            "status": "unverified",
            "targets": targets if self.require_provenance else [],
        }
        try:
            record.update(_check_provenance(self, evidence))
            if record["status"] == "failed":
                record["targets"] = targets
        except Exception as exc:
            record["detail"] = f"Execution verification unavailable: {exc}"
        return record


def _check_provenance(verifier, e):
    import modal

    run_id = verifier.run_id or e.manifest.get("run_id")
    action_id = verifier.action_id or e.manifest.get("action_id")
    if not run_id or not action_id:
        return {
            "status": "unverified",
            "detail": "No controlled-run identifiers; arbitrary scripts cannot attest their own execution",
        }
    # The endpoint is selected by the evaluator, never a submission-supplied app or URL.
    release, app, _ = verifier._configuration()
    if verifier.run_id and e.manifest.get("run_id", run_id) != run_id:
        return {"status": "failed", "detail": "Submission names another execution"}
    roles = {
        "initial_state": "initial.traj",
        "boundary_states": "boundaries.traj",
        "production_trajectory": "production.traj",
        "thermal_trace": "thermal_trace.csv",
    }
    prefix = f"output/verified_md/{action_id}/"
    refs = {
        prefix + name: file_ref(
            e.artifact(
                role, *(("trajectory",) if role == "production_trajectory" else ())
            )
        )
        for role, name in roles.items()
    }
    response = modal.Function.from_name(app, "verify_md_provenance").remote(
        run_id, action_id, release, refs
    )
    if response.get("status") != "passed":
        return {
            "status": response.get("status", "unverified"),
            "detail": response.get("detail", "No backend record"),
        }
    saved = response["record"]
    if (
        saved["run_id"] != run_id
        or saved["action_id"] != action_id
        or saved["release_id"] != release
    ):
        raise RuntimeError("Backend execution identity differs")
    md, config = e.settings["md"], saved["config"]
    if (
        md["thermostat"].lower() != "langevin"
        or md["timestep_fs"] != config["timestep_fs"]
        or md["random_seed"] != config["random_seed"]
        or md["velocity_initializations"] != 1
        or md["manual_velocity_resets"] != 0
        or md["ensemble"] != "NVT"
        or md["initial_temperature_K"] != 300
    ):
        return {
            "status": "failed",
            "detail": "Declared MD settings contradict backend execution",
        }
    if "friction_fs" in md and md["friction_fs"] != config["friction_fs"]:
        return {
            "status": "failed",
            "detail": "Declared friction contradicts backend execution",
        }
    parameters = calculator_settings(e)
    if (
        parameters["default_dtype"] != config["default_dtype"]
        or parameters["dispersion"]
    ):
        return {
            "status": "failed",
            "detail": "Declared calculator settings contradict backend execution",
        }
    if md.get("temperature_dof", 324) != 324:
        return {
            "status": "failed",
            "detail": "Declared temperature degrees of freedom contradict backend execution",
        }
    if len(e.settings["stages"]) != len(saved["stages"]):
        return {
            "status": "failed",
            "detail": "Stage counts contradict backend execution",
        }
    previous = saved["initial_state_sha256"]
    for declared, observed, parameters in zip(
        e.settings["stages"], saved["stages"], config["stages"], strict=True
    ):
        pairs = [
            (declared["start_time_fs"], observed["start_step"] * config["timestep_fs"]),
            (declared["end_time_fs"], observed["end_step"] * config["timestep_fs"]),
            (
                declared["production_start_time_fs"],
                observed["production_start_step"] * config["timestep_fs"],
            ),
            (
                declared["production_end_time_fs"],
                observed["end_step"] * config["timestep_fs"],
            ),
        ]
        if (
            declared["id"] != observed["id"]
            or declared["target_temperature_K"] != parameters["target_temperature_K"]
            or any(a != b for a, b in pairs)
            or observed["start_state_sha256"] != previous
            or observed["equilibration_end_state_sha256"]
            != observed["production_start_state_sha256"]
        ):
            return {
                "status": "failed",
                "detail": "Stage timing or state continuity contradicts backend execution",
            }
        previous = observed["end_state_sha256"]
    return {
        "status": "passed",
        "detail": "Model, actual step counts, boundary continuity and artifact hashes verified",
        "run_id": run_id,
        "action_id": action_id,
        "release_id": release,
        "model_sha256": saved["model_sha256"],
        "total_steps": saved["total_steps"],
    }


def apply_verification(rubric, report):
    """Gate affected existing checks without changing the 100-point denominator."""
    by_name = {check["name"]: check for check in rubric.checks}
    for verification in report["checks"]:
        if (
            verification["id"] == "trusted_md_execution"
            and verification["status"] == "passed"
            and "execution_provenance" in by_name
        ):
            by_name["execution_provenance"].update(
                status="passed", detail=verification["detail"]
            )
        if verification["status"] not in {"failed", "unverified"}:
            continue
        for name in verification.get("targets", []):
            check = by_name.get(name)
            if check is None or check["status"] == "failed":
                continue
            check["status"] = verification["status"]
            check["earned"] = 0.0
            check["detail"] = (
                f"Independent check {verification['id']}: {verification['detail']}"
            )
