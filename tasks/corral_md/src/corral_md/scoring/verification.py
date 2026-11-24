"""Evaluator-owned independent Modal recalculation.

Only geometry, declared calculator parameters and isolated checkpoint bytes go
to calculation sandboxes. Expected answers remain with this evaluator. Sampling
is selected after the evidence is frozen and is recorded in the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from .common import EvidenceError, finite_array

DEFAULT_MAX_PARALLEL_CALCULATIONS = 4
MAX_PARALLEL_CALCULATIONS = 25


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
        depends_on=(),
    ):
        if name in self.expected:
            raise ValueError("Duplicate verification check")
        dependencies = list(depends_on)
        if any(dependency not in self.expected for dependency in dependencies):
            raise ValueError("Verification dependencies must name earlier jobs")
        job = {
            "id": name,
            "operation": operation,
            **inputs,
            "parameters": parameters or {},
        }
        if dependencies:
            job["depends_on"] = dependencies
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
        fallback_operation=None,
        chunk_size=None,
        depends_on=(),
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
        if chunk_size is None:
            chunk_size = len(indices)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Verification chunk_size must be a positive integer")
        chunks = [
            (indices[start : start + chunk_size], expected[start : start + chunk_size])
            for start in range(0, len(indices), chunk_size)
        ]
        for chunk_number, (chunk_indices, chunk_expected) in enumerate(chunks):
            chunk_name = name if len(chunks) == 1 else f"{name}_{chunk_number:03d}"
            inputs = {
                "frames": [geometry(frames[i]) for i in chunk_indices],
                "properties": properties,
                "sample_indices": chunk_indices,
            }
            if reference is not None:
                inputs["reference"] = reference
            if fallback_operation is not None:
                inputs["fallback_operation"] = fallback_operation
            self.add(
                chunk_name,
                operation,
                inputs,
                {"frames": chunk_expected},
                parameters=parameters,
                targets=targets,
                tolerance=tolerance,
                depends_on=depends_on,
            )

    def subset(self, jobs):
        """Return an execution view containing only the selected jobs."""
        selected = list(jobs)
        plan = object.__new__(Plan)
        plan.evidence = self.evidence
        plan.task_number = self.task_number
        plan.challenge = self.challenge
        plan.jobs = selected
        plan.expected = {job["id"]: self.expected[job["id"]] for job in selected}
        plan.files = self.files
        plan.notes = []
        return plan

    def checkpoint(self, path):
        path = Path(path)
        name = "checkpoint-" + file_ref(path)["sha256"] + path.suffix
        self.files[name] = path
        return {"checkpoint": name, "checkpoint_sha256": file_ref(path)["sha256"]}


def build_plan(e, task_number, challenge):
    if task_number in (3, 5, 6, 7, 8, 9, 10):
        raise ValueError(
            f"Task {task_number} uses its own independent calculation plan"
        )
    p = Plan(e, task_number, challenge)

    if task_number == 4:
        from corral_md.scoring.task_4 import build_plan as palladium_plan

        return palladium_plan(e, challenge)
    return p


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
        volume_name=None,
        transport=None,
        max_parallel_calculations=None,
    ):
        self.release_id, self.volume_name = release_id, volume_name
        self.transport = transport
        configured_parallelism = (
            os.getenv(
                "CORRAL_MD_MAX_PARALLEL_CALCULATIONS",
                str(DEFAULT_MAX_PARALLEL_CALCULATIONS),
            )
            if max_parallel_calculations is None
            else max_parallel_calculations
        )
        try:
            self.max_parallel_calculations = int(configured_parallelism)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_parallel_calculations must be an integer from 1 to "
                f"{MAX_PARALLEL_CALCULATIONS}"
            ) from exc
        if (
            isinstance(configured_parallelism, bool)
            or not 1 <= self.max_parallel_calculations <= MAX_PARALLEL_CALCULATIONS
        ):
            raise ValueError(
                "max_parallel_calculations must be an integer from 1 to "
                f"{MAX_PARALLEL_CALCULATIONS}"
            )

    def _configuration(self):
        from corral_md.modal_workspace import (
            configured_app_name,
            configured_runtime,
        )

        if self.release_id and self.volume_name:
            return self.release_id, configured_app_name(), self.volume_name
        current_release, current_volume = configured_runtime()
        release = self.release_id or current_release
        if release != current_release:
            raise RuntimeError(
                "The requested verifier release is not served by SimAgent"
            )
        return (
            release,
            configured_app_name(),
            self.volume_name or current_volume,
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
                    # Restored artifacts may be 0600; the sandbox reads as UID 10001.
                    upload.put_file(
                        source,
                        f"/corral/verifications/{identifier}/input/{name}",
                        mode=0o644,
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

    def _evaluate_job(self, plan, job, fingerprint):
        spec = plan.expected[job["id"]]
        check = {
            "id": job["id"],
            "status": "unverified",
            "targets": spec["targets"],
            "sample_indices": spec["indices"],
        }
        try:
            remote = self._calculate(plan.subset([job]), fingerprint)
            if (
                remote.get("evidence_sha256") != fingerprint
                or remote.get("challenge") != plan.challenge
            ):
                raise RuntimeError(
                    "Verification response is bound to different evidence"
                )
            jobs = remote.get("jobs", [])
            if len(jobs) != 1 or jobs[0].get("id") != job["id"]:
                raise RuntimeError(
                    "Verification response has missing or duplicate jobs"
                )
            result = jobs[0]
            if result.get("status") == "complete":
                actual = result.get("value")
                if "pressure_kinetic_GPa" in spec:
                    from ase import units

                    actual = {
                        "frames": [
                            {
                                "pressure_GPa": -float(
                                    np.mean(finite_array(row["stress"], shape=(6,))[:3])
                                )
                                / units.GPa
                                + kinetic
                            }
                            for row, kinetic in zip(
                                actual["frames"],
                                spec["pressure_kinetic_GPa"],
                                strict=True,
                            )
                        ]
                    }
                check["status"] = (
                    "passed"
                    if compare(spec["values"], actual, spec["tolerance"])
                    else "failed"
                )
            check["detail"] = result.get(
                "detail", "Compared with independently calculated values"
            )
            return check, remote
        except Exception as exc:
            check["detail"] = f"Verification unavailable: {type(exc).__name__}: {exc}"
            return check, None

    @staticmethod
    def _backend(remotes):
        backend = {}
        verification_ids = [
            remote["verification_id"]
            for remote in remotes
            if "verification_id" in remote
        ]
        if len(verification_ids) == 1:
            backend["verification_id"] = verification_ids[0]
        elif verification_ids:
            backend["verification_ids"] = verification_ids
        for key in ("release_id", "model_sha256"):
            values = [remote[key] for remote in remotes if key in remote]
            if values and all(value == values[0] for value in values):
                backend[key] = values[0]
        caches = [remote["cache"] for remote in remotes if "cache" in remote]
        if caches:
            backend["cache"] = {
                "hits": sum(cache.get("hits", 0) for cache in caches),
                "misses": sum(cache.get("misses", 0) for cache in caches),
                "ground_truth": [
                    name for cache in caches for name in cache.get("ground_truth", [])
                ],
            }
        return backend

    def _evaluate_plan(self, plan, fingerprint):
        order = {job["id"]: index for index, job in enumerate(plan.jobs)}
        completed = {}
        remotes = {}
        waiting = list(plan.jobs)
        aborted = any(note.get("status") == "failed" for note in plan.notes)

        def skipped(job, detail):
            spec = plan.expected[job["id"]]
            return {
                "id": job["id"],
                "status": "skipped",
                "targets": spec["targets"],
                "sample_indices": spec["indices"],
                "detail": detail,
            }

        with ThreadPoolExecutor(
            max_workers=self.max_parallel_calculations,
            thread_name_prefix="corral-md-verifier",
        ) as executor:
            running = {}
            while waiting or running:
                if not aborted:
                    ready = []
                    for job in list(waiting):
                        dependencies = job.get("depends_on", [])
                        if not all(name in completed for name in dependencies):
                            continue
                        waiting.remove(job)
                        if any(
                            completed[name]["status"] != "passed"
                            for name in dependencies
                        ):
                            completed[job["id"]] = skipped(
                                job, "Dependency did not pass independent verification."
                            )
                        else:
                            ready.append(job)
                    capacity = self.max_parallel_calculations - len(running)
                    for job in ready[:capacity]:
                        future = executor.submit(
                            self._evaluate_job, plan, job, fingerprint
                        )
                        running[future] = job
                    waiting[:0] = ready[capacity:]

                if not running:
                    if aborted:
                        break
                    if waiting:
                        raise RuntimeError(
                            "Verification jobs contain an unresolved dependency cycle"
                        )
                    break

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: order[running[item]["id"]]):
                    job = running.pop(future)
                    check, remote = future.result()
                    completed[job["id"]] = check
                    if remote is not None:
                        remotes[job["id"]] = remote
                    if check["status"] != "passed":
                        aborted = True

        if waiting:
            reason = "Not started after an earlier independent check did not pass."
            for job in waiting:
                completed[job["id"]] = skipped(job, reason)
        checks = list(plan.notes) + [
            completed[job["id"]] for job in plan.jobs if job["id"] in completed
        ]
        return (
            checks,
            [remotes[job["id"]] for job in plan.jobs if job["id"] in remotes],
            aborted,
        )

    def evaluate(self, evidence, task_number):
        fingerprint = evidence.fingerprint()
        challenge = secrets.token_hex(32)
        plan = build_plan(evidence, task_number, challenge)
        checks, remotes, aborted = self._evaluate_plan(plan, fingerprint)
        if evidence.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during independent verification; retry with frozen evidence"
            )
        return {
            "mode": "modal",
            "evidence_sha256": fingerprint,
            "challenge": challenge,
            "checks": checks,
            "aborted": aborted,
            "max_parallel_calculations": self.max_parallel_calculations,
            "backend": self._backend(remotes),
        }


def apply_verification(rubric, report):
    """Gate affected checks, including task-specific requirements when offline."""
    by_name = {check["name"]: check for check in rubric.checks}
    independent = by_name.get("independent_model_calculation")
    if independent is not None:
        relevant = [
            check
            for check in (report or {}).get("checks", [])
            if "independent_model_calculation" in check.get("targets", [])
        ]
        if any(check.get("status") == "failed" for check in relevant):
            independent.update(
                status="failed",
                earned=0.0,
                detail="An independent model energy or force calculation disagreed with the saved evidence.",
            )
        elif relevant and all(check.get("status") == "passed" for check in relevant):
            independent.update(
                status="passed",
                earned=independent["points"],
                detail="Independent model calculations matched sampled saved energies and forces.",
            )
        else:
            independent.update(
                status="unverified",
                earned=0.0,
                detail="Required independent model calculation is unavailable or incomplete.",
            )
    for verification in report["checks"] if report is not None else ():
        if verification["status"] not in {"failed", "unverified"}:
            continue
        for name in verification.get("targets", []):
            check = by_name.get(name)
            if check is None or check["status"] == "failed":
                continue
            if name == "independent_model_calculation":
                continue
            check["status"] = verification["status"]
            check["earned"] = 0.0
            check["detail"] = (
                f"Independent check {verification['id']}: {verification['detail']}"
            )
