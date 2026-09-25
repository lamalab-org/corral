"""Controller-side isolation and durable records for independent verification."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import modal

CACHE_SCHEMA = 1


class ReferenceUnavailable(RuntimeError):
    """The deployed release has no compatible persistent reference."""


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _job_identity(job, release_id):
    """Bind a reusable result to every scientific input except its report label."""
    normalized = {
        key: value for key, value in job.items() if key not in {"id", "sample_indices"}
    }
    parameters = normalized.get("parameters")
    if isinstance(parameters, dict) and "checkpoint" in parameters:
        normalized["parameters"] = {
            **parameters,
            # The digest is the checkpoint identity; its upload basename is not.
            "checkpoint": "sha256:" + parameters["checkpoint_sha256"],
        }
    return hashlib.sha256(
        _json_bytes(
            {
                "schema": CACHE_SCHEMA,
                "release_id": release_id,
                "job": normalized,
            }
        )
    ).hexdigest()


BOOTSTRAP = """
import ctypes, os, sys
os.chown('/output', 10001, 10001)
os.makedirs('/tmp/corral-verifier', exist_ok=True)
os.chown('/tmp/corral-verifier', 10001, 10001)
os.environ.update(HOME='/tmp/corral-verifier', XDG_CACHE_HOME='/tmp/corral-verifier/.cache')
os.setgroups([])
os.setgid(10001)
os.setuid(10001)
if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
    raise RuntimeError('Cannot disable privilege escalation')
os.chdir('/output')
os.execv(sys.executable, [sys.executable, '-I', *sys.argv[1:]])
"""


class VerificationRuntime:
    def __init__(self, worker):
        self.worker = worker

    def sandbox(
        self, payload, files=None, *, program="verification_worker.py", gpu=False
    ):
        """Expose only geometry/config, isolated checkpoints and read-only assets."""
        w = self.worker
        if program not in {"verification_worker.py", "trusted_md.py"}:
            raise ValueError("Unknown trusted entry point")
        sw_only = (
            program == "verification_worker.py"
            and not files
            and bool(payload.get("jobs"))
            and all(
                job.get("operation") == "sw" and not job.get("parameters")
                for job in payload["jobs"]
            )
        )
        restart_only = (
            program == "verification_worker.py"
            and bool(payload.get("jobs"))
            and all(
                job.get("operation") == "lammps_restart"
                and isinstance(job.get("parameters"), dict)
                and set(job["parameters"])
                == {"checkpoint", "checkpoint_sha256"}
                for job in payload["jobs"]
            )
        )
        # Open MPI's local PMIx listener cannot start with all sockets disabled.
        # Only fixed, data-only LAMMPS operations get loopback access. They
        # remain isolated from submitted executable checkpoints and scripts.
        network = (
            {"cidr_allowlist": ["127.0.0.0/8"]}
            if sw_only or restart_only
            else {"block_network": True}
        )
        with modal.Volume.ephemeral() as inputs, modal.Volume.ephemeral() as outputs:
            with inputs.batch_upload() as upload:
                upload.put_file(
                    BytesIO(json.dumps(payload, allow_nan=False).encode()),
                    "/request.json",
                )
                for name, path in (files or {}).items():
                    if Path(name).name != name:
                        raise ValueError("Checkpoint names must be flat")
                    upload.put_file(path, "/" + name)
            sandbox = modal.Sandbox.create(
                "python",
                "-I",
                "-c",
                BOOTSTRAP,
                f"/opt/corral-md/{program}",
                "/evidence/request.json",
                "/output" if program == "trusted_md.py" else "/output/result.json",
                app=w.app,
                image=w.lammps_image,
                gpu="A100" if gpu else None,
                cpu=w.CPUS,
                memory=16384 if gpu else 5120,
                timeout=7200,
                **network,
                workdir="/output",
                secrets=[],
                volumes={
                    "/evidence": inputs.read_only(),
                    "/output": outputs,
                    "/workspace/models": w.volume_models.read_only(),
                    "/workspace/potentials": w.volume_potential.read_only(),
                },
            )
            try:
                stdout, stderr = sandbox.stdout.read(), sandbox.stderr.read()
                sandbox.wait()
            finally:
                sandbox.terminate(wait=True)
            if sandbox.returncode:
                raise RuntimeError(
                    f"Verification sandbox failed ({sandbox.returncode}): {stderr[-3000:]}"
                )
            with tempfile.TemporaryDirectory(prefix="md-verified-output-") as temporary:
                root = Path(temporary) / "workspace"
                root.mkdir()
                w._collect_workspace(outputs, root)
                yield_result = {
                    "sandbox_id": sandbox.object_id,
                    "network": "loopback_only" if sw_only or restart_only else "disabled",
                    "files": w._manifest(root),
                    "stdout": stdout[-2000:],
                    "stderr": stderr[-2000:],
                }
                # The consumer copies verified files before the temporary tree disappears.
                yield root, yield_result

    def _cache_file(self, job, release_id):
        identity = _job_identity(job, release_id)
        path = (
            self.worker.RUNS.parent
            / "verification-cache"
            / release_id
            / identity[:2]
            / f"{identity}.json"
        )
        return identity, path

    def _cached(self, job, release_id):
        identity, path = self._cache_file(job, release_id)
        if not path.is_file():
            return None
        entry = self.worker._load(path)
        result = entry.get("result")
        if (
            entry.get("schema") != CACHE_SCHEMA
            or entry.get("release_id") != release_id
            or entry.get("job_sha256") != identity
            or not isinstance(result, dict)
            or result.get("status") != "complete"
            or entry.get("result_sha256")
            != hashlib.sha256(_json_bytes(result)).hexdigest()
        ):
            raise RuntimeError(f"Invalid verification cache entry: {path}")
        return {
            "id": job["id"],
            **result,
            "source": "cache",
            "detail": "Compared with a cached independent calculation",
        }

    def _store_cached(self, job, release_id, result):
        if result.get("status") != "complete":
            return
        identity, path = self._cache_file(job, release_id)
        stored = {key: value for key, value in result.items() if key != "id"}
        self.worker._write(
            path,
            {
                "schema": CACHE_SCHEMA,
                "release_id": release_id,
                "job_sha256": identity,
                "result_sha256": hashlib.sha256(_json_bytes(stored)).hexdigest(),
                "result": stored,
            },
        )

    def _checkpoint_files(self, job, frozen, expected_files):
        parameters = job.get("parameters", {})
        if "checkpoint" not in parameters:
            return {}
        name = parameters["checkpoint"]
        path = frozen / self.worker._relative(name)
        if (
            path.parent != frozen
            or name not in expected_files
            or expected_files[name]["sha256"] != parameters.get("checkpoint_sha256")
        ):
            raise ValueError("Checkpoint digest/path mismatch")
        return {name: path}

    def _resolve_reference(self, job, release_id):
        from ground_truth import (  # noqa: PLC0415
            GroundTruthUnavailable,
            resolve_reference,
        )

        path = self.worker.RUNS.parent / "releases" / release_id / "ground_truth.json"
        if not path.is_file():
            raise ReferenceUnavailable("Release ground truth has not been published")
        document = self.worker._load(path)
        try:
            return resolve_reference(job, document, release_id, self.worker.ASSETS)
        except GroundTruthUnavailable as exc:
            raise ReferenceUnavailable(str(exc)) from exc

    @staticmethod
    def _reference_fallback(job):
        fallback = job.get("fallback_operation")
        if fallback not in {"mace", "sw", "soap", "pipeline"}:
            raise ValueError("Reference job has no supported live-calculation fallback")
        return {
            key: value
            for key, value in {**job, "operation": fallback}.items()
            if key not in {"reference", "fallback_operation"}
        }

    def calculations(self, identifier, release_id, expected_files):
        w = self.worker
        w.volume_sim.reload()
        w._release_base(release_id)
        directory = w.RUNS.parent / "verifications" / w._id(identifier)
        source = directory / "input"
        identity = hashlib.sha256(
            json.dumps(expected_files, sort_keys=True).encode()
        ).hexdigest()
        result_file = directory / "result.json"
        if result_file.exists():
            previous = w._load(result_file)
            if (
                previous.get("input_sha256") != identity
                or previous.get("release_id") != release_id
            ):
                raise ValueError("Verification ID was reused for different evidence")
            return previous
        if w._manifest(source) != expected_files:
            raise ValueError("Verification inputs do not match submitted hashes")
        with tempfile.TemporaryDirectory(prefix="md-verification-") as temporary:
            frozen = Path(temporary) / "input"
            shutil.copytree(source, frozen)
            if w._manifest(frozen) != expected_files:
                raise ValueError("Verification inputs changed during snapshot")
            request = w._load(frozen / "request.json")
            jobs = request.get("jobs")
            if not isinstance(jobs, list) or not 1 <= len(jobs) <= 200:
                raise ValueError("Invalid verification job list")
            ids = [job["id"] for job in jobs]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate verification job ID")
            prepared, results, cache_hits = [], {}, 0
            ground_truth_jobs = []
            for original in jobs:
                job = original
                if job.get("operation") == "reference":
                    try:
                        value = self._resolve_reference(job, release_id)
                    except ReferenceUnavailable:
                        job = self._reference_fallback(job)
                    else:
                        results[job["id"]] = {
                            "id": job["id"],
                            "status": "complete",
                            "value": value,
                            "source": "ground_truth",
                            "detail": "Compared with release-owned ground truth",
                        }
                        ground_truth_jobs.append(job["id"])
                        continue
                self._checkpoint_files(job, frozen, expected_files)
                cached = self._cached(job, release_id)
                if cached is not None:
                    results[job["id"]] = cached
                    cache_hits += 1
                else:
                    prepared.append(job)

            trusted, sw, isolated = [], [], []
            for job in prepared:
                if (
                    job["operation"] in {"pipeline", "lammps_restart"}
                    or job.get("parameters", {}).get("model") == "submitted"
                ):
                    isolated.append([job])
                elif job["operation"] == "sw":
                    sw.append(job)
                else:
                    trusted.append(job)
            batches = ([trusted] if trusted else []) + ([sw] if sw else []) + isolated
            sandboxes = []
            for batch in batches:
                files = {}
                for job in batch:
                    files.update(self._checkpoint_files(job, frozen, expected_files))
                gpu = any(job["operation"] == "mace" for job in batch)
                for output, metadata in self.sandbox({"jobs": batch}, files, gpu=gpu):
                    data = w._load(output / "result.json")
                    returned = data.get("jobs", [])
                    if [job.get("id") for job in returned] != [
                        job["id"] for job in batch
                    ]:
                        raise RuntimeError("Verification response job identity differs")
                    for job, result in zip(batch, returned, strict=True):
                        results[job["id"]] = result
                        self._store_cached(job, release_id, result)
                    sandboxes.append(metadata)
            result = {
                "schema": 1,
                "verification_id": identifier,
                "release_id": release_id,
                "input_sha256": identity,
                "evidence_sha256": request["evidence_sha256"],
                "challenge": request["challenge"],
                "jobs": [results[job["id"]] for job in jobs],
                "sandboxes": sandboxes,
                "cache": {
                    "hits": cache_hits,
                    "misses": len(prepared),
                    "ground_truth": ground_truth_jobs,
                },
                "model_sha256": {k: v["sha256"] for k, v in w.ASSETS["models"].items()},
            }
            w._write(result_file, result)
            w.volume_sim.commit()
            return result

    def dynamics(self, config, working, *, run_id, action_id):
        from trusted_md import validate_config

        w = self.worker
        config = validate_config(config)
        prefix = Path("output") / "verified_md" / w._id(action_id)
        destination = working / prefix
        if destination.exists():
            raise ValueError("Verified output directory already exists")
        for output, metadata in self.sandbox(config, program="trusted_md.py", gpu=True):
            observed = w._load(output / "observed.json")
            if (
                observed.get("config") != config
                or observed.get("runner") != "corral.task10.langevin.v1"
            ):
                raise RuntimeError("Controlled MD record differs from its request")
            record = {
                **observed,
                "run_id": run_id,
                "action_id": action_id,
                "release_id": w.RELEASE_ID,
                "sandbox_id": metadata["sandbox_id"],
                "model_sha256": w.ASSETS["models"]["teacher.model"]["sha256"],
                "files": {
                    str(prefix / name): ref for name, ref in metadata["files"].items()
                },
                "prefix": prefix.as_posix(),
            }
            shutil.copytree(output, destination)
            return record

    def provenance(self, run_id, action_id, release_id, expected_files):
        w = self.worker
        w.volume_sim.reload()
        w._release_base(release_id)
        run = w.RUNS / w._id(run_id)
        action = w._id(action_id)
        path = run / "actions" / f"{action}.json"
        if not path.is_file():
            return {"status": "unverified", "detail": "No completed backend action"}
        result = w._load(path)
        record = result.get("trusted_md")
        if result.get("kind") != "verified_md" or not isinstance(record, dict):
            return {
                "status": "unverified",
                "detail": "Action ran agent code; no controlled MD record exists",
            }
        if (
            record.get("run_id") != run_id
            or record.get("action_id") != action_id
            or record.get("release_id") != release_id
            or result.get("release_id") != release_id
        ):
            raise RuntimeError("Backend record identity mismatch")
        working = run / "attempts" / action / w._id(result["attempt_id"]) / "workspace"
        files = w._manifest(working)
        if files != result["files"]:
            raise RuntimeError("Committed backend outputs have changed")
        if not expected_files or any(
            record["files"].get(name) != ref for name, ref in expected_files.items()
        ):
            return {
                "status": "failed",
                "detail": "Submitted file hashes differ from controlled MD outputs",
            }
        return {"status": "passed", "record": record}
