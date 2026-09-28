"""Live release checks for calculators, isolated inference and MD provenance."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import uuid
from pathlib import Path

import modal
import numpy as np
from ase.build import bulk
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ref(path):
    payload = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def frame(atoms):
    return {
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def smoke_verification(app_name, release_id, volume, model_directory):
    from ase import Atoms

    token = uuid.uuid4().hex
    verification_id = "smoke-" + token
    run_id, action_id = "smoke-md-" + token, "smoke-action-" + token
    remote = f"/corral/verifications/{verification_id}"
    with tempfile.TemporaryDirectory(prefix="corral-verification-smoke-") as temporary:
        root = Path(temporary)
        x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(
            x, [1.0, 2.0, 4.0, 6.0]
        )
        checkpoint = root / "pipeline.pkl"
        checkpoint.write_bytes(pickle.dumps(pipeline))
        student = Path(model_directory) / "student.model"
        silicon = frame(bulk("Si", "diamond", a=5.43, cubic=True))
        aluminum = frame(bulk("Al", "fcc", a=4.05, cubic=True))
        primitive = bulk("Si", "diamond", a=5.43)
        displaced = []
        for sign in (1, -1):
            atoms = primitive.copy()
            atoms.positions[0, 0] += sign * 0.001
            displaced.append(frame(atoms))
        jobs = [
            {"id": "sw", "operation": "sw", "frames": [silicon]},
            {
                "id": "hessian",
                "operation": "mace",
                "frames": [frame(primitive)],
                "properties": ["hessian"],
            },
            {
                "id": "hessian_fd",
                "operation": "mace",
                "frames": displaced,
                "properties": ["forces"],
            },
            {
                "id": "mace",
                "operation": "mace",
                "frames": [aluminum],
                "properties": ["energy", "forces", "stress"],
            },
            {
                "id": "dispersion",
                "operation": "mace",
                "frames": [frame(Atoms("Ag2", positions=[[0, 0, 0], [2.5, 0, 0]]))],
                "parameters": {"model": "teacher.model", "dispersion": True},
            },
            {
                "id": "soap",
                "operation": "soap",
                "frames": [silicon],
                "parameters": {
                    "species": ["Si"],
                    "r_cut": 5.0,
                    "n_max": 6,
                    "l_max": 6,
                    "periodic": True,
                    "average": "inner",
                },
            },
            {
                "id": "pipeline",
                "operation": "pipeline",
                "features": x.tolist(),
                "parameters": {
                    "checkpoint": "pipeline.pkl",
                    "checkpoint_sha256": ref(checkpoint)["sha256"],
                },
            },
            {
                "id": "submitted_mace",
                "operation": "mace",
                "frames": [aluminum],
                "parameters": {
                    "model": "submitted",
                    "checkpoint": "student.model",
                    "checkpoint_sha256": ref(student)["sha256"],
                },
            },
        ]
        request = root / "request.json"
        request.write_text(
            json.dumps({"jobs": jobs, "challenge": token, "evidence_sha256": token})
        )
        files = {
            "request.json": request,
            "pipeline.pkl": checkpoint,
            "student.model": student,
        }
        with volume.batch_upload() as upload:
            for name, source in files.items():
                upload.put_file(source, remote + "/input/" + name)
        worker = modal.Function.from_name(app_name, "verify_calculations")
        result = worker.remote(
            verification_id,
            release_id,
            {name: ref(path) for name, path in files.items()},
        )
        values = {job["id"]: job for job in result["jobs"]}
        print(
            f"Verification {verification_id}: "
            + str({name: job["status"] for name, job in values.items()}),
            flush=True,
        )
        for name, job in values.items():
            if job["status"] != "complete":
                raise RuntimeError(f"Verification smoke {name}: {job}")
        assert np.asarray(values["soap"]["value"]["features"]).shape == (1, 147)
        hessian = np.asarray(values["hessian"]["value"]["frames"][0]["hessian"])
        forces = np.asarray(
            [row["forces"] for row in values["hessian_fd"]["value"]["frames"]]
        )
        numerical = -(forces[0] - forces[1]).ravel() / 0.002
        assert hessian.shape == (6, 6)
        assert np.allclose(hessian, hessian.T, rtol=1e-4, atol=1e-6)
        assert np.allclose(hessian[:, 0], numerical, rtol=0.01, atol=0.02)
        assert np.allclose(
            values["pipeline"]["value"]["predictions"],
            pipeline.predict(x),
            rtol=1e-4,
            atol=1e-6,
        )
        for name in ("sw", "mace", "dispersion", "submitted_mace"):
            assert np.isfinite(values[name]["value"]["frames"][0]["energy"])
            assert np.isfinite(values[name]["value"]["frames"][0]["forces"]).all()
        # This short run tests real execution/recording, not physical equilibration.
        config = {
            "random_seed": 42,
            "stages": [
                {
                    "id": f"stage_{i}",
                    "target_temperature_K": t,
                    "equilibration_steps": 2,
                    "production_steps": 2,
                    "sample_interval": 1,
                    "equilibration_interval": 1,
                }
                for i, t in enumerate((300, 400, 500, 600, 700, 800, 900, 300))
            ],
        }
        path = root / "config.json"
        path.write_text(json.dumps(config))
        state = modal.Function.from_name(app_name, "prepare_workspace").remote(
            run_id, release_id
        )
        with volume.batch_upload() as upload:
            upload.put_file(path, f"/corral/runs/{run_id}/workspace/config.json")
        execution = modal.Function.from_name(app_name, "run_verified_md").remote(
            run_id,
            action_id,
            "config.json",
            "/workspace",
            [],
            release_id,
            state["head"],
            {"config.json": ref(path)},
        )
        provenance = modal.Function.from_name(app_name, "verify_md_provenance")
        refs = execution["trusted_md"]["files"]
        assert execution["trusted_md"]["total_steps"] == 32
        assert (
            provenance.remote(run_id, action_id, release_id, refs)["status"] == "passed"
        )
        altered = json.loads(json.dumps(refs))
        altered[next(iter(altered))]["sha256"] = "0" * 64
        assert (
            provenance.remote(run_id, action_id, release_id, altered)["status"]
            == "failed"
        )
        # Retain the smoke records to make release validation auditable.
        return {
            "verification_id": verification_id,
            "run_id": run_id,
            "action_id": action_id,
            "checks": list(values),
            "controlled_steps": 32,
        }
