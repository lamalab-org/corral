"""Run the benchmark's independent calculation worker in local Docker."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from corral_md.local_runtime import assets_directory
from corral_md.resources import asset_descriptors, asset_relative_paths

from .level1_trusted import Level1ModalVerifier
from .verification import ModalVerifier


class LocalCalculationTransport:
    """Give the trusted verifier only its request, pinned assets, and checkpoints."""

    def __init__(self, assets_dir: str | Path | None = None) -> None:
        self.assets = assets_directory(assets_dir)
        self.image = os.getenv(
            "CORRAL_MD_VERIFIER_IMAGE", "corral-md-local:ai-scientist-langfuse"
        )
        configured_gpus = os.getenv("CORRAL_MD_VERIFIER_GPUS")
        if configured_gpus is None:
            self.gpus: tuple[str, ...] | None = None
        else:
            devices = configured_gpus.split(",")
            if (
                not devices
                or any(
                    re.fullmatch(r"(?:0|[1-9][0-9]*)", device) is None
                    for device in devices
                )
                or len(devices) != len(set(devices))
            ):
                raise ValueError(
                    "CORRAL_MD_VERIFIER_GPUS must contain unique GPU indices"
                )
            self.gpus = tuple(devices)

    def __call__(self, plan, fingerprint: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="corral-md-local-verify-") as temporary:
            root = Path(temporary)
            evidence, output = root / "evidence", root / "output"
            evidence.mkdir()
            output.mkdir()
            root.chmod(0o755)
            evidence.chmod(0o755)
            output.chmod(0o777)
            jobs = []
            for original in plan.jobs:
                job = dict(original)
                if job["operation"] == "reference":
                    fallback = job.pop("fallback_operation", None)
                    if fallback not in {"mace", "sw"}:
                        raise ValueError("Local verifier has no calculation fallback")
                    job["operation"] = fallback
                    job.pop("reference", None)
                jobs.append(job)
            request = {
                "schema": 1,
                "task_number": plan.task_number,
                "jobs": jobs,
                "evidence_sha256": fingerprint,
                "challenge": plan.challenge,
            }
            request_path = evidence / "request.json"
            request_path.write_text(json.dumps(request, allow_nan=False))
            request_path.chmod(0o644)
            for name, source in plan.files.items():
                if Path(name).name != name:
                    raise ValueError("Verifier checkpoint names must be flat")
                destination = evidence / name
                shutil.copyfile(source, destination)
                destination.chmod(0o644)
            image_id = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                text=True,
            ).strip()
            operations = {job["operation"] for job in jobs}
            command = [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--user",
                "10001:10001",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--cpus",
                "8",
                "--memory",
                "16g",
                "--pids-limit",
                "512",
                "--shm-size",
                "2g",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=4g",
                "--workdir",
                "/output",
                "--env",
                "HOME=/tmp",
                "--env",
                "XDG_CACHE_HOME=/tmp/.cache",
                "--network",
                "bridge" if operations == {"sw"} else "none",
                "--mount",
                f"type=bind,src={evidence},dst=/evidence,readonly",
                "--mount",
                f"type=bind,src={output},dst=/output",
                "--mount",
                (
                    "type=bind,src="
                    + str(Path(__file__).resolve().parents[3] / "modal_app")
                    + ",dst=/opt/corral-md,readonly"
                ),
            ]
            descriptors = asset_descriptors()
            for name, relative in asset_relative_paths().items():
                if Path(relative).parts[0] in {"models", "potentials"}:
                    command.extend(
                        (
                            "--mount",
                            (
                                f"type=bind,src={self.assets / relative},"
                                f"dst={descriptors[name].public_path},readonly"
                            ),
                        )
                    )
            # A data-disk runtime lets the trusted verifier reuse pinned local
            # scientific packages without putting multi-GB images on home.
            runtime_directory = os.getenv("CORRAL_MD_VERIFIER_RUNTIME")
            if runtime_directory:
                runtime = Path(runtime_directory).expanduser().resolve(strict=True)
                if not (runtime / "local/bin/python").exists():
                    raise ValueError("Verifier runtime must be a complete /usr tree")
                command.extend(
                    ("--mount", f"type=bind,src={runtime},dst=/usr,readonly")
                )
                command.extend(
                    (
                        "--env",
                        "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu",
                    )
                )
            if "mace" in operations:
                gpu_request = "1"
                if self.gpus is not None:
                    selected = self.gpus[(plan.task_number - 1) % len(self.gpus)]
                    gpu_request = f'"device={selected}"'
                command.extend(
                    ("--gpus", gpu_request, "--env", "CUDA_VISIBLE_DEVICES=0")
                )
            command.extend(
                (
                    image_id,
                    "python",
                    "-I",
                    "/opt/corral-md/verification_worker.py",
                    "/evidence/request.json",
                    "/output/result.json",
                )
            )
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=7200, check=False
            )
            if process.returncode:
                raise RuntimeError(
                    f"Local verifier exited {process.returncode}: {process.stderr[-3000:]}"
                )
            jobs = json.loads((output / "result.json").read_text())["jobs"]
            return {
                "verification_id": uuid.uuid4().hex,
                "release_id": image_id,
                "evidence_sha256": fingerprint,
                "challenge": plan.challenge,
                "jobs": jobs,
            }


class Level1LocalVerifier(Level1ModalVerifier):
    def __init__(self, assets_dir: str | Path | None = None) -> None:
        super().__init__(transport=LocalCalculationTransport(assets_dir))

    def evaluate(self, evidence, task_number):
        result = super().evaluate(evidence, task_number)
        result["mode"] = "local_docker"
        return result


class LocalVerifier(ModalVerifier):
    def __init__(self, assets_dir: str | Path | None = None) -> None:
        super().__init__(transport=LocalCalculationTransport(assets_dir))

    def evaluate(self, evidence, task_number):
        result = super().evaluate(evidence, task_number)
        result["mode"] = "local_docker"
        return result
