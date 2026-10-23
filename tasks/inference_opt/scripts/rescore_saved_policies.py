"""Restore audited submissions and rescore them in their original Docker images.

Run with the repository Python environment. Original run folders are read-only
inputs. Results are written separately; only infrastructure-clean evaluations
are eligible to replace original scores in the comparison CSV.
"""

# ruff: noqa: T201 - command-line progress output

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[3]
CONTAINER_SCRIPT = Path(__file__).with_name("rescore_policy_container.py")
LOCK = threading.Lock()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def blob(trial: Path, ref: str) -> Path:
    digest = ref.split(":")[-1]
    return trial / "artifacts" / digest[:2] / digest


def prepare(args) -> list[dict]:
    rows = list(csv.DictReader(args.audit.open()))
    selected = [r for r in rows if r["recommended_action"] == "reevaluate_saved_policy"]
    jobs = []
    for row in selected:
        trial = args.sources / row["source_path"]
        state = read_json(trial / "state-snapshots/final.json")
        ledger = state["environment"]["values"]["resources"]["inference_state"]
        final = ledger["final"]
        sandbox = read_json(trial / "sandbox.json")
        meta = final["result"]["metadata"]
        with tarfile.open(blob(trial, final["private_artifact"])) as archive:
            summaries = [
                json.load(archive.extractfile(member))
                for member in archive
                if member.name.endswith("/summary.json")
            ]
        budgets = {s["calls_reserved"] for s in summaries}
        assert len(budgets) == 1
        job_dir = (
            args.output
            / row["langfuse_session_id"]
            / row["task_id"]
            / row["trial_directory"]
        )
        job_dir.mkdir(parents=True, exist_ok=True)
        workspace = job_dir / "workspace"
        workspace.mkdir(exist_ok=True)
        for name, entry in state["workspace"]["files"].items():
            target = workspace / name
            if not target.resolve().is_relative_to(workspace.resolve()):
                raise ValueError(f"Unsafe saved workspace path: {name}")
            source = blob(trial, entry["blob_ref"])
            content = source.read_bytes()
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        digest = hashlib.sha256()
        policy_dir = workspace / final["policy_dir"]
        for path in sorted(p for p in policy_dir.rglob("*") if p.is_file()):
            if "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(policy_dir)).encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
        assert digest.hexdigest() == final["policy_hash"]
        revealed = set(ledger["revealed_ids"])
        revealed.update((ledger.get("shared_ledger") or {}).get("revealed_ids", []))
        job = {
            "execution_id": row["execution_id"],
            "langfuse_session_id": row["langfuse_session_id"],
            "langfuse_trace_id": row["langfuse_trace_id"],
            "task_id": row["task_id"],
            "trial_index": row["trial_index"],
            "level": int(row["langfuse_session_id"].split("_")[2][1:]),
            "policy_dir": final["policy_dir"],
            "policy_hash": final["policy_hash"],
            "revealed_ids": sorted(revealed),
            "original_score": float(row["score"]),
            "original_budget": next(iter(budgets)),
            "original_models": [s["model"] for s in summaries],
            "content_fingerprint": meta["content_fingerprint"],
            "dataset_version": meta["dataset_version"],
            "image_digest": sandbox["image_digest"],
            "cpus": sandbox["cpus"],
            "memory": sandbox["memory"],
            "pids_limit": sandbox["pids_limit"],
            "host_uid": os.getuid(),
            "host_gid": os.getgid(),
            "source_path": row["source_path"],
            "job_dir": str(job_dir.resolve()),
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(job)
    (args.output / "manifest.json").write_text(json.dumps(jobs, indent=2) + "\n")
    return jobs


def write_summary(args, jobs):
    rows = []
    for job in jobs:
        result_path = Path(job["job_dir"]) / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        valid = result.get("status") == "completed"
        evaluation = result.get("evaluation") or {}
        rows.append(
            {
                "execution_id": job["execution_id"],
                "langfuse_session_id": job["langfuse_session_id"],
                "langfuse_trace_id": job["langfuse_trace_id"],
                "task_id": job["task_id"],
                "trial_index": job["trial_index"],
                "original_score": job["original_score"],
                "rescored_score": evaluation.get("score", "") if valid else "",
                "status": result.get("status", "pending"),
                "service_error_records": sum(
                    result.get("student_service_errors", {}).values()
                ),
                "policy_hash": job["policy_hash"],
                "image_digest": job["image_digest"],
                "result_path": str(result_path),
                "error": result.get("error") or result.get("evaluation_error") or "",
            }
        )
    for destination in (
        args.output / "summary.csv",
        args.report_dir / "rescore_summary.csv",
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            f".csv.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    replacements = {r["execution_id"]: r for r in rows if r["status"] == "completed"}
    scores = list(csv.DictReader((REPO / "reports/inference_opt/trials.csv").open()))
    by_run = collections.defaultdict(list)
    originals = {job["execution_id"]: job["original_score"] for job in jobs}
    for row in scores:
        replacement = replacements.get(row["execution_id"])
        by_run[row["langfuse_session_id"]].append(
            (
                float(originals.get(row["execution_id"], row["score"])),
                float(replacement["rescored_score"])
                if replacement
                else float(row["score"]),
                bool(replacement),
            )
        )
    destination = args.report_dir / "rescore_run_scores.csv"
    temporary = destination.with_suffix(
        f".csv.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "langfuse_session_id",
                "trials",
                "original_average_score",
                "average_with_completed_rescores",
                "completed_rescores",
                "selected_rescores",
            ],
        )
        writer.writeheader()
        for session, values in by_run.items():
            writer.writerow(
                {
                    "langfuse_session_id": session,
                    "trials": len(values),
                    "original_average_score": sum(v[0] for v in values) / len(values),
                    "average_with_completed_rescores": sum(v[1] for v in values)
                    / len(values),
                    "completed_rescores": sum(v[2] for v in values),
                    "selected_rescores": sum(
                        j["langfuse_session_id"] == session for j in jobs
                    ),
                }
            )
    temporary.replace(destination)


def run_one(args, jobs, job, env):
    job_dir = Path(job["job_dir"])
    result_path = job_dir / "result.json"
    if result_path.exists():
        print(
            "EXISTING",
            job["execution_id"],
            read_json(result_path)["status"],
            flush=True,
        )
        return
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        "corral-rescore-" + hashlib.sha256(str(job_dir).encode()).hexdigest()[:18],
        "--cpus",
        str(job["cpus"]),
        "--memory",
        job["memory"],
        "--pids-limit",
        str(job["pids_limit"]),
        "--network",
        "host",
        "--read-only",
        "--user",
        "0:0",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        "apparmor=unconfined",
        "--cap-drop",
        "ALL",
    ]
    for capability in [
        "SETUID",
        "SETGID",
        "CHOWN",
        "DAC_OVERRIDE",
        "KILL",
        "SYS_ADMIN",
        "SYS_CHROOT",
    ]:
        command += ["--cap-add", capability]
    command += [
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,mode=0700,size=256m",
        "--tmpfs",
        "/workspace:rw,nosuid,nodev,mode=0711,size=512m",
        "--tmpfs",
        "/corral-state:rw,nosuid,nodev,mode=0700,size=64m",
        "--mount",
        f"type=bind,src={job_dir},dst=/rescore",
        "--mount",
        f"type=bind,src={CONTAINER_SCRIPT},dst=/rescore_runner.py,readonly",
        "--env",
        "OPENROUTER_API_KEY",
        "--env",
        "HOME=/tmp",
        "--env",
        "LITELLM_LOG=ERROR",
        "--entrypoint",
        "python",
        job["image_digest"],
        "/rescore_runner.py",
    ]
    print("START", job["execution_id"], flush=True)
    try:
        with (job_dir / "container.log").open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=4200,
                check=False,
            )
        if not result_path.exists():
            raise RuntimeError(
                f"Docker exited {result.returncode} without a result; see container.log"
            )
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "stop", "--time", "10", command[5]],
                capture_output=True,
                check=False,
            )
        result_path.write_text(
            json.dumps(
                {"status": "controller_error", "error": f"{type(exc).__name__}: {exc}"}
            )
            + "\n"
        )
    with LOCK:
        write_summary(args, jobs)
        print(
            "FINISH", job["execution_id"], read_json(result_path)["status"], flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit CSV selecting saved policies; required when preparing a new batch.",
    )
    parser.add_argument(
        "--sources", type=Path, default=Path("/data/alamparan/corral_ml_runs")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-dir", type=Path, help="Defaults to the batch output directory."
    )
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.report_dir = args.report_dir or args.output
    if not (args.output / "manifest.json").exists() and args.audit is None:
        parser.error("--audit is required when preparing a new batch")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        arguments = [value for value in sys.argv[1:] if value != "--detach"]
        with (args.output / "batch.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print("STARTED_BACKGROUND_PID", child.pid, flush=True)
        return
    jobs = (
        prepare(args)
        if not (args.output / "manifest.json").exists()
        else read_json(args.output / "manifest.json")
    )
    print(
        "PREPARED",
        len(jobs),
        "policies",
        datetime.now(timezone.utc).isoformat(),
        flush=True,
    )
    write_summary(args, jobs)
    if args.prepare_only:
        return
    env = {
        **os.environ,
        **{k: v for k, v in dotenv_values(REPO / ".env").items() if v is not None},
    }
    if not env.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    selected = jobs[args.start_index :]
    if args.limit:
        selected = selected[: args.limit]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(lambda job: run_one(args, jobs, job, env), selected))
    write_summary(args, jobs)
    print("BATCH_FINISHED", flush=True)


if __name__ == "__main__":
    main()
