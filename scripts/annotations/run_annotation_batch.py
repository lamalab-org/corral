"""Resume a frozen annotation queue; labels and explanations have separate workers.

Each queue record pins its replica and input fingerprint. Publish jobs atomically,
then create source/complete.json when discovery finishes. Run one process per
label replica and one evidence process. This script never starts or stops servers.
"""

from __future__ import annotations

# Worker progress is intentionally written to the durable session log.
# ruff: noqa: E402, T201
import argparse
import asyncio
import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Run snapshots may place this file directly under scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "annotations" else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from corral.epistemic_analysis.annotators.local import (
    OpenAIClient,
    ProviderConfig,
    create_provider,
)
from corral.epistemic_analysis.scheduling import DEFAULT_TRACE_CONCURRENCY
from corral.epistemic_analysis.schema import content_hash
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import (
    accepted_annotation,
    atomic_save,
    read_record,
)
from corral.epistemic_analysis.trace_checks import CheckPipeline


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def publish_evidence(root, job, label):
    """Durably hand off a final positive check, independently of its siblings."""
    if (
        label["state"] not in {"complete", "requires_review"}
        or label["status"] != "present"
    ):
        return
    identifier = f"{job['id']}--{label['fingerprint']}"
    path = root / "queue/evidence" / f"{identifier}.json"
    if not path.exists():
        atomic_save(
            path,
            {
                **job,
                "id": identifier,
                "trace_job_id": job["id"],
                "check": label["name"],
                "label_fingerprint": label["fingerprint"],
                "flagged_at": now(),
            },
        )


def finalize_trace_explanations(root, job, attempt):
    """Join saved per-check text without generating it again at trace level."""
    record = {
        "id": job["id"],
        "gpu": job["gpu"],
        "attempt": attempt,
        "input_fingerprint": job["input_fingerprint"],
        "started_at": now(),
    }
    started = time.time()
    try:
        labels = read(root / "completed/labels" / f"{job['id']}.json")
        if labels["input_fingerprint"] != job["input_fingerprint"]:
            raise ValueError("Label input differs from explanation input")
        artifact = root / "artifacts" / f"gpu-{job['gpu']}"
        source_run = read(artifact / "runs" / f"{labels['run']}.json")
        evidence, failures = {}, []
        for name in labels["positive_labels"]:
            label_key = labels["labels"][name]
            path = root / "completed/evidence-checks" / f"{job['id']}--{label_key}.json"
            if not path.exists():
                failures.append(f"{name}: explanation missing")
                continue
            saved = read(path)
            if saved["input_fingerprint"] != job["input_fingerprint"]:
                raise ValueError("Saved explanation input differs")
            key = saved.get("evidence", {}).get(name)
            value = (
                read_record(artifact / "evidence" / f"{key}.json", key) if key else None
            )
            if value:
                if value["label_fingerprint"] != label_key:
                    raise ValueError("Explanation belongs to a different label")
                evidence[name] = key
            if (
                not value
                or saved["state"] not in {"complete", "requires_review"}
                or value["state"] not in {"complete", "requires_review"}
            ):
                failures.append(f"{name}: explanation unavailable")
        key = content_hash(
            {
                "format": "saved-explanation-rollup-v1",
                "source_run": labels["run"],
                "evidence": evidence,
                "failures": failures,
            }
        )
        state = "failed" if failures else "complete"
        run = {
            **source_run,
            "fingerprint": key,
            "state": state,
            "evidence": evidence,
            "explanation_rollup": {"source_run": labels["run"], "model_called": False},
        }
        atomic_save(artifact / "runs" / f"{key}.json", run)
        record.update(
            state=state,
            run=key,
            labels=labels["labels"],
            evidence=evidence,
            positive_labels=labels["positive_labels"],
        )
        if failures:
            record["error"] = "; ".join(failures)
    except Exception as exc:
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    record.update(
        finished_at=now(),
        finished_epoch=time.time(),
        elapsed_s=round(time.time() - started, 3),
    )
    return record


async def process_job(
    root, stage, job, done_path, previous, clients, evidence, profile
):
    """Persist one trace using the shared provider concurrency limits."""
    attempt = 1 + (previous or {}).get("attempt", 0)
    if stage == "evidence" and "check" not in job:
        record = finalize_trace_explanations(root, job, attempt)
        atomic_save(done_path, record)
        return
    started = time.time()
    record = {
        "id": job["id"],
        "gpu": job["gpu"],
        "attempt": attempt,
        "started_at": now(),
        "input_fingerprint": job["input_fingerprint"],
    }
    try:
        document = read(root / job["input"])
        inputs = CheckInput.from_document(document["trace"], document["context"])
        if inputs.fingerprint != job["input_fingerprint"]:
            raise ValueError("Frozen queue input changed")
        artifact = root / "artifacts" / f"gpu-{job['gpu']}"
        pipeline = CheckPipeline(
            artifact,
            clients[job["gpu"]],
            evidence if stage == "evidence" else None,
            threshold=profile["threshold"],
            windowed=True,
            repair_attempts=profile["repair_attempts"],
            reconcile_windowed=profile.get("reconcile_windowed", False),
            decision_window_tokens=profile.get("decision_window_tokens"),
            share_window_prefill=profile.get("share_window_prefill", False),
            overflow_tool_output_max_chars=profile.get(
                "overflow_tool_output_max_chars", 16000
            ),
        )
        source_run = job.get("reuse_labels_from_run")
        if stage == "evidence" and "check" not in job:
            source_run = read(root / "completed/labels" / f"{job['id']}.json")["run"]
        prior_labels = (
            {job["check"]: job["label_fingerprint"]}
            if "check" in job
            else read(artifact / "runs" / f"{source_run}.json")["labels"]
            if source_run
            else None
        )

        async def handoff(label):
            publish_evidence(root, job, label)

        result = await pipeline.run(
            inputs,
            evidence_only=stage == "evidence",
            label_records=prior_labels,
            **({"checks": [job["check"]]} if "check" in job else {}),
            **({"on_label": handoff} if stage == "labels" else {}),
        )
        labels = [
            read_record(artifact / "labels" / f"{key}.json", key)
            for key in result["labels"].values()
        ]
        record.update(
            state=result["state"],
            run=result["fingerprint"],
            labels=result["labels"],
            evidence=result["evidence"],
            positive_labels=[
                label["name"] for label in labels if label["status"] == "present"
            ],
        )
        if stage == "labels":
            failures = [
                label
                for label in labels
                if label["state"] not in {"complete", "requires_review"}
            ]
            if failures:
                record.update(
                    state="failed",
                    error="; ".join(
                        f"{label['name']}: {label.get('error', label['state'])}"
                        for label in failures
                    ),
                )
        else:
            failures = [
                read_record(artifact / "evidence" / f"{key}.json", key)
                for key in result["evidence"].values()
            ]
            failures = [
                r for r in failures if r["state"] not in {"complete", "requires_review"}
            ]
            if failures:
                record.update(
                    state="failed",
                    error="; ".join(r.get("error", r["state"]) for r in failures),
                )
        if "check" in job:
            record.update(
                check=job["check"],
                trace_job_id=job["trace_job_id"],
                flagged_at=job["flagged_at"],
                handoff_delay_s=round(
                    datetime.fromisoformat(record["started_at"]).timestamp()
                    - datetime.fromisoformat(job["flagged_at"]).timestamp(),
                    3,
                ),
            )
    except Exception as exc:
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    record.update(
        finished_at=now(),
        finished_epoch=time.time(),
        elapsed_s=round(time.time() - started, 3),
    )
    record = accepted_annotation(record)
    atomic_save(done_path, record)
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in {"labels", "evidence"}
            }
        ),
        flush=True,
    )


async def work(root, stage, gpu, *, trace_concurrency=DEFAULT_TRACE_CONCURRENCY):
    if trace_concurrency < 1:
        raise ValueError("trace_concurrency must be positive")
    profile = read(root / "profile.json")
    buffer = profile.get("request_preparation_buffer", 0) if stage == "labels" else 0
    timeout = profile.get("decision_request_timeout") if stage == "labels" else None
    clients = {
        int(key): create_provider(
            ProviderConfig(**value),
            **({"preparation_buffer": buffer} if buffer else {}),
            **({"request_timeout": timeout} if timeout is not None else {}),
        )
        for key, value in profile["replicas"].items()
    }
    evidence = OpenAIClient(ProviderConfig(**profile["evidence"]))
    name = f"labels-gpu-{gpu}" if stage == "labels" else "evidence-gpu-5"
    status_path = root / "status" / f"{name}.json"
    counts = {"completed": 0, "requires_review": 0, "failed": 0}
    pending = {}
    pending_traces = {}
    try:
        if stage == "labels":
            await clients[gpu].preflight()
        else:
            await evidence.preflight()
        while True:
            jobs = [read(p) for p in sorted((root / "queue").glob("*.json"))]
            if stage == "labels":
                jobs = [job for job in jobs if job["gpu"] == gpu]
            ready = []
            check_counts = {"queued": 0, "completed": 0, "failed": 0}
            if stage == "evidence":
                # Check jobs have priority over whole-trace bookkeeping. The
                # latter reuses the same evidence cache after these jobs finish.
                check_jobs = [read(p) for p in (root / "queue/evidence").glob("*.json")]
                check_jobs.sort(key=lambda job: job["flagged_at"])
                check_counts["queued"] = len(check_jobs)
                blocked_traces = set(pending_traces.values())
                for job in check_jobs:
                    done_path = root / "completed/evidence-checks" / f"{job['id']}.json"
                    done = read(done_path) if done_path.exists() else None
                    if done and done["state"] in {"complete", "requires_review"}:
                        check_counts["completed"] += 1
                        continue
                    if done and done.get("attempt", 0) >= 1:
                        check_counts["failed"] += 1
                        continue
                    blocked_traces.add(job["trace_job_id"])
                    if job["id"] in pending:
                        continue
                    if done and time.time() - done.get("finished_epoch", 0) < 60:
                        continue
                    ready.append((job, done_path, done))
            counts = {"completed": 0, "requires_review": 0, "failed": 0}
            for job in jobs:
                if job["id"] in pending:
                    continue
                done_path = root / "completed" / stage / f"{job['id']}.json"
                done = read(done_path) if done_path.exists() else None
                if done and done["state"] in {"complete", "requires_review"}:
                    counts["completed"] += 1
                    counts["requires_review"] += done["state"] == "requires_review"
                    continue
                if done and done.get("attempt", 0) >= (1 if stage == "evidence" else 3):
                    counts["failed"] += 1
                    continue
                if stage == "evidence":
                    if job["id"] in blocked_traces:
                        continue
                    label_path = root / "completed/labels" / f"{job['id']}.json"
                    label_done = read(label_path) if label_path.exists() else None
                    if (
                        label_done
                        and label_done["state"] == "failed"
                        and label_done.get("attempt", 0) >= 3
                    ):
                        atomic_save(
                            done_path,
                            {
                                "id": job["id"],
                                "gpu": job["gpu"],
                                "input_fingerprint": job["input_fingerprint"],
                                "state": "failed",
                                "attempt": 3,
                                "finished_at": now(),
                                "error": "Evidence blocked: label stage exhausted its retries.",
                            },
                        )
                        counts["failed"] += 1
                        continue
                    if not label_done or label_done["state"] not in {
                        "complete",
                        "requires_review",
                    }:
                        continue
                if done and time.time() - done.get("finished_epoch", 0) < 60:
                    continue
                ready.append((job, done_path, done))
            ready.sort(
                key=lambda entry: (
                    "check" not in entry[0],
                    "reuse_labels_from_run" not in entry[0],
                )
            )
            for job, done_path, previous in ready[
                : max(0, trace_concurrency - len(pending))
            ]:
                pending[job["id"]] = asyncio.create_task(
                    process_job(
                        root,
                        stage,
                        job,
                        done_path,
                        previous,
                        clients,
                        evidence,
                        profile,
                    )
                )
                pending_traces[job["id"]] = job.get("trace_job_id", job["id"])
            discovery_done = (root / "source/complete.json").exists()
            finished = (
                not pending
                and discovery_done
                and counts["completed"] + counts["failed"] == len(jobs)
                and check_counts["completed"] + check_counts["failed"]
                == check_counts["queued"]
            )
            atomic_save(
                status_path,
                {
                    "worker": name,
                    "state": "running"
                    if pending
                    else "finished"
                    if finished
                    else "waiting",
                    "updated_at": now(),
                    "trace_id": min(pending) if pending else None,
                    "trace_ids": sorted(pending),
                    "trace_concurrency": trace_concurrency,
                    "queued": len(jobs),
                    **({"check_handoffs": check_counts} if stage == "evidence" else {}),
                    **counts,
                },
            )
            if finished:
                return
            if pending:
                # Refill a freed trace slot immediately; a slow trace must not
                # hold the rest of the queue at an asyncio.gather batch barrier.
                completed, _ = await asyncio.wait(
                    pending.values(), timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                for identifier, task in list(pending.items()):
                    if task in completed:
                        del pending[identifier]
                        del pending_traces[identifier]
                        task.result()
            else:
                await asyncio.sleep(1)
    finally:
        for task in pending.values():
            task.cancel()
        await asyncio.gather(*pending.values(), return_exceptions=True)
        await asyncio.gather(
            *(client.close() for client in clients.values()), evidence.close()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["labels", "evidence"], required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--trace-concurrency", type=int, default=DEFAULT_TRACE_CONCURRENCY
    )
    args = parser.parse_args()
    if args.stage == "labels" and args.gpu is None:
        parser.error("--gpu is required for labels")
    if args.trace_concurrency < 1:
        parser.error("--trace-concurrency must be positive")
    args.root = args.root.resolve()
    control_path = args.root / "control.json"
    if control_path.exists() and not read(control_path).get("execution_enabled", True):
        parser.error("This annotation queue is disabled by control.json")
    lock_dir = args.root / "locks"
    lock_dir.mkdir(exist_ok=True)
    lock = (
        lock_dir
        / f"{args.stage}-{args.gpu if args.stage == 'labels' else 'gemma'}.lock"
    )
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(
            work(
                args.root,
                args.stage,
                args.gpu,
                trace_concurrency=args.trace_concurrency,
            )
        )


if __name__ == "__main__":
    main()
