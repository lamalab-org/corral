"""Run a frozen native KnowLine graph queue, optionally after a predecessor."""

from __future__ import annotations

# ruff: noqa: E402, T201
import argparse
import asyncio
import fcntl
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

# Run snapshots may place this file directly under scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "annotations" else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from corral.epistemic_analysis.annotators.graph import GraphExtractor
from corral.epistemic_analysis.annotators.local import (
    GraphQuestionClient,
    ProviderConfig,
    create_provider,
)
from corral.epistemic_analysis.budget import RequestBudgetExceeded
from corral.epistemic_analysis.context import (
    AnnotationView,
    submission_end,
    through_submission,
)
from corral.epistemic_analysis.graph import iter_windows
from corral.epistemic_analysis.overflow import excerpt_state, view_fragments
from corral.epistemic_analysis.pipeline import annotate_trace
from corral.epistemic_analysis.scheduling import DEFAULT_TRACE_CONCURRENCY, rolling_map
from corral.epistemic_analysis.schema import AnnotationConfig, content_hash
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import accepted_annotation, atomic_save


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessor_status(request):
    if request.get("predecessor") is None:
        return {"ready": True, "completed": {}, "failures": [], "total": 0}
    previous = Path(request["predecessor"]["directory"])
    if (
        hashlib.sha256((previous / "manifest.json").read_bytes()).hexdigest()
        != request["predecessor"]["manifest_sha256"]
    ):
        raise ValueError("Predecessor corpus manifest changed")
    allowed = set(request["predecessor"]["accepted_terminal_states"])
    counts, failures = {}, []
    for stage in request["predecessor"]["required_stages"]:
        counts[stage] = 0
        for job in request["selection"]:
            path = previous / "completed" / stage / f"{job['id']}.json"
            if not path.exists():
                continue
            result = read(path)
            if result.get("input_fingerprint") != job["input_fingerprint"]:
                raise ValueError(f"Predecessor input fingerprint changed: {job['id']}")
            if result["state"] in allowed:
                counts[stage] += 1
            elif result["state"] == "failed":
                failures.append(
                    {"stage": stage, "id": job["id"], "error": result.get("error")}
                )
    return {
        "ready": not failures
        and all(n == len(request["selection"]) for n in counts.values()),
        "completed": counts,
        "failures": failures,
        "total": len(request["selection"]),
    }


def verify_code(root, manifest_path=None):
    manifest = read(manifest_path or root / "code-manifest.json")
    for item in manifest:
        path = ROOT / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen graph implementation changed: {item['path']}")


def next_window(config, message_count):
    """Keep the first halved window that actually changes the failed plan."""
    failed = iter_windows(message_count, config.window, config.overlap)
    window = config.window
    while window > 1:
        window = max(1, window // 2)
        if iter_windows(message_count, window, window // 2) != failed:
            return replace(config, window=window, overlap=window // 2)
    return None


async def process_job(
    root, job, gpu, provider, profile, configuration, request, progress
):
    done_path = root / "completed" / f"{job['id']}.json"
    if done_path.exists():
        saved = read(done_path)
        if saved.get("profile_fingerprint") == configuration and saved["state"] in {
            "complete",
            "requires_review",
        }:
            path = root / saved["graph"]
            if (
                hashlib.sha256(await asyncio.to_thread(path.read_bytes)).hexdigest()
                != saved["graph_sha256"]
            ):
                raise ValueError("Saved graph changed")
            return saved
    source = Path(job["input"])
    if (
        hashlib.sha256(await asyncio.to_thread(source.read_bytes)).hexdigest()
        != job["input_file_sha256"]
    ):
        raise ValueError(f"Frozen graph input changed: {job['id']}")
    document = read(source)
    inputs = CheckInput.from_document(document["trace"], document["context"])
    if inputs.fingerprint != job["input_fingerprint"]:
        raise ValueError("Graph input fingerprint differs")
    prepared_view = None
    public_context = inputs.public_state()
    if job.get("prepared_view"):
        view_path = Path(job["prepared_view"])
        if (
            hashlib.sha256(await asyncio.to_thread(view_path.read_bytes)).hexdigest()
            != job["prepared_view_sha256"]
        ):
            raise ValueError("Frozen graph preprocessing changed")
        prepared = read(view_path)
        if prepared["input_fingerprint"] != inputs.fingerprint:
            raise ValueError("Graph preprocessing input differs")
        if prepared.get("view") is not None:
            prepared_view = through_submission(
                AnnotationView(**prepared["view"]), inputs.trace
            )
            public_context = excerpt_state(
                {
                    **public_context,
                    "messages": view_fragments(inputs.trace, prepared_view),
                },
                prepared_view,
            )
            public_context.pop("messages")
            public_context["scope"] = "graph_through_first_submit_answer"
            public_context["excerpt_coverage"]["all_messages_represented"] = len(
                prepared_view.messages
            ) == len(inputs.trace.messages)
    client = GraphQuestionClient(
        provider,
        root / "artifacts" / f"gpu-{gpu}",
        public_context,
        batch_size=profile["batch_size"],
        progress=progress,
    )
    annotator = GraphExtractor(
        client,
        min_confidence=profile["min_confidence"],
        max_span_chars=profile["max_span_chars"],
    ).backend(stop_at_submission=True)
    config = AnnotationConfig(model=provider.config.model, **profile["annotation"])
    record = {
        "id": job["id"],
        "gpu": gpu,
        "queue_runner_runtime": str(ROOT),
        "input_fingerprint": inputs.fingerprint,
        "profile_fingerprint": configuration,
        "started_at": now(),
        "attempts": [],
    }
    started = time.monotonic()
    for attempt in range(1, 4):
        try:
            while True:
                try:
                    graph = await annotate_trace(
                        inputs.trace,
                        annotator,
                        config,
                        prepared_view=prepared_view,
                    )
                    break
                except RequestBudgetExceeded as exc:
                    record["attempts"].append(
                        {"window": config.window, "error": str(exc)}
                    )
                    smaller = next_window(config, submission_end(inputs.trace))
                    if smaller is None:
                        raise
                    config = smaller
            graph.qc["native_window_reductions"] = record["attempts"]
            graph.provenance["queue_runner_runtime"] = record["queue_runner_runtime"]
            graph.provenance["behavioral_results"] = {
                "directory": (request.get("predecessor") or {}).get("directory"),
                "trace_id": job.get("parent_id", job["id"]),
                "policy": "Behavioral checks run separately from graph extraction.",
            }
            graph.provenance["preprocessing"] = {
                "artifact": job.get("prepared_view"),
                "sha256": job.get("prepared_view_sha256"),
                "parent_id": job.get("parent_id"),
                "scientist_unit": job.get("scientist_unit"),
            }
            path = root / "graphs" / f"{job['id']}.json"
            atomic_save(path, asdict(graph))
            record.update(
                state="complete"
                if graph.annotation_status == "complete"
                and not graph.qc["context_coverage"]["reduced_context"]
                else "requires_review",
                graph=str(path.relative_to(root)),
                graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                configuration_id=graph.provenance["configuration_id"],
                question_count=client.completed_questions,
                attempt=attempt,
            )
            break
        except Exception as exc:
            record.update(
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
                attempt=attempt,
            )
            atomic_save(done_path, record)
            if attempt < 3 and not isinstance(exc, RequestBudgetExceeded):
                await asyncio.sleep(10)
            else:
                break
    record.update(finished_at=now(), elapsed_s=round(time.monotonic() - started, 3))
    record = accepted_annotation(record)
    atomic_save(done_path, record)
    print(json.dumps(record), flush=True)
    return record


async def run(
    root,
    gpu,
    *,
    check_ready=False,
    trace_concurrency=DEFAULT_TRACE_CONCURRENCY,
    code_manifest=None,
    replica=None,
):
    if trace_concurrency < 1:
        raise ValueError("trace_concurrency must be positive")
    request, profile = read(root / "queue-request.json"), read(root / "profile.json")
    configuration = content_hash(profile)
    if configuration != request["profile_fingerprint"]:
        raise ValueError("Frozen graph profile changed")
    if not request["execution_enabled"] or request["method"] != "knowline":
        raise ValueError("Graph queue is not enabled for native KnowLine")
    verify_code(root, code_manifest)
    status_path = root / "status" / f"gpu-{gpu}.json"
    while True:
        gate = predecessor_status(request)
        if check_ready:
            print(json.dumps(gate))
            return
        if gate["ready"]:
            atomic_save(
                root / "predecessor-complete.json", {"verified_at": now(), **gate}
            )
            break
        atomic_save(
            status_path,
            {
                "state": "waiting_for_predecessor",
                "gpu": gpu,
                "updated_at": now(),
                "predecessor": gate,
            },
        )
        await asyncio.sleep(20)
    if replica is None:
        replica = profile["replicas"][str(gpu)]
    else:
        original = next(iter(profile["replicas"].values()))
        endpoints = {"endpoint", "metadata_endpoint", "token_endpoint"}
        if {k: v for k, v in replica.items() if k not in endpoints} != {
            k: v for k, v in original.items() if k not in endpoints
        }:
            raise ValueError(
                "An additional replica must match the frozen model and scoring configuration"
            )
    provider = create_provider(ProviderConfig(**replica))
    try:
        verify_code(root, code_manifest)
        await provider.preflight()
        assigned = [job for job in request["selection"] if job["gpu"] == gpu]
        active = {}

        def publish():
            first = next(iter(active.values()), {})
            atomic_save(
                status_path,
                {
                    **first,
                    "state": "running",
                    "gpu": gpu,
                    "updated_at": now(),
                    "trace_id": next(iter(active), None),
                    "trace_ids": list(active),
                    "active": active,
                    "trace_concurrency": trace_concurrency,
                },
            )

        async def process(job):
            active[job["id"]] = {"started_at": now(), "completed_questions": 0}

            def progress(info):
                active[job["id"]].update(info)
                publish()

            try:
                return await process_job(
                    root, job, gpu, provider, profile, configuration, request, progress
                )
            finally:
                active.pop(job["id"])
                publish()

        await rolling_map(process, assigned, concurrency=trace_concurrency)
        records = [read(root / "completed" / f"{job['id']}.json") for job in assigned]
        atomic_save(
            status_path,
            {
                "state": "finished",
                "gpu": gpu,
                "updated_at": now(),
                "traces": len(records),
                "failed": sum(r["state"] == "failed" for r in records),
                "requires_review": sum(
                    r["state"] == "requires_review" for r in records
                ),
            },
        )
    finally:
        await provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument(
        "--trace-concurrency", type=int, default=DEFAULT_TRACE_CONCURRENCY
    )
    args = parser.parse_args()
    if args.trace_concurrency < 1:
        parser.error("--trace-concurrency must be positive")
    root = args.root.resolve()
    (root / "locks").mkdir(parents=True, exist_ok=True)
    with (root / "locks" / f"gpu-{args.gpu}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(
                run(
                    root,
                    args.gpu,
                    check_ready=args.check_ready,
                    trace_concurrency=args.trace_concurrency,
                    code_manifest=args.code_manifest,
                )
            )
        except Exception as exc:
            atomic_save(
                root / "status" / f"gpu-{args.gpu}.json",
                {
                    "state": "blocked",
                    "gpu": args.gpu,
                    "updated_at": now(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise


if __name__ == "__main__":
    main()
