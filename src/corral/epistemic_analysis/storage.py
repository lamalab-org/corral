"""Storage for epistemic trace analysis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .acceptance import accepted_annotation
from .schema import content_hash


def save_annotation_artifacts(graph, output: Path) -> None:
    """Write portable views and exact Jev exchanges for inspection."""
    if graph.annotation_view is not None:
        digest = content_hash(graph.annotation_view)
        artifact = f"annotation_views/{digest}.json"
        graph.provenance["annotation_view_hash"] = digest
        graph.provenance["annotation_view_artifact"] = artifact
        atomic_save(output / artifact, graph.annotation_view)
    if graph.annotation_requests:
        digest = content_hash(graph.annotation_requests)
        artifact = f"annotation_requests/{digest}.json"
        graph.provenance["annotation_requests_hash"] = digest
        graph.provenance["annotation_requests_artifact"] = artifact
        atomic_save(output / artifact, graph.annotation_requests)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".check-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_record(path, fingerprint):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("fingerprint") != fingerprint:
        raise ValueError("Cached record fingerprint differs")
    sealed = record.pop("record_hash", None)
    if sealed != content_hash(record):
        raise ValueError("Cached record content changed")
    record["record_hash"] = sealed
    return record


def save_record(path, record):
    record.pop("record_hash", None)
    record["record_hash"] = content_hash(record)
    atomic_save(path, record)


def archive_record(path, record):
    """Retain every superseded response when retrying or explicitly rerunning."""
    if record:
        atomic_save(
            path.parent / "history" / f"{path.stem}.{record['record_hash']}.json",
            record,
        )


def export_review(output, run, destination, *, facts_reference=None):
    output, destination = Path(output), Path(destination)
    if destination.exists():
        raise ValueError(
            "Use a new review export directory; existing reviews are never replaced"
        )
    root = output / "inputs" / run["input_fingerprint"]
    trace = json.loads((root / "trace.json").read_text())
    # Batch exports keep the full facts once and reference them from each check.
    facts = (
        facts_reference
        if facts_reference is not None
        else json.loads((root / f"facts-{run['facts_hash']}.json").read_text())
    )
    identifier = run["input_fingerprint"][:24]
    checks, questions = [], {}
    for name, key in run["labels"].items():
        label = read_record(output / "labels" / f"{key}.json", key)
        evidence_key = run["evidence"].get(name)
        evidence = (
            read_record(output / "evidence" / f"{evidence_key}.json", evidence_key)
            if evidence_key
            else {}
        )
        label = accepted_annotation(label)
        evidence = accepted_annotation(evidence)
        definition = label["definition"]
        questions[name] = {
            "type": "choice",
            "instructions": definition["instructions"],
            "criteria": {
                **definition["criteria"],
                "not_applicable": "Assigned only by the code applicability contract.",
            },
        }
        checks.append(
            {
                **label,
                "rationale": label.get("rationale", ""),
                "evidence_record": evidence,
                "evidence": evidence.get("evidence", []),
                "validation_facts": facts,
                "request_count": len(label["exchanges"]),
                "errors": int(label["state"] == "failed"),
            }
        )
    model = "knowline"
    row = {
        "id": identifier,
        "message_count": len(trace["messages"]),
        "reference_tokens": 0,
        "environment": trace.get("grouping", {}).get("environment") or "",
        "agent": "",
        "model": trace.get("grouping", {}).get("agent_model") or "",
        "task_id": trace.get("grouping", {}).get("task") or "",
        "level": trace.get("grouping", {}).get("level") or "",
    }
    manifest = {
        "format": run["format"],
        "selection": [row],
        "models": {model: checks[0]["provenance"] if checks else {}},
        "protocol": {
            "trace_check_version": "content-addressed-checks",
            "questions": questions,
        },
        "code_checks": [],
        "pipeline_run": run["fingerprint"],
        "qualification": run["qualification"],
    }
    plan = {
        "trace_id": identifier,
        "units": [
            {
                "id": "conversation",
                "kind": "conversation",
                "indices": list(range(len(trace["messages"]))),
                "slices": [],
            }
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".review-", dir=destination.parent
    ) as directory:
        staging = Path(directory)
        atomic_save(staging / "traces" / f"{identifier}.json", trace)
        atomic_save(staging / "plans" / f"{identifier}.json", plan)
        row["trace_sha256"] = hashlib.sha256(
            json.dumps(trace["messages"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        atomic_save(staging / "manifest.json", manifest)
        atomic_save(
            staging / "runs" / model / identifier / "assessment.0.json",
            {
                "checks": checks,
                "units": [{"unit_id": "conversation", "checks": checks}],
            },
        )
        staging.rename(destination)
    return destination
