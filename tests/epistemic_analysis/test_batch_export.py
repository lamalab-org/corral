"""Batch snapshots retain parent identities, saved labels, and partial evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from corral.epistemic_analysis.storage import atomic_save, save_record


def test_export_preserves_labels_and_evidence_without_repeating_facts(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from export_annotation_batch import export_batch

    root = tmp_path / "batch"
    artifact = root / "artifacts/gpu-0"
    jobs = []
    for identity, status in [("parent-a", "present"), ("parent-b", "uncertain")]:
        jobs.append(
            {
                "id": identity,
                "gpu": 0,
                "origin": {"environment": "test", "agent": "ai-scientist"},
            }
        )
        atomic_save(
            artifact / "inputs" / identity / "trace.json",
            {"messages": [{"role": "assistant", "content": identity}]},
        )
        atomic_save(
            artifact / "inputs" / identity / "facts-facts.json",
            {"events": [{"content": "large record" * 5000}]},
        )
        save_record(
            artifact / "labels" / f"{identity}.json",
            {
                "fingerprint": identity,
                "name": "check",
                "state": "requires_review",
                "status": status,
                "definition": {
                    "instructions": "Check the trace",
                    "criteria": {
                        "present": "yes",
                        "absent": "no",
                        "uncertain": "unknown",
                    },
                },
                "exchanges": [],
                "provenance": {"model": "test"},
            },
        )
        atomic_save(
            artifact / "runs" / f"{identity}.json",
            {
                "format": "independent-check-stages",
                "fingerprint": identity,
                "input_fingerprint": identity,
                "facts_hash": "facts",
                "labels": {"check": identity},
                "evidence": {},
                "qualification": {},
            },
        )
        atomic_save(
            root / "completed/labels" / f"{identity}.json",
            {"state": "complete", "run": identity},
        )
    jobs.append({"id": "unfinished", "gpu": 0, "origin": {}})
    atomic_save(root / "manifest.json", {"selection": jobs})
    save_record(
        artifact / "evidence/support.json",
        {
            "fingerprint": "support",
            "state": "partial_evidence",
            "evidence": [{"quote": "parent-a"}],
        },
    )
    atomic_save(
        root / "completed/evidence-checks/check.json",
        {"trace_job_id": "parent-a", "evidence": {"check": "support"}},
    )
    original = (artifact / "labels/parent-a.json").read_bytes()
    review_path = root / "fabrication-review/records/review.json"
    save_record(
        review_path,
        {
            "fingerprint": "review",
            "state": "complete",
            "candidates": [{"msg_idx": 0}],
            "review_status": "present",
            "check": "fabricated_evidence",
        },
    )
    atomic_save(
        root / "fabrication-review/completed/parent-a.json",
        {
            "review_record": str(review_path),
            "review_fingerprint": "review",
        },
    )
    output = tmp_path / "review"
    assert export_batch(root, output)["traces"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["id"] for row in manifest["selection"]] == ["parent-a", "parent-b"]
    for identity, status in [("parent-a", "present"), ("parent-b", "uncertain")]:
        assessment = json.loads(
            (output / "runs/knowline" / identity / "assessment.0.json").read_text()
        )
        check = assessment["checks"][0]
        assert check["status"] == status
        assert check["state"] == "complete"
        reference = check["validation_facts"]
        facts = output / reference["artifact"]
        assert hashlib.sha256(facts.read_bytes()).hexdigest() == reference["sha256"]
        assert "large record" not in json.dumps(assessment)
        if identity == "parent-a":
            assert check["evidence_record"]["state"] == "partial_evidence"
            assert check["evidence"] == [{"quote": "parent-a"}]
    assert (artifact / "labels/parent-a.json").read_bytes() == original
    assert json.loads(
        (output / "fabrication-review/parent-a.json").read_text()
    ) == json.loads(review_path.read_text())
    with pytest.raises(ValueError, match="new export directory"):
        export_batch(root, output)


def test_empty_batch_does_not_publish_export(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from export_annotation_batch import export_batch

    root = tmp_path / "batch"
    atomic_save(root / "manifest.json", {"selection": []})
    with pytest.raises(ValueError, match="No completed labels"):
        export_batch(root, tmp_path / "review")
    assert not (tmp_path / "review").exists()
