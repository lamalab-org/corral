"""Export completed batch labels and available evidence to the annotation viewer."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Run snapshots may place this file directly under scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "annotations" else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from corral.epistemic_analysis.storage import atomic_save, export_review, read_record


def read(path):
    return json.loads(path.read_text())


def export_batch(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new export directory; existing reviews are preserved")
    jobs = read(root / "manifest.json")["selection"]
    check_evidence = {}
    for path in (root / "completed/evidence-checks").glob("*.json"):
        record = read(path)
        if record.get("evidence"):
            check_evidence.setdefault(record["trace_job_id"], {}).update(
                record["evidence"]
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".batch-review-", dir=destination.parent
    ) as temp:
        staging = Path(temp)
        manifest = None
        for job in jobs:
            completed = root / "completed/labels" / f"{job['id']}.json"
            if not completed.exists():
                continue
            label_record = read(completed)
            if label_record.get("state") not in {"complete", "requires_review"}:
                continue
            artifact = root / "artifacts" / f"gpu-{job['gpu']}"
            run = read(artifact / "runs" / f"{label_record['run']}.json")
            evidence_path = root / "completed/evidence" / f"{job['id']}.json"
            evidence = (
                read(evidence_path).get("evidence", {})
                if evidence_path.exists()
                else {}
            )
            run = {
                **run,
                "evidence": {
                    **run.get("evidence", {}),
                    **check_evidence.get(job["id"], {}),
                    **evidence,
                },
            }
            facts_source = (
                artifact
                / "inputs"
                / run["input_fingerprint"]
                / f"facts-{run['facts_hash']}.json"
            )
            facts_target = staging / "facts" / f"{job['id']}.json"
            facts_target.parent.mkdir(exist_ok=True)
            shutil.copy2(facts_source, facts_target)
            single = export_review(
                artifact,
                run,
                staging / "single",
                facts_reference={
                    "artifact": f"facts/{job['id']}.json",
                    "sha256": hashlib.sha256(facts_source.read_bytes()).hexdigest(),
                },
            )
            single_manifest = read(single / "manifest.json")
            if manifest is None:
                manifest = {
                    **single_manifest,
                    "selection": [],
                    "pipeline_run": root.name,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "source_batch": str(root),
                    "evidence_snapshot": True,
                }
            elif manifest["protocol"] != single_manifest["protocol"]:
                raise ValueError("Batch contains different check definitions")
            row = single_manifest["selection"][0]
            old_id, new_id = row["id"], job["id"]
            manifest["selection"].append({**row, **job["origin"], "id": new_id})
            for directory in ["traces", "plans"]:
                target = staging / directory / f"{new_id}.json"
                target.parent.mkdir(exist_ok=True)
                (single / directory / f"{old_id}.json").rename(target)
                if directory == "plans":
                    plan = read(target)
                    atomic_save(target, {**plan, "trace_id": new_id})
            target = staging / "runs/knowline" / new_id
            target.parent.mkdir(parents=True, exist_ok=True)
            (single / "runs/knowline" / old_id).rename(target)
            reviewed = root / "fabrication-review/completed" / f"{new_id}.json"
            if reviewed.exists():
                review_index = read(reviewed)
                record = read_record(
                    Path(review_index["review_record"]),
                    review_index["review_fingerprint"],
                )
                if record["candidates"]:
                    atomic_save(
                        staging / "fabrication-review" / f"{new_id}.json", record
                    )
            # The per-trace export is empty except for its manifest and directories.
            shutil.rmtree(single)
        if manifest is None:
            raise ValueError("No completed labels to export")
        atomic_save(staging / "manifest.json", manifest)
        staging.rename(destination)
    return {"traces": len(manifest["selection"]), "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_batch(args.root, args.output)))  # noqa: T201


if __name__ == "__main__":
    main()
