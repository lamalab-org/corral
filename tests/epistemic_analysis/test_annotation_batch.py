"""Resume only ready evidence jobs and overlap them under shared client limits."""

import asyncio
import json
from pathlib import Path

import pytest


def test_trace_explanations_roll_up_saved_text_without_model_calls(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    from corral.epistemic_analysis.storage import atomic_save, save_record

    job = {"id": "trace", "gpu": 0, "input_fingerprint": "input"}
    labels = {
        "input_fingerprint": "input",
        "run": "label-run",
        "labels": {"check": "label-key"},
        "positive_labels": ["check"],
    }
    artifact = tmp_path / "artifacts/gpu-0"
    atomic_save(tmp_path / "completed/labels/trace.json", labels)
    original = (tmp_path / "completed/labels/trace.json").read_bytes()
    atomic_save(
        artifact / "runs/label-run.json",
        {
            "fingerprint": "label-run",
            "input_fingerprint": "input",
            "labels": labels["labels"],
            "facts_hash": "facts",
            "evidence": {},
        },
    )
    atomic_save(
        tmp_path / "completed/evidence-checks/trace--label-key.json",
        {
            "input_fingerprint": "input",
            "state": "complete",
            "evidence": {"check": "text-key"},
        },
    )
    save_record(
        artifact / "evidence/text-key.json",
        {
            "fingerprint": "text-key",
            "label_fingerprint": "label-key",
            "state": "complete",
            "explanation": "The returned plain text.",
        },
    )
    result = batch.finalize_trace_explanations(tmp_path, job, 1)
    assert result["state"] == "complete"
    assert result["evidence"] == {"check": "text-key"}
    run = json.loads((artifact / "runs" / f"{result['run']}.json").read_text())
    assert run["explanation_rollup"]["model_called"] is False
    assert (tmp_path / "completed/labels/trace.json").read_bytes() == original
    (tmp_path / "completed/evidence-checks/trace--label-key.json").unlink()
    failed = batch.finalize_trace_explanations(tmp_path, job, 1)
    assert failed["state"] == "failed"
    assert "missing" in failed["error"]


def test_explanation_queue_does_not_retry_one_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    from corral.epistemic_analysis.storage import atomic_save

    atomic_save(tmp_path / "profile.json", {"replicas": {"0": {}}, "evidence": {}})
    atomic_save(tmp_path / "source/complete.json", {})
    atomic_save(tmp_path / "queue/trace.json", {"id": "trace", "gpu": 0})
    atomic_save(
        tmp_path / "queue/evidence/check.json",
        {"id": "check", "trace_job_id": "trace", "flagged_at": "2026-01-01"},
    )
    atomic_save(tmp_path / "completed/evidence/trace.json", {"state": "complete"})
    atomic_save(
        tmp_path / "completed/evidence-checks/check.json",
        {"state": "failed", "attempt": 1},
    )

    class Client:
        def __init__(self, config):
            pass

        async def preflight(self):
            pass

        async def close(self):
            pass

    async def unexpected(*args):
        raise AssertionError("An exhausted explanation must not be generated again")

    monkeypatch.setattr(batch, "ProviderConfig", lambda **kw: kw)
    monkeypatch.setattr(batch, "create_provider", Client)
    monkeypatch.setattr(batch, "OpenAIClient", Client)
    monkeypatch.setattr(batch, "process_job", unexpected)
    asyncio.run(batch.work(tmp_path, "evidence", None))
    status = json.loads((tmp_path / "status/evidence-gpu-5.json").read_text())
    assert status["state"] == "finished"
    assert status["check_handoffs"]["failed"] == 1


def test_evidence_trace_parallelism_preserves_finished_and_blocked_jobs(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    save(tmp_path / "profile.json", {"replicas": {"0": {}}, "evidence": {}})
    save(tmp_path / "source/complete.json", {})
    for name in ["done", "a", "b", "blocked"]:
        save(
            tmp_path / "queue" / f"{name}.json",
            {"id": name, "gpu": 0, "input_fingerprint": name},
        )
        save(
            tmp_path / "completed/labels" / f"{name}.json",
            {
                "id": name,
                "state": "failed" if name == "blocked" else "complete",
                "attempt": 3 if name == "blocked" else 1,
            },
        )
    finished = tmp_path / "completed/evidence/done.json"
    save(finished, {"id": "done", "state": "complete", "attempt": 1})
    original = finished.read_bytes()

    class Client:
        def __init__(self, config):
            self.closed = False

        async def preflight(self):
            pass

        async def close(self):
            self.closed = True

    monkeypatch.setattr(batch, "ProviderConfig", lambda **kw: kw)
    monkeypatch.setattr(batch, "create_provider", Client)
    monkeypatch.setattr(batch, "OpenAIClient", Client)

    async def exercise():
        entered = []
        clients_seen = []
        both_started = asyncio.Event()

        async def process(
            root, stage, job, done_path, previous, clients, evidence, profile
        ):
            assert stage == "evidence"
            assert previous is None
            entered.append(job["id"])
            clients_seen.append((clients, evidence))
            if len(entered) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 2)
            save(done_path, {"id": job["id"], "state": "complete", "attempt": 1})

        monkeypatch.setattr(batch, "process_job", process)
        await batch.work(tmp_path, "evidence", None, trace_concurrency=2)
        assert sorted(entered) == ["a", "b"]
        assert clients_seen[0][0] is clients_seen[1][0]
        assert clients_seen[0][1] is clients_seen[1][1]
        assert clients_seen[0][1].closed

    asyncio.run(exercise())
    assert finished.read_bytes() == original
    blocked = json.loads((tmp_path / "completed/evidence/blocked.json").read_text())
    assert blocked["state"] == "failed"
    assert blocked["attempt"] == 3
    status = json.loads((tmp_path / "status/evidence-gpu-5.json").read_text())
    assert status["state"] == "finished"
    assert status["completed"] == 3
    assert status["failed"] == 1


@pytest.mark.parametrize("concurrency", [None, 2])
def test_label_queue_refills_before_slow_trace_finishes(
    tmp_path, monkeypatch, concurrency
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save(tmp_path / "profile.json", {"replicas": {"0": {}}, "evidence": {}})
    save(tmp_path / "source/complete.json", {})
    for name in ["a", "b", "c"]:
        save(tmp_path / "queue" / f"{name}.json", {"id": name, "gpu": 0})

    class Client:
        def __init__(self, config):
            pass

        async def preflight(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(batch, "ProviderConfig", lambda **kw: kw)
    monkeypatch.setattr(batch, "create_provider", Client)
    monkeypatch.setattr(batch, "OpenAIClient", Client)

    async def exercise():
        third_started = asyncio.Event()
        seen = []

        async def process(root, stage, job, path, previous, clients, evidence, profile):
            seen.append(job["id"])
            if job["id"] == "a":
                await asyncio.wait_for(third_started.wait(), 2)
            elif job["id"] == "c":
                third_started.set()
            save(path, {"state": "complete"})

        monkeypatch.setattr(batch, "process_job", process)
        options = {} if concurrency is None else {"trace_concurrency": concurrency}
        await batch.work(tmp_path, "labels", 0, **options)
        assert seen == ["a", "b", "c"]

    asyncio.run(exercise())


def test_trace_concurrency_must_be_positive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    with pytest.raises(ValueError, match="positive"):
        asyncio.run(batch.work(tmp_path, "labels", 0, trace_concurrency=0))


def test_check_handoff_runs_before_trace_labels_finish_and_is_idempotent(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    import run_annotation_batch as batch

    save = batch.atomic_save
    save(tmp_path / "profile.json", {"replicas": {"0": {}}, "evidence": {}})
    save(tmp_path / "source/complete.json", {})
    job = {"id": "trace", "gpu": 0, "input_fingerprint": "input"}
    save(tmp_path / "queue/trace.json", job)
    label = {
        "name": "false_reporting",
        "fingerprint": "label",
        "state": "complete",
        "status": "present",
    }
    batch.publish_evidence(tmp_path, job, label)
    event = tmp_path / "queue/evidence/trace--label.json"
    original = event.read_bytes()
    batch.publish_evidence(tmp_path, job, label)
    assert event.read_bytes() == original
    for status in ("uncertain", "absent"):
        batch.publish_evidence(
            tmp_path, job, {**label, "status": status, "fingerprint": status}
        )
    assert len(list(event.parent.glob("*.json"))) == 1

    class Client:
        def __init__(self, config):
            pass

        async def preflight(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(batch, "ProviderConfig", lambda **kw: kw)
    monkeypatch.setattr(batch, "create_provider", Client)
    monkeypatch.setattr(batch, "OpenAIClient", Client)
    seen = []

    async def process(root, stage, queued, path, previous, clients, evidence, profile):
        seen.append(queued["id"])
        if "check" in queued:
            assert not (root / "completed/labels/trace.json").exists()
            save(root / "completed/labels/trace.json", {"state": "complete"})
        save(path, {"state": "complete", "attempt": 1})

    monkeypatch.setattr(batch, "process_job", process)
    asyncio.run(batch.work(tmp_path, "evidence", None, trace_concurrency=4))
    assert seen == ["trace--label", "trace"]
    # A worker restart must not replay either completed job.
    asyncio.run(batch.work(tmp_path, "evidence", None, trace_concurrency=4))
    assert seen == ["trace--label", "trace"]
