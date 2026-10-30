"""Sidecars named by calculation logs remain confined and fingerprinted."""

import json

import pytest
from corral_md.scoring.common import Evidence, EvidenceError
from corral_md.submission import resolve_submission


def evidence_with_log(root):
    output = root / "output"
    output.mkdir(parents=True)
    log = output / "run.jsonl"
    log.write_text(json.dumps({"restart": "restart.traj"}) + "\n")
    restart = output / "restart.traj"
    restart.write_bytes(b"original restart")
    submission = json.dumps({"artifacts": {"run_log": "/workspace/output/run.jsonl"}})
    return Evidence(resolve_submission(submission, root)), log, restart


def test_explicit_log_sidecar_is_included_in_fingerprint(tmp_path):
    evidence, log, restart = evidence_with_log(tmp_path)
    original = evidence.fingerprint()
    assert evidence.linked_path("restart.traj", log, register=True) == restart
    linked = evidence.fingerprint()
    assert linked != original
    assert (
        evidence.linked_path("/workspace/output/restart.traj", log, register=True)
        == restart
    )
    assert evidence.fingerprint() == linked
    restart.write_bytes(b"different restart")
    assert evidence.fingerprint() != linked


def test_registered_sidecar_does_not_need_a_new_submission_key(tmp_path):
    evidence, log, restart = evidence_with_log(tmp_path)
    with pytest.raises(EvidenceError, match=r"listed in manifest\.artifacts"):
        evidence.linked_path("restart.traj", log)
    evidence.linked_path("restart.traj", log, register=True)
    assert evidence.manifest["artifacts"] == {"run_log": str(log)}
    assert restart.read_bytes() == b"original restart"


def test_registered_sidecars_cannot_escape_or_use_unlinked_documents(tmp_path):
    evidence, log, _ = evidence_with_log(tmp_path / "workspace")
    outside = tmp_path / "outside.traj"
    outside.write_bytes(b"outside")
    with pytest.raises(EvidenceError, match="escapes"):
        evidence.linked_path(str(outside), log, register=True)
    link = log.parent / "outside.traj"
    link.symlink_to(outside)
    with pytest.raises(EvidenceError, match="escapes"):
        evidence.linked_path("outside.traj", log, register=True)
    unrelated = log.parent / "unrelated.json"
    unrelated.write_text("{}")
    with pytest.raises(EvidenceError, match="originate in retained evidence"):
        evidence.linked_path("restart.traj", unrelated, register=True)


def test_registration_order_does_not_change_fingerprint(tmp_path):
    first, log, _ = evidence_with_log(tmp_path)
    second = Evidence(resolve_submission(json.dumps(first.manifest), tmp_path))
    (log.parent / "another.traj").write_bytes(b"another")
    for name in ("restart.traj", "another.traj"):
        first.linked_path(name, log, register=True)
    for name in ("another.traj", "restart.traj"):
        second.linked_path(name, log, register=True)
    assert first.fingerprint() == second.fingerprint()
