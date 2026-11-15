"""Acceptance changes workflow state, preserving labels and genuine failures."""

import json
import runpy
from pathlib import Path

import pytest

from corral.epistemic_analysis.storage import accepted_annotation


@pytest.mark.parametrize("status", ["present", "absent", "uncertain"])
def test_user_acceptance_preserves_the_model_label(status):
    original = {"state": "requires_review", "status": status, "record_hash": "sealed"}
    accepted = accepted_annotation(original)
    assert accepted["state"] == "complete"
    assert accepted["status"] == status
    assert accepted["source_record_hash"] == "sealed"
    assert "record_hash" not in accepted
    assert original["state"] == "requires_review"
    assert accepted["acceptance"]["authority"] == "user"


@pytest.mark.parametrize(
    "state", ["failed", "partial_evidence", "pending", "running", "complete"]
)
def test_failures_and_unfinished_jobs_are_not_accepted(state):
    record = {"state": state, "error": "example"}
    assert accepted_annotation(record) is record


def test_existing_and_later_completions_are_accepted_without_changing_artifacts(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/accept_annotation_results.py"
    )
    accept = runpy.run_path(str(script))["accept_results"]
    completed = tmp_path / "completed/labels"
    completed.mkdir(parents=True)
    original = b'{"state":"requires_review","labels":{"check":"original-key"}}'
    (completed / "flagged.json").write_bytes(original)
    failure = b'{"state":"failed","error":"quote mismatch"}'
    (completed / "failed.json").write_bytes(failure)
    artifact = tmp_path / "artifacts/label.json"
    artifact.parent.mkdir()
    artifact.write_bytes(original)
    assert accept(tmp_path) == 1
    assert json.loads((completed / "flagged.json").read_text())["state"] == "complete"
    assert (completed / "failed.json").read_bytes() == failure
    assert artifact.read_bytes() == original
    assert (
        next(
            (tmp_path / "acceptance-history/labels").glob("flagged.*.json")
        ).read_bytes()
        == original
    )
    assert accept(tmp_path) == 0
    (completed / "later.json").write_bytes(original)
    assert accept(tmp_path) == 1
