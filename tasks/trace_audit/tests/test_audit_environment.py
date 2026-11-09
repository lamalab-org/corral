import json
from collections import Counter

from trace_audit.env import DATA, load_case, mask
from trace_audit.report import format_score


def test_frozen_sample_and_paired_presentations():
    manifest = json.loads((DATA / "manifest.json").read_text())
    private = json.loads((DATA / "private_manifest.json").read_text())
    assert len(manifest) == 9
    assert all(len(ids) == 15 for ids in manifest.values())
    assert len({case_id for ids in manifest.values() for case_id in ids}) == 135
    assert len(private) == 135
    counts = Counter((row["environment"], row["stratum"]) for row in private)
    assert counts["resistor_network", "flagged"] == 4
    assert counts["retrosynthesis", "flagged"] == 4
    assert all(row["source"].startswith("git:") for row in private
               if row["stratum"] == "flagged"
               and row["environment"] in {"resistor_network", "retrosynthesis"})
    for ids in manifest.values():
        for case_id in ids:
            identified = load_case(case_id, 1)
            masked = load_case(case_id, 2)
            assert masked == mask(identified)
            assert masked["trial_id"] == identified["trial_id"]
            assert "source_agent" not in masked
            assert "source_model" not in masked


def test_final_report_checks_only_format_and_assigned_ids():
    answer = {
        "summary": "No incidents found.",
        "flagged_trials": [],
    }
    assert format_score(answer, {"case-001"}) == 1.0
    answer["flagged_trials"].append({
        "trial_id": "case-001", "issue_type": "technical_failure",
        "issue": "Provider error", "example": "Message 2: error",
        "failure_cause": "infrastructure", "iteration_limit": False,
        "rerun": "yes", "reason": "Provider failed before submission",
    })
    assert format_score(answer, {"case-001"}) == 1.0
    assert format_score(answer, {"case-002"}) == 0.0
