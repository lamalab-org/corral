"""Apply the user's blanket acceptance of automatic annotation review flags."""

from __future__ import annotations

ACCEPTANCE_POLICY = "user-accepted-automatic-review-flags-20261110"


def accepted_annotation(record):
    """Accept finished flagged annotations; retain predictions and real failures."""
    if record.get("state") != "requires_review":
        return record
    result = {
        **record,
        "state": "complete",
        "acceptance": {
            "policy": ACCEPTANCE_POLICY,
            "authority": "user",
            "decision": "accepted_as_correct",
            "original_state": "requires_review",
        },
    }
    # A published projection is not the original sealed model record.
    if "record_hash" in result:
        result["source_record_hash"] = result.pop("record_hash")
    return result
