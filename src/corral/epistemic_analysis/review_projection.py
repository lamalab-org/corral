"""Standard-library projections of separately sealed second-pass judgments."""

from __future__ import annotations

import copy
import hashlib
import json


def value_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def project_review(original, review):
    """Attach a review; promote confirmed occurrences without erasing provenance."""
    result = copy.deepcopy(original)
    result["second_pass_review"] = review
    if review.get("state") != "complete" or review.get("review_status") != "present":
        return result
    result["original_assessment"] = copy.deepcopy(original)
    if "record_hash" in result:
        result["source_record_hash"] = result.pop("record_hash")
    result.update(
        status="present",
        state="complete",
        decision_source="independent_fabrication_review",
        rationale=review["rationale"],
        support=review["support"],
        evidence=review["support"],
        provenance=review.get("provider", {}),
        request_count=len(review.get("exchanges", [])),
        coverage=review.get("coverage"),
    )
    for key in (
        "probabilities",
        "confidence",
        "threshold",
        "maximum_option_probability",
        "choice",
        "evidence_record",
        "elapsed_s",
        "trigger_slices",
    ):
        result.pop(key, None)
    return result


def apply_review(assessment, review, messages, units):
    """Fail closed on stale/tampered reviews and retain all first-pass details."""
    if review.get("record_hash") != value_hash(
        {k: v for k, v in review.items() if k != "record_hash"}
    ):
        raise ValueError("Second-pass review record changed")
    if review.get("trace_messages_hash") != value_hash(messages):
        raise ValueError("Second-pass review belongs to a different source trace")
    if review.get("check") != "fabricated_evidence":
        raise ValueError("Unexpected second-pass check")
    if review.get("state") not in {"complete", "failed"} or (
        review.get("state") == "complete" and review.get("review_status") is None
    ):
        return assessment
    for support in review.get("support", []):
        i, a, b = (support.get(k) for k in ("msg_idx", "start", "end"))
        if any(type(v) is not int for v in (i, a, b)) or not 0 <= i < len(messages):
            raise ValueError("Invalid second-pass source support")
        if (
            not 0 <= a < b <= len(messages[i]["content"])
            or messages[i]["content"][a:b] != support.get("quote")
            or messages[i]["role"] != support.get("role")
        ):
            raise ValueError("Second-pass support differs from source")
    if review["review_status"] == "present" and not review.get("support"):
        raise ValueError("Positive second-pass review lacks evidence")
    result = copy.deepcopy(assessment)

    def merge(checks):
        checks = copy.deepcopy(checks)
        original = next((c for c in checks if c["name"] == "fabricated_evidence"), None)
        if original is None:
            original = {
                "name": "fabricated_evidence",
                "status": "not_run",
                "state": "pending",
            }
        return [c for c in checks if c["name"] != "fabricated_evidence"] + [
            project_review(original, review)
        ]

    result["checks"] = merge(result.get("checks", []))
    # This review is scoped to the entire conversation. Do not spread its
    # positive label over unrelated actors' or window-local review units.
    if len(units) == 1 and units[0]["indices"] == list(range(len(messages))):
        existing = result.get("units", [{}]) or [{}]
        result["units"] = [
            {
                **existing[0],
                "unit_id": units[0]["id"],
                "checks": merge(existing[0].get("checks", [])),
            }
        ]
    return result
