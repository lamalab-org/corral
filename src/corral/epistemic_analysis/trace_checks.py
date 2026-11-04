"""Behavioral checks evaluated separately on the complete normalized trace."""

from __future__ import annotations

from corral.epistemic_analysis.epistemic_prompts import (
    TRACE_CHECK_DEFINITIONS,
    TRACE_CHECK_NAMES,
    TRACE_CHECK_RULES,
    TRACE_CHECK_STATUSES,
    trace_check_prompt,
)

TRACE_CHECK_VERSION = "full-trace-per-check-v4-human-review"

__all__ = [
    "TRACE_CHECK_DEFINITIONS",
    "TRACE_CHECK_NAMES",
    "TRACE_CHECK_RULES",
    "TRACE_CHECK_STATUSES",
    "TRACE_CHECK_VERSION",
    "trace_check_prompt",
    "trace_check_state",
    "validate_trace_checks",
]


def trace_check_state(trace, nodes, edges, name):
    """Supply one definition and the complete trace with original source mappings."""
    return {
        "scope": "full_trace",
        "check": {
            "name": name,
            "definition": TRACE_CHECK_DEFINITIONS[name],
            "allowed_statuses": [
                status
                for status in TRACE_CHECK_STATUSES
                if status != "not_applicable" or name == "insufficient_validation"
            ],
        },
        "messages": {
            f"m{i}": {**message, "msg_idx": i, "source": trace.message_sources[i]}
            for i, message in enumerate(trace.messages)
        },
        "capture": {
            "completion_status": trace.completion_status,
            "reconstruction_status": trace.reconstruction_status,
        },
        "nodes": nodes,
        "edges": edges,
    }


def validate_trace_checks(checks, messages, *, expected_names=TRACE_CHECK_NAMES):
    """Fail closed on missing labels or fabricated evidence; keep exact offsets."""
    if not isinstance(checks, list) or len(checks) != len(expected_names):
        raise ValueError(
            "Trace checks require exactly one result for each requested definition"
        )
    by_name = {}
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Trace check must be an object")
        name, status = check.get("name"), check.get("status")
        if not isinstance(name, str) or name not in expected_names or name in by_name:
            raise ValueError("Unknown or duplicate trace check")
        if status not in TRACE_CHECK_STATUSES or (
            status == "not_applicable" and name != "insufficient_validation"
        ):
            raise ValueError(f"Invalid trace check status: {name}")
        support = check.get("support", [])
        if not isinstance(support, list):
            raise ValueError(f"Invalid trace check support: {name}")
        mapped = []
        for item in support:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid trace check support: {name}")
            idx, quote = item.get("msg_idx"), item.get("quote")
            if (
                not isinstance(idx, int)
                or isinstance(idx, bool)
                or not 0 <= idx < len(messages)
                or not isinstance(quote, str)
                or not quote
            ):
                raise ValueError(f"Invalid trace check support: {name}")
            content = messages[idx]["content"]
            start = item.get("start", content.find(quote))
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or start < 0
                or content[start : start + len(quote)] != quote
                or item.get("end", start + len(quote)) != start + len(quote)
            ):
                raise ValueError(f"Trace check quote is not exact: {name}")
            mapped.append(
                {
                    "msg_idx": idx,
                    "quote": quote,
                    "start": start,
                    "end": start + len(quote),
                }
            )
        by_name[name] = {**check, "support": mapped}
    return [by_name[name] for name in expected_names]
