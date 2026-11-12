"""Shared cheating issue definitions for both trace audit pilots."""

from corral.epistemic_analysis.epistemic_prompts import TRACE_CHECK_DEFINITIONS

ISSUE_TYPES = (
    "cheating_consideration",
    "cheating_attempt",
    "answer_contamination",
)

CHEATING_ISSUE_DEFINITIONS = "\n\n".join(
    f"{name}: {TRACE_CHECK_DEFINITIONS[name]}" for name in ISSUE_TYPES
)
