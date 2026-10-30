"""Ontology for epistemic trace analysis."""

from __future__ import annotations

from corral.epistemic_analysis.epistemic_prompts import (
    NODE_CRITERIA,
    PASS_A_INSTRUCTIONS,
    PASS_A_SYSTEM,
    PASS_B_INSTRUCTIONS,
    PASS_B_SYSTEM,
)

# Preserve the existing ontology imports while prompt text lives in one module.
__all__ = [
    "ALLOWED_EDGE_TYPE_COMBOS",
    "EDGE_RELATIONS",
    "EDGE_RELATIONS_SET",
    "NODE_CRITERIA",
    "NODE_TYPES",
    "NODE_TYPES_SET",
    "PASS_A_INSTRUCTIONS",
    "PASS_A_SYSTEM",
    "PASS_B_INSTRUCTIONS",
    "PASS_B_SYSTEM",
]

NODE_TYPES = ["H", "T", "E", "J", "C", "N", "F"]


NODE_TYPES_SET = set(NODE_TYPES)


EDGE_RELATIONS = [
    "tests",
    "observes",
    "informs",
    "updates_to",
    "competes_with",
    "contradicts",
]


EDGE_RELATIONS_SET = set(EDGE_RELATIONS)


ALLOWED_EDGE_TYPE_COMBOS: set[tuple[str, str, str]] = {
    # tests (bidirectional)
    ("tests", "H", "T"),
    ("tests", "T", "H"),
    ("tests", "J", "T"),
    ("tests", "T", "J"),
    # observes
    ("observes", "T", "E"),
    # updates_to (same type, bidirectional)
    ("updates_to", "H", "H"),
    # competes_with (same type, bidirectional)
    ("competes_with", "H", "H"),
    # contradicts
    ("contradicts", "E", "H"),
    ("contradicts", "J", "H"),
    # informs
    ("informs", "E", "H"),
    ("informs", "E", "J"),
    ("informs", "E", "C"),
    ("informs", "J", "C"),
    ("informs", "J", "H"),
    ("informs", "J", "J"),
}
