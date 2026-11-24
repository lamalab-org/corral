"""Direct temporal classification of the graph pipeline's final motifs.

Names, descriptions, and families come from the extracted analyze.py registry.
The extra criteria express relationships in observable conversation terms; they
do not reconstruct nodes/edges or claim to reproduce deterministic graph counts.
Structural graph candidates remain useful historical measurements. The direct
reviewer additionally checks current relevance, selective evaluation, and repair;
an old structural match alone does not establish an active behavioral defect.
"""

from __future__ import annotations

from dataclasses import dataclass

from corral.epistemic_analysis.epistemic_prompts import (
    MOTIF_CRITERIA,
    MOTIF_GRAPH_DESCRIPTIONS,
    RELATION_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
    motif_instructions,
)
from corral.epistemic_analysis.patterns import (
    ANTIPATTERN_DESCRIPTIONS,
    ANTIPATTERN_FAMILIES,
    SUBGRAPH_DESCRIPTIONS,
    SUBGRAPH_FAMILIES,
)

# Preserve this trace-wide measurement without treating supported stability as
# a behavioral defect or requiring the actor to manufacture a belief revision.
DESCRIPTIVE_ANTIPATTERNS = frozenset({"fixed_belief_trace"})

__all__ = [
    "DESCRIPTIVE_ANTIPATTERNS",
    "MOTIFS",
    "RELATION_DESCRIPTIONS",
    "STATUS_DESCRIPTIONS",
    "Motif",
    "motif_questions",
]


@dataclass(frozen=True)
class Motif:
    name: str
    kind: str
    family: str
    description: str
    graph_description: str
    criteria: str


MOTIFS = tuple(
    Motif(
        name=name,
        kind=kind,
        family=next(family for family, members in families.items() if name in members),
        description=description,
        graph_description=MOTIF_GRAPH_DESCRIPTIONS[name],
        criteria=MOTIF_CRITERIA[name],
    )
    for kind, descriptions, families in (
        ("productive", SUBGRAPH_DESCRIPTIONS, SUBGRAPH_FAMILIES),
        ("antipattern", ANTIPATTERN_DESCRIPTIONS, ANTIPATTERN_FAMILIES),
    )
    for name, description in descriptions.items()
)


def motif_questions() -> list[dict]:
    """Return all 18 named Decisions choice questions in registry order."""
    return [
        {
            "type": "choice",
            "name": motif.name,
            "instructions": motif_instructions(
                name=motif.name,
                kind=motif.kind,
                family=motif.family,
                graph_description=motif.graph_description,
                description=motif.description,
                criteria=motif.criteria,
            ),
            "choices": [
                {"value": status, "description": description}
                for status, description in STATUS_DESCRIPTIONS.items()
            ],
        }
        for motif in MOTIFS
    ]
