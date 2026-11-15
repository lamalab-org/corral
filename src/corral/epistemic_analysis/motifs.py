"""Names and explanations for deterministically matched graph motifs."""

from __future__ import annotations

from dataclasses import dataclass

from corral.epistemic_analysis.epistemic_prompts import (
    MOTIF_GRAPH_DESCRIPTIONS,
    RELATION_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
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
]


@dataclass(frozen=True)
class Motif:
    name: str
    kind: str
    family: str
    description: str
    graph_description: str


MOTIFS = tuple(
    Motif(
        name=name,
        kind=kind,
        family=next(family for family, members in families.items() if name in members),
        description=description,
        graph_description=MOTIF_GRAPH_DESCRIPTIONS[name],
    )
    for kind, descriptions, families in (
        ("productive", SUBGRAPH_DESCRIPTIONS, SUBGRAPH_FAMILIES),
        ("antipattern", ANTIPATTERN_DESCRIPTIONS, ANTIPATTERN_FAMILIES),
    )
    for name, description in descriptions.items()
)
