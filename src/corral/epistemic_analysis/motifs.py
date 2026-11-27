"""Direct temporal questions for the registered graph patterns."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .epistemic_prompts import (
    MOTIF_CRITERIA,
    RELATION_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
    motif_instructions,
)
from .patterns import DESCRIPTIVE_ANTIPATTERNS
from .patterns import MOTIFS as GRAPH_MOTIFS
from .patterns import Motif as GraphMotif

__all__ = [
    "DESCRIPTIVE_ANTIPATTERNS",
    "MOTIFS",
    "RELATION_DESCRIPTIONS",
    "STATUS_DESCRIPTIONS",
    "Motif",
    "motif_questions",
    "select_motifs",
]


@dataclass(frozen=True)
class Motif(GraphMotif):
    criteria: str


MOTIFS = tuple(
    Motif(**asdict(motif), criteria=MOTIF_CRITERIA[motif.name])
    for motif in GRAPH_MOTIFS
)


def select_motifs(names):
    """Preserve registry order for requests, answers, and stored reports."""
    if names is None:
        return MOTIFS
    if isinstance(names, str):
        names = [names]
    requested = set(names)
    if not requested or len(requested) != len(names):
        raise ValueError("Select at least one motif and specify each name only once")
    unknown = requested.difference(m.name for m in MOTIFS)
    if unknown:
        raise ValueError(f"Unknown motifs: {', '.join(sorted(unknown))}")
    return tuple(m for m in MOTIFS if m.name in requested)


def motif_questions(motifs=MOTIFS) -> list[dict]:
    return [
        {
            "type": "choice",
            "name": motif.name,
            "instructions": motif_instructions(**asdict(motif)),
            "choices": [
                {"value": status, "description": description}
                for status, description in STATUS_DESCRIPTIONS.items()
            ],
        }
        for motif in motifs
    ]
