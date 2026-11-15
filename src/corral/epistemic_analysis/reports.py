"""Public graph feedback records shared by annotations and agent sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from corral.core.messages import EMPTY_RESPONSE_REMINDER, public_message

from .motifs import DESCRIPTIVE_ANTIPATTERNS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPORT_MESSAGE_NAME = "epistemic_report"
SUBMISSION_REVIEW_MESSAGE_NAME = "epistemic_submission_review"


def scientific_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Detach observable messages, excluding scaffolding and prior reports."""
    return [
        public
        for message in messages
        if message.get("role") in {"user", "assistant", "tool", "function"}
        and message.get("name")
        not in {
            REPORT_MESSAGE_NAME,
            SUBMISSION_REVIEW_MESSAGE_NAME,
            EMPTY_RESPONSE_REMINDER,
        }
        and (public := public_message(message)) is not None
    ]


@dataclass(frozen=True)
class MotifAssessment:
    name: str
    kind: str
    family: str
    status: str
    confidence: float | None
    probabilities: dict[str, float]
    refused: bool = False
    explanation: str | None = None
    # Historical reports may contain citations. New explanations use prose only.
    evidence_message_ids: tuple[int, ...] = ()
    explanation_supported: bool | None = None
    selection_probability: float | None = None
    selection_threshold: float | None = None
    selected: bool | None = None

    @property
    def feedback_candidate(self) -> bool:
        """Keep the provider's temporal status separate from threshold selection."""
        return not self.refused and (
            self.selected if self.selected is not None else self.status == "active"
        )


@dataclass(frozen=True)
class EpistemicReport:
    model: str
    assessed_at: str
    message_count: int
    trajectory_sha256: str
    assessments: tuple[MotifAssessment, ...]
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    raw_explanation_response: dict[str, Any] | None = None
    selection_policy: dict[str, Any] | None = None
    # All per-pattern responses, in motif order. The singular field above remains
    # populated for single-request reports and readable in historical reports.
    raw_explanation_responses: tuple[dict[str, Any], ...] = ()
    graph: dict[str, Any] | None = None
    timings_s: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for offline storage or analysis."""
        return asdict(self)

    @property
    def present_antipatterns(self) -> tuple[MotifAssessment, ...]:
        """Selected unconstructive graph matches eligible for feedback.

        New reports derive selection from graph counts. Historical records keep
        their saved status/support gates; descriptive metrics never hold submission.
        """
        return tuple(
            assessment
            for assessment in self.assessments
            if assessment.kind == "antipattern"
            and assessment.name not in DESCRIPTIVE_ANTIPATTERNS
            and assessment.status == "active"
            and assessment.feedback_candidate
            and assessment.explanation_supported is True
            and assessment.explanation
        )

    def format_for_agent(self) -> str:
        """Show only evidenced unconstructive patterns; retain all in storage."""
        present = self.present_antipatterns
        if not present:
            return "No present unconstructive epistemic patterns detected."
        lines = [
            "Graph-matched unconstructive epistemic patterns (check against the observations):"
        ]
        lines.extend(f"- {a.name}: {a.explanation}" for a in present)
        if self.graph is not None:
            # The actor can inspect the annotated relationships behind the
            # signals without receiving raw provider requests or private data.
            lines.append("Annotated graph (fallible labels on the visible trajectory):")
            lines.extend(
                f"{n['node_id']} [{n['type']}, message {n['time']}]: {n['text']}"
                for n in self.graph["nodes"]
            )
            lines.extend(
                f"{e['src']} --{e['relation']}--> {e['dst']}"
                for e in self.graph["edges"]
            )
        return "\n".join(lines)
