"""Global evidence journal and append-only JSONL trace support."""

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeStatus,
    Observation,
)


class JournalClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    disposition: Literal["supported", "contradicted"]
    evidence_node_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class JournalHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    status: Literal["open", "supported", "contradicted"] = "open"
    evidence_node_ids: list[str] = Field(default_factory=list)


class ExperimentFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    hypothesis: str
    errors: list[str] = Field(default_factory=list)


class ResearchJournal:
    """All reliable evidence, independent of which tree branch is best."""

    def __init__(self) -> None:
        self.observations: list[tuple[str, Observation]] = []
        self.claims: list[JournalClaim] = []
        self.hypotheses: list[JournalHypothesis] = []
        self.unresolved_questions: list[str] = []
        self.failed_experiments: list[ExperimentFailure] = []
        self.visual_feedback: list[dict[str, Any]] = []

    def register_hypothesis(self, text: str) -> None:
        if text and all(item.text != text for item in self.hypotheses):
            self.hypotheses.append(JournalHypothesis(text=text))

    def integrate(self, node: ExperimentNode, minimum_validity: float = 0.45) -> None:
        """Retain observations, but promote claims only from successful nodes."""
        for observation in node.observations:
            if observation.success:
                self.observations.append((node.id, observation))

        errors = [
            observation.error or "Unknown tool failure"
            for observation in node.observations
            if not observation.success
        ]
        if node.status in {NodeStatus.FAILED, NodeStatus.INVALID}:
            if not errors and node.evaluation is not None:
                errors.append(node.evaluation.reason)
            self.failed_experiments.append(
                ExperimentFailure(
                    node_id=node.id, hypothesis=node.hypothesis, errors=errors
                )
            )

        evaluation = node.evaluation
        if (
            node.status != NodeStatus.SUCCESSFUL
            or evaluation is None
            or evaluation.validity < minimum_validity
        ):
            return

        if node.visual_artifacts or evaluation.visual_feedback:
            self.visual_feedback.append(
                {
                    "node_id": node.id,
                    "artifacts": list(node.visual_artifacts),
                    "feedback": list(evaluation.visual_feedback),
                }
            )

        confidence = evaluation.evidence_strength
        supported = [*evaluation.conclusions, *evaluation.supported_claims]
        for text in dict.fromkeys(item for item in supported if item):
            self._add_claim(text, "supported", node.id, confidence)
        for text in dict.fromkeys(
            item for item in evaluation.contradicted_claims if item
        ):
            self._add_claim(text, "contradicted", node.id, confidence)
        for question in evaluation.open_questions:
            if question and question not in self.unresolved_questions:
                self.unresolved_questions.append(question)

    def _add_claim(
        self,
        text: str,
        disposition: Literal["supported", "contradicted"],
        node_id: str,
        confidence: float,
    ) -> None:
        for claim in self.claims:
            if claim.text == text and claim.disposition == disposition:
                if node_id not in claim.evidence_node_ids:
                    claim.evidence_node_ids.append(node_id)
                claim.confidence = max(claim.confidence, confidence)
                return
        self.claims.append(
            JournalClaim(
                text=text,
                disposition=disposition,
                evidence_node_ids=[node_id],
                confidence=confidence,
            )
        )
        for hypothesis in self.hypotheses:
            if hypothesis.text == text:
                hypothesis.status = disposition
                if node_id not in hypothesis.evidence_node_ids:
                    hypothesis.evidence_node_ids.append(node_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": [
                {"node_id": node_id, **observation.model_dump(mode="json")}
                for node_id, observation in self.observations
            ],
            "claims": [claim.model_dump(mode="json") for claim in self.claims],
            "hypotheses": [
                hypothesis.model_dump(mode="json") for hypothesis in self.hypotheses
            ],
            "unresolved_questions": self.unresolved_questions,
            "failed_experiments": [
                failure.model_dump(mode="json") for failure in self.failed_experiments
            ],
            "visual_feedback": self.visual_feedback,
        }

    def context(self, max_chars: int = 30_000) -> str:
        rendered = json.dumps(self.as_dict(), indent=2, ensure_ascii=False)
        if len(rendered) <= max_chars:
            return rendered
        keep = max_chars // 2
        return rendered[:keep] + "\n... journal truncated ...\n" + rendered[-keep:]


class JSONLTraceWriter:
    """Append manager events as immediately durable JSON lines."""

    def __init__(self, path: Path | None, task_id: str) -> None:
        self.path = self._resolve(path, task_id) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve(path: Path, task_id: str) -> Path:
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        if path.suffix.lower() == ".jsonl":
            return path
        return path / f"{safe_task_id}.jsonl"

    def write(self, event: str, payload: Any) -> None:
        if self.path is None:
            return
        record = {"event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
