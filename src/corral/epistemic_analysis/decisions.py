"""Standalone, prefix-only epistemic annotation via OpenAI's Decisions API.

This module accepts ordinary message dictionaries and has no agent/session
dependency. It classifies final motifs directly, without extracting a graph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
from openai import APIConnectionError, APIStatusError

from corral.core.messages import EMPTY_RESPONSE_REMINDER, public_message
from corral.epistemic_analysis.decision_thresholds import resolve_pattern_thresholds
from corral.epistemic_analysis.epistemic_prompts import (
    ANNOTATION_RULES,
    MOTIF_ANNOTATION_INPUT,
)
from corral.epistemic_analysis.explanations import (
    ExplanationStreamInterrupted,
    explain_patterns,
)
from corral.epistemic_analysis.motifs import (
    DESCRIPTIVE_ANTIPATTERNS,
    MOTIFS,
    RELATION_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
    motif_questions,
)
from corral.epistemic_analysis.ontology import NODE_CRITERIA

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from openai import AsyncOpenAI

DEFAULT_DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_EXPLANATION_CONCURRENCY = 8
REPORT_MESSAGE_NAME = "epistemic_report"
SUBMISSION_REVIEW_MESSAGE_NAME = "epistemic_submission_review"


class DecisionsAnnotationError(RuntimeError):
    """The endpoint returned an incomplete or invalid motif assessment."""


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for offline storage or analysis."""
        return asdict(self)

    @property
    def present_antipatterns(self) -> tuple[MotifAssessment, ...]:
        """Currently active, evidenced defects eligible to hold submission.

        Thresholds select candidates for explanation, including historical
        occurrences. Explanation support cannot override a nonactive status or
        turn a descriptive trace property into a scientific defect.
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
            "Present unconstructive epistemic patterns (check against the observations):"
        ]
        lines.extend(f"- {a.name}: {a.explanation}" for a in present)
        return "\n".join(lines)


def _probability(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _parse_assessments(payload: Mapping[str, Any]) -> tuple[MotifAssessment, ...]:
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != len(MOTIFS):
        raise DecisionsAnnotationError(
            "Expected exactly one Decisions answer per motif"
        )
    assessments = []
    for motif, answer in zip(MOTIFS, answers, strict=True):
        if not isinstance(answer, dict) or answer.get("name") != motif.name:
            raise DecisionsAnnotationError(
                f"Missing or out-of-order answer: {motif.name}"
            )
        if answer.get("type") == "refusal":
            assessments.append(
                MotifAssessment(
                    motif.name, motif.kind, motif.family, "uncertain", None, {}, True
                )
            )
            continue
        probabilities = answer.get("probabilities")
        if (
            answer.get("type") != "choice"
            or not isinstance(answer.get("choice"), str)
            or answer["choice"] not in STATUS_DESCRIPTIONS
            or not _probability(answer.get("confidence"))
            or not isinstance(probabilities, list)
            or len(probabilities) != len(STATUS_DESCRIPTIONS)
        ):
            raise DecisionsAnnotationError(f"Invalid choice answer: {motif.name}")
        distribution = {}
        for entry in probabilities:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("value"), str)
                or entry["value"] not in STATUS_DESCRIPTIONS
                or entry["value"] in distribution
                or not _probability(entry.get("probability"))
            ):
                raise DecisionsAnnotationError(f"Invalid probabilities: {motif.name}")
            distribution[entry["value"]] = float(entry["probability"])
        if not math.isclose(sum(distribution.values()), 1, abs_tol=0.01):
            raise DecisionsAnnotationError(
                f"Probabilities do not sum to one: {motif.name}"
            )
        assessments.append(
            MotifAssessment(
                motif.name,
                motif.kind,
                motif.family,
                answer["choice"],
                float(answer["confidence"]),
                distribution,
            )
        )
    return tuple(assessments)


class DecisionsAnnotator:
    """Classify motifs, then verify selected candidates concurrently.

    Credentials default to OPENAI_API_KEY. A supplied AsyncOpenAI client remains
    caller-owned; otherwise a client is opened and closed for each annotation.
    SDK retries and timeout are bounded. Errors propagate rather than presenting
    a failed assessment as 'absent'. No conversation is silently truncated.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_DECISIONS_MODEL,
        explanation_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
        include_explanations: bool = True,
        stream_explanations: bool = False,
        explanation_concurrency: int = DEFAULT_EXPLANATION_CONCURRENCY,
        pattern_thresholds: str | Mapping[str, float] | None = None,
    ) -> None:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be finite and positive")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a nonnegative integer")
        if (
            isinstance(explanation_concurrency, bool)
            or not isinstance(explanation_concurrency, int)
            or explanation_concurrency < 1
        ):
            raise ValueError("explanation_concurrency must be a positive integer")
        self.model = model
        self.explanation_model = explanation_model or model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.client = client
        self.include_explanations = include_explanations
        self.stream_explanations = stream_explanations
        self.explanation_concurrency = explanation_concurrency
        self.pattern_thresholds, self.selection_policy = resolve_pattern_thresholds(
            pattern_thresholds
        )

    async def _request(
        self, client: AsyncOpenAI, body: dict[str, Any]
    ) -> dict[str, Any]:
        decisions = getattr(client, "decisions", None)
        if decisions is not None:
            response = await decisions.create(**body)
            payload = response.model_dump(mode="json")
        else:
            # Generic SDK transport also supports releases predating Decisions.
            payload = await client.post("/decisions", body=body, cast_to=dict[str, Any])
        if not isinstance(payload, dict):
            raise DecisionsAnnotationError("Decisions response must be an object")
        return payload

    async def _explain_candidates(
        self,
        client: AsyncOpenAI,
        candidate_names: set[str],
        trajectory: list[dict[str, Any]],
        task: str | list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
        # The limit belongs to this assessment: shared annotators keep concurrent
        # trials independent, and can be reused across event loops.
        gate = asyncio.Semaphore(self.explanation_concurrency)

        async def explain(motif):
            async with gate:
                return await explain_patterns(
                    client,
                    model=self.explanation_model,
                    motifs=[motif],
                    trajectory=trajectory,
                    task=task,
                    **(
                        {
                            "stream": True,
                            "timeout_s": self.timeout_s,
                            "max_retries": self.max_retries,
                        }
                        if self.stream_explanations
                        else {}
                    ),
                )

        pending = [
            asyncio.create_task(explain(motif))
            for motif in MOTIFS
            if motif.name in candidate_names
        ]
        try:
            results = await asyncio.gather(*pending)
        except BaseException:
            # gather does not cancel siblings on a request/validation failure.
            # Drain them before the annotation's client can be closed, including
            # when the caller cancels a trial or an explanation stream fails.
            for request in pending:
                request.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        explanations = {
            name: item for items, _ in results for name, item in items.items()
        }
        return explanations, tuple(payload for _, payload in results)

    async def _assess(
        self,
        client: AsyncOpenAI,
        body: dict[str, Any],
        trajectory: list[dict[str, Any]],
        task: str | list[dict[str, Any]],
    ) -> tuple[dict[str, Any], tuple[MotifAssessment, ...], tuple[dict[str, Any], ...]]:
        try:
            payload = await self._request(client, body)
        except (
            APIConnectionError,
            APIStatusError,
            httpx.TransportError,
            TimeoutError,
        ) as exc:
            raise DecisionsAnnotationError(
                f"Epistemic classification request failed: {type(exc).__name__} "
                f"(model={self.model}, timeout_s={self.timeout_s}, "
                f"max_retries={self.max_retries})"
            ) from None
        assessments = tuple(self._select(a) for a in _parse_assessments(payload))
        candidate_names = {a.name for a in assessments if a.feedback_candidate}
        explanation_payloads = ()
        if candidate_names and self.include_explanations:
            try:
                explanations, explanation_payloads = await self._explain_candidates(
                    client, candidate_names, trajectory, task
                )
            except ValueError as exc:
                raise DecisionsAnnotationError(str(exc)) from exc
            except (
                APIConnectionError,
                APIStatusError,
                httpx.TransportError,
                TimeoutError,
                ExplanationStreamInterrupted,
            ) as exc:
                raise DecisionsAnnotationError(
                    f"Epistemic explanation request failed: {type(exc).__name__} "
                    f"(model={self.explanation_model}, timeout_s={self.timeout_s}, "
                    f"max_retries={self.max_retries}, stream={self.stream_explanations})"
                ) from None
            assessments = tuple(
                replace(
                    a,
                    explanation=explanations[a.name]["explanation"],
                    explanation_supported=explanations[a.name]["supported"],
                )
                if a.name in explanations
                else a
                for a in assessments
            )
        return payload, assessments, explanation_payloads

    def _select(self, assessment: MotifAssessment) -> MotifAssessment:
        if self.pattern_thresholds is None:
            return replace(
                assessment,
                selected=assessment.status == "active" and not assessment.refused,
            )
        threshold = self.pattern_thresholds[assessment.name]
        probability = (
            None
            if assessment.refused
            else round(
                assessment.probabilities["active"]
                + assessment.probabilities["resolved"],
                12,
            )
        )
        return replace(
            assessment,
            selection_probability=probability,
            selection_threshold=threshold,
            selected=probability is not None and probability >= threshold,
        )

    async def annotate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        task: str | list[dict[str, Any]] = "",
    ) -> EpistemicReport:
        """Assess the full prefix provided by the caller, including the first turn."""
        trajectory = scientific_messages(messages)
        serialized = json.dumps(
            {"task": task, "messages": trajectory},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        # Serialize tool calls/results as text: Decisions does not execute tools.
        body = {
            "model": self.model,
            "input": MOTIF_ANNOTATION_INPUT.format(
                rules=ANNOTATION_RULES,
                nodes=json.dumps(NODE_CRITERIA, ensure_ascii=False),
                relations=json.dumps(RELATION_DESCRIPTIONS, ensure_ascii=False),
                trajectory=serialized,
            ),
            "questions": motif_questions(),
        }
        if self.client is not None:
            payload, assessments, explanation_payloads = await self._assess(
                self.client, body, trajectory, task
            )
        else:
            from openai import AsyncOpenAI

            async with AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=self.max_retries,
            ) as client:
                payload, assessments, explanation_payloads = await self._assess(
                    client, body, trajectory, task
                )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "llm_calls": 0,
        }
        for response in (payload, *explanation_payloads):
            raw_usage = response.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raise DecisionsAnnotationError("Invalid annotation usage")
            details = raw_usage.get("output_tokens_details") or {}
            if not isinstance(details, dict):
                raise DecisionsAnnotationError(
                    "Invalid annotation output token details"
                )
            usage["input_tokens"] += raw_usage.get("input_tokens", 0)
            usage["output_tokens"] += raw_usage.get("output_tokens", 0)
            usage["reasoning_tokens"] += details.get("reasoning_tokens", 0)
            usage["llm_calls"] += 1
        return EpistemicReport(
            model=payload.get("model", self.model),
            assessed_at=datetime.now(timezone.utc).isoformat(),
            message_count=len(trajectory),
            trajectory_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
            assessments=assessments,
            usage=usage,
            raw_response=payload,
            raw_explanation_response=(
                explanation_payloads[0] if len(explanation_payloads) == 1 else None
            ),
            raw_explanation_responses=explanation_payloads,
            # Reports are detached snapshots, including nested threshold maps.
            selection_policy=json.loads(json.dumps(self.selection_policy)),
        )


__all__ = [
    "DecisionsAnnotationError",
    "DecisionsAnnotator",
    "EpistemicReport",
    "MotifAssessment",
]
