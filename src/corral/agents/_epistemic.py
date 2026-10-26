"""Shared opt-in epistemic feedback for the native and ReAct session loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from corral.agents.base_agent import BaseAgent
from corral.core.action import with_submit_answer_tool
from corral.epistemic_analysis.decision_thresholds import CALIBRATED_THRESHOLD_POLICY
from corral.epistemic_analysis.decisions import (
    DEFAULT_DECISIONS_MODEL,
    DEFAULT_EXPLANATION_CONCURRENCY,
    REPORT_MESSAGE_NAME,
    SUBMISSION_REVIEW_MESSAGE_NAME,
    DecisionsAnnotator,
    EpistemicReport,
    scientific_messages,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.base_agent import _UsageAccumulator
    from corral.agents.session import AgentSession
    from corral.core.action import Action


class EpistemicSupport(BaseAgent):
    """Opt-in review without actor prompt overrides; trajectory state is per session."""

    def __init__(
        self,
        *,
        epistemic: bool = False,
        annotator: DecisionsAnnotator | None = None,
        actor_refusal_max_retries: int = 1,
        epistemic_model: str = DEFAULT_DECISIONS_MODEL,
        epistemic_explanation_model: str | None = None,
        epistemic_api_key: str | None = None,
        epistemic_base_url: str | None = None,
        epistemic_timeout_s: float = 120,
        epistemic_max_retries: int = 2,
        epistemic_stream_explanations: bool = False,
        epistemic_explanation_concurrency: int = DEFAULT_EXPLANATION_CONCURRENCY,
        epistemic_pattern_thresholds: str | Mapping[str, float] | None = (
            CALIBRATED_THRESHOLD_POLICY
        ),
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(actor_refusal_max_retries, bool)
            or not isinstance(actor_refusal_max_retries, int)
            or actor_refusal_max_retries < 0
        ):
            raise ValueError("actor_refusal_max_retries must be a non-negative integer")
        if not isinstance(epistemic, bool):
            raise TypeError("epistemic must be a boolean")
        super().__init__(**kwargs)
        self.epistemic = epistemic
        self.annotator = None
        if not epistemic:
            return
        self.actor_refusal_max_retries = actor_refusal_max_retries
        self.annotator = (
            annotator
            if annotator is not None
            else DecisionsAnnotator(
                model=epistemic_model,
                explanation_model=epistemic_explanation_model,
                api_key=epistemic_api_key,
                base_url=epistemic_base_url,
                timeout_s=epistemic_timeout_s,
                max_retries=epistemic_max_retries,
                stream_explanations=epistemic_stream_explanations,
                explanation_concurrency=epistemic_explanation_concurrency,
                pattern_thresholds=epistemic_pattern_thresholds,
            )
        )

    def _should_retry_refusal(self, consecutive_refusals: int) -> bool:
        return self.epistemic and consecutive_refusals <= self.actor_refusal_max_retries

    def _tools(self, session: AgentSession) -> list[dict[str, Any]]:
        """Keep the native tool descriptions when enabling epistemic review."""
        return with_submit_answer_tool(session.tools)

    def _text_only_feedback(self) -> str:
        return (
            "A plain-text answer does not finish the task. Call "
            "submit_answer with the complete answer, or call an "
            "available environment tool."
        )

    async def _before_turn(
        self,
        session: AgentSession,
        messages: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        usage: _UsageAccumulator,
    ) -> list[dict[str, Any]]:
        if not self.epistemic:
            return messages
        # The loop supplies canonical history plus local provider calls/results,
        # excluding initial prompts and few-shot demonstrations.
        observed = scientific_messages(trajectory)
        # Task facts and scaffolding alone are not prior agent behavior. Use the
        # trajectory rather than a local iteration counter so resumed runs still
        # receive feedback before their next turn.
        if not any(message.get("role") == "assistant" for message in observed):
            return messages
        await self._record_report(session, messages, observed, usage)
        return messages

    async def _record_report(
        self,
        session: AgentSession,
        messages: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        usage: _UsageAccumulator,
    ) -> EpistemicReport:
        assert self.annotator is not None
        report = await self.annotator.annotate(observed, task=session.prompt)
        model_calls = report.usage.get("llm_calls", 1)
        usage.add(report.usage, llm_calls=model_calls)
        feedback = {
            "role": "user",
            "name": REPORT_MESSAGE_NAME,
            "content": report.format_for_agent(),
        }
        await session.record_message(
            {**feedback, "metadata": {"epistemic_report": report.to_dict()}},
            usage=report.usage,
            llm_calls=model_calls,
        )
        messages.append(feedback)
        return report

    async def _review_submission(
        self,
        action: Action,
        *,
        session: AgentSession,
        messages: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        usage: _UsageAccumulator,
        turn_usage: Mapping[str, Any] | None,
    ) -> bool:
        """Persist and assess a proposal; only the calling loop can execute it."""
        answer = action.arguments["answer"].strip()
        proposal = {
            "role": "assistant",
            "name": "submission_proposal",
            "content": (
                (f"{action.content}\n\n" if action.content else "")
                + f"Proposed final answer (not yet submitted):\n{answer}"
            ),
        }
        # Persist the proposed answer before its assessment so citations refer
        # to an already recorded scientific message, even on annotation failure.
        await session.record_message(
            {
                **proposal,
                "metadata": {"submission_proposal": action.to_tool_call()},
            },
            usage=turn_usage,
        )
        messages.append(proposal)
        trajectory.append(proposal)
        report = await self._record_report(
            session, messages, scientific_messages(trajectory), usage
        )
        if report.present_antipatterns:
            feedback = {
                "role": "user",
                "name": SUBMISSION_REVIEW_MESSAGE_NAME,
                "content": (
                    "Submission held: the proposed answer has not been accepted. "
                    "Address the unconstructive patterns in the preceding report. "
                    "Validate the answer with a targeted experiment when useful, "
                    "interpret the observations, and revise unsupported claims. "
                    "If a pattern does not apply, explain why using concrete "
                    "evidence. Then call submit_answer again with the complete "
                    "answer and a fresh justification."
                ),
            }
            await session.record_message(feedback)
            messages.append(feedback)
            return False
        return True
