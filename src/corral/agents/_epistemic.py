"""Shared opt-in epistemic feedback for the native and ReAct session loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from corral.agents.base_agent import BaseAgent
from corral.core.action import with_submit_answer_tool
from corral.epistemic_analysis.annotation_transcript import scientific_messages
from corral.epistemic_analysis.annotators.local import GraphAnnotator, provider_config
from corral.epistemic_analysis.schema import (
    REPORT_MESSAGE_NAME,
    SUBMISSION_REVIEW_MESSAGE_NAME,
    EpistemicReport,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.base_agent import _UsageAccumulator
    from corral.agents.session import AgentSession
    from corral.core.action import Action
    from corral.epistemic_analysis.decisions import DecisionsAnnotator


class EpistemicSupport(BaseAgent):
    """Opt-in review without actor prompt overrides; trajectory state is per session."""

    def __init__(
        self,
        *,
        epistemic: bool = False,
        annotator: GraphAnnotator | DecisionsAnnotator | None = None,
        epistemic_backend: str = "graph",
        actor_refusal_max_retries: int = 1,
        epistemic_provider_config: Mapping[str, Any] | None = None,
        epistemic_explanation_config: Mapping[str, Any] | None = None,
        epistemic_graph_config: Mapping[str, Any] | None = None,
        epistemic_profile: str | None = None,
        epistemic_model: str | None = None,
        epistemic_base_url: str | None = None,
        epistemic_explanation_model: str | None = None,
        epistemic_explanation_base_url: str | None = None,
        epistemic_timeout_s: float | None = None,
        epistemic_max_retries: int | None = None,
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
        if epistemic_backend not in {"graph", "direct"}:
            raise ValueError("epistemic_backend must be 'graph' or 'direct'")
        if epistemic_backend == "direct":
            if any(
                config is not None
                for config in (
                    epistemic_provider_config,
                    epistemic_explanation_config,
                    epistemic_graph_config,
                    epistemic_explanation_base_url,
                )
            ):
                raise ValueError(
                    "The direct backend uses epistemic_base_url for Decisions and explanations"
                )
            from corral.epistemic_analysis.decision_thresholds import (
                CALIBRATED_THRESHOLD_POLICY,
            )
            from corral.epistemic_analysis.decisions import DecisionsAnnotator

            direct_options = {
                name: kwargs.pop(f"epistemic_{name}")
                for name in (
                    "api_key",
                    "pattern_thresholds",
                    "stream_explanations",
                    "explanation_concurrency",
                    "motif_names",
                    "include_explanations",
                )
                if f"epistemic_{name}" in kwargs
            }
            direct_options.setdefault("pattern_thresholds", CALIBRATED_THRESHOLD_POLICY)
            if epistemic_model is not None:
                direct_options["model"] = epistemic_model
            if epistemic_timeout_s is not None:
                direct_options["timeout_s"] = epistemic_timeout_s
            if epistemic_max_retries is not None:
                direct_options["max_retries"] = epistemic_max_retries
            if epistemic and annotator is None:
                annotator = DecisionsAnnotator(
                    base_url=epistemic_base_url,
                    explanation_model=epistemic_explanation_model,
                    **direct_options,
                )
        obsolete = [key for key in kwargs if key.startswith("epistemic_")]
        if obsolete:
            raise TypeError(
                f"Unsupported options for the {epistemic_backend} backend: {', '.join(obsolete)}"
            )
        if kwargs.get("model") is None:
            if epistemic:
                import os

                config = provider_config("actor", profile=epistemic_profile)
                if config.provider != "openai":
                    raise ValueError(
                        "The actor requires an OpenAI-compatible chat model"
                    )
                kwargs["model"] = "openai/" + config.model
                kwargs["api_endpoint"] = kwargs.get(
                    "api_endpoint"
                ) or config.endpoint.removesuffix("/chat/completions")
                if "api_key" not in kwargs:
                    if config.api_key_env and not os.environ.get(config.api_key_env):
                        raise ValueError(
                            f"Set {config.api_key_env} for the configured actor"
                        )
                    kwargs["api_key"] = (
                        os.environ[config.api_key_env]
                        if config.api_key_env
                        else "local"
                    )
            else:
                kwargs["model"] = "openai/gpt-4o"
        super().__init__(**kwargs)
        self.epistemic = epistemic
        self.annotator = None
        if not epistemic:
            return
        self.actor_refusal_max_retries = actor_refusal_max_retries
        options = dict(epistemic_provider_config or {})
        explanations = dict(epistemic_explanation_config or {})
        if epistemic_base_url is not None:
            options["base_url"] = epistemic_base_url
        if epistemic_explanation_model is not None:
            explanations["model"] = epistemic_explanation_model
        if epistemic_explanation_base_url is not None:
            explanations["base_url"] = epistemic_explanation_base_url
        self.annotator = (
            annotator
            if annotator is not None
            else GraphAnnotator(
                provider_config=options,
                explanation_config=explanations,
                graph_config=epistemic_graph_config,
                profile=epistemic_profile,
                model=epistemic_model,
                timeout_s=epistemic_timeout_s,
                max_retries=epistemic_max_retries,
            )
        )

    def _should_retry_refusal(self, consecutive_refusals: int) -> bool:
        return self.epistemic and consecutive_refusals <= self.actor_refusal_max_retries

    def _tools(self, session: AgentSession) -> list[dict[str, Any]]:
        """Keep the native tool descriptions when enabling epistemic review."""
        return with_submit_answer_tool(session.tools)

    def _text_only_feedback(self) -> str:
        return (
            "A plain-text answer does not finish the task.\n"
            "Call submit_answer with the complete answer, or call an available environment tool."
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
        visible = report.graph is None or bool(report.present_antipatterns)
        await session.record_message(
            {
                **feedback,
                "metadata": {
                    "epistemic_report": report.to_dict(),
                    "agent_visible": visible,
                },
            },
            usage=report.usage,
            llm_calls=model_calls,
        )
        if visible:
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
                    "Submission held: the proposed answer has not been accepted.\n"
                    "Address the unconstructive patterns in the preceding report.\n"
                    "Validate the answer with a targeted experiment when useful, interpret the observations, and revise unsupported claims.\n"
                    "If a pattern does not apply, explain why using concrete evidence.\n"
                    "Then call submit_answer again with the complete answer and a fresh justification."
                ),
            }
            await session.record_message(feedback)
            messages.append(feedback)
            return False
        return True
