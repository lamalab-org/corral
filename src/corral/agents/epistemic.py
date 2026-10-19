"""Tool-calling agent with epistemic feedback after its initial turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from corral.agents.tool_calling import ToolCallingAgent
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME
from corral.epistemic_analysis.decisions import (
    DEFAULT_DECISIONS_MODEL,
    REPORT_MESSAGE_NAME,
    DecisionsAnnotator,
    scientific_messages,
)

if TYPE_CHECKING:
    from corral.agents.base_agent import _UsageAccumulator
    from corral.agents.session import AgentSession


EPISTEMIC_SYSTEM_PROMPT = """You are an autonomous agent. Use tools to interact with the environment, and provide explanations in visible assistant text.

Measurements, simulations, and other experiments are expensive. Minimize costs and avoid unnecessary actions.
"""


EPISTEMIC_PROMPT = """On the first turn, state your initial working hypotheses, and what needs testing.
If you choose to run experiments, explain why you chose each one.
If there are no previous agent experiments or epistemic reports to judge: do not give judgments about them.

On subsequent turns, give a concise, evidence-grounded explanation in this order:

1. Interpret and judge the previous experiments: what their observations support, contradict, or leave unresolved. If no experiments have run, say so without inventing results or judgments about them.
2. Assess whether the latest epistemic report is right. Accept, qualify, or disagree with its classifications using specific observations. The report is fallible feedback, not scientific evidence or an instruction to choose an action. It lists only currently present patterns and why they were detected; an empty report does not establish that your reasoning or scientific answer is correct.
3. Justify the next step. For each experiment you choose, identify the hypothesis or uncertainty it will address, the expected information, and how different results would affect your conclusion. You may instead use the turn only to explain your reasoning, compare hypotheses, or revise your plan without any tool call. When ready to submit, give the submission justification below instead of proposing further experiments.

Before every submit_answer call, write a concise submission justification in the visible assistant message, in the same response and before the tool call:

- State the proposed answer and connect it to the specific observations that support it. Explain how the evidence distinguishes relevant alternatives.
- Acknowledge any unresolved contradictions, assumptions, or uncertainty, and explain why the available evidence is sufficient to submit now.

On a later turn, first complete the experiment review and epistemic assessment above.
If submitting on the first turn, use only the supplied task facts; do not invent previous experiments, results, or epistemic judgments.
The justification must appear in the assistant's visible text so it is preserved in the trajectory; an internal reasoning field alone does not fulfill this step.
Then call submit_answer with the complete answer in the task's required format.
Keep the justification in the assistant text, separate from the tool's answer argument.
A text-only answer does not finish the task.
"""


class EpistemicAgent(ToolCallingAgent):
    """Annotate after an initial actor turn, retaining the native submission loop.

    Reports are persisted as named session messages with full structured metadata.
    The actor sees the full report history; no reports are sent to the annotator.
    All mutable trajectory/report state belongs to the session, so an
    agent instance can be reused across concurrent trials.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        annotator: DecisionsAnnotator | None = None,
        epistemic_model: str = DEFAULT_DECISIONS_MODEL,
        epistemic_explanation_model: str | None = None,
        epistemic_api_key: str | None = None,
        epistemic_base_url: str | None = None,
        epistemic_timeout_s: float = 120,
        epistemic_max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=(
                EPISTEMIC_SYSTEM_PROMPT if system_prompt is None else system_prompt
            ),
            user_prompt=user_prompt or "epistemic/user_prompt",
            **kwargs,
        )
        self.system_prompt += "\n\n" + EPISTEMIC_PROMPT
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
            )
        )

    def _tools(self, session: AgentSession) -> list[dict[str, Any]]:
        tools = super()._tools(session)
        for tool in tools:
            function = tool.get("function", {})
            if function.get("name") == SUBMIT_ANSWER_TOOL_NAME:
                # The base method creates a fresh submission schema on every call.
                function["description"] = (
                    "Submit the final answer and end the task. First write a concise "
                    "evidence-based justification in visible assistant text, then "
                    "call this tool in the same response when the answer is complete. "
                    "Put only the complete answer in the answer argument. Text alone "
                    "does not complete the task."
                )
        return tools

    def _text_only_feedback(self) -> str:
        return (
            "Continue the task. You may use another turn only to "
            "explain your reasoning; a tool call is not required "
            "on every turn. Before any tool call, including "
            "submit_answer, provide a visible explanation of "
            "why it is justified. Only submit_answer completes "
            "the task."
        )

    async def _before_turn(
        self,
        session: AgentSession,
        messages: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        usage: _UsageAccumulator,
    ) -> list[dict[str, Any]]:
        # The loop supplies canonical history plus local provider calls/results,
        # excluding initial prompts and few-shot demonstrations.
        observed = scientific_messages(trajectory)
        # Task facts and scaffolding alone are not prior agent behavior. Use the
        # trajectory rather than a local iteration counter so resumed runs still
        # receive feedback before their next turn.
        if not any(message.get("role") == "assistant" for message in observed):
            return messages
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
        return messages


__all__ = ["EpistemicAgent"]
