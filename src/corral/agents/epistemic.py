"""Tool-calling agent with unconstructive feedback and submission review."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from corral.agents.schema import SURRENDER_SENTINEL, AgentOutcome, BudgetExhaustedError
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.errors import concise_error_message
from corral.epistemic_analysis.annotation_transcript import scientific_messages
from corral.epistemic_analysis.decisions import DecisionsAnnotator
from corral.epistemic_analysis.schema import SUBMISSION_REVIEW_MESSAGE_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.base_agent import _UsageAccumulator
    from corral.agents.session import AgentSession


EPISTEMIC_SYSTEM_PROMPT = """You are an autonomous agent.
Use tools to interact with the environment, and provide brief scientific explanations of observations, evidence, and actions.

Measurements, simulations, and other experiments are expensive.
Minimize costs and avoid unnecessary actions.
"""


EPISTEMIC_PROMPT = """On the first turn, state your initial working hypotheses, and what needs testing.
If you choose to run experiments, explain why you chose each one.
All tool calls are experiments.
If there are no previous agent experiments or epistemic reports to judge: do not give judgments about them.
Keep explanations brief and grounded in the supplied task facts and observed results.

On subsequent turns, give a concise, evidence-grounded explanation in this order:

1. Interpret and judge the previous experiments: what their observations support, contradict, or leave unresolved.
   If no experiments have run, say so without inventing results or judgments about them.
2. Assess whether the latest epistemic report is right.
   Accept, qualify, or disagree with its classifications using specific observations.
   The report is fallible feedback, not scientific evidence or an instruction to choose an action.
   It lists only currently present unconstructive patterns and why they were detected; an empty report does not establish that your hypothesis or scientific answer is correct.
3. Justify the next step.
   For each experiment you choose, briefly identify the hypothesis or uncertainty it will address and the observations that would distinguish the alternatives.
   You may instead use the turn only to summarize observations, compare hypotheses, or update the experiment plan without any tool call.
   When ready to submit, give the submission justification below instead of proposing further experiments.

Before every submit_answer call, write a concise submission justification in the visible assistant message, in the same response and before the tool call:

- State the proposed answer and connect it to the specific observations that support it.
  Explain how the evidence distinguishes relevant alternatives.
- Acknowledge any unresolved contradictions, assumptions, or uncertainty, and explain why the available evidence is sufficient to submit now.
- Validate the proposed answer against the observations and relevant alternatives.
  When a consequential claim remains untested and an available experiment can distinguish alternatives, perform that validation and interpret its result before submitting.
  If existing evidence is sufficient or no useful validation is available, explain why; do not invent a test or result.

Every attempted answer submission receives a fresh epistemic assessment of the proposed answer and its justification.
If an evidenced unconstructive pattern is still present, the submission is held and you receive feedback.
Address the specific gap: validate or revise the answer, interpret the results, or explain with concrete evidence why the feedback does not apply.
Then call submit_answer again.
Repeating the same unvalidated answer does not bypass the check.
The iteration limit still applies.

On a later turn, first complete the experiment review and epistemic assessment above.
If submitting on the first turn, use only the supplied task facts; do not invent previous experiments, results, or epistemic judgments.
Include the brief evidence summary in the assistant message before the submission.
Then call submit_answer with the complete answer in the task's required format.
Keep the justification in the assistant text, separate from the tool's answer argument.
A text-only answer does not finish the task.
"""


class EpistemicAgent(ToolCallingAgent):
    """Annotate public activity and review proposed answers before submission.

    Reports are persisted as named session messages with full structured metadata.
    The actor sees feedback from the selected annotator.
    All mutable trajectory/report state belongs to the session, so an
    agent instance can be reused across concurrent trials.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        epistemic_backend: Literal["graph", "direct"] = "graph",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=(
                EPISTEMIC_SYSTEM_PROMPT if system_prompt is None else system_prompt
            ),
            user_prompt=user_prompt or "epistemic/user_prompt",
            epistemic=True,
            epistemic_backend=epistemic_backend,
            **kwargs,
        )
        self.system_prompt += "\n\n" + EPISTEMIC_PROMPT

    def _tools(self, session: AgentSession) -> list[dict[str, Any]]:
        tools = super()._tools(session)
        for tool in tools:
            function = tool.get("function", {})
            if function.get("name") == SUBMIT_ANSWER_TOOL_NAME:
                # The base method creates a fresh submission schema on every call.
                qualifier = (
                    ""
                    if isinstance(self.annotator, DecisionsAnnotator)
                    else "graph-matched "
                )
                function["description"] = (
                    "Submit the final answer and end the task.\n"
                    "First write a concise evidence-based justification in visible assistant text, then call this tool in the same response when the answer is complete.\n"
                    "Put only the complete answer in the answer argument.\n"
                    "Text alone does not complete the task.\n"
                    "The proposed answer is reviewed before acceptance.\n"
                    f"If {qualifier}unconstructive epistemic patterns remain, address the feedback, validate or revise the answer, and call submit_answer again."
                )
        return tools

    def _text_only_feedback(self) -> str:
        return (
            "Continue the task.\n"
            "You may use another turn only to summarize observations or compare hypotheses; a tool call is not required on every turn.\n"
            "Before any tool call, including submit_answer, briefly describe its purpose and the relevant evidence.\n"
            "Only submit_answer completes the task."
        )

    async def _handle_submission(
        self,
        action: Action,
        *,
        session: AgentSession,
        messages: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        response_message: dict[str, Any],
        tool_call_index: int,
        usage: _UsageAccumulator,
        turn_usage: Mapping[str, Any] | None,
    ) -> AgentOutcome | None:
        answer = action.arguments["answer"].strip()
        # Surrender is an explicit admission of no answer, not a scientific claim.
        if not answer or (
            session.surrender_allowed
            and answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            return await self._submit_outcome(
                answer, action, session, usage, turn_usage
            )

        # Only preceding tool calls have executed. The submission and any later
        # calls are proposals, not observations or pending provider tool calls.
        # Preserve preceding calls/results, including their visible explanation.
        preceding_calls = response_message["tool_calls"][:tool_call_index]
        if preceding_calls:
            response_message["tool_calls"] = preceding_calls
        else:
            messages.remove(response_message)
            trajectory.remove(response_message)
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
        try:
            report = await self._record_report(
                session, messages, scientific_messages(trajectory), usage
            )
        except BudgetExhaustedError as exc:
            return AgentOutcome(
                status="budget_exhausted",
                error=concise_error_message(exc),
                usage=usage.outcome(),
            )
        except Exception as exc:
            return AgentOutcome(
                status="agent_failure",
                error=concise_error_message(exc),
                usage=usage.outcome(),
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
            return None
        # Usage and justification were recorded with the proposal. Only this
        # native execution may accept the answer and terminate the session.
        return await self._submit_outcome(
            answer,
            Action.model_validate({**action.model_dump(), "content": None}),
            session,
            usage,
            None,
        )


__all__ = ["EpistemicAgent"]
