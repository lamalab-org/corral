"""Provider-native tool-calling agent with an agent-owned session loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from corral.agents._epistemic import EpistemicSupport
from corral.agents.base_agent import (
    _UsageAccumulator,
    call_model,
    provider_messages,
)
from corral.agents.prompt_utils import create_prompt
from corral.agents.schema import (
    SURRENDER_SENTINEL,
    AgentOutcome,
    BudgetExhaustedError,
)
from corral.agents.utils import LiteLLMMessage
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.errors import concise_error_message
from corral.core.messages import EMPTY_RESPONSE_REMINDER, public_message
from corral.report.logging import logger

MAX_CONSECUTIVE_EMPTY_RESPONSES = 3

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


class ToolCallingAgent(EpistemicSupport):
    """Own the provider call, tool execution, and observation loop."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        *,
        epistemic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "tool_calling/user_prompt",
            epistemic=epistemic,
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

    def _initial_messages(self, session: AgentSession) -> list[dict[str, Any]]:
        initial = create_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            task_guide=session.prompt,
            examples=list(session.examples),
            surrender_prompt=self.surrender_prompt,
            enable_surrender=session.surrender_allowed,
        )
        return [dict(message) for message in initial] + provider_messages(
            session.messages
        )

    @staticmethod
    def _tool_call_value(call: Any) -> dict[str, Any]:
        if isinstance(call, Mapping):
            return dict(call)
        dump = getattr(call, "model_dump", None)
        if callable(dump):
            value = dump()
            if isinstance(value, Mapping):
                return dict(value)
        function = getattr(call, "function", None)
        return {
            "id": getattr(call, "id", None),
            "function": {
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", {}),
            },
        }

    @staticmethod
    async def _submit_outcome(
        answer: str,
        action: Action,
        session: AgentSession,
        usage: _UsageAccumulator,
        turn_usage: Mapping[str, Any] | None = None,
    ) -> AgentOutcome:
        answer = answer.strip()
        if not answer:
            return AgentOutcome(
                status="protocol_failure",
                error="submit_answer contained an empty answer",
                usage=usage.outcome(),
            )
        result = await session.execute(action, usage=turn_usage)
        if not result.success:
            return AgentOutcome(
                status="protocol_failure",
                error=f"submit_answer failed: {result.error}",
                usage=usage.outcome(),
            )
        if (
            session.surrender_allowed
            and answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            return AgentOutcome(status="surrendered", usage=usage.outcome())
        return AgentOutcome(status="completed", answer=answer, usage=usage.outcome())

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run the complete native tool-calling loop for one task."""
        messages = self._initial_messages(session)
        prompt_message_count = len(messages) - len(provider_messages(session.messages))
        tools = self._tools(session)
        usage = _UsageAccumulator(self._usage)
        iteration_limit = session.iteration_limit
        consecutive_empty = 0
        consecutive_refusals = 0
        refresh_feedback = True

        for _iteration in range(iteration_limit):
            try:
                turn_messages = (
                    await self._before_turn(
                        session, messages, messages[prompt_message_count:], usage
                    )
                    if refresh_feedback
                    else messages
                )
                response = await call_model(
                    model=self.model,
                    messages=turn_messages,
                    tools=tools,
                    temperature=self.temperature,
                    api_endpoint=self.api_endpoint,
                    kwargs=self._call_kwargs(),
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

            usage.add(getattr(response, "usage", None))
            turn_usage = getattr(response, "usage", None) or {}
            raw_calls = list(getattr(response, "tool_calls", None) or [])
            public = public_message(
                {
                    "role": "assistant",
                    "content": getattr(response, "content", None),
                    "content_source": getattr(response, "content_source", "text"),
                    "tool_calls": [self._tool_call_value(call) for call in raw_calls],
                }
            )
            finish_reason = getattr(response, "finish_reason", None)
            if finish_reason in {"content_filter", "refusal"}:
                consecutive_refusals += 1
                if _iteration + 1 < iteration_limit and self._should_retry_refusal(
                    consecutive_refusals
                ):
                    # Retain usage without feeding a refused/partial response
                    # back to the actor or executing any of its tool calls.
                    await session.record_messages((), usage=turn_usage)
                    refresh_feedback = False
                    logger.warning(
                        f"Actor provider stopped the response "
                        f"(finish_reason={finish_reason!r}); retrying the unchanged "
                        f"request (refusal retry {consecutive_refusals})"
                    )
                    continue
                if public is not None and public.get("content"):
                    await session.record_message(
                        {"role": "assistant", "content": public["content"]},
                        usage=turn_usage,
                    )
                else:
                    await session.record_messages((), usage=turn_usage)
                return AgentOutcome(
                    status="agent_failure",
                    error=f"actor provider stopped the response (finish_reason={finish_reason!r})",
                    usage=usage.outcome(),
                )
            consecutive_refusals = 0
            if public is None:
                # Account for the request without inventing an assistant message.
                # No new public evidence means no new epistemic assessment.
                await session.record_messages((), usage=turn_usage)
                consecutive_empty += 1
                refresh_feedback = False
                logger.warning(
                    f"Actor returned no public text or tool calls "
                    f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY_RESPONSES}, "
                    f"finish_reason={finish_reason!r})"
                )
                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY_RESPONSES:
                    return AgentOutcome(
                        status="protocol_failure",
                        error=(
                            f"actor returned {consecutive_empty} consecutive responses "
                            "without public text or tool calls "
                            f"(finish_reason={finish_reason!r})"
                        ),
                        usage=usage.outcome(),
                    )
                if consecutive_empty == 1:
                    feedback = {
                        "role": "user",
                        "name": EMPTY_RESPONSE_REMINDER,
                        "content": (
                            "Your last response contained no public text or tool call. "
                            "Please continue with visible assistant text or an "
                            "available tool call. Use submit_answer when finished."
                        ),
                    }
                    messages.append(feedback)
                    await session.record_message(feedback)
                continue
            consecutive_empty = 0
            refresh_feedback = True
            content = public.get("content")

            if not raw_calls:
                assistant = dict(
                    LiteLLMMessage(
                        role="assistant",
                        content=content or "",
                        id=getattr(response, "id", None),
                    )
                )
                messages.append(assistant)
                await session.record_message(assistant, usage=turn_usage)
                feedback = dict(
                    LiteLLMMessage(
                        role="user",
                        content=self._text_only_feedback(),
                    )
                )
                messages.append(feedback)
                await session.record_message(feedback)
                continue

            messages.append(public)

            protocol_error: str | None = None
            for index, raw_call in enumerate(raw_calls):
                try:
                    value = self._tool_call_value(raw_call)
                    action = Action.from_tool_call(
                        value,
                        content=content if index == 0 else None,
                    )
                except Exception as exc:
                    protocol_error = f"invalid provider tool call: {exc}"
                    break

                if action.name == SUBMIT_ANSWER_TOOL_NAME:
                    answer = action.arguments.get("answer")
                    if not isinstance(answer, str):
                        protocol_error = (
                            "submit_answer requires a string argument named answer"
                        )
                        break
                    outcome = await self._handle_submission(
                        action,
                        session=session,
                        messages=messages,
                        trajectory=messages[prompt_message_count:],
                        response_message=public,
                        tool_call_index=index,
                        usage=usage,
                        turn_usage=turn_usage if index == 0 else None,
                    )
                    if outcome is not None:
                        return outcome
                    # A submission review already supplied feedback. Let the
                    # actor address it before assessing that same prefix again.
                    refresh_feedback = False
                    break

                observation = await session.execute(
                    action,
                    usage=(turn_usage if index == 0 else None),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.id,
                        "name": action.name,
                        "content": str(
                            observation.result
                            if observation.success
                            else observation.error
                        ),
                    }
                )

            if protocol_error is not None:
                feedback = dict(
                    LiteLLMMessage(
                        role="user", content=f"Protocol error: {protocol_error}"
                    )
                )
                messages.append(feedback)
                await session.record_message(feedback)

        error = f"agent exhausted its {iteration_limit} interaction budget"
        await session.record_message(
            dict(
                LiteLLMMessage(
                    role="assistant",
                    content=error,
                    name="tool-calling-error",
                )
            )
        )
        return AgentOutcome(
            status="iteration_limit",
            error=error,
            usage=usage.outcome(),
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
        if (
            not self.epistemic
            or not answer
            or (
                session.surrender_allowed
                and answer.casefold() == SURRENDER_SENTINEL.casefold()
            )
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
        try:
            accepted = await self._review_submission(
                action,
                session=session,
                messages=messages,
                trajectory=trajectory,
                usage=usage,
                turn_usage=turn_usage,
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
        if not accepted:
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


__all__ = ["ToolCallingAgent"]
