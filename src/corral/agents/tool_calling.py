"""Provider-native tool-calling agent with an agent-owned session loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from corral.agents.base_agent import (
    BaseAgent,
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
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action, with_submit_answer_tool
from corral.core.errors import concise_error_message

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


class ToolCallingAgent(BaseAgent):
    """Own the provider call, tool execution, and observation loop."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "tool_calling/user_prompt",
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
        tools = with_submit_answer_tool(session.tools)
        usage = _UsageAccumulator(self._usage)
        iteration_limit = session.iteration_limit

        for _iteration in range(iteration_limit):
            try:
                response = await call_model(
                    model=self.model,
                    messages=messages,
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
            content = getattr(response, "content", None)

            if not raw_calls:
                assistant = dict(
                    LiteLLMMessage(
                        role="assistant",
                        content=content or "",
                        id=getattr(response, "id", None),
                    )
                )
                messages.append(assistant)
                await session.record_message(
                    assistant,
                    usage=turn_usage,
                )
                feedback = dict(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            "A plain-text answer does not finish the task. Call "
                            "submit_answer with the complete answer, or call an "
                            "available environment tool."
                        ),
                    )
                )
                messages.append(feedback)
                await session.record_message(feedback)
                continue

            provider_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": [self._tool_call_value(call) for call in raw_calls],
            }
            messages.append(provider_message)

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
                    return await self._submit_outcome(
                        answer,
                        action,
                        session,
                        usage,
                        turn_usage,
                    )

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


__all__ = ["ToolCallingAgent"]
