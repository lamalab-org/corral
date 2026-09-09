"""Hierarchical planner implemented as an agent-owned session loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from corral.agents.base_agent import (
    BaseAgent,
    _UsageAccumulator,
    call_model,
    provider_messages,
)
from corral.agents.prompt_utils import create_prompt
from corral.agents.react import ReActAgent
from corral.agents.schema import AgentOutcome, AgentUsage, BudgetExhaustedError
from corral.agents.utils import LiteLLMMessage
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.errors import concise_error_message

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


def _merge_usage(planner: AgentUsage, executor: AgentUsage) -> AgentUsage:
    return AgentUsage(
        input_tokens=planner.input_tokens + executor.input_tokens,
        output_tokens=planner.output_tokens + executor.output_tokens,
        reasoning_tokens=planner.reasoning_tokens + executor.reasoning_tokens,
        llm_calls=planner.llm_calls + executor.llm_calls,
    )


class LLMPlanner(BaseAgent):
    """Generate a high-level plan, then let a session agent execute it."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "planner_prompt/user_prompt",
            surrender_prompt=None,
            temperature=temperature,
            **kwargs,
        )

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Own planning and delegate low-level actions through the same session."""
        iteration_limit = session.iteration_limit
        initial = create_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            task_guide=session.prompt,
            max_iterations=iteration_limit,
            examples=list(session.examples),
            tools=json.dumps(session.tools, ensure_ascii=False),
        )
        messages = [dict(message) for message in initial] + provider_messages(
            session.messages
        )
        usage = _UsageAccumulator(self._usage)

        try:
            response = await call_model(
                model=self.model,
                messages=messages,
                tools=None,
                temperature=self.temperature,
                api_endpoint=self.api_endpoint,
                kwargs=self._call_kwargs(),
            )
        except BudgetExhaustedError as exc:
            return AgentOutcome(
                status="budget_exhausted", error=concise_error_message(exc)
            )
        except Exception as exc:
            return AgentOutcome(
                status="agent_failure", error=concise_error_message(exc)
            )

        usage.add(getattr(response, "usage", None))
        turn_usage = getattr(response, "usage", None) or {}
        plan = (getattr(response, "content", None) or "").strip()
        planner_message = dict(
            LiteLLMMessage(
                role="assistant",
                content=plan,
                name="high-level-planner",
                id=getattr(response, "id", None),
            )
        )
        await session.record_message(
            planner_message,
            usage=turn_usage,
        )

        if "Final Answer:" in plan:
            answer = plan.rsplit("Final Answer:", 1)[1].strip()
            if answer:
                submission = await session.execute(
                    Action(
                        name=SUBMIT_ANSWER_TOOL_NAME,
                        arguments={"answer": answer},
                    )
                )
                if not submission.success:
                    return AgentOutcome(
                        status="protocol_failure",
                        error=f"submit_answer failed: {submission.error}",
                        usage=usage.outcome(),
                    )
                return AgentOutcome(
                    status="completed",
                    answer=answer,
                    usage=usage.outcome(),
                )

        if not plan:
            return AgentOutcome(
                status="protocol_failure",
                error="the planner returned an empty plan",
                usage=usage.outcome(),
            )

        remaining_iterations = iteration_limit - usage.llm_calls
        if remaining_iterations < 1:
            return AgentOutcome(
                status="iteration_limit",
                error=(
                    "the planner used the task's complete interaction budget "
                    "before low-level execution"
                ),
                usage=usage.outcome(),
            )

        await session.record_message(
            dict(
                LiteLLMMessage(
                    role="user",
                    content=(
                        "Execute the following high-level plan while still solving "
                        f"the original task:\n\n{plan}"
                    ),
                    name="high-level-plan",
                )
            )
        )
        executor = ReActAgent(
            model=self.model,
            api_endpoint=self.api_endpoint,
            temperature=self.temperature,
            **self._call_kwargs(),
        )
        outcome = await session.run_delegate(
            executor,
            max_iterations=remaining_iterations,
        )
        return AgentOutcome(
            status=outcome.status,
            answer=outcome.answer,
            error=outcome.error,
            usage=_merge_usage(usage.outcome(), outcome.usage),
            metadata={"executor": type(executor).__name__, **dict(outcome.metadata)},
        )


__all__ = ["LLMPlanner"]
