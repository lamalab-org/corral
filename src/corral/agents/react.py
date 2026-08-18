"""A ReAct agent that owns its complete task session loop."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

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
    Thought,
)
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes
from corral.core.action import (
    SUBMIT_ANSWER_TOOL_NAME,
    Action,
    with_submit_answer_tool,
)

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


class ReActAgent(BaseAgent):
    """Run a text-based thought/action/observation loop inside an AgentSession."""

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
            user_prompt=user_prompt or "react/user_prompt",
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

    def parse_llm_response(
        self, response: str
    ) -> tuple[list[Thought] | None, list[Action] | None]:
        """Parse the XML-like ReAct response into thoughts and actions."""
        thoughts = [
            Thought(match.group(1).strip())
            for match in re.finditer(r"<thought>(.*?)</thought>", response, re.DOTALL)
        ]
        actions: list[Action] = []
        for match in re.finditer(
            r"<action>(.*?)</action>(?:.*?<action_input>(.*?)</action_input>)?",
            response,
            re.DOTALL,
        ):
            raw_arguments = match.group(2)
            if raw_arguments is None:
                return thoughts or None, None
            try:
                converted = convert_outermost_triple_quotes(raw_arguments.strip())
                converted = converted.replace("True", "true").replace("False", "false")
                arguments = json.loads(converted)
                if not isinstance(arguments, dict):
                    raise ValueError("action_input must be a JSON object")
                actions.append(Action(name=match.group(1).strip(), arguments=arguments))
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                logger.error(f"Parsing error: {exc}")
        return thoughts or None, actions or None

    def _initial_messages(self, session: AgentSession) -> list[dict[str, Any]]:
        messages = create_prompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            task_guide=session.prompt,
            examples=list(session.examples),
            surrender_prompt=self.surrender_prompt,
            enable_surrender=session.surrender_allowed,
        )
        catalog = json.dumps(
            with_submit_answer_tool(session.tools),
            indent=2,
            ensure_ascii=False,
        )
        messages.append(
            LiteLLMMessage(
                role="user",
                content=(
                    "Available tool schemas (including the required completion "
                    f"tool):\n{catalog}"
                ),
            )
        )
        return [dict(message) for message in messages] + provider_messages(
            session.messages
        )

    @staticmethod
    async def _submit_outcome(
        answer: str,
        session: AgentSession,
        usage: _UsageAccumulator,
        *,
        action: Action | None = None,
    ) -> AgentOutcome:
        answer = answer.strip()
        if not answer:
            return AgentOutcome(
                status="protocol_failure",
                error="the model produced an empty final answer",
                usage=usage.outcome(),
            )
        submission = action or Action(
            name=SUBMIT_ANSWER_TOOL_NAME,
            arguments={"answer": answer},
        )
        result = await session.execute(submission)
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
        """Own the ReAct model/tool loop for one task-bound session."""
        messages = self._initial_messages(session)
        usage = _UsageAccumulator()
        iteration_limit = session.iteration_limit

        for _iteration in range(iteration_limit):
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
                    status="budget_exhausted",
                    error=str(exc),
                    usage=usage.outcome(),
                )
            except Exception as exc:
                return AgentOutcome(
                    status="agent_failure",
                    error=str(exc),
                    usage=usage.outcome(),
                )

            usage.add(getattr(response, "usage", None))
            content = getattr(response, "content", None) or ""
            assistant = dict(
                LiteLLMMessage(
                    role="assistant",
                    content=content,
                    id=getattr(response, "id", None),
                )
            )
            messages.append(assistant)

            if session.surrender_allowed and re.search(
                r"<surrender>.*?</surrender>", content, re.DOTALL | re.IGNORECASE
            ):
                return await self._submit_outcome(
                    SURRENDER_SENTINEL,
                    session,
                    usage,
                    action=Action(
                        name=SUBMIT_ANSWER_TOOL_NAME,
                        arguments={"answer": SURRENDER_SENTINEL},
                        content=content,
                    ),
                )

            _thoughts, actions = self.parse_llm_response(content)

            if actions:
                for index, parsed in enumerate(actions):
                    if parsed.name == SUBMIT_ANSWER_TOOL_NAME:
                        answer = parsed.arguments.get("answer")
                        if not isinstance(answer, str):
                            feedback = dict(
                                LiteLLMMessage(
                                    role="user",
                                    content=(
                                        "submit_answer requires a string argument "
                                        "named answer."
                                    ),
                                )
                            )
                            messages.append(feedback)
                            await session.record_message(assistant)
                            await session.record_message(feedback)
                            break
                        return await self._submit_outcome(
                            answer,
                            session,
                            usage,
                            action=Action(
                                id=parsed.id,
                                name=parsed.name,
                                arguments=parsed.arguments,
                                content=content if index == 0 else None,
                            ),
                        )

                    action = Action(
                        id=parsed.id,
                        name=parsed.name,
                        arguments=parsed.arguments,
                        content=content if index == 0 else None,
                    )
                    observation = await session.execute(action)
                    rendered = (
                        f"Observation: {observation.result}"
                        if observation.success
                        else f"Error: {observation.error}"
                    )
                    messages.append(
                        dict(
                            LiteLLMMessage(
                                role="user",
                                content=rendered,
                                name=action.name,
                            )
                        )
                    )
                continue

            await session.record_message(assistant)
            feedback = dict(
                LiteLLMMessage(
                    role="user",
                    content=(
                        "No valid action was found. Plain-text final answers do "
                        "not complete the task. Return one or more complete "
                        "<action>/<action_input> pairs; when finished, call "
                        "submit_answer with a string answer."
                    ),
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
                    name="react-error",
                )
            )
        )
        return AgentOutcome(
            status="iteration_limit",
            error=error,
            usage=usage.outcome(),
        )


__all__ = ["ReActAgent"]
