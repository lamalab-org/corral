"""A ReAct agent that owns its complete task session loop."""

from __future__ import annotations

import json
import re
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
    Thought,
)
from corral.agents.utils import LiteLLMMessage, convert_outermost_triple_quotes
from corral.core.action import (
    SUBMIT_ANSWER_TOOL_NAME,
    Action,
)
from corral.core.errors import concise_error_message
from corral.core.messages import EMPTY_RESPONSE_REMINDER, public_message
from corral.report.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.session import AgentSession


_PYTHON_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _loads_arguments(text: str) -> Any:
    """Parse action arguments, accepting Python's bare True/False/None.

    Only literals outside JSON strings are converted, so code the agent writes
    (``write_file`` content, shell commands) keeps its ``True`` and ``False``.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_python_literals_to_json(text))


def _python_literals_to_json(text: str) -> str:
    out: list[str] = []
    i, in_string = 0, False
    while i < len(text):
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        else:
            match = re.match(r"(True|False|None)\b", text[i:])
            if match and not (i and (text[i - 1].isalnum() or text[i - 1] == "_")):
                out.append(_PYTHON_LITERALS[match.group(1)])
                i += len(match.group(1))
                continue
            out.append(char)
        i += 1
    return "".join(out)


# OpenAI accepts a message name of at most 64 letters, digits, "_" or "-".
_MESSAGE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


# OpenAI accepts a message name of at most 64 letters, digits, "_" or "-".
_MESSAGE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


class ReActAgent(EpistemicSupport):
    """Run a text-based thought/action/observation loop inside an AgentSession."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 1.0,
        *,
        epistemic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "react/user_prompt",
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            epistemic=epistemic,
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
                arguments = _loads_arguments(converted)
                if not isinstance(arguments, dict):
                    raise ValueError("action_input must be a JSON object")
                actions.append(Action(name=match.group(1).strip(), arguments=arguments))
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                logger.warning(f"Ignoring malformed agent action: {exc}")
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
            self._tools(session),
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
        turn_usage: Mapping[str, Any] | None = None,
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
        result = await session.execute(submission, usage=turn_usage)
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

    async def _handle_submission(
        self,
        action: Action,
        *,
        session: AgentSession,
        messages: list[dict[str, Any]],
        usage: _UsageAccumulator,
        turn_usage: Mapping[str, Any] | None,
    ) -> AgentOutcome | None:
        answer = action.arguments["answer"].strip()
        if (
            self.epistemic
            and answer
            and not (
                session.surrender_allowed
                and answer.casefold() == SURRENDER_SENTINEL.casefold()
            )
        ):
            try:
                # The committed history contains only actions that actually ran,
                # with canonical tool results, even after resuming a session.
                accepted = await self._review_submission(
                    action,
                    session=session,
                    messages=messages,
                    trajectory=provider_messages(session.messages),
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
            action = Action.model_validate({**action.model_dump(), "content": None})
            turn_usage = None  # The proposal already recorded the actor usage.
        return await self._submit_outcome(
            answer, session, usage, action=action, turn_usage=turn_usage
        )

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Own the ReAct model/tool loop for one task-bound session."""
        messages = self._initial_messages(session)
        usage = _UsageAccumulator(self._usage)
        iteration_limit = session.iteration_limit
        refresh_feedback = True
        consecutive_empty = 0
        consecutive_refusals = 0

        for _iteration in range(iteration_limit):
            try:
                if refresh_feedback:
                    await self._before_turn(
                        session, messages, provider_messages(session.messages), usage
                    )
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
            refresh_feedback = True
            turn_usage = getattr(response, "usage", None) or {}
            content = getattr(response, "content", None) or ""
            if self.epistemic:
                public = public_message(
                    {
                        "role": "assistant",
                        "content": content,
                        "content_source": getattr(response, "content_source", "text"),
                    }
                )
                content = public["content"] if public else ""
                finish_reason = getattr(response, "finish_reason", None)
                if finish_reason in {"content_filter", "refusal"}:
                    consecutive_refusals += 1
                    if _iteration + 1 < iteration_limit and self._should_retry_refusal(
                        consecutive_refusals
                    ):
                        await session.record_messages((), usage=turn_usage)
                        refresh_feedback = False
                        continue
                    await session.record_messages(
                        (public,) if public else (), usage=turn_usage
                    )
                    return AgentOutcome(
                        status="agent_failure",
                        error=f"actor provider stopped the response (finish_reason={finish_reason!r})",
                        usage=usage.outcome(),
                    )
                consecutive_refusals = 0
                if public is None:
                    await session.record_messages((), usage=turn_usage)
                    refresh_feedback = False
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        return AgentOutcome(
                            status="protocol_failure",
                            error="actor returned 3 consecutive responses without public text",
                            usage=usage.outcome(),
                        )
                    if consecutive_empty == 1:
                        feedback = {
                            "role": "user",
                            "name": EMPTY_RESPONSE_REMINDER,
                            "content": (
                                "Your last response contained no public text. Continue "
                                "with <thought> and, when useful, <action>/<action_input>. "
                                "Use submit_answer when finished."
                            ),
                        }
                        messages.append(feedback)
                        await session.record_message(feedback)
                    continue
                consecutive_empty = 0
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
                    turn_usage=turn_usage,
                )

            thoughts, actions = self.parse_llm_response(content)
            explanation = (
                re.sub(
                    r"<action>.*?</action>(?:.*?<action_input>.*?</action_input>)?",
                    "",
                    content,
                    flags=re.DOTALL,
                )
                .replace("<thought>", "")
                .replace("</thought>", "")
                .strip()
                if self.epistemic
                else content
            )

            if actions:
                for index, parsed in enumerate(actions):
                    action = Action(
                        id=parsed.id,
                        name=parsed.name,
                        arguments=parsed.arguments,
                        content=explanation if index == 0 else None,
                    )
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
                            await session.record_message(
                                assistant,
                                usage=turn_usage if index == 0 else None,
                            )
                            await session.record_message(feedback)
                            break
                        if self.epistemic:
                            # Do not retain a held submission or any later action
                            # as an executed action in the actor's conversation.
                            if index:
                                assistant["content"] = (
                                    f"<thought>{explanation}</thought>\n"
                                    + "\n".join(
                                        f"<action>{a.name}</action>"
                                        f"<action_input>{json.dumps(a.arguments)}</action_input>"
                                        for a in actions[:index]
                                    )
                                )
                            else:
                                messages.remove(assistant)
                        outcome = await self._handle_submission(
                            action,
                            session=session,
                            messages=messages,
                            usage=usage,
                            turn_usage=turn_usage if index == 0 else None,
                        )
                        if outcome is not None:
                            return outcome
                        refresh_feedback = False  # Review already supplied feedback.
                        break

                    observation = await session.execute(
                        action,
                        usage=(turn_usage if index == 0 else None),
                    )
                    rendered = (
                        f"Observation: {observation.result}"
                        if observation.success
                        else f"Error: {observation.error}"
                    )
                    observation_message = LiteLLMMessage(role="user", content=rendered)
                    # The action name is model text, and providers reject a
                    # message name that is not a short identifier, which would
                    # end the trial on what is only a malformed action.
                    if _MESSAGE_NAME.fullmatch(action.name):
                        observation_message["name"] = action.name
                    messages.append(dict(observation_message))
                continue

            await session.record_message(
                assistant,
                usage=turn_usage,
            )
            feedback = dict(
                LiteLLMMessage(
                    role="user",
                    content=(
                        self._text_only_feedback()
                        if self.epistemic and thoughts and "<action>" not in content
                        else "No valid action was found. Plain-text final answers do "
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
