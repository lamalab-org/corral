"""Terminus-style structured agent with an agent-owned session loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from jsonschema import Draft202012Validator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from corral.agents.base_agent import (
    BaseAgent,
    _UsageAccumulator,
    call_model,
    provider_messages,
)
from corral.agents.schema import (
    SURRENDER_SENTINEL,
    AgentOutcome,
    BudgetExhaustedError,
)
from corral.agents.utils import LiteLLMMessage
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.errors import concise_error_message
from corral.report.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.session import AgentSession


class TerminusToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: Annotated[
        dict[str, Any],
        WithJsonSchema(
            {
                "type": "string",
                "description": "Tool arguments encoded as one JSON object string.",
            }
        ),
    ] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value.strip() or "{}")
        return value


class TerminusResponse(BaseModel):
    """Validated response returned on every Terminus turn."""

    model_config = ConfigDict(extra="forbid")
    analysis: str
    plan: str
    tool_calls: list[TerminusToolCall] = Field(default_factory=list)
    surrender: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> TerminusResponse:
        if sum((bool(self.tool_calls), self.surrender)) != 1:
            raise ValueError("Exactly one of tool_calls or surrender must be provided.")
        return self


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^`+[^\n]*\n?", "", text)
        text = re.sub(r"\n?`+\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_code_fence(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in the response.")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object.")
    return value


@dataclass(slots=True)
class _RunState:
    messages: list[dict[str, Any]]
    usage: _UsageAccumulator
    last_total_tokens: int = 0
    structured_output: bool = True
    pending_answer: str | None = None
    affirmations: int = 0


class TerminusAgent(BaseAgent):
    """Drive a schema-validated reasoning/tool loop through AgentSession."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_actions_per_turn: int = 1,
        max_observation_chars: int | None = 20_000,
        max_consecutive_parse_failures: int = 3,
        use_structured_output: bool = True,
        confirmations_required: int = 1,
        summarize_after_tokens: int | None = 60_000,
        keep_recent_messages: int = 6,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        if max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be at least 1")
        if max_consecutive_parse_failures < 1:
            raise ValueError("max_consecutive_parse_failures must be at least 1")
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "tool_calling/user_prompt",
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )
        self.max_actions_per_turn = max_actions_per_turn
        self.max_observation_chars = max_observation_chars
        self.max_consecutive_parse_failures = max_consecutive_parse_failures
        self.use_structured_output = use_structured_output
        self.confirmations_required = confirmations_required
        self.summarize_after_tokens = summarize_after_tokens
        self.keep_recent_messages = keep_recent_messages

    def _initial_prompt(
        self,
        task: str | list[dict[str, Any]],
        tools: list[dict[str, Any]],
        enable_surrender: bool = False,
        examples: list[str] | None = None,
    ) -> str:
        task_text = (
            task if isinstance(task, str) else json.dumps(task, ensure_ascii=False)
        )
        surrender_rule = "Do not surrender; keep working until you have a final answer."
        if enable_surrender:
            surrender_rule = (
                self.surrender_prompt.fill({})
                if self.surrender_prompt is not None
                else "Surrender only if the task is genuinely impossible."
            )
        examples_block = (
            "\nWorked examples:\n" + "\n\n".join(map(str, examples)) if examples else ""
        )
        return f"""
You are a Terminus-style autonomous scientific agent.

Complete the task using only the Corral tools listed below.

Task:
{task_text}
{examples_block}

Available Corral tools:
{json.dumps(tools, indent=2, ensure_ascii=False)}

Return exactly one JSON object matching this schema:
{json.dumps(TerminusResponse.model_json_schema(), indent=2)}

Rules:
- Never invent a tool.
- Tool arguments must satisfy the supplied JSON Schema.
- Provide exactly one of tool_calls or surrender.
- Finish only by calling submit_answer; never place the final answer in plain text.
- Request at most {self.max_actions_per_turn} tool call(s) per response.
- Tool calls execute sequentially in listed order.
- {surrender_rule}
- Do not include Markdown fences or surrounding prose.
""".strip()

    def _parse(self, content: str) -> TerminusResponse:
        try:
            return TerminusResponse.model_validate_json(_strip_code_fence(content))
        except ValidationError:
            return TerminusResponse.model_validate(_extract_json_object(content))

    @staticmethod
    def _tool_definitions(
        tools: tuple[dict[str, Any], ...],
    ) -> dict[str, dict[str, Any]]:
        definitions: dict[str, dict[str, Any]] = {}
        for tool in tools:
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name"):
                definitions[str(function["name"])] = function
        return definitions

    @staticmethod
    def _validate_arguments(
        definitions: dict[str, dict[str, Any]],
        name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        schema = definitions.get(name, {}).get("parameters")
        if not schema:
            return None
        errors = sorted(
            Draft202012Validator(schema).iter_errors(arguments),
            key=lambda item: list(item.path),
        )
        return None if not errors else "; ".join(error.message for error in errors[:5])

    def _truncate_observation(self, value: Any) -> Any:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        limit = self.max_observation_chars
        if limit is None or len(text) <= limit:
            return value
        head = limit // 2
        omitted = len(text) - limit
        return (
            f"{text[:head]}\n...[truncated {omitted} chars]...\n{text[-(limit-head):]}"
        )

    async def _call(self, run: _RunState) -> Any:
        kwargs = self._call_kwargs()
        if run.structured_output:
            try:
                return await call_model(
                    model=self.model,
                    messages=run.messages,
                    tools=None,
                    temperature=self.temperature,
                    api_endpoint=self.api_endpoint,
                    kwargs={**kwargs, "response_format": TerminusResponse},
                )
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Structured output failed ({exc}); falling back to JSON text."
                )
                run.structured_output = False
        return await call_model(
            model=self.model,
            messages=run.messages,
            tools=None,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            kwargs=kwargs,
        )

    async def _maybe_compact(self, run: _RunState) -> None:
        threshold = self.summarize_after_tokens
        if threshold is None or run.last_total_tokens < threshold:
            return
        head, tail = 2, self.keep_recent_messages
        if len(run.messages) <= head + tail + 1:
            return
        middle = run.messages[head:-tail]
        prompt = (
            "Summarize this transcript into a concise progress note. Preserve "
            "concrete observations and remaining work; invent nothing.\n\n"
            + "\n\n".join(
                f"[{message.get('role')}] {message.get('content')}"
                for message in middle
            )
        )
        try:
            response = await call_model(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
                api_endpoint=self.api_endpoint,
                kwargs=self._call_kwargs(),
            )
        except Exception as exc:
            logger.warning(f"Terminus context summarization failed: {exc}")
            return
        run.usage.add(getattr(response, "usage", None))
        summary = getattr(response, "content", None)
        if summary:
            run.messages = [
                *run.messages[:head],
                {"role": "user", "content": f"Summary of earlier progress:\n{summary}"},
                *run.messages[-tail:],
            ]

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run Terminus's complete structured loop against `session`."""
        tools = tuple(dict(tool) for tool in session.tools)
        definitions = self._tool_definitions(tools)
        iteration_limit = session.iteration_limit
        messages = [
            dict(LiteLLMMessage(role="system", content=self.system_prompt)),
            dict(
                LiteLLMMessage(
                    role="user",
                    content=self._initial_prompt(
                        session.prompt,
                        list(tools),
                        session.surrender_allowed,
                        list(session.examples),
                    ),
                )
            ),
            *provider_messages(session.messages),
        ]
        run = _RunState(
            messages=messages,
            usage=_UsageAccumulator(self._usage),
            structured_output=self.use_structured_output,
        )
        parse_failures = 0

        for _iteration in range(iteration_limit):
            await self._maybe_compact(run)
            try:
                response = await self._call(run)
            except BudgetExhaustedError as exc:
                return AgentOutcome(
                    status="budget_exhausted",
                    error=concise_error_message(exc),
                    usage=run.usage.outcome(),
                )
            except Exception as exc:
                return AgentOutcome(
                    status="agent_failure",
                    error=concise_error_message(exc),
                    usage=run.usage.outcome(),
                )

            raw_usage = getattr(response, "usage", None)
            turn_usage = raw_usage or {}
            run.usage.add(raw_usage)
            if raw_usage:
                run.last_total_tokens = int(raw_usage.get("total_tokens", 0) or 0)
            content = getattr(response, "content", None) or ""
            assistant = dict(
                LiteLLMMessage(
                    role="assistant",
                    content=content,
                    id=getattr(response, "id", None),
                )
            )
            run.messages.append(assistant)

            try:
                parsed = self._parse(content)
            except (ValueError, ValidationError) as exc:
                parse_failures += 1
                await session.record_message(assistant, usage=turn_usage)
                if parse_failures >= self.max_consecutive_parse_failures:
                    return AgentOutcome(
                        status="protocol_failure",
                        error=f"model returned {parse_failures} invalid responses: {exc}",
                        usage=run.usage.outcome(),
                    )
                feedback = {
                    "role": "user",
                    "content": f"Invalid response: {exc}. Return corrected JSON only.",
                }
                run.messages.append(feedback)
                await session.record_message(feedback)
                continue
            parse_failures = 0

            if parsed.surrender:
                if session.surrender_allowed:
                    result = await session.execute(
                        Action(
                            name=SUBMIT_ANSWER_TOOL_NAME,
                            arguments={"answer": SURRENDER_SENTINEL},
                            content=content,
                        ),
                        usage=turn_usage,
                    )
                    if result.success:
                        return AgentOutcome(
                            status="surrendered", usage=run.usage.outcome()
                        )
                    return AgentOutcome(
                        status="protocol_failure",
                        error=f"submit_answer failed: {result.error}",
                        usage=run.usage.outcome(),
                    )
                feedback = {
                    "role": "user",
                    "content": "Surrender is disabled. Continue using the available tools.",
                }
                await session.record_message(assistant, usage=turn_usage)
                run.messages.append(feedback)
                await session.record_message(feedback)
                continue

            if len(parsed.tool_calls) > self.max_actions_per_turn:
                await session.record_message(assistant, usage=turn_usage)
                feedback = {
                    "role": "user",
                    "content": (
                        f"Requested {len(parsed.tool_calls)} calls; at most "
                        f"{self.max_actions_per_turn} are allowed per turn."
                    ),
                }
                run.messages.append(feedback)
                await session.record_message(feedback)
                continue

            observations: list[dict[str, Any]] = []
            executed = False
            pending_usage: Mapping[str, Any] | None = turn_usage
            for index, call in enumerate(parsed.tool_calls):
                if call.name not in definitions:
                    observations.append(
                        {
                            "tool_name": call.name,
                            "success": False,
                            "error": "unknown tool",
                        }
                    )
                    continue
                argument_error = self._validate_arguments(
                    definitions, call.name, call.arguments
                )
                if argument_error:
                    observations.append(
                        {
                            "tool_name": call.name,
                            "arguments": call.arguments,
                            "success": False,
                            "error": f"invalid arguments: {argument_error}",
                        }
                    )
                    continue
                if call.name == SUBMIT_ANSWER_TOOL_NAME:
                    answer = str(call.arguments.get("answer", "")).strip()
                    if not answer:
                        observations.append(
                            {
                                "tool_name": call.name,
                                "arguments": call.arguments,
                                "success": False,
                                "error": "submit_answer requires a non-empty answer",
                            }
                        )
                        continue
                    if self.confirmations_required > 0:
                        if answer != run.pending_answer:
                            run.pending_answer = answer
                            run.affirmations = 1
                        else:
                            run.affirmations += 1
                        if run.affirmations <= self.confirmations_required:
                            observations.append(
                                {
                                    "tool_name": call.name,
                                    "arguments": call.arguments,
                                    "success": False,
                                    "error": (
                                        "Double-check the answer, then call "
                                        "submit_answer again with the exact same value."
                                    ),
                                }
                            )
                            continue
                result = await session.execute(
                    Action(
                        name=call.name,
                        arguments=call.arguments,
                        content=content if index == 0 else None,
                    ),
                    usage=pending_usage,
                )
                pending_usage = None
                executed = True
                if call.name == SUBMIT_ANSWER_TOOL_NAME:
                    if not result.success:
                        return AgentOutcome(
                            status="protocol_failure",
                            error=f"submit_answer failed: {result.error}",
                            usage=run.usage.outcome(),
                        )
                    if (
                        session.surrender_allowed
                        and answer.casefold() == SURRENDER_SENTINEL.casefold()
                    ):
                        return AgentOutcome(
                            status="surrendered", usage=run.usage.outcome()
                        )
                    return AgentOutcome(
                        status="completed",
                        answer=answer,
                        usage=run.usage.outcome(),
                    )
                observations.append(
                    {
                        "tool_name": call.name,
                        "arguments": call.arguments,
                        "success": result.success,
                        "result": self._truncate_observation(result.result),
                        "error": result.error,
                    }
                )

            if not executed:
                await session.record_message(assistant, usage=turn_usage)
            observation_message = {
                "role": "user",
                "content": json.dumps(
                    {"observations": observations}, ensure_ascii=False, default=str
                ),
            }
            run.messages.append(observation_message)
            if not executed:
                await session.record_message(observation_message)

        error = f"agent exhausted its {iteration_limit} interaction budget"
        await session.record_message(
            {"role": "assistant", "content": error, "name": "terminus-error"}
        )
        return AgentOutcome(
            status="iteration_limit", error=error, usage=run.usage.outcome()
        )


__all__ = ["TerminusAgent", "TerminusResponse", "TerminusToolCall"]
