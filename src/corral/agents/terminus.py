"""Corral-native Terminus agent.

This is a Corral-native reimplementation of the Terminal-Bench "Terminus"
scaffold. Terminus itself is tightly coupled to Terminal-Bench (its action is a
batch of terminal keystrokes executed against a `TmuxSession` and its
observations are captured terminal panes), which does not map onto Corral where
tools are retrieved per task and dispatched exclusively through
`CorralRouter`.

Rather than importing the Terminal-Bench class, this agent ports its
control-flow idea: on each turn the model returns a single structured JSON
object with `analysis` and `plan` fields plus exactly one terminal action
(`tool_calls`, `final_answer`, or `surrender`). Tool calls are executed
through the router and the structured observations are fed back for the next
turn. Plain JSON is used instead of provider-native function calling so the
scaffold behaves comparably across models; when the provider supports it the
schema is additionally enforced through LiteLLM's `response_format`.

It also ports the two behaviors that characterize the Terminus-2 revision of
the scaffold — context summarization once the transcript grows large, and
completion confirmation (a proposed `final_answer` must be re-affirmed before
it is accepted). Both are opt-out via `__init__` arguments.

The Terminal-Bench / Harbor projects that inspired this control loop are
Apache-2.0 licensed; this file is an independent reimplementation and copies no
code from them.
"""

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from corral.agents.base_agent import BaseAgent
from corral.agents.hooks import HookPoint
from corral.agents.schema import SURRENDER_SENTINEL
from corral.agents.utils import LiteLLMMessage, llm_call
from corral.router.routes import CorralRouter

# `SURRENDER_SENTINEL` is the shared sentinel returned by `run` when the agent
# gives up; `CorralRunner` checks for exactly this string before calling
# `surrender_task()`. Re-exported here for backwards compatibility.
__all__ = ["SURRENDER_SENTINEL", "TerminusAgent"]


class TerminusToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TerminusResponse(BaseModel):
    """Structured response the model must return on every turn.

    Exactly one *terminal action* must be present: either the model requests one
    or more `tool_calls`, or it provides a `final_answer`, or it decides to
    `surrender`. The `analysis` and `plan` fields make the model's
    reasoning explicit and are recorded in the transcript.
    """

    model_config = ConfigDict(extra="forbid")

    analysis: str
    plan: str
    tool_calls: list[TerminusToolCall] = Field(default_factory=list)
    final_answer: str | None = None
    surrender: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> "TerminusResponse":
        terminal_actions = sum(
            [
                bool(self.tool_calls),
                self.final_answer is not None,
                self.surrender,
            ]
        )
        if terminal_actions != 1:
            raise ValueError(
                "Exactly one of tool_calls, final_answer, or surrender "
                "must be provided."
            )
        return self


def _strip_code_fence(text: str) -> str:
    """Remove a single outer Markdown code fence, if present.

    Correctly strips a triple-backtick fence (with an optional language tag)
    from both ends; text without a fence is returned unchanged. The backtick
    runs are matched with ``` `+ ``` so a standard three-backtick fence is
    consumed whole rather than leaving a stray backtick behind.
    """
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line (backticks optionally followed by a language).
        text = re.sub(r"^`+[^\n]*\n?", "", text)
        # Drop the closing fence.
        text = re.sub(r"\n?`+\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from a model response.

    Used as a fallback when strict whole-payload parsing fails: it tolerates a
    code fence and prose surrounding the object so a small formatting slip does
    not waste a turn.
    """
    text = _strip_code_fence(text)

    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the response.")

    value, _ = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object.")

    return value


class TerminusAgent(BaseAgent):
    """Terminus-2-style agent restricted to Corral tools.

    Unlike :class:`~corral.agents.react.ReActAgent` (XML-like text parsing) and
    :class:`~corral.agents.tool_calling.ToolCallingAgent` (provider-native
    function calling), this agent drives the model with a validated structured
    JSON protocol. Each turn yields explicit `analysis`/`plan` fields and a
    single terminal action, and observations are returned as structured JSON.

    In addition to that core loop it implements the two behaviors that
    distinguish Terminal-Bench's Terminus-2 scaffold:

    * **Context summarization.** Once the running prompt grows past
      `summarize_after_tokens`, the older middle of the transcript is replaced
      by a model-generated progress summary, keeping the system prompt, the
      original task, and the most recent turns intact. This lets long episodes
      continue without blowing the context window.
    * **Completion confirmation.** A proposed `final_answer` is not accepted
      immediately. The model is asked to double-check it against the task and
      the observed tool results and must re-affirm the *same* answer
      `confirmations_required` more time(s) before the run returns it. This
      guards against premature completion.

    Both features are opt-out: set `summarize_after_tokens=None` to disable
    summarization or `confirmations_required=0` to accept the first final answer.

    Args:
        model (str): The model to use for running the agent. Defaults to
            `"openai/gpt-4o"`.
        max_iterations (int, optional): The maximum number of turns (LLM calls)
            to run. Defaults to 30.
        max_actions_per_turn (int, optional): The maximum number of `tool_calls`
            the model may request per turn. Must be >= 1. Defaults to 1, which
            forces the model to inspect each result before acting again — the
            safest setting for scientific environments where a tool call may
            mutate hidden state. A response that requests *more* than this is
            rejected with feedback (never silently truncated). Increase (e.g. 4)
            to allow Terminus-style batching of independent operations as a
            scaffold ablation.
        max_observation_chars (int | None, optional): Per-observation character
            budget. Larger tool results are head/tail truncated before being fed
            back so a single big result cannot blow the context window. `None`
            disables truncation. Defaults to 20000.
        max_consecutive_parse_failures (int, optional): How many consecutive
            unparseable/invalid responses to tolerate before aborting the
            episode, mirroring Terminus-1's bounded retry. Defaults to 3.
        use_structured_output (bool, optional): When True, request the schema via
            LiteLLM's `response_format` so providers that support it constrain
            the output. Automatically disabled for the rest of the run if the
            provider rejects it, falling back to the tolerant text parser.
            Defaults to True.
        confirmations_required (int, optional): How many times the model must
            re-affirm the same `final_answer` before it is accepted. `1`
            (the default) yields the classic Terminus-2 double confirmation
            (propose once, confirm once). `0` accepts immediately.
        summarize_after_tokens (int | None, optional): Token threshold (measured
            from the previous turn's total prompt+completion tokens) above which
            the transcript is summarized. `None` disables summarization.
            Defaults to 60000.
        keep_recent_messages (int, optional): How many of the most recent
            messages to preserve verbatim (never summarized). Defaults to 6.
        api_endpoint (str, optional): The API endpoint URL for the LLM provider.
            Defaults to None, meaning LiteLLM's default routing is used.
        system_prompt (str, optional): The system prompt to use. If None, uses
            the default corral system prompt.
        user_prompt (str, optional): Unused by this agent (the turn prompt is
            built internally) but kept for interface compatibility with the base
            machinery. Defaults to `"tool_calling/user_prompt"`.
        extractor_prompt (str, optional): The extractor prompt for cleaning final
            answers. If None, uses the default extractor prompt.
        surrender_prompt (str, optional): Instructions for surrendering an
            unsolvable task, inserted into the turn prompt when
            `enable_surrender=True`.
        temperature (float, optional): The sampling temperature. Defaults to 0.7.
        **kwargs: Additional keyword arguments forwarded to the LiteLLM API.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 30,
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
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """Initialize the agent."""
        if max_actions_per_turn < 1:
            raise ValueError(
                f"max_actions_per_turn must be >= 1, got {max_actions_per_turn}."
            )

        # This agent builds its own per-turn prompt and does not use
        # `self.user_prompt`, but the base class still resolves it, so give it
        # a valid default rather than letting `None` crash initialization.
        if user_prompt is None:
            user_prompt = "tool_calling/user_prompt"

        super().__init__(
            model=model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
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
        # Stored for verbose transcript logging (see BaseAgent.run_agent).
        self._available_tools: list[dict[str, Any]] = []
        # Populated per run: tool name -> the tool's `function` definition.
        self._tools_by_name: dict[str, dict[str, Any]] = {}
        # Confirmation state for the currently-pending answer. Reset at the
        # start of every `run()` so it never leaks across tasks.
        self._pending_answer: str | None = None
        self._affirmations = 0

    def _initial_prompt(
        self,
        task: str | list[dict[str, Any]],
        tools: list[dict[str, Any]],
        enable_surrender: bool = False,
        examples: list[str] | None = None,
    ) -> str:
        """Build the first user message describing the task and protocol."""
        task_text = (
            task if isinstance(task, str) else json.dumps(task, ensure_ascii=False)
        )

        schema = TerminusResponse.model_json_schema()

        if enable_surrender:
            if self.surrender_prompt is not None:
                surrender_rule = self.surrender_prompt.fill({})
            else:
                surrender_rule = (
                    "Set surrender=true only if the task is genuinely impossible "
                    "with the available tools."
                )
        else:
            surrender_rule = (
                "Do not surrender; keep working until you can give a final_answer."
            )

        examples_block = ""
        if examples:
            joined = "\n\n".join(str(example) for example in examples)
            examples_block = f"\nWorked examples:\n{joined}\n"

        return f"""
You are a Terminus-style autonomous scientific agent.

Complete the task using only the Corral tools listed below. You do not
have access to a shell, terminal, filesystem, Python interpreter, browser,
or any other tools unless one is explicitly listed.

Task:
{task_text}
{examples_block}
Available Corral tools:
{json.dumps(tools, indent=2, ensure_ascii=False)}

Return exactly one JSON object matching this schema:
{json.dumps(schema, indent=2)}

Rules:
- Never invent a tool.
- Tool arguments must satisfy the provided JSON Schema.
- Use final_answer only when no more tools are needed.
- Provide exactly one of tool_calls, final_answer, or surrender per response.
- Request at most {self.max_actions_per_turn} tool call(s) per response.
- Tool calls execute sequentially in the listed order.
- Do not batch calls when a later call depends on an earlier result.
- {surrender_rule}
- Do not include Markdown fences or any text outside the JSON object.
""".strip()

    def _parse(self, content: str) -> TerminusResponse:
        """Parse and validate a raw model response into a TerminusResponse.

        Tries a strict parse of the whole (de-fenced) payload first — which is
        what a structured-output response yields — then falls back to tolerant
        object extraction for models that wrap the JSON in prose.
        """
        stripped = _strip_code_fence(content)
        try:
            return TerminusResponse.model_validate_json(stripped)
        except ValidationError:
            return TerminusResponse.model_validate(_extract_json_object(content))

    def _call_model(self) -> Any:
        """Get one model response, using structured output when available.

        If the provider rejects `response_format` the failure is caught,
        structured output is disabled for the remainder of the run, and the
        request is retried once without it so a single unsupported provider does
        not abort the episode.
        """
        if self.use_structured_output:
            try:
                return self.get_llm_response(response_format=TerminusResponse)
            except Exception as exc:  # provider may not support response_format
                logger.warning(
                    f"Structured output failed ({exc}); disabling it for this run "
                    "and falling back to text parsing."
                )
                self.use_structured_output = False
        return self.get_llm_response()

    def _handle_final_answer(self, answer: str) -> str | None:
        """Require the model to re-affirm the same answer before accepting it.

        Returns the answer string to accept and return from `run`, or `None`
        to keep looping until the same answer has been re-affirmed
        `confirmations_required` times. With `confirmations_required=0` the
        first proposal is accepted immediately.
        """
        if self.confirmations_required <= 0:
            return answer

        if answer != self._pending_answer:
            # A new (or changed) proposal: start counting confirmations afresh.
            self._pending_answer = answer
            self._affirmations = 1
        else:
            self._affirmations += 1

        if self._affirmations > self.confirmations_required:
            return answer

        self.messages.append(
            LiteLLMMessage(
                role="user",
                content=(
                    "You proposed a final answer:\n"
                    f"{answer}\n\n"
                    "Before completing, double-check it against the task and the "
                    "tool results so far. If you are certain it is correct and "
                    "complete, resubmit the exact same final_answer to confirm. "
                    "Otherwise, keep working with the available Corral tools."
                ),
            )
        )
        return None

    def _maybe_compact_context(self) -> None:
        """Summarize the middle of the transcript once it grows too large."""
        if self.summarize_after_tokens is None:
            return

        total_tokens = self.token_usage.get("total_tokens", 0)
        if total_tokens < self.summarize_after_tokens:
            return

        # Preserve the system prompt + original task prompt (head) and the most
        # recent turns (tail); summarize everything in between.
        head = 2
        tail = self.keep_recent_messages
        if len(self.messages) <= head + tail + 1:
            return

        to_summarize = self.messages[head:-tail]
        if not to_summarize:
            return

        summary = self._summarize(to_summarize)
        if not summary:
            return

        self.messages = [
            *self.messages[:head],
            LiteLLMMessage(
                role="user",
                content=f"Summary of earlier progress so far:\n{summary}",
            ),
            *self.messages[-tail:],
        ]
        logger.info(
            f"Terminus compacted context: summarized {len(to_summarize)} messages"
        )

    def _summarize(self, messages: list[dict[str, Any]]) -> str | None:
        """Produce a concise progress summary of `messages` via the model.

        Returns None on failure so the caller leaves the transcript untouched
        rather than discarding history it could not summarize.
        """
        transcript = "\n\n".join(
            f"[{message.get('role')}] {message.get('content')}" for message in messages
        )
        prompt = (
            "Summarize the following agent transcript into a concise progress "
            "note. Capture what has been tried, the key tool results and "
            "observations (preserve concrete values that may be needed later), "
            "and what remains to be done. Do not invent information.\n\n"
            f"{transcript}"
        )
        try:
            response = llm_call(
                model=self.model,
                messages=[LiteLLMMessage(role="user", content=prompt)],
                temperature=0.0,
                api_endpoint=self.api_endpoint,
                return_usage=True,
                **self.kwargs,
            )
        except Exception as exc:  # keep the run going on summarization failure
            logger.warning(f"Terminus context summarization failed: {exc}")
            return None
        # Count the summarization call against the run's token budget so
        # compaction cost is not hidden from benchmark accounting.
        if response.usage:
            self._accumulate_token_usage(response.usage)
        return response.content or None

    def _truncate_observation(self, value: Any) -> Any:
        """Bound a tool result so a single large output cannot exhaust context.

        Small results are returned unchanged (preserving their original type);
        oversized results are serialized and head/tail truncated with a marker.
        """
        limit = self.max_observation_chars
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        if limit is None or len(text) <= limit:
            return value
        head = limit // 2
        tail = limit - head
        omitted = len(text) - limit
        return f"{text[:head]}\n...[truncated {omitted} chars]...\n{text[-tail:]}"

    def _validate_arguments(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Validate tool arguments against the tool's JSON Schema locally.

        Returns an error string if the arguments are invalid, or None if they
        are valid (or cannot be checked). This gives the model cleaner feedback
        and avoids a doomed round-trip to the environment.
        """
        schema = self._tools_by_name.get(name, {}).get("parameters")
        if not schema:
            return None
        errors = sorted(
            Draft202012Validator(schema).iter_errors(arguments),
            key=lambda e: list(e.path),
        )
        if not errors:
            return None
        return "; ".join(error.message for error in errors[:5])

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        enable_surrender: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        """Run the Terminus loop to solve a task.

        Args:
            interface (CorralRouter): The interface to the environment server.
            task_id (str): The task ID to solve.
            task_prompt (str, optional): The task prompt to use. If None, the
                task prompt is fetched from the environment. Defaults to None.
            examples (list[str], optional): Worked examples included in the turn
                prompt. Defaults to None.
            enable_surrender (bool, optional): Whether to allow the agent to give
                up on an unsolvable task. Defaults to False.

        Returns:
            str: The final answer to the task, `"SURRENDER"` if the agent gave
            up (and surrender is enabled), or an `"Error solving the task: ..."`
            string if the iteration or parse-failure limit is reached.
        """
        tool_payload = interface.get_available_tools_for_task(task_id)
        self._available_tools = tool_payload.get("tools", [])
        self._tools_by_name = {
            tool["function"]["name"]: tool["function"]
            for tool in self._available_tools
            if "function" in tool
        }
        allowed_names = set(self._tools_by_name)

        if self._initial_messages is not None:
            self.messages = list(self._initial_messages)
        else:
            task = (
                task_prompt
                if task_prompt is not None
                else interface.get_task_prompt(task_id)
            )
            self.messages = [
                LiteLLMMessage(role="system", content=self.system_prompt),
                LiteLLMMessage(
                    role="user",
                    content=self._initial_prompt(
                        task, self._available_tools, enable_surrender, examples
                    ),
                ),
            ]

        self._execute_hooks(HookPoint.BEFORE_TASK, interface, task_id)

        # Reset completion-confirmation state so a pending answer from an
        # earlier task solved by this same instance never leaks into this run.
        self._pending_answer = None
        self._affirmations = 0

        consecutive_parse_failures = 0

        for iteration in range(self.max_iterations):
            self._current_iteration = iteration

            self._execute_hooks(HookPoint.BEFORE_ITERATION, interface, task_id)

            self._maybe_compact_context()

            response = self._call_model()
            content = response.content or ""

            self.messages.append(
                LiteLLMMessage(role="assistant", content=content, id=response.id)
            )

            try:
                parsed = self._parse(content)
            except (ValueError, ValidationError) as exc:
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= self.max_consecutive_parse_failures:
                    self.messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=(
                                "Error: too many consecutive invalid responses "
                                f"({consecutive_parse_failures})."
                            ),
                            name="terminus-error",
                        )
                    )
                    logger.error(
                        "Terminus agent aborting after "
                        f"{consecutive_parse_failures} invalid responses"
                    )
                    return (
                        "Error solving the task: the model repeatedly returned "
                        "invalid responses."
                    )
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            "Your previous response was invalid.\n"
                            f"Validation error: {exc}\n"
                            "Return a corrected JSON object only."
                        ),
                    )
                )
                continue

            consecutive_parse_failures = 0

            if parsed.surrender:
                if enable_surrender:
                    logger.info(f"Agent surrendering from task {task_id}")
                    return SURRENDER_SENTINEL

                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            "Surrender is disabled. Continue attempting the task "
                            "using the available Corral tools."
                        ),
                    )
                )
                continue

            if parsed.final_answer is not None:
                accepted = self._handle_final_answer(parsed.final_answer)
                self._execute_hooks(
                    HookPoint.AFTER_ITERATION,
                    interface,
                    task_id,
                    llm_response=response,
                )
                if accepted is not None:
                    return accepted
                continue

            # Reject over-budget batches instead of silently dropping calls, so
            # the model is never left believing a discarded call executed.
            if len(parsed.tool_calls) > self.max_actions_per_turn:
                self.messages.append(
                    LiteLLMMessage(
                        role="user",
                        content=(
                            f"You requested {len(parsed.tool_calls)} tool calls, "
                            f"but at most {self.max_actions_per_turn} may be "
                            "executed per turn. Return a corrected response "
                            "containing only the next call(s) to execute."
                        ),
                    )
                )
                continue

            observations: list[dict[str, Any]] = []

            for call in parsed.tool_calls:
                if call.name not in allowed_names:
                    observations.append(
                        {
                            "tool_name": call.name,
                            "success": False,
                            "error": "Unknown or unavailable Corral tool.",
                        }
                    )
                    continue

                arg_error = self._validate_arguments(call.name, call.arguments)
                if arg_error is not None:
                    observations.append(
                        {
                            "tool_name": call.name,
                            "arguments": call.arguments,
                            "success": False,
                            "error": f"Invalid arguments: {arg_error}",
                        }
                    )
                    continue

                result = interface.execute_tool(task_id, call.name, call.arguments)

                observations.append(
                    {
                        "tool_name": call.name,
                        "arguments": call.arguments,
                        "success": result.success,
                        "result": self._truncate_observation(result.result),
                        "error": result.error,
                    }
                )

            self.messages.append(
                LiteLLMMessage(
                    role="user",
                    content=json.dumps(
                        {"observations": observations},
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            )

            self._execute_hooks(
                HookPoint.AFTER_ITERATION,
                interface,
                task_id,
                llm_response=response,
            )

        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content="Error: Maximum iterations reached without a final answer.",
                name="terminus-error",
            )
        )
        logger.error("Terminus agent reached its iteration limit")
        return "Error solving the task: unable to complete it in the iteration limit"
