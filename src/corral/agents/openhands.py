import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal

from loguru import logger

# The OpenHands SDK ships as the optional `corral[openhands]` extra, which is
# gated to Python >= 3.12. Wrap the import so an environment without the extra
# gets an actionable install hint instead of a bare ModuleNotFoundError deep in
# the import chain.
try:
    from openhands.sdk import (
        LLM,
        Agent,
        AgentContext,
        Conversation,
        LLMConvertibleEvent,
    )
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.event import (
        ActionEvent,
        MessageEvent,
        ObservationEvent,
        SystemPromptEvent,
    )
    from openhands.sdk.event.conversation_error import ConversationErrorEvent
    from openhands.sdk.mcp import MCPServer
    from openhands.sdk.mcp.exceptions import MCPError
    from openhands.sdk.mcp.tool import MCP_TOOL_TIMEOUT_SECONDS
    from openhands.sdk.tool.builtins import FinishAction
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extras
    raise ModuleNotFoundError(
        "OpenHandsAgent requires the OpenHands SDK, which ships as the optional "
        "'openhands' extra (Python >= 3.12 only). Install it with "
        "`pip install 'corral[openhands]'`."
    ) from exc
from pydantic import SecretStr

from corral.agents.base_agent import BaseAgent, prompt_with_state_history
from corral.agents.schema import SURRENDER_SENTINEL, AgentOutcome, AgentUsage
from corral.agents.session import AgentSession
from corral.agents.utils import LiteLLMMessage

# Name under which the corral MCP server is registered with the OpenHands
# harness (the `mcp_config` table key).
_MCP_SERVER_NAME = "corral"

# OpenHands' internal-only tools: `FinishTool` lets the agent terminate with a
# final answer and `ThinkTool` lets it reason. Neither can act outside the
# process, so the corral MCP endpoint stays the only actionable surface.
_INCLUDED_DEFAULT_TOOLS = ["FinishTool", "ThinkTool"]

# Runtime tool *names* the two included default tools resolve to (OpenHands maps
# the `FinishTool`/`ThinkTool` specs to tools named `finish`/`think`). Used to
# build the expected runtime tool allowlist for the isolation self-check.
_INCLUDED_DEFAULT_TOOL_NAMES = frozenset({"finish", "think"})

# OpenHands 1.35.0 caps every MCP tool call at a fixed *executor* timeout and
# does NOT propagate `MCPServer.timeout` into that executor — the server-level
# timeout only bounds the HTTP transport, while `MCPToolExecutor` wraps each call
# in its own `MCP_TOOL_TIMEOUT_SECONDS` deadline and cancels first. So a per-call
# timeout above this cap cannot actually be honored. The value is read from the
# installed SDK (rather than hard-coded) so the recorded provenance tracks the
# pinned version, and `tool_timeout_s` defaults to it so the configured timeout
# is enforceable end-to-end out of the box.
_OPENHANDS_EXECUTOR_TIMEOUT_S = float(MCP_TOOL_TIMEOUT_SECONDS)

# Pinned agent identity, forwarded as `Agent.system_prompt_kwargs["soul_content"]`
# so the system prompt cannot silently inherit a machine-local
# `~/.openhands/SOUL.md`. Keeping the identity fixed makes the benchmarked prompt
# reproducible across machines.
_PINNED_SOUL_CONTENT = (
    "You are OpenHands agent, a helpful AI assistant that can interact with the "
    "provided tools to solve tasks."
)

# Terminal status of a single harness run. Distinguishing these lets the
# benchmark record *why* a run ended instead of collapsing infrastructure
# failures (SDK crashes and MCP transport failures) into a wrong model answer.
HarnessStatus = Literal[
    "success",
    "surrender",
    "max_iterations",
    "tool_failure",
    "sdk_failure",
]


@dataclass
class HarnessRunResult:
    """Structured outcome of one OpenHands harness run.

    ``OpenHandsAgent.run_session`` maps this provider result onto
    ``AgentOutcome`` so infrastructure failures are never scored as wrong
    answers.
    """

    status: HarnessStatus
    answer: str | None = None
    error: str | None = None
    num_events: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RunState:
    """Mutable scratch owned by one invocation, never by the agent instance."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    result: HarnessRunResult | None = None


class _OpenHandsError(RuntimeError):
    """Raised when the OpenHands harness ends in a non-success state.

    Carries a `subtype` so the caller can map it onto a :data:`HarnessStatus`
    instead of treating every failure the same way.
    """

    def __init__(self, message: str, subtype: str | None = None):
        super().__init__(message)
        self.subtype = subtype


def _sdk_version() -> str | None:
    """Best-effort version of the installed `openhands-sdk` package."""
    try:
        return _pkg_version("openhands-sdk")
    except PackageNotFoundError:  # pragma: no cover - only without the SDK
        return None


def _content_to_text(content: Any) -> str:
    """Flatten an OpenHands message's content for the corral transcript.

    The content is either a plain string or a list of content parts (each of
    which usually exposes a `.text` attribute); anything non-textual is
    stringified so it round-trips through the saved transcript.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            parts.append(text if isinstance(text, str) else str(part))
        return "\n".join(parts)
    return str(content)


def _truncate(text: Any, limit: int = 200) -> str:
    """Collapse whitespace and cap `text` to one readable log line.

    Tool results (e.g. NMR/mass-spectra payloads) can be large JSON blobs, so
    live progress logs must be bounded to stay legible.
    """
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"


class OpenHandsAgent(BaseAgent):
    """Agent that delegates solving a task to the OpenHands harness.

    Args:
        model (str): The model the OpenHands harness should use, as a
            LiteLLM-style route (e.g. `"openai/gpt-5.6"`). Passed to the
            harness `LLM` verbatim. Defaults to `"openai/gpt-5.6"`.
        api_key (str, optional): API key for the harness `LLM`. If None, falls
            back to the `LLM_API_KEY` environment variable. Defaults to None.
        api_endpoint (str, optional): Base URL for the harness `LLM` (mapped to
            the SDK `base_url`). Defaults to None.
        temperature (float, optional): Sampling temperature for the harness
            `LLM`. Defaults to 0.7.
        reasoning_effort (str, optional): Reasoning effort forwarded to the
            harness `LLM` (`"low"`, `"medium"`, `"high"`, `"xhigh"`, or
            `"none"`). If None, the OpenHands SDK default (`"high"`) applies and
            the value is left unset on the `LLM`. Defaults to None.
        tool_timeout_s (float, optional): MCP transport (HTTP request) timeout
            passed to `MCPServer.timeout`. NOTE: OpenHands separately caps each
            MCP tool *call* at a fixed executor timeout
            (`MCP_TOOL_TIMEOUT_SECONDS`) that `MCPServer.timeout` does not
            override, so the effective per-call ceiling is
            `min(tool_timeout_s, executor cap)`. Values above the cap are honored
            only at the transport layer and log a warning. Defaults to the
            executor cap (300) so the configured timeout is enforceable
            end-to-end.
        system_prompt (str, optional): Instructions appended to the OpenHands
            harness system prompt (via `AgentContext.system_message_suffix`). If
            None, uses the default corral system prompt. Tool-usage and
            final-answer instructions are appended automatically.
        **kwargs: Additional provider configuration retained as provenance.

    ``run_session`` drives ``Conversation.arun()`` directly on the scheduler's
    event loop and exposes environment tools exclusively through the session's
    task-local MCP endpoint.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5.6",
        api_key: str | None = None,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: Literal["low", "medium", "high", "xhigh", "none"]
        | None = None,
        tool_timeout_s: float = _OPENHANDS_EXECUTOR_TIMEOUT_S,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        **kwargs,
    ):
        removed_limits = {"wall_clock_timeout_s", "interrupt_grace_s"} & kwargs.keys()
        if removed_limits:
            names = ", ".join(sorted(removed_limits))
            raise TypeError(
                f"{names} is no longer supported by OpenHandsAgent; "
                "configure limits in BenchmarkTaskMetadata"
            )
        if user_prompt is None:
            user_prompt = "tool_calling/user_prompt"

        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

        # Model string handed to the OpenHands harness itself.
        self.harness_model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.tool_timeout_s = tool_timeout_s
        # OpenHands caps each MCP call at its own executor timeout regardless of
        # `MCPServer.timeout`, so the value actually enforced per call is the
        # smaller of the two. Warn loudly when a caller asks for more than the
        # SDK can honor rather than silently under-delivering.
        if tool_timeout_s > _OPENHANDS_EXECUTOR_TIMEOUT_S:
            logger.warning(
                f"tool_timeout_s={tool_timeout_s}s exceeds the OpenHands MCP "
                f"executor cap of {_OPENHANDS_EXECUTOR_TIMEOUT_S}s; OpenHands does "
                "not propagate MCPServer.timeout into its executor, so each MCP "
                f"call will still be cancelled after {_OPENHANDS_EXECUTOR_TIMEOUT_S}s."
            )
        self.effective_tool_timeout_s = min(
            tool_timeout_s, _OPENHANDS_EXECUTOR_TIMEOUT_S
        )

    def _system_message_suffix(self, enable_surrender: bool) -> str:
        """Compose the suffix appended to the OpenHands harness system prompt.

        This does not *replace* the OpenHands system prompt (that is left to the
        harness so this condition measures the real harness), it augments it via
        `AgentContext.system_message_suffix`.
        """
        final_answer_directive = (
            "When you have solved the task, call the `submit_answer` MCP tool "
            "with the complete answer. A plain-text final response does not "
            "complete the task."
        )
        suffix = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed evaluation environment. "
            f"You may ONLY interact with it through the provided `{_MCP_SERVER_NAME}` "
            "MCP tools; do not attempt to use the shell, filesystem, web, or any "
            "other tool, and do not request additional permissions. Do not invent "
            "tool outputs. " + final_answer_directive
        )
        if enable_surrender:
            if self.surrender_prompt is not None:
                suffix += "\n\n" + self.surrender_prompt.fill({})
            suffix += (
                "\nTo surrender, call `submit_answer` with the exact answer "
                f"`{SURRENDER_SENTINEL}`; do not return the sentinel as text."
            )
        return suffix

    def _make_llm(self) -> Any:
        """Build the OpenHands `LLM` handed to the harness agent."""
        key = self.api_key or os.getenv("LLM_API_KEY")

        llm_kwargs: dict[str, Any] = {
            "usage_id": "corral-openhands",
            "model": self.harness_model,
            "temperature": self.temperature,
            "stream": True,
        }
        # Only set `reasoning_effort` when explicitly requested; leaving it unset
        # defers to the SDK default rather than silently pinning a value.
        if self.reasoning_effort is not None:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort
        if key:
            llm_kwargs["api_key"] = SecretStr(key)
        if self.api_endpoint:
            llm_kwargs["base_url"] = self.api_endpoint
        return LLM(**llm_kwargs)

    def _build_agent(self, llm: Any, mcp_url: str, enable_surrender: bool) -> Any:
        """Assemble the fully-isolated OpenHands `Agent` for a run."""
        return Agent(
            llm=llm,
            # Critical: register no OpenHands terminal/file-editor/browser tools
            # and do NOT use the default tool preset. All externally-acting tools
            # come exclusively from the corral MCP endpoint.
            tools=[],
            # Keep only OpenHands' internal reasoning/termination tools.
            include_default_tools=list(_INCLUDED_DEFAULT_TOOLS),
            mcp_config={
                _MCP_SERVER_NAME: MCPServer(
                    url=mcp_url,
                    transport="streamable-http",
                    timeout=self.tool_timeout_s,
                )
            },
            # Augment (do not replace) the OpenHands harness system prompt.
            agent_context=AgentContext(
                # Pin the datetime off: `AgentContext.current_datetime` otherwise
                # defaults to the machine's local wall-clock time and is injected
                # into the system prompt, so two runs of the same task would see
                # different prompts depending on the minute/timezone. `None`
                # omits the datetime block, keeping the benchmarked prompt
                # reproducible across machines. The actual rendered system prompt
                # is additionally hashed from the `SystemPromptEvent` (see
                # `_record_system_prompt_event`) to prove reproducibility.
                current_datetime=None,
                system_message_suffix=self._system_message_suffix(enable_surrender),
            ),
            # Pin the agent identity so the system prompt cannot inherit a
            # machine-local `~/.openhands/SOUL.md` (which OpenHands would
            # otherwise load when `soul_content` is absent), keeping the
            # benchmarked prompt reproducible across machines.
            system_prompt_kwargs={"soul_content": _PINNED_SOUL_CONTENT},
        )

    def _harness_metadata(
        self,
        mcp_url: str,
        verbosity: str,
        tools: list[dict[str, Any]],
        mcp_schema_sha256: str | None,
        iteration_limit: int,
    ) -> dict[str, Any]:
        """Static provenance for the run so the harness stack is reproducible.

        Two schema hashes are recorded and named for what they are: the
        REST/OpenAI function-calling schema the agent fetched, and — when the
        server exposes it — the *MCP* schema (`tools/list`) that OpenHands
        actually receives. They are logically related but not byte-for-byte
        equal, so the MCP hash is the authoritative record of what the harness
        saw.
        """
        rest_tool_schema = json.dumps(
            [t.get("function", {}) for t in tools], sort_keys=True, default=str
        )
        return {
            "model_requested": self.harness_model,
            # None records that the SDK default reasoning effort was left in place.
            "reasoning_effort": self.reasoning_effort,
            "openhands_sdk_version": _sdk_version(),
            "python_version": platform.python_version(),
            "max_iterations_configured": iteration_limit,
            "run_limit_policy": "iterations_only",
            "stuck_detection_enabled": False,
            "streaming_enabled": True,
            # Timeout provenance, named for what OpenHands actually enforces. The
            # configured value is the MCP transport (HTTP request) timeout passed
            # to `MCPServer.timeout`; per-call execution is separately capped by
            # the SDK's fixed executor timeout, which `MCPServer.timeout` does not
            # override. The effective per-call ceiling is the smaller of the two.
            "configured_mcp_timeout_s": self.tool_timeout_s,
            "openhands_executor_timeout_s": _OPENHANDS_EXECUTOR_TIMEOUT_S,
            "effective_tool_timeout_s": self.effective_tool_timeout_s,
            "mcp_url": mcp_url,
            "tool_verbosity": verbosity,
            "included_default_tools": list(_INCLUDED_DEFAULT_TOOLS),
            "system_prompt_identity_pinned": True,
            "mcp_tools_enabled": sorted(
                t["function"]["name"]
                for t in tools
                if t.get("function", {}).get("name")
            ),
            "rest_openai_tool_schema_sha256": hashlib.sha256(
                rest_tool_schema.encode("utf-8")
            ).hexdigest(),
            "mcp_tool_schema_sha256": mcp_schema_sha256,
        }

    def _normalize_prompt(self, task_guide: Any) -> tuple[str, bool]:
        """Flatten a raw task prompt into the plain-text OpenHands input.

        The initial implementation forwards text; multimodal parts are flattened
        to their text and image parts are dropped with a warning (the corral MCP
        tools, not the prompt, are the intended channel for task data). Returns
        the flattened text and whether any non-text parts were dropped, so the
        run metadata can flag that the harness saw a reduced view of the task.
        """
        if isinstance(task_guide, list):
            dropped_images = any(
                isinstance(p, dict) and p.get("type", "text") != "text"
                for p in task_guide
            )
            if dropped_images:
                logger.warning(
                    "OpenHandsAgent received multimodal prompt content; image "
                    "parts are not forwarded to the OpenHands harness. If the task "
                    "depends on the image and no MCP tool exposes it, the task is "
                    "not faithfully supported by this agent."
                )
            text = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in task_guide
            )
            return text, dropped_images
        return str(task_guide), False

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run OpenHands's native loop against the task-local MCP session."""
        run = _RunState()
        iteration_limit = session.iteration_limit
        tools = [dict(tool) for tool in session.tools]
        run.available_tools = tools
        prompt, dropped_images = self._normalize_prompt(session.prompt)
        prompt = prompt_with_state_history(prompt, session.messages)
        run.messages.append(dict(LiteLLMMessage(role="user", content=prompt)))
        task_id = str(
            session.initial_state.metadata.task.get("id") or session.execution_id
        )

        async with session.open_mcp() as mcp:
            run.metadata = self._harness_metadata(
                mcp.url,
                "full",
                tools,
                mcp_schema_sha256=None,
                iteration_limit=iteration_limit,
            )
            run.metadata["dropped_image_parts"] = dropped_images
            # Keep a conservative fallback for adapters/tests that do not expose
            # SDK metrics. The real OpenHands path replaces this with the count
            # of per-completion token-usage records in `_record_usage`.
            run.metadata["sdk_turns"] = 1
            await self._execute_harness(
                task_id=task_id,
                prompt=prompt,
                mcp_url=mcp.url,
                enable_surrender=session.surrender_allowed,
                iteration_limit=iteration_limit,
                run=run,
            )

        for message in run.messages:
            await session.record_message(message)
        usage = AgentUsage(
            input_tokens=int(run.usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(run.usage.get("completion_tokens", 0) or 0),
            llm_calls=int(run.metadata.get("sdk_turns", 0) or 0),
        )
        result = run.result
        if result is None:
            return AgentOutcome(
                status="harness_failure",
                error="OpenHands returned no structured result",
                usage=usage,
                metadata=dict(run.metadata),
            )
        metadata = {
            **dict(result.metadata),
            "harness_status": result.status,
        }
        submission = session.submission
        if submission is not None:
            if session.submission_status == "surrendered":
                return AgentOutcome(
                    status="surrendered", usage=usage, metadata=metadata
                )
            return AgentOutcome(
                status="completed",
                answer=submission,
                usage=usage,
                metadata=metadata,
            )
        if result.status in {"success", "surrender"}:
            return AgentOutcome(
                status="protocol_failure",
                error="OpenHands finished without calling submit_answer",
                usage=usage,
                metadata=metadata,
            )
        status_map = {
            "max_iterations": "iteration_limit",
            "tool_failure": "tool_failure",
            "sdk_failure": "harness_failure",
        }
        return AgentOutcome(
            status=status_map[result.status],
            error=result.error or "OpenHands harness failed",
            usage=usage,
            metadata=metadata,
        )

    async def _execute_harness(
        self,
        *,
        task_id: str,
        prompt: str,
        mcp_url: str,
        enable_surrender: bool,
        iteration_limit: int,
        run: _RunState,
    ) -> str:
        """Drive the harness and retain its terminal text for diagnostics.

        Owns the per-episode working directory and maps a drive failure to a
        precise terminal status via :meth:`_harness_failure_answer`. Completion
        still requires the session's ``submit_answer`` MCP tool.
        """
        # Fresh, empty, per-episode working directory; cleaned up afterwards.
        cwd = tempfile.mkdtemp(prefix="corral-openhands-")

        try:
            final_answer = await self._execute_openhands(
                prompt=prompt,
                mcp_url=mcp_url,
                cwd=cwd,
                enable_surrender=enable_surrender,
                iteration_limit=iteration_limit,
                run=run,
            )
        except Exception as e:  # surface as infra failure, not a model answer
            return self._harness_failure_answer(e, run)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)

        return self._finalize_answer(final_answer, task_id, enable_surrender, run)

    def _harness_failure_answer(self, exc: Exception, run: _RunState) -> str:
        """Map a harness-drive exception onto a `_fail` answer string.

        A typed :class:`_OpenHandsError` carries the precise terminal subtype
        (tool_failure / max_iterations / sdk_failure); any
        other exception is classified into a tool-vs-SDK failure so a broken
        benchmark run is never scored as a wrong answer.
        """
        if isinstance(exc, _OpenHandsError):
            status_map: dict[str, HarnessStatus] = {
                "tool_failure": "tool_failure",
                "max_iterations": "max_iterations",
                "sdk_failure": "sdk_failure",
            }
            return self._fail(
                run,
                status_map.get(exc.subtype or "", "sdk_failure"),
                str(exc),
                exc,
            )
        return self._fail(run, self._classify_error(exc), str(exc), exc)

    def _finalize_answer(
        self,
        final_answer: str,
        task_id: str,
        enable_surrender: bool,
        run: _RunState,
    ) -> str:
        """Normalize the harness output into diagnostic terminal text.

        This post-processing belongs to the single session adapter entry point.
        """
        final_answer = final_answer.strip()

        # Surrender only on an *exact* sentinel answer, so the marker appearing
        # inside a longer response is not mistaken for giving up.
        if (
            enable_surrender
            and final_answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            logger.info(f"Agent retiring from task {task_id}")
            run.result = self._result(run, "surrender", answer=SURRENDER_SENTINEL)
            return SURRENDER_SENTINEL

        if not final_answer:
            error = "OpenHands completed without a FinishAction message"
            run.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=f"Error: {error}.",
                    name="openhands-error",
                )
            )
            run.result = self._result(run, "sdk_failure", error=error)
            return f"Error solving the task: {error}"

        # Ensure the harness terminal text is in the transcript exactly once. The
        # harness's final message is usually already recorded, so only append when
        # it differs (ignoring surrounding whitespace).
        last_content = run.messages[-1].get("content") if run.messages else None
        already_recorded = (
            isinstance(last_content, str) and last_content.strip() == final_answer
        )
        if not already_recorded:
            run.messages.append(LiteLLMMessage(role="assistant", content=final_answer))
        run.result = self._result(run, "success", answer=final_answer)
        return final_answer

    async def _execute_openhands(
        self,
        *,
        prompt: str,
        mcp_url: str,
        cwd: str,
        enable_surrender: bool,
        iteration_limit: int,
        run: _RunState,
    ) -> str:
        """Drive the OpenHands conversation to completion and return its answer.

        Owns the full conversation lifecycle: builds the isolated agent, drives
        it until completion or the iteration limit, records usage/transcript,
        validates the runtime tool set and terminal state, extracts the terminal
        `FinishAction` message, and always closes the conversation.
        """
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)
            # Live progress first, so the run is observable even if transcript
            # recording of an event fails.
            self._log_event(event)
            try:
                self._record_event(event, run)
            except Exception:
                logger.debug(
                    "Could not record OpenHands event into transcript",
                    exc_info=True,
                )

        llm = self._make_llm()
        agent = self._build_agent(llm, mcp_url, enable_surrender=enable_surrender)
        conversation = Conversation(
            agent=agent,
            callbacks=[on_event],
            # OpenHands deliberately falls back to a blocking response when
            # streaming is enabled without a token callback. A no-op callback
            # keeps the SDK on its streaming transport while completed events
            # continue to be recorded through ``on_event`` above.
            token_callbacks=[lambda _chunk: None],
            workspace=cwd,
            max_iteration_per_run=iteration_limit,
            # The configured iteration count is the only agent-run limit. Tool
            # and transport timeouts remain local safeguards for one operation;
            # there is no wall-clock or stuck-detector cutoff for the run.
            stuck_detection=False,
            persistence_dir=None,
            # Silence the SDK's full-screen console visualizer; `on_event`
            # emits concise per-event progress lines via `_log_event` instead,
            # so the run stays observable without the TUI noise.
            visualizer=None,
        )

        try:
            conversation.send_message(prompt)
            await conversation.arun()
            # Fail loudly on a capability leak or a non-clean terminal state
            # (including max-iterations, which OpenHands does not raise for) so
            # it is recorded precisely instead of as a generic no-answer
            # `sdk_failure`.
            self._verify_runtime_tools(conversation, run)
            self._require_clean_completion(conversation, events)
        finally:
            self._record_usage(llm, run)
            run.metadata["num_events"] = len(events)
            try:
                conversation.close()
            except Exception:
                logger.warning("Failed to close OpenHands conversation", exc_info=True)

        return self._extract_finish_message(events)

    def _log_event(self, event: Any) -> None:
        """Emit one concise loguru line for a meaningful OpenHands event.

        The SDK's own console visualizer is disabled (`visualizer=None`) to
        avoid its full-screen TUI, which otherwise leaves the harness a black
        box while it runs — the only visible signal is the server-side MCP log.
        This surfaces the key beats (the agent's reasoning, each tool call and
        its result, run errors, and the final answer) as single lines so a run
        is observable. Best-effort: live logging must never break a run.
        """
        try:
            if isinstance(event, ActionEvent):
                action = getattr(event, "action", None)
                if isinstance(action, FinishAction):
                    message = (getattr(action, "message", "") or "").strip()
                    logger.info(f"[openhands] finish → {_truncate(message)}")
                    return
                thought = _content_to_text(getattr(event, "thought", "") or "").strip()
                if thought:
                    logger.info(f"[openhands] thinking: {_truncate(thought)}")
                tool_name = getattr(event, "tool_name", "") or "?"
                args = _truncate(json.dumps(self._action_arguments(event), default=str))
                logger.info(f"[openhands] call {tool_name}({args})")
            elif isinstance(event, ObservationEvent):
                observation = getattr(event, "observation", None)
                is_error = bool(getattr(event, "error", None)) or bool(
                    getattr(observation, "is_error", False)
                )
                tool_name = getattr(event, "tool_name", "") or "tool"
                text = _truncate(self._event_text(event))
                if is_error:
                    logger.warning(f"[openhands] {tool_name} error: {text}")
                else:
                    logger.info(f"[openhands] result {tool_name}: {text}")
            elif isinstance(event, ConversationErrorEvent):
                code = getattr(event, "code", None)
                detail = getattr(event, "detail", None) or ""
                logger.warning(f"[openhands] run error {code}: {_truncate(detail)}")
        except Exception:
            logger.debug("Could not log OpenHands event", exc_info=True)

    def _record_event(self, event: Any, run: _RunState) -> None:
        """Fold one OpenHands event into the inspectable corral transcript.

        Structured tool calls and their observations are preserved (name,
        call id, arguments, result, error flag) rather than collapsed to
        role+content, so tool-level trace analysis stays possible — matching
        the Codex/Claude Code transcripts.
        """
        if isinstance(event, ActionEvent):
            self._record_action_event(event, run)
        elif isinstance(event, ObservationEvent):
            self._record_observation_event(event, run)
        elif isinstance(event, SystemPromptEvent):
            self._record_system_prompt_event(event, run)
        elif isinstance(event, MessageEvent):
            self._record_message_event(event, run)
        elif isinstance(event, LLMConvertibleEvent):
            # Any other convertible event: keep its text, skipping the initial
            # user prompt (already seeded as the first transcript message).
            self._record_generic_event(event, run)

    def _record_action_event(self, event: Any, run: _RunState) -> None:
        """Record the model's reasoning and structured tool call."""
        # The terminal `FinishAction` carries the final answer, which is
        # appended exactly once by `_finalize_answer`; recording it here as a
        # nameless tool call would only duplicate it.
        if isinstance(getattr(event, "action", None), FinishAction):
            return
        thought = _content_to_text(getattr(event, "thought", "") or "")
        if thought:
            run.messages.append(
                LiteLLMMessage(role="assistant", content=thought, name="thinking")
            )
        run.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": getattr(event, "tool_call_id", "") or "",
                        "function": {
                            "name": getattr(event, "tool_name", "") or "",
                            "arguments": json.dumps(
                                self._action_arguments(event), default=str
                            ),
                        },
                    }
                ],
            }
        )

    def _record_system_prompt_event(self, event: Any, run: _RunState) -> None:
        """Record the actual rendered system prompt for reproducibility.

        Hashing the *rendered* prompt OpenHands built — not just the pinned
        soul/suffix — captures every input the model saw, so two runs of the
        same task are demonstrably identical. `dropped`/`dynamic_context` is
        flagged so any machine-dependent block (e.g. runtime info or secrets)
        that would break reproducibility is visible in the run provenance rather
        than silently changing the prompt.
        """
        prompt = getattr(event, "system_prompt", None)
        text = getattr(prompt, "text", None)
        if isinstance(text, str):
            run.metadata["system_prompt_sha256"] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            run.metadata["system_prompt_has_dynamic_context"] = (
                getattr(event, "dynamic_context", None) is not None
            )
        # Keep the system prompt itself in the transcript, at the *front*. The
        # task prompt is pre-seeded as `messages[0]` before the harness runs, but
        # the system message precedes it in the real conversation (OpenHands
        # emits the `SystemPromptEvent` as its first event), so insert it ahead of
        # the seeded prompt rather than appending — otherwise the saved transcript
        # records the system message second, out of order.
        message = event.to_llm_message()
        run.messages.insert(
            0,
            LiteLLMMessage(
                role=getattr(message, "role", "system"),
                content=_content_to_text(getattr(message, "content", "")),
            ),
        )

    def _record_observation_event(self, event: Any, run: _RunState) -> None:
        """Record an MCP tool result as a `tool`-role transcript message.

        The error flag reads the observation's `is_error` field: OpenHands
        `Observation`s (including `MCPToolObservation`, built with
        `is_error=result.isError`) expose the failure flag there, not as an
        `error` attribute, so a failed MCP call is recorded — and counted — as an
        error instead of silently passing as success.
        """
        observation = getattr(event, "observation", None)
        is_error = bool(getattr(event, "error", None)) or bool(
            getattr(observation, "is_error", False)
        )
        if is_error:
            run.metadata["tool_errors"] = run.metadata.get("tool_errors", 0) + 1
        tool_name = getattr(event, "tool_name", "") or ""
        run.messages.append(
            {
                "role": "tool",
                "tool_call_id": getattr(event, "tool_call_id", "") or "",
                "content": self._event_text(event),
                # Name the result after the actual tool, not a generic
                # "tool_result", so tool-level analysis stays possible.
                "name": tool_name or "tool_result",
                "is_error": is_error,
            }
        )

    def _record_message_event(self, event: Any, run: _RunState) -> None:
        """Record an assistant message (skipping echoed user messages)."""
        self._record_generic_event(event, run)

    def _record_generic_event(self, event: Any, run: _RunState) -> None:
        """Record any convertible event as a role+content transcript message."""
        message = event.to_llm_message()
        if getattr(message, "role", None) == "user":
            return
        run.messages.append(
            LiteLLMMessage(
                role=getattr(message, "role", "assistant"),
                content=_content_to_text(getattr(message, "content", "")),
            )
        )

    @staticmethod
    def _action_arguments(event: Any) -> Any:
        """Best-effort structured arguments of a tool-call action.

        Dynamically-created MCP tools wrap the caller's arguments in
        `MCPToolAction.data`, so return that verbatim: the transcript then
        records the arguments actually sent to the tool (e.g. `{"query": ...}`)
        rather than the SDK action envelope (`{"kind": ..., "data": {...}}`).
        Non-MCP actions fall back to a `model_dump()` with the non-argument
        envelope keys (`kind`/`summary`) stripped.
        """
        action = getattr(event, "action", None)
        if action is None:
            return {}
        data = getattr(action, "data", None)
        if isinstance(data, dict):
            return data
        dump = getattr(action, "model_dump", None)
        if callable(dump):
            try:
                payload = dump(mode="json")
            except Exception:
                return {"repr": str(action)}
            if isinstance(payload, dict):
                return {
                    key: value
                    for key, value in payload.items()
                    if key not in {"kind", "summary"}
                }
            return payload
        return {"repr": str(action)}

    @staticmethod
    def _event_text(event: Any) -> str:
        """Flatten a convertible event's message content to text."""
        try:
            message = event.to_llm_message()
            return _content_to_text(getattr(message, "content", ""))
        except Exception:
            return _content_to_text(getattr(event, "observation", ""))

    def _verify_runtime_tools(self, conversation: Any, run: _RunState) -> None:
        """Record and validate the actual runtime tool set OpenHands built.

        `tools=[]` + `include_default_tools=[...]` is necessary but not
        sufficient: OpenHands can still add ambient tools (e.g. a
        `VisionInspectTool` for a non-vision model, or plugin-provided tools),
        and default tools are not governed by `filter_tools_regex`. The only
        actionable surface must be the corral MCP endpoint plus the two internal
        reasoning/termination tools, so any *unexpected* runtime tool is treated
        as a broken benchmark condition rather than a silent capability leak.
        """
        runtime = self._runtime_tool_names(conversation)
        if runtime is None:
            return  # tools not introspectable (e.g. a stubbed conversation)
        run.metadata["runtime_tools"] = sorted(runtime)
        corral_tools = {
            t["function"]["name"]
            for t in run.available_tools
            if t.get("function", {}).get("name")
        }
        expected = set(_INCLUDED_DEFAULT_TOOL_NAMES) | corral_tools
        unexpected = runtime - expected
        if unexpected:
            raise _OpenHandsError(
                f"OpenHands initialized unexpected tools {sorted(unexpected)}; "
                f"expected only {sorted(expected)}",
                subtype="sdk_failure",
            )

    @staticmethod
    def _runtime_tool_names(conversation: Any) -> set[str] | None:
        """Return the runtime tool-name set, or None if not introspectable."""
        state = getattr(conversation, "state", None)
        agent = getattr(state, "agent", None) if state is not None else None
        if agent is None:
            return None
        try:
            tools_map = agent.tools_map
        except Exception:
            return None
        if not isinstance(tools_map, dict):
            return None
        return {str(name) for name in tools_map}

    def _require_clean_completion(self, conversation: Any, events: list[Any]) -> None:
        """Reject a non-clean terminal state that OpenHands does not raise for.

        Reaching `max_iteration_per_run` returns normally, transitions the
        conversation to ERROR, and emits
        `ConversationErrorEvent(code="MaxIterationsReached")`. Without this
        check it would look like a completion with no `FinishAction` and be
        misreported as a generic `sdk_failure`.
        """
        errors = [e for e in events if isinstance(e, ConversationErrorEvent)]
        latest = errors[-1] if errors else None
        if latest is not None:
            code = getattr(latest, "code", None)
            detail = getattr(latest, "detail", None) or code or "OpenHands run error"
            if code == "MaxIterationsReached":
                raise _OpenHandsError(str(detail), subtype="max_iterations")
            raise _OpenHandsError(
                f"{code}: {detail}",
                subtype=self._classify_error_code(code, detail),
            )

        status = self._execution_status(conversation)
        if status is not None and status != ConversationExecutionStatus.FINISHED:
            raise _OpenHandsError(
                f"OpenHands ended with unexpected execution status {status!r}",
                subtype="sdk_failure",
            )

    @staticmethod
    def _execution_status(conversation: Any) -> Any:
        """Return `conversation.state.execution_status`, or None if absent."""
        state = getattr(conversation, "state", None)
        return getattr(state, "execution_status", None) if state is not None else None

    @staticmethod
    def _classify_error_code(code: Any, detail: Any) -> HarnessStatus:
        """Map a `ConversationErrorEvent` code onto a terminal status.

        MCP/tool-execution failures are `tool_failure` (the corral endpoint
        failed); everything else is a generic `sdk_failure`. Deliberately
        narrower than "any mention of `connect`": a bare connection-error string
        also matches unrelated LLM-provider or proxy failures, which are not the
        corral tool surface and should stay `sdk_failure`.
        """
        text = f"{code} {detail}".casefold()
        if "mcp" in text or "tool execution" in text:
            return "tool_failure"
        return "sdk_failure"

    def _extract_finish_message(self, events: list[Any]) -> str:
        """Return the message from the most recent terminal `FinishAction`."""
        for event in reversed(events):
            if isinstance(event, ActionEvent) and isinstance(
                getattr(event, "action", None), FinishAction
            ):
                message = getattr(event.action, "message", "") or ""
                return message.strip()
        return ""

    def _record_usage(self, llm: Any, run: _RunState) -> None:
        """Map the OpenHands `LLM` metrics onto the base token-usage schema."""
        metrics = getattr(llm, "metrics", None)
        if metrics is None:
            return
        token_usages = getattr(metrics, "token_usages", None)
        if isinstance(token_usages, list):
            # OpenHands records one TokenUsage for every completion call. Treat
            # each SDK turn as one comparable LLM call, matching Claude's
            # `num_turns` and Codex's single `thread.turn(...)` accounting.
            run.metadata["sdk_turns"] = len(token_usages)
        token_metrics = getattr(metrics, "accumulated_token_usage", None)
        prompt_tokens = int(getattr(token_metrics, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(token_metrics, "completion_tokens", 0) or 0)
        run.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        cost = getattr(metrics, "accumulated_cost", None)
        if cost is not None:
            run.metadata["total_cost_usd"] = cost

    @staticmethod
    def _classify_error(exc: BaseException) -> HarnessStatus:
        """Classify an unexpected harness exception into a terminal status.

        MCP/tool failures are reported as `tool_failure` (the corral endpoint
        failed), everything else as a generic `sdk_failure`, so a broken
        benchmark run is not scored as a wrong model answer. Typed `MCPError`s
        are authoritative; the string fallback is deliberately restricted to
        MCP/tool-execution evidence, because a bare "connect" also matches
        unrelated LLM-provider or proxy connection errors.
        """
        if isinstance(exc, MCPError):
            return "tool_failure"
        text = f"{type(exc).__name__}: {exc}".casefold()
        if "mcp" in text or "tool execution" in text:
            return "tool_failure"
        return "sdk_failure"

    def _result(
        self,
        run: _RunState,
        status: HarnessStatus,
        *,
        answer: str | None = None,
        error: str | None = None,
    ) -> HarnessRunResult:
        """Build a `HarnessRunResult` from the current run scratch state."""
        return HarnessRunResult(
            status=status,
            answer=answer,
            error=error,
            num_events=run.metadata.get("num_events"),
            usage=dict(run.usage),
            total_cost_usd=run.metadata.get("total_cost_usd"),
            metadata=dict(run.metadata),
        )

    def _fail(
        self,
        run: _RunState,
        status: HarnessStatus,
        error: str,
        exc: BaseException,
    ) -> str:
        """Record an infrastructure failure and return an error answer string."""
        logger.error(f"OpenHands harness {status}: {error}")
        run.messages.append(
            LiteLLMMessage(
                role="assistant",
                content=f"Error running OpenHands harness: {exc}",
                name="openhands-error",
            )
        )
        run.result = self._result(run, status, error=error)
        return f"Error solving the task: {error}"
