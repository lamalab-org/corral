import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal
from urllib.parse import quote, urlencode

from loguru import logger

# The OpenHands SDK is declared as the `corral[openhands]` extra and assumed
# installed.
from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    LLMConvertibleEvent,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.mcp import MCPServer
from openhands.sdk.tool.builtins import FinishAction
from pydantic import SecretStr

from corral.agents.base_agent import BaseAgent
from corral.agents.hooks import HookPoint
from corral.agents.schema import SURRENDER_SENTINEL
from corral.agents.utils import LiteLLMMessage
from corral.router.routes import CorralRouter

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
# failures (timeouts, SDK crashes, MCP transport failures) into a wrong model
# answer. `stuck` is OpenHands' stuck-loop detection firing — an agent/harness
# outcome, kept distinct from a raw `sdk_failure` crash.
HarnessStatus = Literal[
    "success",
    "surrender",
    "max_iterations",
    "stuck",
    "timeout",
    "tool_failure",
    "sdk_failure",
]


@dataclass
class HarnessRunResult:
    """Structured outcome of one OpenHands harness run.

    :meth:`OpenHandsAgent.run` still returns a plain string for interface
    compatibility, but the benchmark record should read the structured status
    from here so that, for example, a `timeout` is not scored as a wrong answer.
    """

    status: HarnessStatus
    answer: str | None = None
    error: str | None = None
    num_events: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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


class OpenHandsAgent(BaseAgent):
    """Agent that delegates solving a task to the OpenHands harness.

    Args:
        model (str): The model the OpenHands harness should use, as a
            LiteLLM-style route (e.g. `"openai/gpt-5.6"`). Passed to the
            harness `LLM` verbatim. Defaults to `"openai/gpt-5.6"`.
        max_iterations (int, optional): Maximum number of harness steps (mapped
            to the SDK's `max_iteration_per_run`). Defaults to 30.
        api_key (str, optional): API key for the harness `LLM`. If None, falls
            back to the `LLM_API_KEY` environment variable. Defaults to None.
        api_endpoint (str, optional): Base URL for the harness `LLM` (mapped to
            the SDK `base_url`); also kept for :class:`BaseAgent` interface
            compatibility. Defaults to None.
        temperature (float, optional): Sampling temperature for the harness
            `LLM`. Defaults to 0.7.
        tool_timeout_s (float, optional): Per-call timeout OpenHands applies to
            the corral MCP server. Defaults to 600.
        wall_clock_timeout_s (float, optional): Hard wall-clock deadline for the
            whole harness run. On expiry the run is interrupted (cancelling the
            in-flight `arun()` task, mid-LLM-call safe) and the run is reported
            with `status="timeout"`. Defaults to None (no timeout).
        interrupt_grace_s (float, optional): After a wall-clock interrupt, how
            long to wait for the worker to actually unwind before the
            conversation is closed and its workspace deleted. Waiting avoids
            racing cleanup against a still-running harness; if the worker does
            not stop within this grace window the run is reported as an
            `sdk_failure` instead of a `timeout`. Defaults to 30.
        system_prompt (str, optional): Instructions appended to the OpenHands
            harness system prompt (via `AgentContext.system_message_suffix`). If
            None, uses the default corral system prompt. Tool-usage and
            final-answer instructions are appended automatically.
        extractor_model (str, optional): LiteLLM-compatible model used only by
            the base class machinery. If None it is derived from `model`.
        **kwargs: Additional keyword arguments forwarded to :class:`BaseAgent`.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5.6",
        max_iterations: int = 30,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        temperature: float = 0.7,
        tool_timeout_s: float = 600.0,
        wall_clock_timeout_s: float | None = None,
        interrupt_grace_s: float = 30.0,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        extractor_model: str | None = None,
        **kwargs,
    ):
        # OpenHands models are already LiteLLM routes (e.g. "openai/gpt-5.6"),
        # so the base machinery (token counting + answer extraction, only used
        # if ever invoked) can reuse the same string. Still allow an explicit
        # override for parity with the other harness agents.
        if extractor_model is None:
            extractor_model = model if "/" in model else f"openai/{model}"

        if user_prompt is None:
            user_prompt = "tool_calling/user_prompt"

        super().__init__(
            model=extractor_model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

        # Model string handed to the OpenHands harness itself.
        self.harness_model = model
        self.api_key = api_key
        self.tool_timeout_s = tool_timeout_s
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self.interrupt_grace_s = interrupt_grace_s
        self._available_tools = None
        # Structured outcome of the most recent run (see `HarnessRunResult`).
        self.harness_result: HarnessRunResult | None = None
        # Per-run scratch populated during a run for the benchmark record.
        self._run_meta: dict[str, Any] = {}

    @property
    def requires_answer_extraction(self) -> bool:
        """Submit the harness answer verbatim, without a second model call.

        OpenHands' `FinishAction` already carries the submit-ready answer, so
        running the base class's LiteLLM extractor would add an extra,
        separately-billed call (a *different* model) whose output could differ
        from the harness's actual answer. Bypassing it keeps this a faithful
        measurement of the OpenHands harness.
        """
        return False

    def _system_message_suffix(self, enable_surrender: bool) -> str:
        """Compose the suffix appended to the OpenHands harness system prompt.

        This does not *replace* the OpenHands system prompt (that is left to the
        harness so this condition measures the real harness), it augments it via
        `AgentContext.system_message_suffix`.
        """
        # The harness output is submitted verbatim unless answer extraction is
        # enabled, so only ask for the `Final Answer:` marker when something will
        # actually strip it; otherwise that prefix would end up in the submitted
        # answer. See :attr:`requires_answer_extraction`.
        if self.requires_answer_extraction:
            final_answer_directive = (
                "When you have solved the task, finish with your final answer "
                "prefixed exactly by 'Final Answer:' and nothing else."
            )
        else:
            final_answer_directive = (
                "When you have solved the task, finish with your final answer "
                "and nothing else."
            )
        suffix = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed evaluation environment. "
            f"You may ONLY interact with it through the provided `{_MCP_SERVER_NAME}` "
            "MCP tools; do not attempt to use the shell, filesystem, web, or any "
            "other tool, and do not request additional permissions. Do not invent "
            "tool outputs. " + final_answer_directive
        )
        if enable_surrender and self.surrender_prompt is not None:
            suffix += "\n\n" + self.surrender_prompt.fill({})
        return suffix

    def _mcp_url(self, interface: CorralRouter, task_id: str, verbosity: str) -> str:
        """Build the task-scoped MCP endpoint URL OpenHands connects to.

        Points OpenHands at the environment server's own task-scoped MCP
        endpoint. The `verbosity` is forwarded as a query parameter so it matches
        exactly the verbosity the REST allowlist was fetched at (otherwise
        OpenHands could see full tool descriptions while the allowlist metadata
        reflects a briefer condition, silently breaking ablations).
        """
        base_url = interface.base_url.rstrip("/")
        encoded_task_id = quote(str(task_id), safe="")
        query = urlencode({"verbosity": verbosity})
        return f"{base_url}/tasks/{encoded_task_id}/mcp?{query}"

    def _make_llm(self) -> Any:
        """Build the OpenHands `LLM` handed to the harness agent."""
        key = self.api_key or os.getenv("LLM_API_KEY")

        llm_kwargs: dict[str, Any] = {
            "usage_id": "corral-openhands",
            "model": self.harness_model,
            "temperature": self.temperature,
        }
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
                system_message_suffix=self._system_message_suffix(enable_surrender)
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
            "openhands_sdk_version": _sdk_version(),
            "python_version": platform.python_version(),
            "max_iterations_configured": self.max_iterations,
            "wall_clock_timeout_s": self.wall_clock_timeout_s,
            "tool_timeout_s": self.tool_timeout_s,
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

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,  # noqa: ARG002
        enable_surrender: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Run the OpenHands harness to solve the task.

        Args:
            interface (CorralRouter): The interface to the environment server.
            task_id (str): The task ID to solve.
            task_prompt (str, optional): The task prompt to use. If None, it is
                fetched from the environment. Defaults to None.
            examples (list[str], optional): Unused; the harness manages its own
                context. Kept for interface compatibility.
            enable_surrender (bool, optional): Whether to allow the agent to give
                up on an unsolvable task. Defaults to False.

        Returns:
            str: The final answer produced by the harness, or the
            `SURRENDER_SENTINEL` value if the agent surrendered.
            Infrastructure failures return an `"Error solving the task: ..."`
            string; inspect `self.harness_result` for the structured status.
        """
        self.harness_result = None
        self.reset_token_usage()

        # Resolve verbosity once and use the *same* value for the REST allowlist
        # and the MCP endpoint, so what OpenHands sees and what the metadata
        # records cannot drift apart (important for faithful ablations).
        verbosity = getattr(interface, "current_verbosity", None) or "brief"
        tools = interface.get_available_tools_for_task(
            task_id, verbosity=verbosity
        ).get("tools", [])
        self._available_tools = tools

        task_guide = (
            task_prompt
            if task_prompt is not None
            else interface.get_task_prompt(task_id)
        )
        prompt, dropped_images = self._normalize_prompt(task_guide)

        # Seed the message history so transcript saving has the task prompt first.
        self.messages = [LiteLLMMessage(role="user", content=prompt)]

        # Record the hash of the MCP tool schema the harness actually receives
        # (`tools/list` == `Tool.to_mcp`), so the run provenance well and truly
        # captures what tools the agent saw.
        mcp_schema_sha256 = self._fetch_mcp_schema_digest(interface, task_id, verbosity)
        mcp_url = self._mcp_url(interface, task_id, verbosity)
        self._run_meta = self._harness_metadata(
            mcp_url, verbosity, tools, mcp_schema_sha256
        )
        self._run_meta["dropped_image_parts"] = dropped_images

        self._execute_hooks(HookPoint.BEFORE_TASK, interface, task_id)

        try:
            return self._run_and_extract(
                task_id=task_id,
                prompt=prompt,
                mcp_url=mcp_url,
                enable_surrender=enable_surrender,
            )
        finally:
            # Lifecycle hooks run on *every* exit path (success, surrender,
            # timeout, failure).
            self._execute_hooks(HookPoint.AFTER_TASK, interface, task_id)

    def _run_and_extract(
        self,
        *,
        task_id: str,
        prompt: str,
        mcp_url: str,
        enable_surrender: bool,
    ) -> str:
        """Drive the harness and turn its output into a submit-ready answer."""
        # Fresh, empty, per-episode working directory; cleaned up afterwards.
        cwd = tempfile.mkdtemp(prefix="corral-openhands-")

        try:
            final_answer = self._run_openhands(
                prompt=prompt,
                mcp_url=mcp_url,
                cwd=cwd,
                enable_surrender=enable_surrender,
            )
        except _OpenHandsError as e:
            status_map: dict[str, HarnessStatus] = {
                "timeout": "timeout",
                "tool_failure": "tool_failure",
                "max_iterations": "max_iterations",
                "stuck": "stuck",
                "sdk_failure": "sdk_failure",
            }
            return self._fail(status_map.get(e.subtype or "", "sdk_failure"), str(e), e)
        except Exception as e:  # surface as infra failure, not a model answer
            return self._fail(self._classify_error(e), str(e), e)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)

        # Normalize the harness output into a submit-ready answer. Answer
        # extraction is disabled by default (`requires_answer_extraction` is
        # False), so the harness output is submitted verbatim. When extraction is
        # enabled, strip the declared "Final Answer:" marker, using the *last*
        # occurrence: earlier ones may appear in intermediate messages that
        # mention the required output format.
        if self.requires_answer_extraction:
            matches = list(
                re.finditer(
                    r"Final Answer:\s*(.*)", final_answer, re.DOTALL | re.IGNORECASE
                )
            )
            final_answer = (
                matches[-1].group(1).strip() if matches else final_answer.strip()
            )
        else:
            final_answer = final_answer.strip()

        # Surrender only on an *exact* sentinel answer, so the marker appearing
        # inside a longer response is not mistaken for giving up.
        if (
            enable_surrender
            and final_answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            logger.info(f"Agent retiring from task {task_id}")
            self.harness_result = self._result("surrender", answer=SURRENDER_SENTINEL)
            return SURRENDER_SENTINEL

        if not final_answer:
            error = "OpenHands completed without a FinishAction message"
            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content=f"Error: {error}.",
                    name="openhands-error",
                )
            )
            self.harness_result = self._result("sdk_failure", error=error)
            return f"Error solving the task: {error}"

        # Ensure the submit-ready answer is in the transcript exactly once. The
        # harness's final message is usually already recorded, so only append when
        # it differs (ignoring surrounding whitespace).
        last_content = self.messages[-1].get("content") if self.messages else None
        already_recorded = (
            isinstance(last_content, str) and last_content.strip() == final_answer
        )
        if not already_recorded:
            self.messages.append(LiteLLMMessage(role="assistant", content=final_answer))
        self.harness_result = self._result("success", answer=final_answer)
        return final_answer

    def _run_openhands(
        self, *, prompt: str, mcp_url: str, cwd: str, enable_surrender: bool
    ) -> str:
        """Drive the OpenHands conversation to completion and return its answer.

        Owns the full conversation lifecycle: builds the isolated agent, runs it
        (under the wall-clock watchdog), records usage/transcript, validates the
        runtime tool set and terminal state, extracts the terminal `FinishAction`
        message, and always closes the conversation.
        """
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)
            try:
                self._record_event(event)
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
            workspace=cwd,
            max_iteration_per_run=self.max_iterations,
            stuck_detection=True,
            persistence_dir=None,
            # Silence the default console visualizer: benchmark runs are
            # non-interactive and its output is noise.
            visualizer=None,
        )

        try:
            conversation.send_message(prompt)
            self._drive_with_timeout(conversation)
            # Fail loudly on a capability leak or a non-clean terminal state
            # (max-iterations / stuck) that OpenHands does *not* raise for, so
            # they are recorded precisely instead of as a generic no-answer
            # `sdk_failure`.
            self._verify_runtime_tools(conversation)
            self._require_clean_completion(conversation, events)
        finally:
            self._record_usage(llm)
            self._run_meta["num_events"] = len(events)
            try:
                conversation.close()
            except Exception:
                logger.warning("Failed to close OpenHands conversation", exc_info=True)

        return self._extract_finish_message(events)

    # -- Transcript recording --------------------------------------------

    def _record_event(self, event: Any) -> None:
        """Fold one OpenHands event into the inspectable corral transcript.

        Structured tool calls and their observations are preserved (name,
        call id, arguments, result, error flag) rather than collapsed to
        role+content, so tool-level trace analysis stays possible — matching
        the Codex/Claude Code transcripts.
        """
        if isinstance(event, ActionEvent):
            self._record_action_event(event)
        elif isinstance(event, ObservationEvent):
            self._record_observation_event(event)
        elif isinstance(event, MessageEvent):
            self._record_message_event(event)
        elif isinstance(event, LLMConvertibleEvent):
            # Any other convertible event: keep its text, skipping the initial
            # user prompt (already seeded as the first transcript message).
            self._record_generic_event(event)

    def _record_action_event(self, event: Any) -> None:
        """Record the model's reasoning and structured tool call."""
        # The terminal `FinishAction` carries the final answer, which is
        # appended exactly once at the end of `_run_and_extract`; recording it
        # here as a nameless tool call would only duplicate it.
        if isinstance(getattr(event, "action", None), FinishAction):
            return
        thought = _content_to_text(getattr(event, "thought", "") or "")
        if thought:
            self.messages.append(
                LiteLLMMessage(role="assistant", content=thought, name="thinking")
            )
        self.messages.append(
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

    def _record_observation_event(self, event: Any) -> None:
        """Record an MCP tool result as a `tool`-role transcript message."""
        is_error = bool(getattr(event, "error", None)) or bool(
            getattr(getattr(event, "observation", None), "error", None)
        )
        if is_error:
            self._run_meta["tool_errors"] = self._run_meta.get("tool_errors", 0) + 1
        tool_name = getattr(event, "tool_name", "") or ""
        self.messages.append(
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

    def _record_message_event(self, event: Any) -> None:
        """Record an assistant message (skipping echoed user messages)."""
        self._record_generic_event(event)

    def _record_generic_event(self, event: Any) -> None:
        """Record any convertible event as a role+content transcript message."""
        message = event.to_llm_message()
        if getattr(message, "role", None) == "user":
            return
        self.messages.append(
            LiteLLMMessage(
                role=getattr(message, "role", "assistant"),
                content=_content_to_text(getattr(message, "content", "")),
            )
        )

    @staticmethod
    def _action_arguments(event: Any) -> Any:
        """Best-effort structured arguments of a tool-call action."""
        action = getattr(event, "action", None)
        if action is None:
            return {}
        dump = getattr(action, "model_dump", None)
        if callable(dump):
            try:
                return dump(mode="json")
            except Exception:
                return {"repr": str(action)}
        return {"repr": str(action)}

    @staticmethod
    def _event_text(event: Any) -> str:
        """Flatten a convertible event's message content to text."""
        try:
            message = event.to_llm_message()
            return _content_to_text(getattr(message, "content", ""))
        except Exception:
            return _content_to_text(getattr(event, "observation", ""))

    # -- Driving / termination -------------------------------------------

    def _drive_with_timeout(self, conversation: Any) -> None:
        """Run the conversation, enforcing the wall-clock deadline.

        A thread timeout alone is insufficient: it merely stops *waiting*, it
        does not terminate the underlying OpenHands run. On expiry we call
        `conversation.interrupt()`, which cancels the in-flight `arun()` task
        (effective even mid-LLM-call, unlike the cooperative `pause()`), then
        wait for the worker to actually unwind before returning so the caller
        can close the conversation and delete the workspace without racing a
        still-running harness.
        """
        if self.wall_clock_timeout_s is None:
            conversation.run()
            return

        error: dict[str, BaseException] = {}
        done = threading.Event()

        def _worker() -> None:
            try:
                # Drive the *async* entrypoint so the wall-clock interrupt can
                # cancel the tracked task mid-LLM-call.
                asyncio.run(conversation.arun())
            except BaseException as exc:  # propagate to the caller thread
                error["exc"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        if not done.wait(self.wall_clock_timeout_s):
            self._interrupt(conversation)
            # Do not let the caller close the conversation / delete the
            # workspace until the worker has unwound, else cleanup races a live
            # run (leaking paid LLM/MCP calls past the reported timeout).
            if not done.wait(self.interrupt_grace_s):
                raise _OpenHandsError(
                    "OpenHands did not terminate within "
                    f"{self.interrupt_grace_s}s of the wall-clock interrupt",
                    subtype="sdk_failure",
                )
            raise _OpenHandsError(
                f"OpenHands did not finish within {self.wall_clock_timeout_s}s",
                subtype="timeout",
            )

        if "exc" in error:
            raise error["exc"]

    def _interrupt(self, conversation: Any) -> None:
        """Cancel the in-flight run (thread-safe), best effort."""
        interrupt = getattr(conversation, "interrupt", None) or getattr(
            conversation, "pause", None
        )
        if interrupt is None:
            return
        try:
            interrupt()
        except Exception as exc:  # best effort
            logger.warning(f"OpenHands interrupt (timeout) failed: {exc}")

    def _verify_runtime_tools(self, conversation: Any) -> None:
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
        self._run_meta["runtime_tools"] = sorted(runtime)
        corral_tools = {
            t["function"]["name"]
            for t in (self._available_tools or [])
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

        Two outcomes return normally instead of raising: reaching
        `max_iteration_per_run` transitions the conversation to ERROR and emits
        `ConversationErrorEvent(code="MaxIterationsReached")`, and stuck-loop
        detection transitions it to STUCK. Without this check both would look
        like a completion with no `FinishAction` and be misreported as a generic
        `sdk_failure`.
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
        if status == ConversationExecutionStatus.STUCK:
            raise _OpenHandsError(
                "OpenHands stuck-loop detection terminated the run",
                subtype="stuck",
            )
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

        MCP transport/connection failures are `tool_failure` (the corral
        endpoint was unreachable); everything else is a generic `sdk_failure`.
        """
        text = f"{code} {detail}".casefold()
        if "mcp" in text or "connect" in text:
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

    def _record_usage(self, llm: Any) -> None:
        """Map the OpenHands `LLM` metrics onto the base token-usage schema."""
        metrics = getattr(llm, "metrics", None)
        if metrics is None:
            return
        token_metrics = getattr(metrics, "accumulated_token_usage", None)
        prompt_tokens = int(getattr(token_metrics, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(token_metrics, "completion_tokens", 0) or 0)
        self.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # The harness reports usage already aggregated for the whole run, so
        # mirror it into the run total that `get_total_token_usage` reads rather
        # than summing (which would double-count on repeated calls).
        self.cumulative_token_usage = dict(self.token_usage)
        cost = getattr(metrics, "accumulated_cost", None)
        if cost is not None:
            self._run_meta["total_cost_usd"] = cost

    def _fetch_mcp_schema_digest(
        self, interface: CorralRouter, task_id: str, verbosity: str
    ) -> str | None:
        """Fetch the digest of the MCP schema OpenHands will see, if available."""
        fetch = getattr(interface, "get_mcp_tool_schema", None)
        if fetch is None:
            return None
        try:
            return fetch(task_id, verbosity=verbosity).get("mcp_schema_sha256")
        except Exception as exc:
            logger.warning(f"Could not fetch MCP tool schema digest: {exc}")
            return None

    @staticmethod
    def _classify_error(exc: BaseException) -> HarnessStatus:
        """Classify an unexpected harness exception into a terminal status.

        MCP transport/connection failures are reported as `tool_failure` (the
        corral endpoint was unreachable), everything else as a generic
        `sdk_failure`, so a broken benchmark run is not scored as a wrong model
        answer.
        """
        text = f"{type(exc).__name__}: {exc}".casefold()
        if "mcp" in text or "connect" in text:
            return "tool_failure"
        return "sdk_failure"

    def _result(
        self,
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
            num_events=self._run_meta.get("num_events"),
            usage=dict(self.token_usage),
            total_cost_usd=self._run_meta.get("total_cost_usd"),
            metadata=dict(self._run_meta),
        )

    def _fail(self, status: HarnessStatus, error: str, exc: BaseException) -> str:
        """Record an infrastructure failure and return an error answer string."""
        logger.error(f"OpenHands harness {status}: {error}")
        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content=f"Error running OpenHands harness: {exc}",
                name="openhands-error",
            )
        )
        self.harness_result = self._result(status, error=error)
        return f"Error solving the task: {error}"
