import asyncio
import hashlib
import importlib.resources
import json
import platform
import re
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import anyio
from promptstore import PromptStore

from corral.logging import logger

# The Claude Agent SDK ships as the optional `corral[claude]` extra. Wrap the
# import so an environment without the extra gets an actionable install hint
# instead of a bare ModuleNotFoundError deep in the import chain.
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extras
    raise ModuleNotFoundError(
        "ClaudeCodeAgent requires the Claude Agent SDK, which ships as the "
        "optional 'claude' extra. Install it with `pip install 'corral[claude]'`."
    ) from exc

from corral.agents.base_agent import provider_messages
from corral.agents.hooks import AgentHooks
from corral.agents.prompt_utils import ensure_jinja_compatible, get_prompt
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.session import AgentSession
from corral.agents.utils import LiteLLMMessage

# Name under which the corral MCP server is registered with the Claude Code
# harness. Tool names are namespaced by the SDK as
# `mcp__<server_name>__<tool_name>` when building the allowed-tools list.
_MCP_SERVER_NAME = "corral"


@dataclass(slots=True)
class _RunState:
    """Mutable scratch owned by one invocation, never by the agent instance."""

    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    last_assistant_text: str | None = None


class _HarnessError(RuntimeError):
    """Raised when the SDK reports a non-success ResultMessage.

    Carries the SDK `subtype` (e.g. `"error_max_turns"`) so the caller can
    map it onto a typed :class:`AgentOutcome` instead of treating every failure
    the same way.
    """

    def __init__(self, message: str, subtype: str | None):
        super().__init__(message)
        self.subtype = subtype


def _sdk_version() -> str | None:
    """Best-effort version of the installed `claude-agent-sdk` package."""
    try:
        return _pkg_version("claude-agent-sdk")
    except PackageNotFoundError:  # pragma: no cover - only without the SDK
        return None


def _stringify_tool_content(content: Any) -> str:
    """Render an SDK `ToolResultBlock.content` value as text for the trace.

    The content is either a plain string or a list of content parts; anything
    non-string is JSON-encoded so it round-trips through the saved transcript.
    """
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _parse_data_uri(url: str) -> tuple[str, str] | None:
    """Split a `data:` URI into `(media_type, base64_payload)`.

    Returns `None` for anything that is not a base64 data URI.
    """
    match = re.fullmatch(r"data:([^;,]+);base64,(.*)", url, re.DOTALL)
    if not match:
        return None
    return match.group(1), match.group(2)


def _to_sdk_content_blocks(parts: list[Any]) -> list[dict[str, Any]]:
    """Convert LiteLLM-style prompt parts into Anthropic SDK content blocks.

    Text parts become `{"type": "text", ...}` and image parts become
    `{"type": "image", "source": {...}}` (base64 for `data:` URIs, otherwise
    a URL source). This lets multimodal tasks be forwarded through the harness's
    streaming input instead of being rejected.
    """
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue

        part_type = part.get("type", "text")
        if part_type == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            parsed = _parse_data_uri(url)
            if parsed is not None:
                media_type, data = parsed
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        elif part_type == "image":
            # Already an Anthropic-style image block; forward as-is.
            blocks.append(part)
        else:
            raise ValueError(
                f"ClaudeCodeAgent cannot forward content part of type "
                f"{part_type!r} to the Claude Code harness."
            )
    return blocks


def _static_prompt(value: str | None, default_id: str) -> str:
    """Resolve one static harness prompt."""
    if value is not None:
        return ensure_jinja_compatible(value).fill({})
    with importlib.resources.path("corral.agents", "") as package_path:
        store = PromptStore(f"{package_path}/prompts")
    return get_prompt(store, None, default_id).fill({})


class ClaudeCodeAgent:
    """First-class session agent backed by the native Claude Code harness.

    Claude owns its SDK loop, context management, and result extraction. Corral
    owns the task-bound MCP endpoint, canonical tool transitions, usage folding,
    and final `submit_answer` action. The adapter has no legacy `run` or
    `arun_agent` entry point and keeps no task transcript on the instance.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        system_prompt: str | None = None,
        surrender_prompt: str | None = None,
        reasoning_effort: str | None = "high",
        thinking: dict[str, Any] | None = None,
        wall_clock_timeout_s: float | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        if thinking is None:
            thinking = {"type": "adaptive"}

        self.model = model
        self.harness_model = model
        self.system_prompt = _static_prompt(
            system_prompt,
            "system_prompt/system_prompt",
        )
        self.surrender_prompt = (
            ensure_jinja_compatible(surrender_prompt).fill({})
            if surrender_prompt is not None
            else None
        )
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self.hooks = hooks or AgentHooks()

    def _build_system_prompt(self, enable_surrender: bool) -> dict[str, Any]:
        """Compose the harness system prompt.

        Returns the Claude Code *preset* system prompt with the corral
        instructions appended, so this condition measures the real Claude Code
        harness prompt rather than replacing it with a custom string. (Passing a
        bare string to the SDK replaces the preset entirely.)
        """
        final_answer_directive = (
            "When you have solved the task, call the `submit_answer` MCP tool "
            "with the complete final answer. A plain-text final response does "
            "not complete the task."
        )
        append = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed environment. You may ONLY "
            f"interact with it through the provided `{_MCP_SERVER_NAME}` MCP tools; "
            "do not attempt to use the filesystem, shell, web, or subagents. "
            + final_answer_directive
        )
        if enable_surrender:
            if self.surrender_prompt is not None:
                append += "\n\n" + self.surrender_prompt
            append += (
                "\nTo surrender, call `submit_answer` with the exact answer "
                "`SURRENDER`; do not return the sentinel as text."
            )
        return {"type": "preset", "preset": "claude_code", "append": append}

    def _build_options(
        self,
        server: Any,
        allowed_tools: list[str],
        enable_surrender: bool,
        cwd: str,
        max_turns: int,
    ) -> Any:
        """Assemble the fully-isolated `ClaudeAgentOptions` for a run."""
        opts: dict[str, Any] = {
            "system_prompt": self._build_system_prompt(enable_surrender),
            "model": self.harness_model,
            # Fail closed: expose *no* built-in tools (empty base tool set) and
            # deny anything that was not explicitly pre-approved. Only the
            # `mcp__corral__*` task tools reach the agent. This does not rely on
            # a denylist, which would silently go stale as the SDK adds built-ins.
            "tools": [],
            "mcp_servers": {_MCP_SERVER_NAME: server},
            # Ignore project `.mcp.json`, user settings, and plugin MCP servers
            # so only the corral task endpoint is loaded.
            "strict_mcp_config": True,
            "allowed_tools": allowed_tools,
            "permission_mode": "dontAsk",
            # Load *no* filesystem configuration: `[]` (unlike `None`, which
            # loads user/project/local settings, CLAUDE.md, hooks, and MCP
            # config) keeps runs reproducible across machines.
            "setting_sources": [],
            "skills": [],
            "plugins": [],
            # A fresh empty working directory per episode so the preset's
            # dynamic (cwd/git/memory) context cannot leak between runs.
            "cwd": cwd,
            "max_turns": max_turns,
            # Ask the SDK to forward partial model events as they arrive. The
            # receive loop deliberately records only completed messages, so
            # partials keep the transport active without duplicating transcript
            # content.
            "include_partial_messages": True,
        }
        if self.reasoning_effort is not None:
            opts["effort"] = self.reasoning_effort
        if self.thinking is not None:
            opts["thinking"] = self.thinking
        return ClaudeAgentOptions(**opts)

    def _harness_metadata(
        self, options: Any, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Static provenance for the run so the harness stack is reproducible."""
        system_prompt = getattr(options, "system_prompt", None)
        append = (
            system_prompt.get("append", "") if isinstance(system_prompt, dict) else ""
        )
        # These are the OpenAI-function-format schemas from the REST allowlist,
        # not the MCP `tools/list` payload the harness receives; named
        # accordingly. The MCP-side digest is recorded by `_preflight_mcp`.
        rest_tool_schema = json.dumps(
            [t.get("function", {}) for t in tools], sort_keys=True, default=str
        )
        return {
            "model_requested": self.harness_model,
            "claude_agent_sdk_version": _sdk_version(),
            "python_version": platform.python_version(),
            "system_prompt_mode": "preset:claude_code",
            "system_prompt_append_sha256": hashlib.sha256(
                append.encode("utf-8")
            ).hexdigest(),
            "rest_tool_schema_sha256": hashlib.sha256(
                rest_tool_schema.encode("utf-8")
            ).hexdigest(),
            "reasoning_effort": self.reasoning_effort,
            "thinking": self.thinking,
            "max_turns_configured": getattr(options, "max_turns", None),
            "wall_clock_timeout_s": self.wall_clock_timeout_s,
            "streaming_enabled": bool(
                getattr(options, "include_partial_messages", False)
            ),
        }

    async def _execute_harness(
        self,
        session: AgentSession,
        mcp_url: str,
        prompt_input: Any,
        tools: list[dict[str, Any]],
        enable_surrender: bool,
        cwd: str,
        max_turns: int,
        run: _RunState,
    ) -> str:
        """Drive the Claude Code harness to completion and return its answer."""
        server = {"type": "http", "url": mcp_url}
        allowed_tools = [
            f"mcp__{_MCP_SERVER_NAME}__{t['function']['name']}"
            for t in tools
            if t.get("function", {}).get("name")
        ]
        options = self._build_options(
            server,
            allowed_tools,
            enable_surrender,
            cwd,
            max_turns,
        )
        run.metadata = self._harness_metadata(options, tools)
        run.metadata["mcp_url"] = mcp_url

        assistant_text: list[str] = []
        result_text: str | None = None

        # Use the persistent client (rather than the one-shot `query()`) so the
        # corral MCP endpoint can be inspected before the model starts solving.
        async with ClaudeSDKClient(options=options) as client:
            await self._preflight_mcp(client, tools, run)
            await client.query(self._sdk_prompt(prompt_input))
            async for message in client.receive_response():
                result_text, transcript = self._consume_message(
                    message,
                    assistant_text,
                    result_text,
                    run,
                )
                for item in transcript:
                    await session.record_message(item)

        return result_text or ("\n".join(assistant_text)).strip()

    async def _preflight_mcp(
        self,
        client: ClaudeSDKClient,
        tools: list[dict[str, Any]],
        run: _RunState,
    ) -> None:
        """Fail fast if the corral MCP endpoint is not usable for this run.

        The server-side per-task tool check remains the real security boundary;
        this client-side preflight only turns a broken benchmark run (endpoint
        unreachable, or exposing a different tool set than the REST allowlist)
        into an explicit infrastructure error *before* the model spends turns,
        rather than a silent wrong answer. It also records the tool metadata the
        harness actually receives via MCP `tools/list`.
        """
        corral = await self._wait_for_corral_mcp(client)

        exposed_tools = corral.get("tools", []) or []
        exposed = {self._bare_tool_name(t.get("name", "")) for t in exposed_tools}
        expected = {
            t["function"]["name"] for t in tools if t.get("function", {}).get("name")
        }
        if exposed != expected:
            raise _HarnessError(
                "Corral MCP tool mismatch: "
                f"expected={sorted(expected)!r}, exposed={sorted(exposed)!r}",
                subtype="error_mcp_tool_mismatch",
            )

        # Provenance: hash the tool *metadata* the harness sees over MCP
        # (`name`/`description`/`annotations`). `get_mcp_status` does not expose
        # the input schemas, so this is deliberately not called a schema digest;
        # the REST-format schema digest is recorded separately as
        # `rest_tool_schema_sha256`.
        run.metadata["mcp_tools_exposed"] = sorted(exposed)
        run.metadata["mcp_tool_metadata_sha256"] = hashlib.sha256(
            json.dumps(
                exposed_tools, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()

    async def _wait_for_corral_mcp(
        self, client: ClaudeSDKClient, timeout_s: float = 15.0
    ) -> dict[str, Any]:
        """Poll MCP status until the corral server connects (or fails).

        A Streamable-HTTP server can report a transient pending state before
        it finishes connecting, so a single status snapshot would spuriously fail
        an otherwise healthy run. Terminal-bad states (failed/needs-auth/
        disabled) and a missing server are raised immediately; pending is
        retried within a bounded deadline.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last: dict[str, Any] | None = None
        while True:
            status = await client.get_mcp_status()
            servers = status.get("mcpServers", []) if status else []
            corral = next(
                (s for s in servers if s.get("name") == _MCP_SERVER_NAME), None
            )
            if corral is None:
                raise _HarnessError(
                    f"Corral MCP server missing from status: {servers!r}",
                    subtype="error_mcp_connection",
                )
            last = corral
            state = corral.get("status")
            if state == "connected":
                return corral
            if state in {"failed", "needs-auth", "disabled"}:
                raise _HarnessError(
                    "Corral MCP server did not connect: "
                    f"status={state!r}, error={corral.get('error')!r}",
                    subtype="error_mcp_connection",
                )
            # state == "pending" (or unknown): retry until the deadline.
            if loop.time() >= deadline:
                raise _HarnessError(
                    f"Timed out waiting for Corral MCP server to connect: {last!r}",
                    subtype="error_mcp_connection",
                )
            await asyncio.sleep(0.25)

    @staticmethod
    def _bare_tool_name(name: str) -> str:
        """Strip the SDK `mcp__<server>__` namespace prefix if present."""
        prefix = f"mcp__{_MCP_SERVER_NAME}__"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def _consume_message(
        self,
        message: Any,
        assistant_text: list[str],
        result_text: str | None,
        run: _RunState,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Translate one SDK event into result state and provider messages.

        Raises :class:`_HarnessError` on a non-success `ResultMessage`.
        """
        transcript: list[dict[str, Any]] = []
        if isinstance(message, AssistantMessage):
            reported = getattr(message, "model", None)
            if reported:
                run.metadata["model_reported"] = reported
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    assistant_text.append(block.text)
                    run.last_assistant_text = block.text
                    transcript.append(
                        dict(LiteLLMMessage(role="assistant", content=block.text))
                    )
                elif isinstance(block, ThinkingBlock):
                    transcript.append(
                        dict(
                            LiteLLMMessage(
                                role="assistant",
                                content=block.thinking,
                                name="thinking",
                            )
                        )
                    )
                elif isinstance(block, ToolUseBlock):
                    transcript.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": block.id,
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(
                                            block.input, default=str
                                        ),
                                    },
                                }
                            ],
                        }
                    )
        elif isinstance(message, UserMessage):
            content = getattr(message, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        is_error = bool(getattr(block, "is_error", False))
                        if is_error:
                            # Count (but don't terminate on) MCP tool errors:
                            # recovering from them is legitimate agent behaviour,
                            # so this is provenance, not a failure by itself.
                            run.metadata["tool_errors"] = (
                                run.metadata.get("tool_errors", 0) + 1
                            )
                        transcript.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": _stringify_tool_content(block.content),
                                "name": "tool_result",
                                "is_error": is_error,
                            }
                        )
        elif isinstance(message, ResultMessage):
            self._record_result_meta(message, run)
            usage = getattr(message, "usage", None)
            if usage:
                self._record_usage(usage, run)
            # Record metadata *before* raising: the SDK may emit an error
            # ResultMessage (e.g. `error_max_turns`) and only then raise.
            subtype = getattr(message, "subtype", "success")
            if getattr(message, "is_error", False) or subtype != "success":
                raise _HarnessError(
                    "Claude Agent SDK run did not succeed: "
                    f"subtype={subtype!r}, "
                    f"stop_reason={getattr(message, 'stop_reason', None)!r}, "
                    f"result={getattr(message, 'result', None)!r}",
                    subtype=subtype,
                )
            if getattr(message, "result", None):
                result_text = message.result

        return result_text, transcript

    def _sdk_prompt(self, prompt_input: Any) -> Any:
        """Return the value to hand to `query(prompt=...)`.

        Text tasks use the plain string interface. Multimodal tasks are sent
        through streaming input (an async iterable of message dicts), the only
        way the SDK accepts image content.
        """
        if isinstance(prompt_input, str):
            return prompt_input

        content_blocks = _to_sdk_content_blocks(prompt_input)

        async def _messages() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "parent_tool_use_id": None,
                "session_id": "default",
            }

        return _messages()

    def _record_result_meta(self, message: Any, run: _RunState) -> None:
        """Capture the SDK ResultMessage fields for the benchmark record."""
        for attr in (
            "subtype",
            "num_turns",
            "total_cost_usd",
            "duration_ms",
            "duration_api_ms",
            "session_id",
            "stop_reason",
            "model_usage",
            "permission_denials",
        ):
            value = getattr(message, attr, None)
            if value is not None:
                run.metadata[attr] = value

    @staticmethod
    def _record_usage(usage: dict[str, Any], run: _RunState) -> None:
        """Normalize the SDK's aggregate usage for the typed session outcome."""
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        prompt_tokens = input_tokens + cache_read + cache_creation
        run.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }

    def _normalize_prompt(self, task_guide: Any) -> tuple[Any, Any]:
        """Split a raw task prompt into `(transcript_content, sdk_prompt)`.

        Text prompts yield a string for both. Multimodal prompts keep the
        LiteLLM-style content list for the transcript and forward the same parts
        to the SDK as a structured prompt (images are no longer rejected).
        """
        if isinstance(task_guide, list):
            if any(
                isinstance(p, dict) and p.get("type", "text") != "text"
                for p in task_guide
            ):
                # Multimodal: preserve the structured content on both sides.
                return task_guide, list(task_guide)
            # Text-only list: flatten to a single string.
            text = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in task_guide
            )
            return text, text
        text = str(task_guide)
        return text, text

    @staticmethod
    def _prompt_with_state_history(
        prompt_input: Any,
        messages: tuple[Mapping[str, Any], ...] | list[dict[str, Any]],
    ) -> Any:
        """Add the canonical State conversation to a fresh harness query.

        Claude Code owns a new SDK session for each Corral task attempt, so an
        existing immutable State chain cannot be restored through an SDK session
        id.  Supplying its provider-safe messages in the query gives the harness
        the same resumable context without introducing adapter-owned memory.
        """
        if not messages:
            return prompt_input
        history = json.dumps(
            list(messages),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        context = (
            "Continue from this canonical Corral State.messages conversation. "
            "Treat it as prior context and do not repeat completed tool calls:\n"
            f"{history}\n\nCurrent task input follows."
        )
        if isinstance(prompt_input, str):
            return f"{context}\n\n{prompt_input}"
        return [{"type": "text", "text": context}, *list(prompt_input)]

    def _usage(self, run: _RunState) -> AgentUsage:
        llm_calls = int(run.metadata.get("num_turns", 0) or 0)
        if llm_calls == 0 and "subtype" in run.metadata:
            llm_calls = 1
        return AgentUsage(
            input_tokens=int(run.usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(run.usage.get("completion_tokens", 0) or 0),
            llm_calls=llm_calls,
            metadata={
                "input_tokens": int(run.usage.get("input_tokens", 0) or 0),
                "cache_read_input_tokens": int(
                    run.usage.get("cache_read_input_tokens", 0) or 0
                ),
                "cache_creation_input_tokens": int(
                    run.usage.get("cache_creation_input_tokens", 0) or 0
                ),
            },
        )

    async def _failure_outcome(
        self,
        session: AgentSession,
        exc: Exception,
        run: _RunState,
    ) -> AgentOutcome:
        if isinstance(exc, TimeoutError):
            status = "timeout"
            error = f"harness timed out after {self.wall_clock_timeout_s}s"
        elif isinstance(exc, _HarnessError):
            statuses = {
                "error_max_turns": "iteration_limit",
                "error_mcp_connection": "tool_failure",
                "error_mcp_tool_mismatch": "protocol_failure",
            }
            status = statuses.get(exc.subtype or "", "harness_failure")
            error = str(exc)
        else:
            status = "harness_failure"
            error = str(exc)

        await session.record_message(
            dict(
                LiteLLMMessage(
                    role="assistant",
                    content=f"Claude Code harness failed: {error}",
                    name="claude-code-error",
                )
            )
        )
        return AgentOutcome(
            status=status,
            error=error,
            usage=self._usage(run),
            metadata=dict(run.metadata),
        )

    async def _finalize_outcome(
        self,
        session: AgentSession,
        final_answer: str,
        run: _RunState,
    ) -> AgentOutcome:
        submission = session.submission
        if submission is None:
            answer = final_answer.strip()
            if answer:
                await session.record_message(
                    dict(LiteLLMMessage(role="assistant", content=answer))
                )
            return AgentOutcome(
                status="protocol_failure",
                error="Claude Code finished without calling submit_answer",
                usage=self._usage(run),
                metadata=dict(run.metadata),
            )
        if session.submission_status == "surrendered":
            logger.debug(f"Agent retiring from execution {session.execution_id}")
            return AgentOutcome(
                status="surrendered",
                usage=self._usage(run),
                metadata=dict(run.metadata),
            )
        return AgentOutcome(
            status="completed",
            answer=submission,
            usage=self._usage(run),
            metadata=dict(run.metadata),
        )

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run Claude's native loop against one task-bound Corral session."""
        run = _RunState()
        tools = [dict(tool) for tool in session.tools]
        history = provider_messages(session.messages)
        transcript_content, sdk_prompt = self._normalize_prompt(session.prompt)
        sdk_prompt = self._prompt_with_state_history(sdk_prompt, history)
        max_turns = session.iteration_limit
        await session.record_message(
            dict(LiteLLMMessage(role="user", content=transcript_content))
        )

        try:
            with tempfile.TemporaryDirectory(prefix="corral-claude-") as cwd:
                async with session.open_mcp() as mcp:
                    call = self._execute_harness(
                        session,
                        mcp.url,
                        sdk_prompt,
                        tools,
                        session.surrender_allowed,
                        cwd,
                        max_turns,
                        run,
                    )
                    if self.wall_clock_timeout_s is None:
                        final_answer = await call
                    else:
                        with anyio.fail_after(self.wall_clock_timeout_s):
                            final_answer = await call
        except ImportError:
            raise
        except Exception as exc:
            return await self._failure_outcome(session, exc, run)

        return await self._finalize_outcome(session, final_answer, run)
