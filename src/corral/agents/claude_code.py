import asyncio
import hashlib
import json
import platform
import re
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal
from urllib.parse import quote, urlencode

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
from loguru import logger

from corral.agents.base_agent import BaseAgent
from corral.agents.hooks import HookPoint
from corral.agents.schema import SURRENDER_SENTINEL
from corral.agents.utils import LiteLLMMessage
from corral.router.routes import CorralRouter

# Name under which the corral MCP server is registered with the Claude Code
# harness. Tool names are namespaced by the SDK as
# `mcp__<server_name>__<tool_name>` when building the allowed-tools list.
_MCP_SERVER_NAME = "corral"

# Terminal status of a single harness run. Distinguishing these lets the
# benchmark record *why* a run ended instead of collapsing infrastructure
# failures (timeouts, SDK/CLI crashes, budget exhaustion) into an ordinary
# wrong model answer.
HarnessStatus = Literal[
    "success",
    "surrender",
    "max_turns",
    "timeout",
    "tool_failure",
    "sdk_failure",
]


@dataclass
class HarnessRunResult:
    """Structured outcome of one Claude Code harness run.

    `run()` still returns a plain string for interface compatibility, but the
    benchmark record should read the structured status from here so that, for
    example, `max_turns` (budget exhaustion) is not scored as a wrong answer.
    """

    status: HarnessStatus
    answer: str | None = None
    error: str | None = None
    num_turns: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _HarnessError(RuntimeError):
    """Raised when the SDK reports a non-success ResultMessage.

    Carries the SDK `subtype` (e.g. `"error_max_turns"`) so the caller can
    map it onto a :data:`HarnessStatus` instead of treating every failure the
    same way.
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


def _run_coroutine(coro: Any, timeout: float | None = None) -> Any:
    """Run `coro` to completion from synchronous code.

    `BaseAgent.run` is synchronous while the Claude Agent SDK is async. When
    no event loop is running we use :func:`asyncio.run`; if one is already
    running (e.g. the agent is driven from async code) we execute the coroutine
    in a dedicated loop on a background thread to avoid clobbering the caller's
    loop.

    Exceptions raised inside the coroutine are re-raised to the caller (rather
    than surfacing as a confusing `KeyError`), and `timeout` enforces a hard
    wall-clock deadline so a stalled CLI, tool, or network call cannot block the
    benchmark indefinitely — a :class:`TimeoutError` is raised instead.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None:
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout)
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    finished = threading.Event()

    def _worker() -> None:
        try:
            inner = asyncio.wait_for(coro, timeout) if timeout is not None else coro
            result["value"] = asyncio.run(inner)
        except BaseException as exc:  # propagate to the caller thread
            result["exception"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    # `finished` is set even on timeout inside the worker, but guard the join
    # with the same deadline in case the worker itself wedges.
    finished.wait(timeout)

    if not finished.is_set():
        raise TimeoutError(f"Claude Code harness did not finish within {timeout}s.")
    if "exception" in result:
        raise result["exception"]
    return result["value"]


class ClaudeCodeAgent(BaseAgent):
    """Agent that delegates solving a task to the Claude Code harness.

    Unlike the other agents in this package, `ClaudeCodeAgent` does not
    implement its own reasoning loop. Instead it runs the Claude Code harness
    (via the `Claude Agent SDK <https://code.claude.com/docs/en/agent-sdk>`_)
    as a black box: the harness owns the planning/acting loop, context
    management, and tool orchestration. The corral task tools are exposed to it
    through the environment server's own task-scoped MCP endpoint
    (`{base_url}/tasks/{task_id}/mcp`), which the harness connects to directly
    over Streamable HTTP, and the harness's final message is returned as the
    answer. Routing tool calls through the server's MCP transport (rather than
    an in-process bridge) means schemas and execution reuse the same
    `Tool.to_mcp` / `Environment.call_tool` code paths as the REST API, with
    no second tool-definition or dispatch layer to keep in sync.

    Because the harness is a black box, the `max_iterations`/prompt-format
    machinery used by the other agents does not directly apply. Instead the
    harness turn budget is controlled by `max_iterations` (mapped to the SDK's
    `max_turns`) and the built-in Claude Code tools (`Bash`, `Read`,
    `Write`, ...) are disabled so the agent can only use the corral tools.

    Args:
        model (str): The model the Claude Code harness should use. Accepts any
            model string the harness understands (e.g. `"claude-opus-4-8"` or
            an alias like `"sonnet"`). Defaults to `"claude-opus-4-8"`.
        max_iterations (int, optional): Maximum number of harness turns
            (mapped to the SDK `max_turns`). Defaults to 30.
        api_endpoint (str, optional): Unused by the harness; kept for interface
            compatibility with :class:`BaseAgent` (used only by the answer
            extractor). Defaults to None.
        system_prompt (str, optional): System prompt handed to the harness. If
            None, uses the default corral system prompt. Tool-usage and
            final-answer instructions are appended automatically.
        user_prompt (str, optional): Prompt template used only for the base
            machinery. If None, defaults to `"tool_calling/user_prompt"`.
        extractor_prompt (str, optional): The extractor prompt for cleaning the
            harness's final answer. If None, uses the default extractor prompt.
        surrender_prompt (str, optional): Instructions for surrendering an
            unsolvable task. Only used when `enable_surrender=True`.
        temperature (float, optional): Kept for interface compatibility (the
            harness manages its own sampling). Defaults to 0.7.
        reasoning_effort (str, optional): Effort level passed to the harness
            (`"low"`/`"medium"`/`"high"`/`"xhigh"`/`"max"`). Set
            explicitly so the benchmark condition does not depend on a
            version-dependent SDK default. Defaults to `"high"`.
        thinking (dict, optional): Thinking configuration for the harness (e.g.
            `{"type": "adaptive"}` or `{"type": "enabled", "budget_tokens": N}`).
            Defaults to `{"type": "adaptive"}`; pass `{"type": "disabled"}`
            to turn extended thinking off.
        tool_timeout_s (float, optional): Recorded in the run metadata for
            provenance. Tool execution is now delegated to the environment
            server's MCP endpoint, so this is no longer enforced per-tool by the
            agent; use `wall_clock_timeout_s` to bound the whole run. Kept for
            interface compatibility. Defaults to None.
        wall_clock_timeout_s (float, optional): Hard wall-clock deadline for the
            whole harness run. On expiry the run is cancelled and reported with
            `status="timeout"`. Defaults to None (no timeout).
        extractor_model (str, optional): LiteLLM-compatible model used by the
            base class for the answer-extraction step. If None, it is derived
            from `model` (prefixing `"anthropic/"` when `model` has no
            provider prefix).
        **kwargs: Additional keyword arguments forwarded to :class:`BaseAgent`.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_iterations: int = 30,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = "high",
        thinking: dict[str, Any] | None = None,
        tool_timeout_s: float | None = None,
        wall_clock_timeout_s: float | None = None,
        extractor_model: str | None = None,
        **kwargs,
    ):
        """Initialize the agent."""
        # Configure thinking explicitly (default adaptive) rather than relying
        # on a version-dependent SDK default. To turn extended thinking off,
        # pass `thinking={"type": "disabled"}`.
        if thinking is None:
            thinking = {"type": "adaptive"}

        # The harness model (e.g. "claude-opus-4-8") is not a LiteLLM route on
        # its own. Derive a LiteLLM-compatible model for the base machinery
        # (token counting + answer extraction) so those steps keep working.
        if extractor_model is None:
            extractor_model = model if "/" in model else f"anthropic/{model}"

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

        # Model string handed to the Claude Code harness itself.
        self.harness_model = model
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.tool_timeout_s = tool_timeout_s
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self._available_tools = None
        # Structured outcome of the most recent run (see `HarnessRunResult`).
        self.harness_result: HarnessRunResult | None = None
        # Per-run scratch populated by `_run_harness` for the benchmark record.
        self._run_meta: dict[str, Any] = {}

    @property
    def requires_answer_extraction(self) -> bool:
        """Submit the harness answer verbatim, without a second model call.

        The harness is already instructed to emit Final Answer: and
        :meth:`run` deterministically extracts and returns it, so running the
        base class's LiteLLM answer extractor would add an extra, separately
        billed model call whose output could differ from — or fail on — the
        harness's actual answer. Bypassing it keeps this condition a faithful
        measurement of the Claude Code harness alone.
        """
        return False

    def _build_system_prompt(self, enable_surrender: bool) -> dict[str, Any]:
        """Compose the harness system prompt.

        Returns the Claude Code *preset* system prompt with the corral
        instructions appended, so this condition measures the real Claude Code
        harness prompt rather than replacing it with a custom string. (Passing a
        bare string to the SDK replaces the preset entirely.)
        """
        # The harness output is submitted verbatim unless answer extraction is
        # enabled, so only ask for the `Final Answer:` marker when something will
        # actually strip it; otherwise that prefix would end up in the submitted
        # answer. See :attr:`requires_answer_extraction`.
        if self.requires_answer_extraction:
            final_answer_directive = (
                "When you have solved the task, reply with your final answer "
                "prefixed exactly by 'Final Answer:' and nothing else."
            )
        else:
            final_answer_directive = (
                "When you have solved the task, reply with your final answer and "
                "nothing else."
            )
        append = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed environment. You may ONLY "
            f"interact with it through the provided `{_MCP_SERVER_NAME}` MCP tools; "
            "do not attempt to use the filesystem, shell, web, or subagents. "
            + final_answer_directive
        )
        if enable_surrender and self.surrender_prompt is not None:
            append += "\n\n" + self.surrender_prompt.fill({})
        return {"type": "preset", "preset": "claude_code", "append": append}

    def _mcp_server_config(
        self, interface: CorralRouter, task_id: str
    ) -> tuple[str, dict[str, Any], str]:
        """Build the Streamable-HTTP MCP server config for this task.

        Points the harness at the environment server's own task-scoped MCP
        endpoint. Execution and tool schemas are served there (via
        `Environment.call_tool` / `Tool.to_mcp`), so the agent does not bridge
        tools itself.

        The REST allowlist is fetched at the router's current tool verbosity, so
        the same verbosity is forwarded to the MCP endpoint as a query parameter.
        Otherwise the harness could see full tool descriptions while the
        allowlist metadata reflects a briefer condition, silently breaking
        tool-description ablations.

        Returns the endpoint URL, the SDK `mcp_servers` config entry, and the
        resolved verbosity.
        """
        base_url = interface.base_url.rstrip("/")
        # Percent-encode the task id so ids containing slashes or other reserved
        # characters cannot produce a malformed path segment.
        encoded_task_id = quote(str(task_id), safe="")
        # Mirror the REST tool verbosity (defaults to the router's own "brief").
        verbosity = getattr(interface, "current_verbosity", None) or "brief"
        query = urlencode({"verbosity": verbosity})
        # Trailing slash on `/mcp/` avoids a 307 redirect: the endpoint is a
        # Starlette Mount, so a bare `/mcp` bounces to `/mcp/` (an extra
        # round-trip per call). Hit the canonical path directly.
        url = f"{base_url}/tasks/{encoded_task_id}/mcp/?{query}"
        return url, {"type": "http", "url": url}, verbosity

    def _build_options(
        self,
        server: Any,
        allowed_tools: list[str],
        enable_surrender: bool,
        cwd: str,
    ) -> Any:
        """Assemble the fully-isolated `ClaudeAgentOptions` for a run."""
        opts: dict[str, Any] = {
            "system_prompt": self._build_system_prompt(enable_surrender),
            "model": self.harness_model,
            # Fail closed: expose *no* built-in tools (empty base tool set) and
            # deny anything that was not explicitly pre-approved. Only the
            # `mcp__corral__*` tools reach the agent. This does not rely on a
            # denylist, which would silently go stale as the SDK adds built-ins.
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
            "max_turns": self.max_iterations,
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
            "max_turns_configured": self.max_iterations,
            "wall_clock_timeout_s": self.wall_clock_timeout_s,
            "tool_timeout_s": self.tool_timeout_s,
        }

    async def _run_harness(
        self,
        interface: CorralRouter,
        task_id: str,
        prompt_input: Any,
        tools: list[dict[str, Any]],
        enable_surrender: bool,
        cwd: str,
    ) -> str:
        """Drive the Claude Code harness to completion and return its answer."""
        mcp_url, server, verbosity = self._mcp_server_config(interface, task_id)
        allowed_tools = [
            f"mcp__{_MCP_SERVER_NAME}__{t['function']['name']}"
            for t in tools
            if t.get("function", {}).get("name")
        ]

        options = self._build_options(server, allowed_tools, enable_surrender, cwd)
        self._run_meta = self._harness_metadata(options, tools)
        self._run_meta["mcp_url"] = mcp_url
        self._run_meta["tool_verbosity"] = verbosity

        assistant_text: list[str] = []
        result_text: str | None = None

        # Use the persistent client (rather than the one-shot `query()`) so the
        # corral MCP endpoint can be inspected before the model starts solving.
        async with ClaudeSDKClient(options=options) as client:
            await self._preflight_mcp(client, tools)
            await client.query(self._sdk_prompt(prompt_input))
            async for message in client.receive_response():
                result_text = self._consume_message(
                    message, assistant_text, result_text
                )

        return result_text or ("\n".join(assistant_text)).strip()

    async def _preflight_mcp(
        self, client: ClaudeSDKClient, tools: list[dict[str, Any]]
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
        self._run_meta["mcp_tools_exposed"] = sorted(exposed)
        self._run_meta["mcp_tool_metadata_sha256"] = hashlib.sha256(
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
        self, message: Any, assistant_text: list[str], result_text: str | None
    ) -> str | None:
        """Fold one SDK message into the transcript; return the final result.

        Raises :class:`_HarnessError` on a non-success `ResultMessage`.
        """
        if isinstance(message, AssistantMessage):
            reported = getattr(message, "model", None)
            if reported:
                self._run_meta["model_reported"] = reported
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    assistant_text.append(block.text)
                    self.messages.append(
                        LiteLLMMessage(role="assistant", content=block.text)
                    )
                elif isinstance(block, ThinkingBlock):
                    self.messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=block.thinking,
                            name="thinking",
                        )
                    )
                elif isinstance(block, ToolUseBlock):
                    self.messages.append(
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
                            self._run_meta["tool_errors"] = (
                                self._run_meta.get("tool_errors", 0) + 1
                            )
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": _stringify_tool_content(block.content),
                                "name": "tool_result",
                                "is_error": is_error,
                            }
                        )
        elif isinstance(message, ResultMessage):
            self._record_result_meta(message)
            usage = getattr(message, "usage", None)
            if usage:
                self._record_usage(usage)
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

        return result_text

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

    def _record_result_meta(self, message: Any) -> None:
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
                self._run_meta[attr] = value

    def _record_usage(self, usage: dict[str, Any]) -> None:
        """Map the harness usage dict onto the base token-usage schema.

        `prompt_tokens`/`completion_tokens`/`total_tokens` keep the base
        class working, while the cache-read and cache-creation components are
        preserved separately because they have different cost semantics.
        """
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        prompt_tokens = input_tokens + cache_read + cache_creation
        self.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }
        # The harness reports usage already aggregated for the whole session, so
        # mirror it into the run total that `get_total_token_usage` reads
        # rather than summing (which would double-count on repeated results).
        self.cumulative_token_usage = dict(self.token_usage)

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

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,  # noqa: ARG002
        enable_surrender: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Run the Claude Code harness to solve the task.

        Args:
            interface (CorralRouter): The interface to the environment server.
            task_id (str): The task ID to solve.
            task_prompt (str, optional): The task prompt to use. If None, the
                task prompt is fetched from the environment. Defaults to None.
            examples (list[str], optional): Unused; the harness manages its own
                context. Kept for interface compatibility. Defaults to None.
            enable_surrender (bool, optional): Whether to allow the agent to give
                up on an unsolvable task. Defaults to False.

        Returns:
            str: The final answer produced by the harness, or the
            `SURRENDER_SENTINEL` value if the agent surrendered. Infrastructure
            failures return an `"Error solving the task: ..."` string; inspect
            `self.harness_result` for the structured status.
        """
        tools = interface.get_available_tools_for_task(task_id).get("tools", [])
        self._available_tools = tools
        self.harness_result = None

        if task_prompt is None:
            task_guide = interface.get_task_prompt(task_id)
        else:
            task_guide = task_prompt

        # `get_task_prompt` may return multimodal content. Text stays a plain
        # string; images are forwarded through the SDK's streaming input.
        transcript_content, sdk_prompt = self._normalize_prompt(task_guide)

        # Seed the message history so the base-class answer extractor and
        # transcript saving have the task prompt as the first message.
        self.messages = [LiteLLMMessage(role="user", content=transcript_content)]

        self._execute_hooks(HookPoint.BEFORE_TASK, interface, task_id)

        # Fresh, empty working directory per episode; cleaned up afterwards.
        cwd = tempfile.mkdtemp(prefix="corral-claude-")
        try:
            final_answer = _run_coroutine(
                self._run_harness(
                    interface, task_id, sdk_prompt, tools, enable_surrender, cwd
                ),
                timeout=self.wall_clock_timeout_s,
            )
        except ImportError:
            raise
        except TimeoutError as e:
            return self._fail(
                "timeout", f"harness timed out after {self.wall_clock_timeout_s}s", e
            )
        except _HarnessError as e:
            # Map the SDK/preflight subtype onto a precise terminal status so a
            # budget exhaustion, an MCP transport failure, and a tool-set
            # mismatch are not all collapsed into a generic SDK crash.
            subtype_status: dict[str, HarnessStatus] = {
                "error_max_turns": "max_turns",
                "error_mcp_connection": "tool_failure",
                "error_mcp_tool_mismatch": "sdk_failure",
            }
            status: HarnessStatus = subtype_status.get(e.subtype or "", "sdk_failure")
            return self._fail(status, str(e), e)
        except Exception as e:  # surface as infra failure, not a model answer
            return self._fail("sdk_failure", str(e), e)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)

        # Normalize the harness output into a submit-ready answer. Answer
        # extraction is disabled by default (`requires_answer_extraction` is
        # False), so the harness output is submitted verbatim. When extraction is
        # enabled, strip the declared "Final Answer:" marker, then reason about
        # that normalized value.
        if self.requires_answer_extraction:
            final_answer_match = re.search(
                r"Final Answer:\s*(.*)", final_answer, re.DOTALL | re.IGNORECASE
            )
            final_answer = (
                final_answer_match.group(1).strip()
                if final_answer_match
                else final_answer.strip()
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
            self.harness_result = HarnessRunResult(
                status="surrender",
                answer=SURRENDER_SENTINEL,
                num_turns=self._run_meta.get("num_turns"),
                usage=dict(self.token_usage),
                total_cost_usd=self._run_meta.get("total_cost_usd"),
                metadata=dict(self._run_meta),
            )
            return SURRENDER_SENTINEL

        if not final_answer:
            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content="Error: harness returned no answer.",
                    name="claude-code-error",
                )
            )
            self.harness_result = HarnessRunResult(
                status="sdk_failure",
                error="the harness returned no answer",
                num_turns=self._run_meta.get("num_turns"),
                usage=dict(self.token_usage),
                total_cost_usd=self._run_meta.get("total_cost_usd"),
                metadata=dict(self._run_meta),
            )
            return "Error solving the task: the harness returned no answer."

        # Ensure the submit-ready answer is in the transcript exactly once. The
        # harness `ResultMessage` is not otherwise recorded, so append it unless
        # the last streamed assistant message already equals it (ignoring
        # surrounding whitespace). In verbatim mode this avoids the duplicate the
        # `Final Answer:`-stripping used to produce, while still keeping the
        # answer in the trace.
        last_content = self.messages[-1].get("content") if self.messages else None
        already_recorded = (
            isinstance(last_content, str) and last_content.strip() == final_answer
        )
        if not already_recorded:
            self.messages.append(LiteLLMMessage(role="assistant", content=final_answer))
        self.harness_result = HarnessRunResult(
            status="success",
            answer=final_answer,
            num_turns=self._run_meta.get("num_turns"),
            usage=dict(self.token_usage),
            total_cost_usd=self._run_meta.get("total_cost_usd"),
            metadata=dict(self._run_meta),
        )
        return final_answer

    def _fail(self, status: HarnessStatus, error: str, exc: BaseException) -> str:
        """Record an infrastructure failure and return an error answer string.

        The failure is stored on `self.harness_result` with a precise status
        so the benchmark can distinguish a timeout / budget exhaustion / SDK
        crash from an ordinary wrong model answer.
        """
        logger.error(f"Claude Code harness {status}: {error}")
        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content=f"Error running Claude Code harness: {exc}",
                name="claude-code-error",
            )
        )
        self.harness_result = HarnessRunResult(
            status=status,
            error=error,
            num_turns=self._run_meta.get("num_turns"),
            usage=dict(self.token_usage),
            total_cost_usd=self._run_meta.get("total_cost_usd"),
            metadata=dict(self._run_meta),
        )
        return f"Error solving the task: {error}"
