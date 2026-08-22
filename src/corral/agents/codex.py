import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Literal

import anyio

from corral.report.logging import logger

# The Codex SDK ships as the optional `corral[codex]` extra. Wrap the import so
# an environment without the extra gets an actionable install hint instead of a
# bare ModuleNotFoundError deep in the import chain.
try:
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extras
    raise ModuleNotFoundError(
        "CodexAgent requires the OpenAI Codex SDK, which ships as the optional "
        "'codex' extra. Install it with `pip install 'corral[codex]'`."
    ) from exc

from corral.agents.base_agent import BaseAgent, prompt_with_state_history
from corral.agents.schema import (
    SURRENDER_SENTINEL,
    AgentOutcome,
    AgentUsage,
)
from corral.agents.session import AgentSession
from corral.agents.usage import usage_field
from corral.agents.utils import LiteLLMMessage

# Name under which the corral MCP server is registered with the Codex harness
# (the `[mcp_servers.<name>]` config table key).
_MCP_SERVER_NAME = "corral"

# MCP tools that create/place a file the model may later submit a path to. When
# any of these is available, the model is told to use relative paths: the corral
# filesystem tools and the scorer both resolve a relative path against the
# server-side execution workspace, whereas an absolute path built from Codex's
# advertised sandbox cwd points at a throwaway temp dir this agent deletes before
# scoring (so the file — and the submission referencing it — is lost).
_FILE_WRITING_TOOLS = frozenset({"write_file", "copy_file", "move_file", "mkdir"})

# Terminal status of a single harness run. Distinguishing these lets the
# benchmark record *why* a run ended instead of collapsing infrastructure
# failures (timeouts, SDK crashes) into a wrong model answer.
HarnessStatus = Literal[
    "success",
    "surrender",
    "timeout",
    "sdk_failure",
]


@dataclass
class HarnessRunResult:
    """Structured outcome of one Codex harness run.

    `CodexAgent.run_session` maps this provider result onto `AgentOutcome`
    so, for example, a timeout is never scored as a wrong answer.
    """

    status: HarnessStatus
    answer: str | None = None
    error: str | None = None
    num_tool_calls: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RunState:
    """Mutable scratch owned by one invocation, never by the agent instance."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    result: HarnessRunResult | None = None


class _CodexError(RuntimeError):
    """Raised when the Codex harness ends in a non-success state.

    Carries a `subtype` so the caller can map it onto a :data:`HarnessStatus`
    instead of treating every failure the same way.
    """

    def __init__(self, message: str, subtype: str | None = None):
        super().__init__(message)
        self.subtype = subtype


def _sdk_version() -> str | None:
    """Best-effort version of the installed `openai-codex` package."""
    try:
        return _pkg_version("openai-codex")
    except PackageNotFoundError:  # pragma: no cover - only without the SDK
        return None


def _toml_str(value: str) -> str:
    """Render a Python string as a TOML basic string.

    A JSON string is a valid TOML basic string for our ASCII-safe inputs
    (URLs, tool names, paths), so `json.dumps` produces correct escaping.
    """
    return json.dumps(value)


def _toml_num(value: float) -> str:
    """Render a timeout as a TOML number without silently truncating floats.

    `int()` would turn a sub-second timeout like `0.5` into `0` (i.e. *no*
    timeout / an instant one). Whole values are emitted as integers so the
    generated config stays clean; fractional values are preserved as floats.
    """
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class CodexAgent(BaseAgent):
    """Agent that delegates solving a task to the OpenAI Codex harness.

    Args:
        model (str): The model the Codex harness should use (e.g.
            `"gpt-5.4"`). Passed to the harness verbatim. Defaults to
            `"gpt-5.4"`.
        reasoning_effort (str, optional): Effort passed to the Codex turn
            (`"minimal"`/`"low"`/`"medium"`/`"high"`/`"xhigh"`). Set
            explicitly so the benchmark condition does not depend on a
            version-dependent SDK default. Defaults to `"high"`.
        wall_clock_timeout_s (float, optional): Hard wall-clock deadline for the
            whole harness run. On expiry the turn is interrupted and the run is
            reported with `status="timeout"`. Defaults to None (no timeout).
        startup_timeout_s (float, optional): Timeout Codex waits for the corral
            MCP server to initialise. Defaults to 30.
        tool_timeout_s (float, optional): Per-tool-call timeout Codex applies to
            the corral MCP server. Defaults to 600.
        api_endpoint (str, optional): Unused by the native harness.
        temperature (float, optional): Ignored by the Codex harness (the harness
            controls its own sampling; `reasoning_effort` is the tuning knob).
            Accepted for configuration consistency; the harness ignores it.
        system_prompt (str, optional): Developer instructions handed to the
            harness. If None, uses the default corral system prompt.
        **kwargs: Additional provider configuration retained as provenance.

    `run_session` opens the task-local MCP endpoint and offloads the Codex
    SDK's synchronous event stream without blocking the task runtime.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        reasoning_effort: str | None = "high",
        wall_clock_timeout_s: float | None = None,
        startup_timeout_s: float = 30.0,
        tool_timeout_s: float = 600.0,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
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

        # Model string handed to the Codex harness itself.
        self.harness_model = model
        self.reasoning_effort = reasoning_effort
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.tool_timeout_s = tool_timeout_s

    def _developer_instructions(
        self, enable_surrender: bool, tool_names: list[str] | None = None
    ) -> str:
        """Compose the developer instructions handed to the Codex thread."""
        final_answer_directive = (
            "When you have solved the task, call the `submit_answer` MCP tool "
            "with the complete answer. A plain-text final response does not "
            "complete the task."
        )
        instructions = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed evaluation environment. "
            "You may use Codex's built-in tools as well as the provided "
            f"`{_MCP_SERVER_NAME}` MCP tools. Built-in filesystem and shell tools "
            "operate in an isolated per-run workspace; use the MCP tools for task "
            "environment data and actions. Do not invent tool outputs. "
        )
        # Only mention file paths when a file-writing tool is actually available,
        # so a task without one sees the exact same instructions as before. The
        # harness advertises a sandbox working directory, but that directory is a
        # throwaway temp dir — files must be addressed relative to the task
        # workspace (handled by the corral filesystem tools), not by an absolute
        # path built from the advertised cwd.
        if tool_names and _FILE_WRITING_TOOLS.intersection(tool_names):
            instructions += (
                "When a tool writes a file and you submit or reference its path, "
                "use a RELATIVE path (a bare filename like `result.cif`), never an "
                "absolute path: paths are resolved against the task workspace, and "
                "an absolute path built from the working directory shown to you "
                "points outside it and will not be found. "
            )
        instructions += final_answer_directive
        if enable_surrender:
            if self.surrender_prompt is not None:
                instructions += "\n\n" + self.surrender_prompt.fill({})
            instructions += (
                "\nTo surrender, call `submit_answer` with the exact answer "
                f"`{SURRENDER_SENTINEL}`; do not return the sentinel as text."
            )
        return instructions

    def _render_config_toml(self, mcp_url: str, tool_names: list[str]) -> str:
        """Generate the isolated Codex `config.toml` for a run.

        Leaves Codex's native built-in tool set enabled and registers the corral
        MCP server, restricted to the exact task tools. Codex loads this from
        the run's isolated `CODEX_HOME`, so no user plugins, skills, or MCP
        servers are inherited.
        """
        lines = [
            # Keep native Codex tool defaults. Only the external task endpoint
            # is narrowed to the tools exposed by this Corral session.
            f"[mcp_servers.{_MCP_SERVER_NAME}]",
            f"url = {_toml_str(mcp_url)}",
            "required = true",
            f"enabled_tools = [{', '.join(_toml_str(n) for n in tool_names)}]",
            # The allowlisted corral tools do not require interactive approval
            # (this is a non-interactive benchmark); it does NOT widen access
            # beyond `enabled_tools` / the task-scoped endpoint.
            'default_tools_approval_mode = "approve"',
            f"startup_timeout_sec = {_toml_num(self.startup_timeout_s)}",
            f"tool_timeout_sec = {_toml_num(self.tool_timeout_s)}",
            "",
        ]
        return "\n".join(lines)

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
        server exposes it — the *MCP* schema (`tools/list`) that Codex actually
        receives. They are logically related but not byte-for-byte equal, so the
        MCP hash is the authoritative record of what the harness saw.
        """
        rest_tool_schema = json.dumps(
            [t.get("function", {}) for t in tools], sort_keys=True, default=str
        )
        return {
            "model_requested": self.harness_model,
            "openai_codex_version": _sdk_version(),
            "python_version": platform.python_version(),
            "reasoning_effort": self.reasoning_effort,
            "max_sdk_turns": iteration_limit,
            "streaming_enabled": True,
            "wall_clock_timeout_s": self.wall_clock_timeout_s,
            "tool_timeout_s": self.tool_timeout_s,
            "startup_timeout_s": self.startup_timeout_s,
            "mcp_url": mcp_url,
            "tool_verbosity": verbosity,
            "mcp_tools_enabled": sorted(
                t["function"]["name"]
                for t in tools
                if t.get("function", {}).get("name")
            ),
            "rest_openai_tool_schema_sha256": hashlib.sha256(
                rest_tool_schema.encode("utf-8")
            ).hexdigest(),
            "mcp_tool_schema_sha256": mcp_schema_sha256,
            "sdk_internal_tools_policy": "native_defaults",
            "sandbox": Sandbox.read_only.value,
            "approval_mode": ApprovalMode.auto_review.value,
        }

    def _execute_codex_turn(
        self,
        *,
        prompt: str,
        developer_instructions: str,
        mcp_url: str,
        tool_names: list[str],
        codex_home: Path,
        workspace: Path,
        run: _RunState,
    ) -> str:
        """Drive the Codex harness to completion and return its answer."""
        (codex_home / "config.toml").write_text(
            self._render_config_toml(mcp_url, tool_names), encoding="utf-8"
        )

        # Isolate CODEX_HOME so the run cannot inherit the developer's global
        # config, MCP servers, plugins, skills, or memories. `CodexConfig.env`
        # is merged over the inherited process environment by the SDK, so
        # OPENAI_API_KEY still reaches the `codex app-server` subprocess.
        config = CodexConfig(
            cwd=str(workspace),
            env={"CODEX_HOME": str(codex_home)},
        )

        with Codex(config=config) as codex:
            # The isolated CODEX_HOME has no `auth.json`, and the harness does
            # not auto-authenticate the `/v1/responses` API from the
            # OPENAI_API_KEY env var alone. Explicitly log the key in so the
            # app-server attaches a bearer token (otherwise every request 401s
            # with "Missing bearer or basic authentication"). `run()` has
            # already verified OPENAI_API_KEY is present.
            codex.login_api_key(os.environ["OPENAI_API_KEY"])
            thread = codex.thread_start(
                model=self.harness_model,
                cwd=str(workspace),
                developer_instructions=developer_instructions,
                approval_mode=ApprovalMode.auto_review,
                sandbox=Sandbox.read_only,
                ephemeral=True,
            )
            return self._drive_turn(thread, prompt, run)

    def _drive_turn(self, thread: Any, prompt: str, run: _RunState) -> str:
        """Stream one Codex turn: record the transcript, enforce the deadline.

        Consuming the turn's event stream (rather than the blocking
        `thread.run`) lets the agent (a) build the transcript incrementally,
        (b) select the final answer with the SDK's phase-aware logic, and
        (c) enforce a wall-clock deadline via a watchdog that interrupts the turn.
        """
        turn = thread.turn(
            prompt,
            effort=self.reasoning_effort,
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.auto_review,
        )

        # Watchdog: interrupt the turn if it outlives the wall-clock deadline.
        timed_out = threading.Event()

        def _on_timeout() -> None:
            timed_out.set()
            try:
                turn.interrupt()
            except Exception as exc:  # best effort
                logger.warning(f"Codex turn interrupt (timeout) failed: {exc}")

        timer: threading.Timer | None = None
        if self.wall_clock_timeout_s is not None:
            timer = threading.Timer(self.wall_clock_timeout_s, _on_timeout)
            timer.daemon = True
            timer.start()

        # Select the answer the way the official SDK does: prefer the most recent
        # message explicitly in the final-answer phase, and fall back to the most
        # recent phase-less assistant message. Never concatenate every message.
        final_phase_text: str | None = None
        phase_less_text: str | None = None
        tool_calls = 0
        turn_status: str | None = None

        stream = turn.stream()
        try:
            for event in stream:
                method = getattr(event, "method", None)
                payload = getattr(event, "payload", None)

                if method == "item/completed":
                    item = getattr(payload, "item", None)
                    root = getattr(item, "root", item)
                    kind = getattr(root, "type", None)
                    if kind == "mcpToolCall":
                        tool_calls += 1
                    elif kind == "agentMessage":
                        phase = getattr(getattr(root, "phase", None), "value", None)
                        text = getattr(root, "text", "") or ""
                        if phase in {"final_answer", "finalAnswer"}:
                            final_phase_text = text
                        elif phase is None:
                            phase_less_text = text
                    self._record_item(root, run)
                elif method == "thread/tokenUsage/updated":
                    usage = getattr(payload, "token_usage", None)
                    if usage is not None:
                        self._record_usage(usage, run)
                elif method == "turn/completed":
                    turn_status = self._on_turn_completed(payload, run)
        finally:
            stream.close()
            if timer is not None:
                timer.cancel()

        run.metadata["num_tool_calls"] = tool_calls

        # A wall-clock timeout legitimately ends the turn as `interrupted`; check
        # it before requiring a clean completion so that expected interrupt is
        # not treated as an unexplained failure.
        if timed_out.is_set():
            raise _CodexError(
                f"Codex turn exceeded the wall-clock deadline of "
                f"{self.wall_clock_timeout_s}s",
                subtype="timeout",
            )
        self._require_completed_turn(turn_status)

        if final_phase_text is not None:
            return final_phase_text
        return phase_less_text or ""

    def _on_turn_completed(self, payload: Any, run: _RunState) -> str | None:
        """Handle a `turn/completed` event: record status, fail on `failed`.

        Returns the turn's status string so the caller can require a genuinely
        `completed` turn (a `failed` turn raises here immediately).
        """
        turn = getattr(payload, "turn", None)
        status = getattr(getattr(turn, "status", None), "value", None)
        run.metadata["duration_ms"] = getattr(turn, "duration_ms", None)
        run.metadata["turn_status"] = status
        if status == "failed":
            error = getattr(turn, "error", None)
            message = getattr(error, "message", None) or "Codex turn failed."
            raise _CodexError(message, subtype="sdk_failure")
        return status

    def _require_completed_turn(self, status: str | None) -> None:
        """Reject any terminal turn status other than a clean `completed`.

        A wall-clock timeout is handled by the caller before this check, so any
        remaining `interrupted` (or otherwise non-`completed`) status is
        unexplained: treating it as success could return partial text as a real
        answer. Fail loudly instead.
        """
        if status == "completed":
            return
        if status == "interrupted":
            raise _CodexError(
                "Codex turn was interrupted unexpectedly (no local timeout "
                "explains it).",
                subtype="sdk_failure",
            )
        raise _CodexError(
            f"Codex turn ended with unexpected status {status!r}.",
            subtype="sdk_failure",
        )

    def _record_item(self, item: Any, run: _RunState) -> None:
        """Fold one completed Codex thread item into the transcript."""
        kind = getattr(item, "type", None)
        if kind == "agentMessage":
            text = getattr(item, "text", "") or ""
            run.messages.append(LiteLLMMessage(role="assistant", content=text))
        elif kind == "reasoning":
            content = getattr(item, "content", None) or getattr(item, "summary", None)
            text = "\n".join(content) if isinstance(content, list) else str(content)
            if text:
                run.messages.append(
                    LiteLLMMessage(role="assistant", content=text, name="thinking")
                )
        elif kind == "mcpToolCall":
            self._record_tool_call(item, run)

    def _record_tool_call(self, item: Any, run: _RunState) -> None:
        """Record an MCP tool call and its result as transcript messages."""
        tool = getattr(item, "tool", "")
        call_id = getattr(item, "id", "")
        arguments = getattr(item, "arguments", None)
        run.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(arguments, default=str),
                        },
                    }
                ],
            }
        )
        error = getattr(item, "error", None)
        is_error = error is not None
        if is_error:
            run.metadata["tool_errors"] = run.metadata.get("tool_errors", 0) + 1
            content = getattr(error, "message", None) or str(error)
        else:
            content = json.dumps(getattr(item, "result", None), default=str)
        run.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
                # Name the transcript result after the actual tool, not a
                # generic "tool_result", so tool-level analysis stays possible.
                "name": tool or "tool_result",
                "is_error": is_error,
            }
        )

    def _record_usage(self, usage: Any, run: _RunState) -> None:
        """Record canonical scratch usage from a Codex token-usage update."""
        normalized = self._usage(usage)
        total = usage_field(usage, "total", None)
        total = usage if total is None else total
        cached = int(usage_field(total, "cached_input_tokens", 0) or 0)
        run.usage = {
            "input_tokens": normalized.input_tokens,
            "output_tokens": normalized.output_tokens,
            "reasoning_tokens": normalized.reasoning_tokens,
            "total_tokens": normalized.input_tokens + normalized.output_tokens,
            "cached_input_tokens": cached,
        }

    def _usage(
        self,
        raw_usage: Any,
        *,
        llm_calls: int = 0,
    ) -> AgentUsage:
        """Extract Codex's nested `usage.total` token fields."""
        total = usage_field(raw_usage, "total", None)
        total = raw_usage if total is None else total
        return super()._usage(
            {
                "input_tokens": usage_field(
                    total,
                    "input_tokens",
                    usage_field(total, "prompt_tokens", 0),
                ),
                "output_tokens": usage_field(
                    total,
                    "output_tokens",
                    usage_field(total, "completion_tokens", 0),
                ),
                "reasoning_tokens": usage_field(
                    total,
                    "reasoning_tokens",
                    usage_field(total, "reasoning_output_tokens", 0),
                ),
            },
            llm_calls=llm_calls,
        )

    def _normalize_prompt(self, task_guide: Any) -> tuple[str, bool]:
        """Flatten a raw task prompt into the plain-text Codex input.

        Codex turns take text input; multimodal parts are flattened to their
        text and image parts are dropped with a warning (the corral MCP tools,
        not the prompt, are the intended channel for task data). Returns the
        flattened text and whether any non-text parts were dropped, so the run
        metadata can flag that the harness saw a reduced view of the task.
        """
        if isinstance(task_guide, list):
            dropped_images = any(
                isinstance(p, dict) and p.get("type", "text") != "text"
                for p in task_guide
            )
            if dropped_images:
                logger.warning(
                    "CodexAgent received multimodal prompt content; image parts "
                    "are not forwarded to the Codex harness. If the task depends "
                    "on the image and no MCP tool exposes it, the task is not "
                    "faithfully supported by this agent."
                )
            text = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in task_guide
            )
            return text, dropped_images
        return str(task_guide), False

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run Codex's native harness against the task-local MCP session."""
        # Each run uses an isolated CODEX_HOME, so an existing interactive Codex
        # login (stored under the developer's normal CODEX_HOME) is *not*
        # inherited. Require API-key auth explicitly for reproducible runs and a
        # clear failure rather than a confusing mid-run auth error.
        if not os.environ.get("OPENAI_API_KEY"):
            return AgentOutcome(
                status="harness_failure",
                error=(
                    "CodexAgent requires OPENAI_API_KEY: each run uses an "
                    "isolated CODEX_HOME"
                ),
            )

        run = _RunState()
        iteration_limit = session.iteration_limit

        # Resolve verbosity once and use the *same* value for the REST allowlist
        # and the MCP endpoint, so what Codex sees and what the metadata records
        # cannot drift apart (important for faithful ablations).
        verbosity = "full"
        tools = [dict(tool) for tool in session.tools]
        tool_names = [
            t["function"]["name"] for t in tools if t.get("function", {}).get("name")
        ]

        prompt, dropped_images = self._normalize_prompt(session.prompt)
        prompt = prompt_with_state_history(prompt, session.messages)

        # Seed the message history so transcript saving has the task prompt first.
        run.messages.append(LiteLLMMessage(role="user", content=prompt))

        # Record the hash of the MCP tool schema the harness actually receives
        # (`tools/list` == `Tool.to_mcp`), so the run provenance well and truly
        # captures what tools the agent saw.
        task_id = str(getattr(session, "task_id", session.execution_id))
        developer_instructions = self._developer_instructions(
            session.surrender_allowed, tool_names
        )
        async with session.open_mcp() as mcp:
            run.metadata = self._harness_metadata(
                mcp.url,
                verbosity,
                tools,
                mcp_schema_sha256=None,
                iteration_limit=iteration_limit,
            )
            run.metadata["dropped_image_parts"] = dropped_images
            # A Codex thread.turn(...) is one SDK turn for Corral usage
            # accounting, even though it can contain many internal events.
            run.metadata["sdk_turns"] = 1
            await anyio.to_thread.run_sync(
                lambda: self._execute_harness(
                    task_id=task_id,
                    prompt=prompt,
                    developer_instructions=developer_instructions,
                    mcp_url=mcp.url,
                    tool_names=tool_names,
                    enable_surrender=session.surrender_allowed,
                    run=run,
                )
            )

        for message in run.messages:
            await session.record_message(message)
        result = run.result
        usage = self._usage(
            run.usage,
            llm_calls=int(run.metadata.get("sdk_turns", 0) or 0),
        )
        if result is None:
            return AgentOutcome(
                status="harness_failure",
                error="Codex harness returned no structured result",
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
                error="Codex finished without calling submit_answer",
                usage=usage,
                metadata=metadata,
            )
        status = "timeout" if result.status == "timeout" else "harness_failure"
        return AgentOutcome(
            status=status,
            error=result.error or "Codex harness failed",
            usage=usage,
            metadata=metadata,
        )

    def _execute_harness(
        self,
        *,
        task_id: str,
        prompt: str,
        developer_instructions: str,
        mcp_url: str,
        tool_names: list[str],
        enable_surrender: bool,
        run: _RunState,
    ) -> str:
        """Drive the harness and retain its terminal text for diagnostics.

        Completion still requires the harness to call the session's
        `submit_answer` MCP tool; this returned text is never submitted by the
        adapter.
        """
        # CODEX_HOME is always a throwaway temp dir this agent owns and deletes,
        # so the run never inherits the developer's global Codex config.
        root = Path(tempfile.mkdtemp(prefix="corral-codex-"))
        codex_home = root / "codex-home"
        codex_home.mkdir()

        # Codex never receives the server-side execution path. Its built-in
        # filesystem/shell tools operate in an empty per-run directory, while
        # task file operations remain available through the capability-scoped
        # MCP endpoint. This prevents project discovery from exposing sibling
        # task workspaces without disabling the native tools.
        run_workspace = root / "workspace"
        run_workspace.mkdir()
        run.metadata["codex_cwd"] = str(run_workspace)
        run.metadata["codex_cwd_is_execution_workspace"] = False
        run.metadata["workspace_access"] = "isolated_sdk_workspace_and_mcp"

        try:
            final_answer = self._execute_codex_turn(
                prompt=prompt,
                developer_instructions=developer_instructions,
                mcp_url=mcp_url,
                tool_names=tool_names,
                codex_home=codex_home,
                workspace=run_workspace,
                run=run,
            )
        except _CodexError as e:
            status_map: dict[str, HarnessStatus] = {
                "timeout": "timeout",
                "sdk_failure": "sdk_failure",
            }
            return self._fail(
                run,
                status_map.get(e.subtype or "", "sdk_failure"),
                str(e),
                e,
            )
        except Exception as e:  # surface as infra failure, not a model answer
            return self._fail(run, "sdk_failure", str(e), e)
        finally:
            shutil.rmtree(root, ignore_errors=True)

        final_answer = final_answer.strip()

        # Surrender only on an *exact* sentinel answer.
        if (
            enable_surrender
            and final_answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            logger.debug(f"Agent retiring from task {task_id}")
            run.result = self._result(run, "surrender", answer=SURRENDER_SENTINEL)
            return SURRENDER_SENTINEL

        if not final_answer:
            run.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content="Error: harness returned no answer.",
                    name="codex-error",
                )
            )
            run.result = self._result(
                run, "sdk_failure", error="the harness returned no answer"
            )
            return "Error solving the task: the harness returned no answer."

        # Ensure the harness terminal text is in the transcript exactly once. The
        # harness's final message is usually already recorded, so only append when
        # it differs (ignoring surrounding whitespace). In verbatim mode the
        # recorded message equals the answer, so nothing is appended and the trace
        # is not duplicated; extraction that rewrote the text appends the
        # normalized value.
        last_content = run.messages[-1].get("content") if run.messages else None
        already_recorded = (
            isinstance(last_content, str) and last_content.strip() == final_answer
        )
        if not already_recorded:
            run.messages.append(LiteLLMMessage(role="assistant", content=final_answer))
        run.result = self._result(run, "success", answer=final_answer)
        return final_answer

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
            num_tool_calls=run.metadata.get("num_tool_calls"),
            usage=dict(run.usage),
            duration_ms=run.metadata.get("duration_ms"),
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
        run.messages.append(
            LiteLLMMessage(
                role="assistant",
                content=f"Error running Codex harness: {exc}",
                name="codex-error",
            )
        )
        run.result = self._result(run, status, error=error)
        return f"Error solving the task: {error}"
