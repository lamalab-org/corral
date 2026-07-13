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
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode

from loguru import logger

# The Codex SDK is declared as the `corral[codex]` extra and assumed installed.
from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

from corral.agents.base_agent import BaseAgent
from corral.agents.hooks import HookPoint
from corral.agents.schema import SURRENDER_SENTINEL
from corral.agents.utils import LiteLLMMessage
from corral.router.routes import CorralRouter

# Name under which the corral MCP server is registered with the Codex harness
# (the `[mcp_servers.<name>]` config table key).
_MCP_SERVER_NAME = "corral"

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

    :meth:`CodexAgent.run` still returns a plain string for interface
    compatibility, but the benchmark record should read the structured status
    from here so that, for example, a `timeout` is not scored as a wrong answer.
    """

    status: HarnessStatus
    answer: str | None = None
    error: str | None = None
    num_tool_calls: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
        max_iterations (int, optional): Unused by Codex (one Codex turn contains
            many internal reasoning/tool steps, so it does not map to a corral
            iteration count). Kept for interface compatibility. Bound a run with
            `wall_clock_timeout_s`. Defaults to 1.
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
        api_endpoint (str, optional): Unused by the harness; kept for interface
            compatibility with :class:`BaseAgent`.
        temperature (float, optional): Ignored by the Codex harness (the harness
            controls its own sampling; `reasoning_effort` is the tuning knob).
            Accepted only for :class:`BaseAgent` interface compatibility.
        system_prompt (str, optional): Developer instructions handed to the
            harness. If None, uses the default corral system prompt.
        extractor_model (str, optional): LiteLLM-compatible model used only by
            the base class machinery. If None it is derived from `model`
            (prefixing `"openai/"` when `model` has no provider prefix).
        **kwargs: Additional keyword arguments forwarded to :class:`BaseAgent`.
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        max_iterations: int = 1,
        reasoning_effort: str | None = "high",
        wall_clock_timeout_s: float | None = None,
        startup_timeout_s: float = 30.0,
        tool_timeout_s: float = 600.0,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        extractor_model: str | None = None,
        **kwargs,
    ):
        # The harness model (e.g. "gpt-5.4") is not a LiteLLM route on its own.
        # Derive a LiteLLM-compatible model for the (unused-by-default) base
        # machinery so token counting / answer extraction keep working if ever
        # invoked.
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

        # Model string handed to the Codex harness itself.
        self.harness_model = model
        self.reasoning_effort = reasoning_effort
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.tool_timeout_s = tool_timeout_s
        self._available_tools = None
        # Structured outcome of the most recent run (see `HarnessRunResult`).
        self.harness_result: HarnessRunResult | None = None
        # Per-run scratch populated by `_run_codex` for the benchmark record.
        self._run_meta: dict[str, Any] = {}

    @property
    def requires_answer_extraction(self) -> bool:
        """Submit the harness answer verbatim, without a second model call.

        The harness is instructed to emit `Final Answer:` and :meth:`run`
        deterministically extracts it, so running the base class's LiteLLM
        extractor would add an extra, separately-billed call (a *different*
        model) whose output could differ from the harness's actual answer.
        Bypassing it keeps this a faithful measurement of the Codex harness.
        """
        return False

    def _developer_instructions(self, enable_surrender: bool) -> str:
        """Compose the developer instructions handed to the Codex thread."""
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
        instructions = (
            self.system_prompt
            + "\n\nYou are solving a task in a sandboxed evaluation environment. "
            f"You may ONLY interact with it through the provided `{_MCP_SERVER_NAME}` "
            "MCP tools; do not attempt to use the shell, filesystem, web search, or "
            "any other tool, and do not request additional permissions. Do not invent "
            "tool outputs. " + final_answer_directive
        )
        if enable_surrender and self.surrender_prompt is not None:
            instructions += "\n\n" + self.surrender_prompt.fill({})
        return instructions

    def _mcp_url(self, interface: CorralRouter, task_id: str, verbosity: str) -> str:
        """Build the task-scoped MCP endpoint URL Codex connects to.

        Points Codex at the environment server's own task-scoped MCP endpoint.
        The `verbosity` is forwarded as a query parameter so it matches exactly
        the verbosity the REST allowlist was fetched at (otherwise Codex could
        see full tool descriptions while the allowlist metadata reflects a
        briefer condition, silently breaking ablations).
        """
        base_url = interface.base_url.rstrip("/")
        encoded_task_id = quote(str(task_id), safe="")
        query = urlencode({"verbosity": verbosity})
        # Trailing slash on `/mcp/` avoids a 307 redirect: the endpoint is a
        # Starlette Mount, so a bare `/mcp` bounces to `/mcp/` (an extra
        # round-trip per call). Hit the canonical path directly.
        return f"{base_url}/tasks/{encoded_task_id}/mcp/?{query}"

    def _render_config_toml(self, mcp_url: str, tool_names: list[str]) -> str:
        """Generate the isolated Codex `config.toml` for a run.

        Disables every built-in capability and registers only the corral MCP
        server, restricted to the exact tools available for this task. Codex
        loads this from the run's isolated `CODEX_HOME`.
        """
        lines = [
            # Remove the built-in web-search tool.
            'web_search = "disabled"',
            "",
            # Disable execution / multi-agent / plugin capabilities. Shell,
            # unified exec, apps and multi-agent are otherwise on by default.
            "[features]",
            "shell_tool = false",
            "unified_exec = false",
            "apps = false",
            "multi_agent = false",
            "",
            "[tools]",
            "view_image = false",
            "web_search = false",
            "",
            # The only actionable tools: the corral task endpoint over HTTP.
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
        }

    def _run_codex(
        self,
        *,
        prompt: str,
        developer_instructions: str,
        mcp_url: str,
        tool_names: list[str],
        codex_home: Path,
        workspace: Path,
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
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.read_only,
                ephemeral=True,
            )
            return self._drive_turn(thread, prompt)

    def _drive_turn(self, thread: Any, prompt: str) -> str:
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
            approval_mode=ApprovalMode.deny_all,
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
                    self._record_item(root)
                elif method == "thread/tokenUsage/updated":
                    usage = getattr(payload, "token_usage", None)
                    if usage is not None:
                        self._record_usage(usage)
                elif method == "turn/completed":
                    turn_status = self._on_turn_completed(payload)
        finally:
            stream.close()
            if timer is not None:
                timer.cancel()

        self._run_meta["num_tool_calls"] = tool_calls

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

    def _on_turn_completed(self, payload: Any) -> str | None:
        """Handle a `turn/completed` event: record status, fail on `failed`.

        Returns the turn's status string so the caller can require a genuinely
        `completed` turn (a `failed` turn raises here immediately).
        """
        turn = getattr(payload, "turn", None)
        status = getattr(getattr(turn, "status", None), "value", None)
        self._run_meta["duration_ms"] = getattr(turn, "duration_ms", None)
        self._run_meta["turn_status"] = status
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

    def _record_item(self, item: Any) -> None:
        """Fold one completed Codex thread item into the transcript."""
        kind = getattr(item, "type", None)
        if kind == "agentMessage":
            text = getattr(item, "text", "") or ""
            self.messages.append(LiteLLMMessage(role="assistant", content=text))
        elif kind == "reasoning":
            content = getattr(item, "content", None) or getattr(item, "summary", None)
            text = "\n".join(content) if isinstance(content, list) else str(content)
            if text:
                self.messages.append(
                    LiteLLMMessage(role="assistant", content=text, name="thinking")
                )
        elif kind == "mcpToolCall":
            self._record_tool_call(item)

    def _record_tool_call(self, item: Any) -> None:
        """Record an MCP tool call and its result as transcript messages."""
        tool = getattr(item, "tool", "")
        call_id = getattr(item, "id", "")
        arguments = getattr(item, "arguments", None)
        self.messages.append(
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
            self._run_meta["tool_errors"] = self._run_meta.get("tool_errors", 0) + 1
            content = getattr(error, "message", None) or str(error)
        else:
            content = json.dumps(getattr(item, "result", None), default=str)
        self.messages.append(
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

    def _record_usage(self, usage: Any) -> None:
        """Map the Codex thread token usage onto the base token-usage schema."""
        total = getattr(usage, "total", None) or usage
        input_tokens = int(getattr(total, "input_tokens", 0) or 0)
        cached = int(getattr(total, "cached_input_tokens", 0) or 0)
        completion_tokens = int(getattr(total, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(total, "total_tokens", 0) or (input_tokens + completion_tokens)
        )
        self.token_usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
        }
        # Codex reports usage already aggregated for the whole thread, so mirror
        # it into the run total rather than summing (which would double-count on
        # repeated updates).
        self.cumulative_token_usage = dict(self.token_usage)

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

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,  # noqa: ARG002
        enable_surrender: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Run the Codex harness to solve the task.

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
        # Each run uses an isolated CODEX_HOME, so an existing interactive Codex
        # login (stored under the developer's normal CODEX_HOME) is *not*
        # inherited. Require API-key auth explicitly for reproducible runs and a
        # clear failure rather than a confusing mid-run auth error.
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "CodexAgent requires OPENAI_API_KEY: each run uses an isolated "
                "CODEX_HOME and does not inherit an existing Codex login."
            )

        self.harness_result = None

        # Resolve verbosity once and use the *same* value for the REST allowlist
        # and the MCP endpoint, so what Codex sees and what the metadata records
        # cannot drift apart (important for faithful ablations).
        verbosity = getattr(interface, "current_verbosity", None) or "brief"
        tools = interface.get_available_tools_for_task(
            task_id, verbosity=verbosity
        ).get("tools", [])
        self._available_tools = tools
        tool_names = [
            t["function"]["name"] for t in tools if t.get("function", {}).get("name")
        ]

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
        developer_instructions = self._developer_instructions(enable_surrender)

        self._execute_hooks(HookPoint.BEFORE_TASK, interface, task_id)

        try:
            return self._run_and_extract(
                task_id=task_id,
                prompt=prompt,
                developer_instructions=developer_instructions,
                mcp_url=mcp_url,
                tool_names=tool_names,
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
        developer_instructions: str,
        mcp_url: str,
        tool_names: list[str],
        enable_surrender: bool,
    ) -> str:
        """Drive the harness and turn its output into a submit-ready answer."""
        # Fresh, empty, per-episode isolated directories; cleaned up afterwards.
        root = Path(tempfile.mkdtemp(prefix="corral-codex-"))
        codex_home = root / "codex-home"
        workspace = root / "workspace"
        codex_home.mkdir()
        workspace.mkdir()

        try:
            final_answer = self._run_codex(
                prompt=prompt,
                developer_instructions=developer_instructions,
                mcp_url=mcp_url,
                tool_names=tool_names,
                codex_home=codex_home,
                workspace=workspace,
            )
        except _CodexError as e:
            status_map: dict[str, HarnessStatus] = {
                "timeout": "timeout",
                "sdk_failure": "sdk_failure",
            }
            return self._fail(status_map.get(e.subtype or "", "sdk_failure"), str(e), e)
        except Exception as e:  # surface as infra failure, not a model answer
            return self._fail("sdk_failure", str(e), e)
        finally:
            shutil.rmtree(root, ignore_errors=True)

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

        # Surrender only on an *exact* sentinel answer.
        if (
            enable_surrender
            and final_answer.casefold() == SURRENDER_SENTINEL.casefold()
        ):
            logger.info(f"Agent retiring from task {task_id}")
            self.harness_result = self._result("surrender", answer=SURRENDER_SENTINEL)
            return SURRENDER_SENTINEL

        if not final_answer:
            self.messages.append(
                LiteLLMMessage(
                    role="assistant",
                    content="Error: harness returned no answer.",
                    name="codex-error",
                )
            )
            self.harness_result = self._result(
                "sdk_failure", error="the harness returned no answer"
            )
            return "Error solving the task: the harness returned no answer."

        # Ensure the submit-ready answer is in the transcript exactly once. The
        # harness's final message is usually already recorded, so only append when
        # it differs (ignoring surrounding whitespace). In verbatim mode the
        # recorded message equals the answer, so nothing is appended and the trace
        # is not duplicated; extraction that rewrote the text appends the
        # normalized value.
        last_content = self.messages[-1].get("content") if self.messages else None
        already_recorded = (
            isinstance(last_content, str) and last_content.strip() == final_answer
        )
        if not already_recorded:
            self.messages.append(LiteLLMMessage(role="assistant", content=final_answer))
        self.harness_result = self._result("success", answer=final_answer)
        return final_answer

    def _fetch_mcp_schema_digest(
        self, interface: CorralRouter, task_id: str, verbosity: str
    ) -> str | None:
        """Fetch the digest of the MCP schema Codex will see, if available."""
        fetch = getattr(interface, "get_mcp_tool_schema", None)
        if fetch is None:
            return None
        try:
            return fetch(task_id, verbosity=verbosity).get("mcp_schema_sha256")
        except Exception as exc:
            logger.warning(f"Could not fetch MCP tool schema digest: {exc}")
            return None

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
            num_tool_calls=self._run_meta.get("num_tool_calls"),
            usage=dict(self.token_usage),
            duration_ms=self._run_meta.get("duration_ms"),
            metadata=dict(self._run_meta),
        )

    def _fail(self, status: HarnessStatus, error: str, exc: BaseException) -> str:
        """Record an infrastructure failure and return an error answer string."""
        logger.error(f"Codex harness {status}: {error}")
        self.messages.append(
            LiteLLMMessage(
                role="assistant",
                content=f"Error running Codex harness: {exc}",
                name="codex-error",
            )
        )
        self.harness_result = self._result(status, error=error)
        return f"Error solving the task: {error}"
