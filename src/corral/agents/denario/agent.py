"""Run Denario's native research workflow in a task-isolated worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from corral.agents._transcript import event_clock
from corral.agents.base_agent import BaseAgent, _UsageAccumulator
from corral.agents.denario._workflow import completed_context
from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSessionCapabilities
from corral.core.action import Action
from corral.core.errors import concise_error_message
from corral.report.logging import logger

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


# The sandbox deliberately hides Corral's source tree from fresh interpreters.
# Carry only these public adapter modules across the serialization boundary;
# never expose the controller package or private environment implementations.
_WORKER_FILES = {
    name: Path(__file__).with_name(name).read_text(encoding="utf-8")
    for name in ("_worker.py", "_bridge.py", "_responses.py", "_workflow.py")
}


class DenarioAgent(BaseAgent):
    """Use Denario idea/method generation and its CMBagent experiment loop.

    Corral supplies tools, durable observations, and a request budget. Denario
    owns planning, handoffs, code execution, review, and retries. Its local code
    executor requires Corral's Docker sandbox; it is never run on the host.
    """

    session_capabilities = AgentSessionCapabilities(native_workspace=True)

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        temperature: float | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        surrender_prompt: str | None = None,
        denario_model: str | None = None,
        denario_iterations: int = 4,
        max_plan_steps: int = 6,
        max_n_attempts: int = 10,
        wall_clock_timeout_s: float | None = None,
        openrouter_zdr: bool = False,
        openrouter_provider: dict[str, Any] | None = None,
        hooks=None,
        **kwargs: Any,
    ) -> None:
        provider, separator, native_model = model.partition("/")
        if not separator:
            native_model = model
            provider = "anthropic" if model.startswith("claude-") else "openai"
        if provider not in {"openai", "anthropic", "openrouter"} or not native_model:
            raise ValueError(
                "Denario supports openai/, anthropic/, and openrouter/ models"
            )
        if provider != "openrouter" and "/" in native_model:
            raise ValueError("Use openrouter/<author>/<model> for OpenRouter models")
        if not isinstance(openrouter_zdr, bool):
            raise TypeError("openrouter_zdr must be a boolean")
        if openrouter_provider is not None and not isinstance(
            openrouter_provider, dict
        ):
            raise TypeError("openrouter_provider must be an object")
        if provider != "openrouter" and (openrouter_zdr or openrouter_provider):
            raise ValueError("OpenRouter routing settings require an openrouter/ model")
        if denario_model is not None and denario_model != native_model:
            raise ValueError(
                "denario_model must match --model; the native workflow uses "
                "one configured model for all roles"
            )
        for name, value in {
            "denario_iterations": denario_iterations,
            "max_plan_steps": max_plan_steps,
            "max_n_attempts": max_n_attempts,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if wall_clock_timeout_s is not None and wall_clock_timeout_s <= 0:
            raise ValueError("wall_clock_timeout_s must be positive")
        supported = {
            "reasoning_effort",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "frequency_penalty",
            "presence_penalty",
        }
        if unknown := kwargs.keys() - supported:
            raise TypeError(
                f"Unsupported Denario settings: {', '.join(sorted(unknown))}"
            )
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "tool_calling/user_prompt",
            surrender_prompt=surrender_prompt,
            default_surrender_prompt="surrender_prompt/surrender_prompt_tool_calling@tool_calling",
            temperature=temperature,
            hooks=hooks,
            **kwargs,
        )
        self.denario_model = native_model
        self.provider = provider
        self.openrouter_zdr = openrouter_zdr
        self.openrouter_provider = dict(openrouter_provider or {})
        self.denario_iterations = denario_iterations
        self.max_plan_steps = max_plan_steps
        self.max_n_attempts = max_n_attempts
        self.wall_clock_timeout_s = wall_clock_timeout_s
        self._worker_files = dict(_WORKER_FILES)

    def _worker_command(self) -> list[str]:
        # Redirect the OS descriptor before importing Corral or optional SDKs:
        # some dependencies install stdout log handlers during import.
        return [
            sys.executable,
            "-c",
            "import os; writer = os.fdopen(os.dup(1), 'w'); os.dup2(2, 1); "
            "from corral.agents.denario._worker import main; main(writer=writer)",
        ]

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        usage = _UsageAccumulator()
        metadata = {
            "denario_mode": "native_experiment",
            "denario_model": self.denario_model,
            "provider": self.provider,
            "denario_iterations": self.denario_iterations,
            "usage_scope": "all_native_model_requests",
        }

        def outcome(status, error=None):
            if session.submission is not None:
                surrendered = session.submission_status == "surrendered"
                return AgentOutcome(
                    status="surrendered" if surrendered else "completed",
                    answer=None if surrendered else session.submission,
                    usage=usage.outcome(),
                    metadata=metadata,
                )
            return AgentOutcome(
                status=status, error=error, usage=usage.outcome(), metadata=metadata
            )

        if session.submission is not None:
            return outcome("completed")
        workspace = session.native_tool_workspace
        if workspace is None:
            return outcome(
                "harness_failure",
                "Native Denario requires Corral's Docker sandbox. Use corral bench --sandbox docker --build-sandbox-image.",
            )
        if session.get_agent_state("denario"):
            return outcome(
                "harness_failure",
                "An existing Denario run cannot be restarted safely: native checkpoint resume is not supported. Start a new trial to avoid replaying experiments.",
            )

        # Denario owns its role prompts. Supply the environment's task text
        # directly, including its constraints, inputs and submission format.
        task = session.prompt
        if not isinstance(task, str):
            return outcome(
                "harness_failure",
                "Native Denario requires a text task description; multimodal task prompts are not supported.",
            )
        project = Path(workspace) / f"denario-{uuid4().hex}"
        project.mkdir(parents=True)
        metadata["project_dir"] = str(project)
        config = {
            "project_dir": str(project),
            "task": task,
            "tools": list(session.tools),
            "model": self.denario_model,
            "provider": self.provider,
            "openrouter_zdr": self.openrouter_zdr,
            "openrouter_provider": self.openrouter_provider,
            "api_endpoint": self.api_endpoint,
            "model_settings": {**self.kwargs, "temperature": self.temperature},
            "denario_iterations": self.denario_iterations,
            "max_plan_steps": self.max_plan_steps,
            "max_n_attempts": self.max_n_attempts,
        }
        await session.set_agent_state("denario", {**metadata, "phase": "starting"})
        process = None
        drain = None
        runtime = TemporaryDirectory(prefix="denario-runtime-")
        tool_names = {tool["function"]["name"] for tool in session.tools}
        pending: dict[str, dict[str, Any]] = {}
        native_calls: dict[str, str] = {}
        candidate: dict[str, Any] | None = None

        async def record_event(event, *, llm_calls=0):
            event = {"sdk": "denario", **event}
            public_message = event.get("message")
            content = (
                public_message.get("content")
                if isinstance(public_message, dict)
                else public_message
            )
            await session.record_message(
                {
                    "role": "system",
                    "name": f"denario.{event['type']}",
                    "content": content or event["type"],
                    "sdk_event": event,
                },
                usage={} if llm_calls else None,
                llm_calls=llm_calls,
            )

        def track_native(messages):
            for message in messages:
                for call in message.get("tool_calls") or []:
                    native_calls[call["id"]] = call["function"]["name"]
                if message.get("role") == "tool":
                    native_calls.pop(message.get("tool_call_id"), None)

        async def consume():
            nonlocal candidate
            assert process is not None
            assert process.stdout is not None
            assert process.stdin is not None
            while line := await process.stdout.readline():
                event = json.loads(line)
                kind = event["kind"]
                response: Any = None
                if kind == "model_start":
                    if usage.llm_calls >= session.iteration_limit:
                        await record_event(
                            {
                                "type": "model_request_rejected",
                                "request_id": event["id"],
                                "request": event.get("request", {}),
                                "reason": "iteration_limit",
                            }
                        )
                        return outcome(
                            "iteration_limit",
                            "Denario exhausted the model-request budget (including planning and execution).",
                        )
                    pending[event["id"]] = {
                        "request_id": event["id"],
                        "agent": event.get("agent"),
                        "request": event.get("request", {}),
                        "endpoint": event.get("endpoint", "chat/completions"),
                    }
                    usage.add({})
                    await record_event(
                        {"type": "model_request", **pending[event["id"]]}, llm_calls=1
                    )
                elif kind == "model_end":
                    if event["id"] not in pending:
                        raise ValueError("Unmatched Denario model response")
                    pending.pop(event["id"])
                    raw_usage = event.get("usage") or {}
                    usage.add(raw_usage, llm_calls=0)
                    message = dict(event["message"])
                    message["model_output"] = dict(event["message"])
                    if event.get("provider_response") is not None:
                        message["provider_response"] = event["provider_response"]
                    message["request_id"] = event["id"]
                    message["is_error"] = event.get("is_error", False)
                    if "status_code" in event:
                        message["status_code"] = event["status_code"]
                    message["chronology"] = event_clock(sdk="denario")
                    calls = [
                        call
                        for call in (message.pop("tool_calls", None) or [])
                        if call["function"]["name"] not in tool_names
                    ]
                    if calls:
                        message.update(tool_calls=calls, sdk_native=True)
                        track_native([message])
                    message.update(
                        name=f"denario.{event['agent']}", model_input=event["input"]
                    )
                    message["content"] = message.get("content") or ""
                    await session.record_message(message, usage=raw_usage, llm_calls=0)
                    if event.get("error_code") == "insufficient_quota":
                        return outcome(
                            "budget_exhausted",
                            "OpenAI rejected the Denario request: insufficient_quota",
                        )
                elif kind == "tool":
                    action = Action.from_tool_call(event)
                    if action.is_submission:
                        return outcome(
                            "protocol_failure",
                            "Native subagents cannot submit; use propose_answer and finish the native plan.",
                        )
                    result = await session.execute(action)
                    observation = next(
                        m
                        for m in reversed(session.messages)
                        if m.get("role") == "tool"
                        and m.get("tool_call_id") == action.id
                    )
                    response = {
                        "success": result.success,
                        "result": result.result,
                        "error": result.error,
                        "status": observation.get("metadata", {}).get(
                            "status", "success" if result.success else "execution_error"
                        ),
                    }
                elif kind == "proposal":
                    arguments = event.get("arguments")
                    valid = (
                        isinstance(arguments, dict)
                        and set(arguments) == {"answer"}
                        and isinstance(arguments["answer"], str)
                    )
                    response = {
                        "success": valid,
                        "status": "success" if valid else "invalid_args",
                        "result": "Candidate recorded. Finish the native plan and researcher results before submission."
                        if valid
                        else None,
                        "error": None
                        if valid
                        else "propose_answer requires exactly one string argument named 'answer'",
                    }
                    if valid:
                        candidate = {
                            "answer": arguments["answer"],
                            "proposal_id": event["id"],
                            "role": event.get("role"),
                            "step": event.get("step"),
                        }
                        await record_event({"type": "answer_proposed", **candidate})
                elif kind == "messages":
                    await session.record_messages(
                        [
                            {**message, "chronology": event_clock(sdk="denario")}
                            for message in event["messages"]
                        ],
                        llm_calls=0,
                    )
                    track_native(event["messages"])
                elif kind == "event":
                    await record_event(event["event"])
                elif kind == "phase":
                    await record_event({"type": "phase", "phase": event["phase"]})
                    await session.set_agent_state(
                        "denario", {**metadata, "phase": event["phase"]}
                    )
                elif kind == "done":
                    context = event.get("context") or {}
                    await record_event({"type": "finished", "context": context})
                    if candidate is None:
                        return outcome(
                            "protocol_failure",
                            "Denario finished without proposing an answer.",
                        )
                    if candidate["role"] != "researcher" or candidate[
                        "step"
                    ] != context.get("number_of_steps_in_plan"):
                        return outcome(
                            "protocol_failure",
                            "The final researcher step did not confirm the candidate answer.",
                        )
                    report = project / "input_files" / "results.md"
                    if (
                        not completed_context(context)
                        or not report.is_file()
                        or not report.read_text().strip()
                    ):
                        return outcome(
                            "protocol_failure",
                            "Denario has not completed its native plan and saved nonempty results.",
                        )
                    result = await session.execute(
                        Action(
                            name="submit_answer",
                            arguments={"answer": candidate["answer"]},
                            metadata={
                                "denario_proposal_id": candidate["proposal_id"],
                                "denario_results": str(report),
                            },
                        )
                    )
                    return outcome(
                        "completed" if result.success else "protocol_failure",
                        result.error,
                    )
                elif kind == "error":
                    await record_event(
                        {"type": "error", "message": event["error"], "is_error": True}
                    )
                    return outcome("agent_failure", f"Denario failed: {event['error']}")
                else:
                    raise ValueError(f"Unknown Denario worker event: {kind}")
                process.stdin.write(
                    (
                        json.dumps({"result": response}, ensure_ascii=False) + "\n"
                    ).encode()
                )
                await process.stdin.drain()
            return outcome(
                "agent_failure",
                "Denario worker exited unexpectedly; see denario-worker.log in its project directory.",
            )

        async def drain_logs():
            assert process is not None
            assert process.stderr is not None
            with (project / "denario-worker.log").open("wb") as log:
                while chunk := await process.stderr.read(8192):
                    log.write(chunk)
                    log.flush()
                    logger.debug("Denario: {}", chunk.decode(errors="replace").rstrip())

        try:
            package = Path(runtime.name) / "corral" / "agents" / "denario"
            package.mkdir(parents=True)
            for directory in (package, package.parent, package.parent.parent):
                (directory / "__init__.py").touch()
            for filename, source in self._worker_files.items():
                (package / filename).write_text(source, encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                *self._worker_command(),
                cwd=workspace,
                env={**os.environ, "PYTHONPATH": runtime.name},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=64 * 1024 * 1024,
            )
            drain = asyncio.create_task(drain_logs())
            process.stdin.write(
                (json.dumps(config, ensure_ascii=False) + "\n").encode()
            )
            await process.stdin.drain()
            return await asyncio.wait_for(consume(), timeout=self.wall_clock_timeout_s)
        except asyncio.TimeoutError:
            return outcome("timeout", "Denario exceeded wall_clock_timeout_s.")
        except Exception as exc:
            return outcome("agent_failure", concise_error_message(exc))
        finally:
            if process is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            if drain is not None:
                await drain
            runtime.cleanup()
            for request in pending.values():
                await record_event(
                    {"type": "model_interrupted", **request, "is_error": True}
                )
            if native_calls:
                # Close any native call interrupted when the worker stops,
                # without claiming its execution finished.
                await session.record_messages(
                    [
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": "Native call interrupted after accepted submission."
                            if session.submission is not None
                            else "Native call interrupted when the Denario worker stopped.",
                            "sdk_native": True,
                            "is_error": session.submission is None,
                        }
                        for call_id, name in native_calls.items()
                    ],
                    llm_calls=0,
                )
