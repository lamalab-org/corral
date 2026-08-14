"""The `TrialRuntime` interface behind every `/trials/{id}/*` endpoint.

Environment Concurrency Isolation, Phase 1. The trial REST/MCP handlers used to
call a bare :class:`~corral.backend.env.Environment` directly. This module lifts
the exact method surface those handlers touch into a small :class:`TrialRuntime`
protocol and provides :class:`InProcessTrialRuntime`, the default backend that
simply wraps a live `Environment` (today's behaviour, byte-for-byte).

Naming the interface is the seam a later process-worker backend (Phase 2) plugs
into: it will implement the same protocol by RPC to a worker process, so the
HTTP layer never learns which backend is behind a given `trial_runtime_id`.

The payload builders (`build_openai_tools_payload`, `build_mcp_tools_payload`,
`build_status_payload`) and `finalize_trial_runtime` live here because the
in-process runtime is defined in terms of them; the task-scoped REST routes in
`server.py` import the tool-payload builders too, so both surfaces emit
byte-identical schemas.
"""

import hashlib
import json
from copy import deepcopy
from typing import Protocol

from corral.backend.env import Environment
from corral.backend.schema import ToolCall, TrialCompletionResponse
from corral.router.verbosity import (
    ToolVerbosity,
    VerbosityConfig,
    get_tools_guide_with_verbosity,
)


def build_openai_tools_payload(env: Environment, verbosity: ToolVerbosity) -> dict:
    """Build the OpenAI function-calling tool list for an environment.

    Shared by the task (`/tasks/{id}/tools`) and trial
    (`/trials/{rid}/tools`) endpoints so both expose byte-identical schemas.
    """
    tools_info = []
    for tool_obj in env.tools.values():
        filtered_description = VerbosityConfig.filter_tool_description(
            tool_obj.description, verbosity
        )

        schema = deepcopy(tool_obj.params_json_schema)
        schema.pop("additionalProperties", None)
        schema.pop("title", None)

        # Filter argument descriptions based on verbosity
        if verbosity == ToolVerbosity.MINIMAL:
            for prop in schema.get("properties", {}).values():
                prop.pop("description", None)
        else:
            for prop in schema.get("properties", {}).values():
                if "description" in prop:
                    prop["description"] = VerbosityConfig.filter_argument_description(
                        prop["description"], verbosity
                    )

        tools_info.append(
            {
                "type": "function",
                "function": {
                    "name": tool_obj.name,
                    "description": filtered_description,
                    "parameters": schema,
                },
            }
        )

    return {"tools": tools_info}


def build_mcp_tools_payload(env: Environment, verbosity: ToolVerbosity) -> dict:
    """Build the MCP tool list plus its schema digest for an environment."""
    tools = [tool_obj.to_mcp(verbosity) for tool_obj in env.tools.values()]
    canonical = json.dumps(tools, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"tools": tools, "mcp_schema_sha256": digest}


def build_status_payload(env: Environment) -> dict:
    """Build the task/trial status payload for an environment."""
    return {
        "is_attempted": env.state.is_attempted,
        "score": env.state.score,
        "submitted_answer": env.state.submitted_answer,
        "tool_statistics": env.state.tool_statistics(),
    }


def finalize_trial_runtime(
    env: Environment, score: float, surrendered: bool = False
) -> TrialCompletionResponse:
    """Finalize a single-use *trial runtime* without starting a next trial.

    Unlike `server._finalize_trial` (which resets the shared task template for
    its next sequential trial), a trial runtime is discarded after one trial, so
    we capture its completed state and finalize in place — no fresh workspace is
    created. The runtime stays in the registry until an explicit `DELETE` so
    its final status/trials remain queryable.
    """
    completed = env.get_completed_trial_data()
    env.state.finalize()
    return TrialCompletionResponse(
        score=score,
        state=completed["state"],
        trial_id=completed["trial_id"],
        surrendered=surrendered,
    )


class TrialRuntime(Protocol):
    """The fixed method surface every `/trials/{id}/*` endpoint drives.

    A trial runtime owns one isolated execution of a task — its own state,
    workspace, tools, and (optionally) background jobs. The handlers only ever
    call the members below, so any backend that implements them can stand behind
    a `trial_runtime_id` without the HTTP layer changing: today's in-process
    :class:`InProcessTrialRuntime`, and a future process-worker proxy (Phase 2).
    """

    @property
    def task_id(self) -> str:
        """Id of the task this runtime is executing."""
        ...

    @property
    def workspace(self) -> str | None:
        """Filesystem workspace of the active trial (may be `None`)."""
        ...

    def get_task_prompt(self) -> str | list[dict]:
        """Render the task prompt for the agent."""
        ...

    def guide(self, verbosity: ToolVerbosity) -> str:
        """Render the task prompt plus the tools guide at `verbosity`."""
        ...

    def tools_payload(self, verbosity: ToolVerbosity) -> dict:
        """The OpenAI function-calling tool list at `verbosity`."""
        ...

    def mcp_tools_payload(self, verbosity: ToolVerbosity) -> dict:
        """The MCP tool list plus its schema digest at `verbosity`."""
        ...

    def call_tool(self, name: str, arguments: dict) -> ToolCall:
        """Execute a tool and return its recorded :class:`ToolCall`."""
        ...

    def snapshot(self) -> dict:
        """Live state snapshot with in-flight background jobs folded in."""
        ...

    def status(self) -> dict:
        """Completion status (attempted / score / answer / tool stats)."""
        ...

    def configure(self) -> dict:
        """Configure external apps for the trial; return status + trial id."""
        ...

    def submit(self, answer: str) -> TrialCompletionResponse:
        """Submit an answer, score it, and finalize the runtime."""
        ...

    def surrender(self) -> TrialCompletionResponse:
        """Surrender the trial without an answer and finalize the runtime."""
        ...

    def close(self) -> None:
        """Release the runtime's resources (background-job threads, ...)."""
        ...


class InProcessTrialRuntime:
    """The default `TrialRuntime`: a thin wrapper over a live `Environment`.

    This is today's behaviour unchanged — every method delegates straight to the
    wrapped environment (or the module helpers defined above), which run in the
    server process on its thread pool. Envs whose tools are pure/reentrant are
    correct here; ones that keep process-global mutable state need the Phase 2
    process-worker backend instead.

    The wrapped `env` is exposed so the shared `/trials` MCP mount — whose
    background-job task surface is defined in terms of a live `Environment` —
    can reach it (see `mcp_server._current_trial_env`).
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

    @property
    def task_id(self) -> str:
        return self.env.task_id

    @property
    def workspace(self) -> str | None:
        return self.env.current_work_dir

    def get_task_prompt(self) -> str | list[dict]:
        return self.env.get_task_prompt()

    def guide(self, verbosity: ToolVerbosity) -> str:
        tools_guide = get_tools_guide_with_verbosity(self.env, verbosity)
        return f"Task: {self.env.get_task_prompt()}\n\n{tools_guide}"

    def tools_payload(self, verbosity: ToolVerbosity) -> dict:
        return build_openai_tools_payload(self.env, verbosity)

    def mcp_tools_payload(self, verbosity: ToolVerbosity) -> dict:
        return build_mcp_tools_payload(self.env, verbosity)

    def call_tool(self, name: str, arguments: dict) -> ToolCall:
        return self.env.call_tool(name, arguments)

    def snapshot(self) -> dict:
        # Reflect any in-flight background jobs in the live state snapshot.
        self.env.refresh_jobs()
        return self.env.state.snapshot()

    def status(self) -> dict:
        return build_status_payload(self.env)

    def configure(self) -> dict:
        status = self.env.configure_additional_apps()
        return {
            "status": status,
            "task_id": self.env.task_id,
            "trial_id": self.env.state.trial_id,
        }

    def submit(self, answer: str) -> TrialCompletionResponse:
        score = self.env.submit_answer(answer)
        return finalize_trial_runtime(self.env, score, surrendered=False)

    def surrender(self) -> TrialCompletionResponse:
        score = self.env.surrender()
        return finalize_trial_runtime(self.env, score, surrendered=True)

    def close(self) -> None:
        # Release any background-job threads so nothing outlives the trial.
        self.env.shutdown_jobs()
