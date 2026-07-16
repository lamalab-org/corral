from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from corral.backend.tool import Tool, ToolConcurrency

if TYPE_CHECKING:
    from collections.abc import Callable

    from corral.backend.env import Environment


class _CallableTool(Tool):
    """A :class:`Tool` whose behaviour is a bound Python callable.

    Used for the generated background/control tools, whose logic is a closure
    over the trial's environment and job manager rather than a module-level
    `@tool` function.
    """

    def __init__(
        self,
        name: str,
        description: str,
        params_json_schema: dict[str, Any],
        fn: Callable[..., Any],
        concurrency: ToolConcurrency = ToolConcurrency.SERIAL,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            params_json_schema=params_json_schema,
            concurrency=concurrency,
        )
        self._fn = fn

    def execute(self, **kwargs: Any) -> str:
        result = self._fn(**kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def _job_id_schema(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """A JSON schema for a control tool that takes a `job_id` plus extras."""
    properties: dict[str, Any] = {
        "job_id": {
            "type": "string",
            "description": "Handle returned by a start_<tool> call.",
        }
    }
    required = ["job_id"]
    if extra:
        properties.update(extra)
    return {"type": "object", "properties": properties, "required": required}


def _start_description(tool: Tool) -> str:
    """Compose the description for a generated `start_<tool>` variant."""
    first_line = (tool.description or "").strip().splitlines()
    summary = first_line[0] if first_line else tool.name
    return (
        f"Start '{tool.name}' as a background job and return a job handle "
        f"immediately instead of blocking until it finishes. Poll the handle's "
        f"job_id with get_job_status / get_job_result, or block with "
        f"wait_for_job. Underlying tool: {summary}"
    )


def _make_start_tool(env: Environment, tool: Tool) -> _CallableTool:
    """Build the `start_<tool>` background variant for one tool.

    The variant advertises the *same* argument schema as the original (hidden
    arguments already stripped by the `@tool` decorator); the environment
    re-injects hidden arguments and resolves the workspace at submit time.
    """
    schema = deepcopy(tool.params_json_schema)
    schema.pop("additionalProperties", None)
    schema.pop("title", None)

    def _start(**kwargs: Any) -> dict[str, Any]:
        return env.submit_job(tool.name, kwargs)

    return _CallableTool(
        name=f"start_{tool.name}",
        description=_start_description(tool),
        params_json_schema=schema,
        fn=_start,
    )


def _make_control_tools(env: Environment) -> list[_CallableTool]:
    """Build the fixed job-control tools bound to `env`'s job manager."""

    def _guard(call: Callable[[], Any]) -> dict[str, Any]:
        """Run a manager call, turning an unknown job_id into a tool error."""
        try:
            return call()
        except KeyError as exc:
            return {"error": f"Unknown job_id {exc.args[0]!r}"}

    def _get_job_status(job_id: str) -> dict[str, Any]:
        return _guard(
            lambda: {"job_id": job_id, "status": env.job_manager.status(job_id).value}
        )

    def _get_job_result(job_id: str, wait: bool = False) -> dict[str, Any]:
        return _guard(lambda: env.job_manager.result(job_id, wait=wait))

    def _wait_for_job(job_id: str, timeout_seconds: float = 60.0) -> dict[str, Any]:
        return _guard(
            lambda: env.job_manager.result(job_id, wait=True, timeout=timeout_seconds)
        )

    def _cancel_job(job_id: str) -> dict[str, Any]:
        return _guard(lambda: env.job_manager.cancel(job_id))

    def _list_jobs() -> dict[str, Any]:
        return {"jobs": env.job_manager.list_jobs()}

    # The job-query tools only read the (independently thread-safe) JobManager,
    # so they are READ_ONLY: they must never hold the runtime's exclusive state
    # lock. This matters most for `wait_for_job`, which blocks — as SERIAL it
    # would stall every other tool call for the whole wait, defeating the point
    # of background jobs. `start_<tool>` and `cancel_job` stay SERIAL: they are
    # quick mutations that return immediately (Section 9).
    return [
        _CallableTool(
            name="get_job_status",
            description=(
                "Return the current status (queued/running/succeeded/failed/"
                "cancelled/timed_out) of a background job by its job_id."
            ),
            params_json_schema=_job_id_schema(),
            fn=_get_job_status,
            concurrency=ToolConcurrency.READ_ONLY,
        ),
        _CallableTool(
            name="get_job_result",
            description=(
                "Fetch a background job's result. With wait=false this polls "
                "and returns immediately (no result yet if still running); with "
                "wait=true it blocks until the job finishes."
            ),
            params_json_schema=_job_id_schema(
                {
                    "wait": {
                        "type": "boolean",
                        "description": "Block until the job finishes.",
                        "default": False,
                    }
                }
            ),
            fn=_get_job_result,
            concurrency=ToolConcurrency.READ_ONLY,
        ),
        _CallableTool(
            name="wait_for_job",
            description=(
                "Block until a background job finishes or timeout_seconds "
                "elapses, then return its state (still running on timeout)."
            ),
            params_json_schema=_job_id_schema(
                {
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum seconds to wait.",
                        "default": 60.0,
                    }
                }
            ),
            fn=_wait_for_job,
            concurrency=ToolConcurrency.READ_ONLY,
        ),
        _CallableTool(
            name="cancel_job",
            description=(
                "Cancel a background job (best-effort). A queued job is dropped; "
                "a running job is marked cancelled and its result discarded."
            ),
            params_json_schema=_job_id_schema(),
            fn=_cancel_job,
        ),
        _CallableTool(
            name="list_jobs",
            description="List all background jobs submitted in this trial.",
            params_json_schema={"type": "object", "properties": {}, "required": []},
            fn=_list_jobs,
            concurrency=ToolConcurrency.READ_ONLY,
        ),
    ]


def attach_background_tools(env: Environment) -> None:
    """Add `start_<tool>` + control tools to `env` for its background tools.

    A no-op unless the trial's resolved toolset contains at least one
    background-capable tool. The original (blocking) tools are left in place, so
    an agent may still call them directly; the generated variants are additive.
    Must be called after :attr:`Environment.job_manager` is set.
    """
    background = [
        tool
        for tool in list(env.tools.values())
        if getattr(tool, "background_capable", False)
    ]
    if not background:
        return

    for tool in background:
        start_tool = _make_start_tool(env, tool)
        env.tools[start_tool.name] = start_tool

    for control_tool in _make_control_tools(env):
        env.tools[control_tool.name] = control_tool
