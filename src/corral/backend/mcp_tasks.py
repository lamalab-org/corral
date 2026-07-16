from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
from loguru import logger
from mcp.shared.exceptions import McpError
from mcp.types import (
    INVALID_PARAMS,
    TASK_STATUS_CANCELLED,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_WORKING,
    CallToolResult,
    CancelTaskRequest,
    CancelTaskResult,
    CreateTaskResult,
    ErrorData,
    GetTaskPayloadRequest,
    GetTaskPayloadResult,
    GetTaskRequest,
    GetTaskResult,
    ListTasksResult,
    RelatedTaskMetadata,
    Task,
    TextContent,
)

from corral.backend.jobs import DEFAULT_POLL_AFTER_SECONDS, JobStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.lowlevel import Server

    from corral.backend.env import Environment

    # A callable that resolves the environment a request targets. The task-scoped
    # server binds one env; the shared trial server resolves the per-request
    # runtime from the registry, so both reuse the same handlers.
    EnvResolver = Callable[[], Environment | None]

# Advisory poll interval (ms) handed back in every task handle, mirroring the
# fallback tools' `poll_after_seconds`. Advisory only — the client may poll
# whenever it likes.
POLL_INTERVAL_MS = DEFAULT_POLL_AFTER_SECONDS * 1000

# Per-spec `_meta` key that ties a `tasks/result` payload back to its task.
RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"

# JobStatus → MCP task status. The Tasks extension has no "timed_out"; a
# deadline-elapsed job is reported as failed (its error message explains why).
_JOB_TO_TASK_STATUS: dict[str, str] = {
    JobStatus.QUEUED.value: TASK_STATUS_WORKING,
    JobStatus.RUNNING.value: TASK_STATUS_WORKING,
    JobStatus.SUCCEEDED.value: TASK_STATUS_COMPLETED,
    JobStatus.FAILED.value: TASK_STATUS_FAILED,
    JobStatus.CANCELLED.value: TASK_STATUS_CANCELLED,
    JobStatus.TIMED_OUT.value: TASK_STATUS_FAILED,
}


def task_status_for(job_status: str) -> str:
    """Map a JobStatus value to its MCP task status (defaulting to working)."""
    return _JOB_TO_TASK_STATUS.get(job_status, TASK_STATUS_WORKING)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _job_timestamps(job: dict[str, Any]) -> tuple[datetime, datetime]:
    """Derive (createdAt, lastUpdatedAt) from a JobManager record dict."""
    created = _parse_dt(job.get("submitted_at")) or datetime.now(timezone.utc)
    updated = (
        _parse_dt(job.get("ended_at")) or _parse_dt(job.get("started_at")) or created
    )
    return created, updated


def _status_message(job: dict[str, Any]) -> str | None:
    """A short human-readable status line — the error on a failed job."""
    error = job.get("error")
    return str(error)[:500] if error else None


def _task_from_job(job: dict[str, Any]) -> Task:
    """Build a :class:`Task` snapshot from a JobManager record dict."""
    created, updated = _job_timestamps(job)
    return Task(
        taskId=job["job_id"],
        status=task_status_for(job["status"]),
        statusMessage=_status_message(job),
        createdAt=created,
        lastUpdatedAt=updated,
        ttl=None,
        pollInterval=POLL_INTERVAL_MS,
    )


def _payload_from_job(job: dict[str, Any]) -> GetTaskPayloadResult:
    """Render a terminal job into a `tasks/result` payload.

    The payload is exactly the :class:`CallToolResult` the tool call would have
    returned synchronously — success text for a succeeded job, an `isError`
    message otherwise — carried through `GetTaskPayloadResult` (a `Result`
    with `extra="allow"`) with the spec-mandated related-task `_meta`.
    """
    status = job.get("status")
    if status == JobStatus.SUCCEEDED.value:
        result = job.get("result")
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        call_result = CallToolResult(content=[TextContent(type="text", text=text)])
    else:
        message = job.get("error") or f"Job ended in status {status!r}."
        call_result = CallToolResult(
            content=[TextContent(type="text", text=str(message))], isError=True
        )

    data = call_result.model_dump(by_alias=True)
    related = {
        RELATED_TASK_METADATA_KEY: RelatedTaskMetadata(taskId=job["job_id"]).model_dump(
            by_alias=True
        )
    }
    data["_meta"] = {**(data.get("_meta") or {}), **related}
    return GetTaskPayloadResult.model_validate(data)


def _not_found(task_id: str) -> McpError:
    return McpError(
        ErrorData(code=INVALID_PARAMS, message=f"Task not found: {task_id}")
    )


async def maybe_start_task(
    env: Environment | None,
    name: str,
    arguments: dict[str, Any] | None,
    experimental: Any | None,
) -> CreateTaskResult | CallToolResult | None:
    """Handle a possibly task-augmented `tools/call`.

    Returns `None` when the request is not task-augmented, or targets a tool
    that is not background-capable — the caller then runs the tool synchronously
    exactly as before. When it *is* a task-augmented call of a background tool,
    the tool is submitted as a background job (bound to this runtime's provenance
    at submit time) and a :class:`CreateTaskResult` handle is returned
    immediately. A submission error comes back as an `isError` result so the
    model sees it right away rather than as a stuck task.
    """
    if (
        env is None
        or experimental is None
        or not getattr(experimental, "is_task", False)
    ):
        return None

    tool = env.tools.get(name)
    if tool is None or not getattr(tool, "background_capable", False):
        # Task-augmenting a non-background tool: ignore the augmentation and let
        # the caller run it synchronously (its `execution.taskSupport` never
        # advertised task support, so a well-behaved client won't reach here).
        return None

    handle = await anyio.to_thread.run_sync(
        partial(env.submit_job, name, arguments or {})
    )
    if "error" in handle:
        return CallToolResult(
            content=[TextContent(type="text", text=str(handle["error"]))], isError=True
        )

    metadata = getattr(experimental, "task_metadata", None)
    ttl = getattr(metadata, "ttl", None)
    now = datetime.now(timezone.utc)
    task = Task(
        taskId=handle["job_id"],
        status=TASK_STATUS_WORKING,
        createdAt=now,
        lastUpdatedAt=now,
        ttl=ttl,
        pollInterval=POLL_INTERVAL_MS,
    )
    logger.debug(f"Started MCP task {handle['job_id']} for background tool {name!r}")
    return CreateTaskResult(task=task)


def enable_job_task_surface(server: Server, env_resolver: EnvResolver) -> None:
    """Register `tasks/get` / `tasks/result` / `tasks/list` / `tasks/cancel`.

    Each handler resolves the target runtime via `env_resolver` (per-request
    for the shared trial server, fixed for a task server) and dispatches to that
    runtime's :class:`~corral.backend.jobs.JobManager`, mapping job state onto
    task state. Registering these handlers is what makes the low-level server
    advertise the `tasks` capability during initialize.
    """

    def _manager():
        env = env_resolver()
        manager = env.job_manager if env is not None else None
        if manager is None:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="No background-task surface is active for this runtime.",
                )
            )
        return manager

    def _job_snapshot(manager, task_id: str) -> dict[str, Any]:
        try:
            return manager.result(task_id, wait=False)
        except KeyError:
            raise _not_found(task_id) from None

    def _wait_snapshot(manager, task_id: str) -> dict[str, Any]:
        # Runs in a worker thread: block on the job's future until it reaches a
        # terminal state (the executor's own deadline/cancel govern completion),
        # then return the final record for rendering.
        try:
            return manager.result(task_id, wait=True, timeout=None)
        except KeyError:
            raise _not_found(task_id) from None

    @server.experimental.get_task()
    async def _get_task(req: GetTaskRequest) -> GetTaskResult:
        job = _job_snapshot(_manager(), req.params.taskId)
        created, updated = _job_timestamps(job)
        return GetTaskResult(
            taskId=job["job_id"],
            status=task_status_for(job["status"]),
            statusMessage=_status_message(job),
            createdAt=created,
            lastUpdatedAt=updated,
            ttl=None,
            pollInterval=POLL_INTERVAL_MS,
        )

    @server.experimental.get_task_result()
    async def _get_task_result(req: GetTaskPayloadRequest) -> GetTaskPayloadResult:
        manager = _manager()
        task_id = req.params.taskId
        _job_snapshot(manager, task_id)  # surface an unknown id before blocking
        job = await anyio.to_thread.run_sync(partial(_wait_snapshot, manager, task_id))
        return _payload_from_job(job)

    @server.experimental.list_tasks()
    async def _list_tasks() -> ListTasksResult:
        # No pagination: one runtime's job set is small, so every job is a task.
        env = env_resolver()
        manager = env.job_manager if env is not None else None
        if manager is None:
            return ListTasksResult(tasks=[])
        tasks = [_task_from_job(job) for job in manager.list_jobs()]
        return ListTasksResult(tasks=tasks, nextCursor=None)

    @server.experimental.cancel_task()
    async def _cancel_task(req: CancelTaskRequest) -> CancelTaskResult:
        manager = _manager()
        task_id = req.params.taskId
        try:
            status = manager.status(task_id)
        except KeyError:
            raise _not_found(task_id) from None
        # Per spec, cancelling a terminal task is an INVALID_PARAMS error.
        if status.is_terminal:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Cannot cancel task in terminal state "
                    f"{task_status_for(status.value)!r}",
                )
            )
        job = manager.cancel(task_id)
        created, updated = _job_timestamps(job)
        return CancelTaskResult(
            taskId=job["job_id"],
            status=task_status_for(job["status"]),
            statusMessage=_status_message(job),
            createdAt=created,
            lastUpdatedAt=updated,
            ttl=None,
            pollInterval=POLL_INTERVAL_MS,
        )
