"""Worker construction for Corral Temporal Workflows and Activities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from corral.orchestration.workflows import BenchmarkWorkflow, TaskWorkflow

if TYPE_CHECKING:
    from temporalio.client import Client

    from corral.orchestration.activities import CorralActivities


def create_worker(
    client: Client,
    *,
    task_queue: str,
    activities: CorralActivities,
    max_concurrent_activities: int | None = None,
    **worker_options: Any,
) -> Worker:
    """Build a worker polling the configured Corral task queue."""
    options = dict(worker_options)
    # OpenHands installs beartype's process-wide import hook. Re-importing
    # beartype inside Temporal's isolated module table can then observe a
    # partially initialized ``beartype.claw`` module and prevent otherwise
    # unrelated Corral workflows from loading. The hook is not workflow logic;
    # reuse the already imported package while keeping Corral workflow modules
    # sandboxed and deterministic. Callers can still provide their own runner.
    options.setdefault(
        "workflow_runner",
        SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(
                "beartype"
            )
        ),
    )
    if max_concurrent_activities is not None:
        options["max_concurrent_activities"] = max_concurrent_activities
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[TaskWorkflow, BenchmarkWorkflow],
        activities=activities.definitions,
        **options,
    )


__all__ = ["create_worker"]
