"""Evaluate a task using its persisted workspace snapshot."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from corral.evaluation import TaskScorer
from corral.observability import (
    Observation,
    ObservationContext,
    observe_safely,
    update_safely,
)
from corral.orchestration.models import EvaluationRef

if TYPE_CHECKING:
    from corral.core.state import ExecutionState
    from corral.observability import Observer
    from corral.orchestration.models import EvaluateTaskInput
    from corral.orchestration.registry import RuntimeRegistry
    from corral.persistence import CommitStore


@contextlib.asynccontextmanager
async def _evaluation_workspace(manager: Any, state: ExecutionState):
    if manager is not None:
        async with manager.temporary_materialization(state.workspace) as path:
            yield path
        return
    if state.workspace.files:
        raise RuntimeError(
            "evaluation cannot restore workspace files without their ArtifactStore"
        )
    with tempfile.TemporaryDirectory(prefix="corral-evaluation-") as temporary:
        yield Path(temporary)


async def evaluate_task(
    request: EvaluateTaskInput,
    *,
    state_store: CommitStore,
    registry: RuntimeRegistry,
    observer: Observer,
) -> EvaluationRef:
    """Score a final commit without depending on the live task workspace."""
    execution_store = state_store.for_execution(request.execution_id)
    state = await execution_store.materialize(request.branch_id, request.commit_hash)
    observation_context = ObservationContext(
        execution_id=request.execution_id,
        task_id=request.task_id,
        benchmark_run_id=request.benchmark_run_id,
    )
    with observe_safely(
        observer,
        Observation(
            name="restore",
            context=observation_context,
            based_on_hash=state.through_commit_hash,
            input={"commit_hash": state.through_commit_hash},
        ),
    ) as restore:
        update_safely(restore)
    environment = registry.environment(request.environment_id, request.execution_id)
    manager_factory = getattr(state_store, "workspace_manager", None)
    workspace_manager = (
        manager_factory(request.execution_id)
        if manager_factory is not None
        else environment.workspace_manager
    )
    with observe_safely(
        observer,
        Observation(
            name="task.evaluate",
            context=observation_context,
            as_type="evaluator",
            based_on_hash=state.through_commit_hash,
            metadata={"recoverable": True},
        ),
    ) as evaluation:
        async with _evaluation_workspace(
            workspace_manager, state
        ) as evaluation_workspace:
            result = await asyncio.to_thread(
                TaskScorer(
                    environment.current_task,
                    workspace=evaluation_workspace,
                ).evaluate,
                state,
            )
        if request.task_id is not None:
            registry.record_evaluation(
                request.task_id,
                request.execution_id,
                result.model_dump(mode="json"),
            )
        update_safely(
            evaluation,
            output=result.model_dump(mode="json"),
        )
    return EvaluationRef(
        commit_hash=result.commit_hash,
        score=result.score,
        metrics=dict(result.metrics),
        scorer_version=result.scorer_version,
        feedback=result.feedback,
        metadata=dict(result.metadata),
    )
