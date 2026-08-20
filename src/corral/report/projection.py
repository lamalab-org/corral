"""Projection from durable Temporal results into benchmark report models."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from corral.evaluation import EvaluationResult
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults

if TYPE_CHECKING:
    from collections.abc import Iterable

    from corral.core.state import ExecutionState
    from corral.orchestration.models import (
        BenchmarkWorkflowResult,
        EvaluationRef,
        StateRef,
        TaskWorkflowResult,
    )
    from corral.persistence import CommitStore
    from corral.report.metrics import Metric


def normalise_k_values(
    k_values: int | Iterable[int] | None,
    trials_per_task: int,
) -> list[int]:
    """Validate and normalize the pass@k values used by reporting."""
    if k_values is None:
        return list(range(1, min(5, trials_per_task) + 1))
    values = (
        list(range(1, k_values + 1)) if isinstance(k_values, int) else list(k_values)
    )
    if not values:
        raise ValueError("k_values cannot be empty")
    if any(value < 1 for value in values):
        raise ValueError("k values must be at least 1")
    if max(values) > trials_per_task:
        raise ValueError(
            f"k value ({max(values)}) is greater than the number of trials "
            f"({trials_per_task})"
        )
    return sorted(set(values))


def _evaluation_result(evaluation: EvaluationRef | None) -> EvaluationResult | None:
    if evaluation is None:
        return None
    return EvaluationResult(
        commit_hash=evaluation.commit_hash,
        score=evaluation.score,
        metrics=dict(evaluation.metrics),
        feedback=evaluation.feedback,
        scorer_version=evaluation.scorer_version,
        metadata=dict(evaluation.metadata),
    )


def _state_ref_data(state_ref: StateRef | None) -> dict[str, Any]:
    return {} if state_ref is None else asdict(state_ref)


def _duration_seconds(state: ExecutionState | None) -> float | None:
    if state is None:
        return None
    started_at = state.runtime.started_at
    ended_at = state.runtime.ended_at
    if started_at is None or ended_at is None:
        return None
    return max(0.0, (ended_at - started_at).total_seconds())


def _token_usage(state: ExecutionState | None) -> dict[str, int] | None:
    if state is None:
        return None
    usage = state.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "llm_calls": usage.llm_calls,
        "tool_calls": usage.tool_calls,
        "agent_steps": usage.agent_steps,
    }


def _tool_statistics(state: ExecutionState | None) -> dict[str, Any]:
    if state is None:
        return {}

    calls: list[dict[str, Any]] = []
    for action_state in state.actions.values():
        action = action_state.action
        invocation = next(
            (
                state.tool_invocations[invocation_id]
                for invocation_id in reversed(action_state.invocation_ids)
                if invocation_id in state.tool_invocations
            ),
            None,
        )
        status = action_state.status
        content = None if invocation is None else invocation.observation
        calls.append(
            {
                "tool_name": action.name,
                "arguments": dict(action.arguments),
                "result": content,
                "status": status,
                "error": status == "failed",
                "error_message": (
                    invocation.error
                    if invocation is not None and status == "failed"
                    else None
                ),
                "duration": (
                    None
                    if invocation is None or invocation.duration_ms is None
                    else invocation.duration_ms / 1000.0
                ),
                "timestamp": None,
                "action_id": action.id,
                "actor_id": action.actor_id,
            }
        )
    return {"tool_calls": calls, "by_tool": state.tool_statistics}


def _runtime_error(
    result: TaskWorkflowResult, state: ExecutionState | None
) -> str | None:
    if result.error is not None:
        return result.error
    if state is None or state.runtime.status != "failed":
        return None
    error = state.runtime.metadata.get("error")
    return str(error) if error is not None else "task execution failed"


async def _project_trial(
    result: TaskWorkflowResult,
    state_store: CommitStore | None,
) -> TaskTrialResult:
    state: ExecutionState | None = None
    if state_store is not None and result.state is not None:
        store = state_store.for_execution(result.state.execution_id)
        state = await store.materialize(
            result.state.branch_id, result.state.commit_hash
        )

    evaluation = _evaluation_result(result.evaluation)
    unreachable = result.unreachable_dependency
    if state is not None:
        state_data = state.model_dump(mode="json")
        messages = [
            {"agent_run_id": run_id, **dict(message)}
            for run_id, conversation in state.conversations.items()
            for message in conversation
        ]
    else:
        state_data = _state_ref_data(result.state)
        messages = None

    if unreachable is not None:
        state_data.update({"unreachable": True, "missing_dependency": unreachable})

    state_ref = result.state
    output = (
        dict(state_ref.output) if state_ref is not None and state_ref.output else None
    )
    surrendered = bool(
        (state is not None and state.runtime.status == "surrendered")
        or (state_ref is not None and state_ref.status == "surrendered")
    )

    return TaskTrialResult(
        task_id=result.task_id,
        trial_id=result.execution_id,
        score=evaluation.score if evaluation is not None else 0.0,
        state=state_data,
        tool_statistics=_tool_statistics(state),
        output=output,
        evaluation=evaluation,
        evaluation_error=result.evaluation_error,
        messages=messages,
        duration=_duration_seconds(state),
        token_usage=_token_usage(state),
        error_message=_runtime_error(result, state),
        surrendered=surrendered,
    )


async def project_benchmark_result(
    result: BenchmarkWorkflowResult,
    *,
    state_store: CommitStore | None = None,
    k_values: int | Iterable[int] | None = None,
    total_duration: float | None = None,
    verbose: bool = False,
    metrics: Iterable[Metric] | None = None,
) -> BenchmarkResult:
    """Load final States and project a Workflow result into report models."""
    normalised_k = normalise_k_values(k_values, result.trials_per_task)
    task_results = {
        task_id: TaskTrialResults(task_id=task_id) for task_id in result.task_ids
    }
    task_order = {task_id: index for index, task_id in enumerate(result.task_ids)}
    ordered_trials = sorted(
        result.trials,
        key=lambda trial: (
            task_order.get(trial.task_id, len(task_order)),
            trial.trial_index,
        ),
    )
    for workflow_trial in ordered_trials:
        if workflow_trial.task_id not in task_results:
            raise ValueError(
                f"Workflow returned unknown task {workflow_trial.task_id!r}"
            )
        task_results[workflow_trial.task_id].trials.append(
            await _project_trial(workflow_trial, state_store)
        )

    return BenchmarkResult(
        task_results=task_results,
        k=normalised_k,
        total_duration=total_duration,
        verbose=verbose,
        metrics=list(metrics) if metrics is not None else None,
    )


__all__ = ["normalise_k_values", "project_benchmark_result"]
