"""An execution without a committed state is not an observed model failure."""

import pytest

from corral.orchestration.models import StateRef, TaskExecutionResult
from corral.report.projection import _project_trial


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_unstarted_execution_is_unscored_and_preserves_error():
    result = TaskExecutionResult(
        task_id="t",
        trial_index=0,
        execution_id="e",
        state=None,
        error="container failed to start",
    )
    trial = await _project_trial(result, None)
    assert trial.score is None
    assert trial.error_message == "container failed to start"


@pytest.mark.anyio
async def test_started_attempt_without_submission_still_scores_zero():
    state = StateRef(
        commit_hash="c",
        execution_id="e",
        branch_id="main",
        sequence=1,
        status="budget_exhausted",
        agent_steps=10,
    )
    trial = await _project_trial(
        TaskExecutionResult(
            task_id="t",
            trial_index=0,
            execution_id="e",
            state=state,
            error="agent exhausted its interaction budget",
        ),
        None,
    )
    assert trial.score == 0.0


@pytest.mark.anyio
async def test_evaluation_failure_remains_unscored():
    trial = await _project_trial(
        TaskExecutionResult(
            task_id="t",
            trial_index=0,
            execution_id="e",
            state=None,
            evaluation_error="catalog unavailable",
        ),
        None,
    )
    assert trial.score is None
