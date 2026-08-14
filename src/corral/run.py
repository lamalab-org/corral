import json
import math
import pickle
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import anyio
from loguru import logger

from corral.agents import SURRENDER_SENTINEL, AgentRunResult, BaseAgent
from corral.agents.hooks import AgentHooks
from corral.agents.utils import LiteLLMMessage
from corral.backend.task import assert_dependencies_selected, order_selected
from corral.concurrency import (
    AgentFactory,
    ConcurrencyConfig,
    TrialCompleted,
    TrialContext,
    TrialWorkItem,
)
from corral.episode import run_episode_dag
from corral.report import (
    BenchmarkResult,
    CorralWandbLogger,
    TaskTrialResult,
    TaskTrialResults,
)
from corral.report.metrics import Metric, get_default_metrics
from corral.report.metrics.base import TaskMetric
from corral.report.metrics.registry import MetricRegistry
from corral.router import CorralRouter, acall, as_sync_interface
from corral.types import BudgetExhaustedError


def _reraise_budget_exhausted(exc: BaseException) -> None:
    """Re-raise a :class:`BudgetExhaustedError` hidden inside an exception group.

    `anyio` task groups surface child failures as an `ExceptionGroup` (even
    for a single failure). A `BudgetExhaustedError` must still stop the whole
    benchmark, so unwrap the group and re-raise it bare when present; otherwise
    return and let the caller re-raise the original group unchanged.
    """
    if isinstance(exc, BudgetExhaustedError):
        raise exc
    for sub in getattr(exc, "exceptions", ()) or ():
        _reraise_budget_exhausted(sub)


# Headroom kept above the trial concurrency limit for anyio's default worker
# pool. On the native-async path each in-flight trial may hold one pool thread
# at a time — a non-native agent's `run()` offload, or the short surrender/
# submit HTTP offload — so the pool must have at least `global_trials` tokens or
# those offloads could starve. This keeps a comfortable margin above that (and
# never lowers the pool below anyio's own default of 40).
_THREAD_POOL_HEADROOM = 40


def _ensure_thread_capacity(global_trials: int) -> None:
    """Raise anyio's default thread-pool size to fit the trial concurrency.

    Must be called from within the event loop. Only ever *raises* the token
    count, so it cannot shrink a pool another component sized up.
    """
    limiter = anyio.to_thread.current_default_thread_limiter()
    needed = global_trials + _THREAD_POOL_HEADROOM
    if limiter.total_tokens < needed:
        limiter.total_tokens = needed


def _build_per_model_gates(
    per_model: Mapping[str, int] | None,
) -> dict[str, anyio.Semaphore] | None:
    """Build one :class:`anyio.Semaphore` per model in `config.per_model`.

    Returns `None` when no per-model caps are configured (the common case), so
    callers skip the extra gating entirely. A model *absent* from the map gets no
    semaphore and is therefore bounded only by the global limiter — the plan's
    "no extra cap" default. Must be called from within the event loop.
    """
    if not per_model:
        return None
    return {model: anyio.Semaphore(limit) for model, limit in per_model.items()}


def _model_gate(per_model_gate: dict[str, anyio.Semaphore] | None, model: str | None):
    """Context manager gating one trial on its model's simultaneous-trial cap.

    Returns the model's semaphore when `config.per_model` capped that model,
    otherwise a no-op context (a model absent from the map, or no per-model caps
    at all, is bounded only by the global limiter). The result is always an
    async context manager so schedulers can `async with` it uniformly and
    acquire it *before* the global limiter — so a trial queued on its model cap
    holds no global concurrency slot, mirroring the per-task gate.
    """
    if per_model_gate is not None and model is not None:
        gate = per_model_gate.get(model)
        if gate is not None:
            return gate
    return nullcontext()


async def _fetch_concurrency_modes(interface: Any) -> dict[str, str]:
    """Fetch the server's `{task_id: declared_concurrency_mode}` map.

    Interfaces without the method (test fakes) or servers predating the
    `/concurrency` endpoint yield an empty map, so callers treat every task as
    the safe-to-overlap default `"thread"` and the historical behaviour is
    unchanged.
    """
    getter = getattr(interface, "get_concurrency_modes", None)
    if getter is None:
        return {}
    try:
        return await acall(getter)
    except Exception as exc:
        logger.warning(
            f"Could not fetch env concurrency modes ({exc!r}); treating every "
            "env as thread-safe. Envs that keep process-global state may be "
            "corrupted by concurrent trials."
        )
        return {}


async def _worker_backend_active(interface: Any) -> bool:
    """Whether the server has a live process-worker pool (Phase 3).

    When active, independent `"process"` trials are each isolated in their own
    worker process, so the scheduler may let them overlap rather than serialise
    them. Interfaces without the method (test fakes) or servers predating the
    `/trial_worker` endpoint report `False`, so `"process"` envs stay serialised
    exactly as in Phase 0.
    """
    getter = getattr(interface, "get_trial_worker_active", None)
    if getter is None:
        return False
    try:
        return bool(await acall(getter))
    except Exception as exc:
        logger.warning(
            f"Could not fetch trial-worker status ({exc!r}); assuming no worker "
            "pool, so process-stateful envs will be serialised."
        )
        return False


async def _serialised_task_ids(interface: Any, *, process_isolated: bool) -> set[str]:
    """Task ids whose env keeps process-global state and must run serialised.

    Reads the server's `/concurrency` map and returns the tasks the concurrent
    scheduler must gate so they never overlap another globally-stateful trial
    (Environment Concurrency Isolation).

    * `"serial"` envs are always serialised — the explicit escape hatch for envs
      that must never overlap anything.
    * `"process"` envs are serialised **unless** `process_isolated` is set. When
      the caller routes each `"process"` trial into its own worker process (the
      independent path with a live worker pool, Phase 3), the process boundary —
      not the serial gate — provides isolation, so they may overlap. Without that
      (no worker pool, or the chained path where an episode's nodes share one
      pinned worker), they stay serialised (Phase-0 behaviour).
    """
    modes = await _fetch_concurrency_modes(interface)
    serialised: set[str] = set()
    for task_id, mode in modes.items():
        if mode == "serial" or mode == "process" and not process_isolated:
            serialised.add(task_id)
    return serialised


def _serial_lease(serial_gate: anyio.Semaphore, is_serialised: bool):
    """Return the benchmark-wide serial gate for a globally-stateful trial.

    `serial_gate` is one `anyio.Semaphore(1)` shared by every worker in a run:
    acquiring it lets at most one `"process"`/`"serial"` trial execute at a time,
    so those envs never overlap another globally-stateful trial. A `"thread"`
    trial (`is_serialised` false) gets a no-op context and is bounded only by the
    global/per-task limiters, exactly as before. Acquired *before* the global
    limiter so a trial queued on the serial gate holds no global slot.
    """
    return serial_gate if is_serialised else nullcontext()


def create_session_id() -> str:
    """Create a unique session ID"""
    return f"session_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def sanitize_model_name(model_name: str, max_length: int = 50) -> str:
    """Sanitize model name for use in filenames.

    Replaces invalid filesystem characters with underscores and limits length.

    Args:
        model_name: The raw model name to sanitize
        max_length: Maximum length for the sanitized name (default: 50)

    Returns:
        Sanitized model name safe for use in filenames
    """
    # Replace invalid filesystem characters with underscores
    invalid_chars = r"[/\\:*?\"<>|\s\-]+"
    sanitized = re.sub(invalid_chars, "_", model_name)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")

    # Ensure we have a valid name
    if not sanitized:
        sanitized = "unknown_model"

    return sanitized


def validate_k_values(
    k_values: int | list[int] | None, trials_per_task: int
) -> list[int]:
    """Validate and normalize k_values."""
    if k_values is None:
        # Default to range 1 to min(5, trials_per_task)
        max_k = min(5, trials_per_task)
        return list(range(1, max_k + 1))
    elif isinstance(k_values, int):
        # Single int means max k, generate range 1 to k
        if k_values > trials_per_task:
            raise ValueError(
                f"k value ({k_values}) is greater than the number of trials ({trials_per_task})"
            )
        return list(range(1, k_values + 1))
    else:
        # List of k values - validate all are within bounds
        max_k = max(k_values)
        if max_k > trials_per_task:
            raise ValueError(
                f"k value ({max_k}) is greater than the number of trials ({trials_per_task})"
            )
        return sorted(k_values)


def initialize_task_results(task_ids: list[str]) -> dict[str, TaskTrialResults]:
    """Initialize empty task results"""
    return {task_id: TaskTrialResults(task_id=task_id) for task_id in task_ids}


def filter_incomplete_tasks(
    task_results: dict[str, TaskTrialResults], trials_per_task: int
) -> list[str]:
    """Get list of tasks that still need trials"""
    return [
        task_id
        for task_id, results in task_results.items()
        if len(results.trials) < trials_per_task
    ]


def get_score_from_state(interface: CorralRouter, task_id: str) -> float:
    """Retrieve current score from task state, defaulting to 0.0 if unavailable"""
    try:
        status = interface.get_task_status(task_id)
        return status.get("score", 0.0) or 0.0  # Handle None case
    except Exception as e:
        logger.warning(f"Failed to retrieve score from state: {e}")
        return 0.0


def exception_trial_result(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    error: Exception,
    error_type: str,
    token_usage: dict[str, Any],
    surrendered: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> TaskTrialResult:
    """Create a TaskTrialResult when there is an exception during trial execution. Benchmark continues.

    Args:
        task_id: The task identifier
        trial_index: The trial index
        interface: Router interface to retrieve task state
        error: The exception that occurred
        error_type: Type of error (e.g., "Surrender Error", "Submission Error", "Agent Error")
        token_usage: Token usage statistics
        surrendered: Whether this was a surrender operation
        messages: Optional agent messages

    Returns:
        TaskTrialResult with score retrieved from state or 0.0 if unavailable
    """
    score = get_score_from_state(interface, task_id)
    return _build_exception_result(
        task_id,
        trial_index,
        score,
        error,
        error_type,
        token_usage,
        surrendered,
        messages,
    )


def _build_exception_result(
    task_id: str,
    trial_index: int,
    score: float,
    error: Exception,
    error_type: str,
    token_usage: dict[str, Any],
    surrendered: bool,
    messages: list[dict[str, Any]] | None,
) -> TaskTrialResult:
    """Assemble the exception :class:`TaskTrialResult` from an already-fetched score.

    Pure (no I/O): the score is resolved by the caller so the same builder backs
    both the synchronous :func:`exception_trial_result` and the asynchronous
    :func:`aexception_trial_result`, which differ only in how they fetch the
    fallback score.
    """
    return TaskTrialResult(
        task_id=task_id,
        trial_id=f"attempt_{trial_index + 1}",
        score=score,
        state={"error": str(error), "attempt": trial_index + 1},
        tool_statistics={"error": str(error)},
        messages=messages,
        duration=None,
        token_usage=token_usage,
        error_message=f"{error_type}: {error}",
        surrendered=surrendered,
    )


async def aget_score_from_state(interface: CorralRouter, task_id: str) -> float:
    """Async twin of :func:`get_score_from_state`.

    Fetches the status through :func:`acall`, so an
    :class:`~corral.router.AsyncCorralRouter` is queried directly on the loop
    while a synchronous router is offloaded to a worker thread.
    """
    try:
        status = await acall(interface.get_task_status, task_id)
        return status.get("score", 0.0) or 0.0
    except Exception as e:
        logger.warning(f"Failed to retrieve score from state: {e}")
        return 0.0


async def aexception_trial_result(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    error: Exception,
    error_type: str,
    token_usage: dict[str, Any],
    surrendered: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> TaskTrialResult:
    """Async twin of :func:`exception_trial_result` (async fallback-score lookup)."""
    score = await aget_score_from_state(interface, task_id)
    return _build_exception_result(
        task_id,
        trial_index,
        score,
        error,
        error_type,
        token_usage,
        surrendered,
        messages,
    )


def unreachable_trial_result(
    task_id: str, trial_index: int, missing_dependency: str
) -> TaskTrialResult:
    """Record a task whose upstream chain broke, without invoking the agent.

    A broken chain is a normal workflow failure (most often the upstream task
    exhausted its iteration budget without a valid answer), not a harness error.
    So the task is scored 0 (counts as a pass@k failure: the workflow never
    reached this step) but records **0 iterations / 0 tokens and no
    `error_message`** — it must not distort the iteration or error-rate
    metrics. The explicit `unreachable` / `missing_dependency` markers let
    analysis tell "chain broke upstream" from "agent tried and failed".
    """
    return TaskTrialResult(
        task_id=task_id,
        trial_id=f"attempt_{trial_index + 1}",
        score=0.0,
        state={"unreachable": True, "missing_dependency": missing_dependency},
        tool_statistics={},
        messages=None,
        duration=0.0,
        token_usage={},
        error_message=None,
        surrendered=False,
    )


def _unsatisfied_dependency(
    task_id: str,
    graph: Mapping[str, list[str]],
    task_results: dict[str, TaskTrialResults],
    trial_round: int,
) -> str | None:
    """Return the first dependency that did not produce a usable output this round.

    A dependency is "usable" when its trial for this round succeeded (a real,
    scored answer). A failed, surrendered, errored, or itself-unreachable
    dependency means the chain is broken and the dependent is unreachable.
    """
    for dep_id in graph.get(task_id, []):
        dep_trials = task_results[dep_id].trials
        if trial_round >= len(dep_trials) or not dep_trials[trial_round].success:
            return dep_id
    return None


def _elapsed_seconds(trial_start_time: datetime) -> float:
    """Wall-clock seconds since `trial_start_time` (UTC)."""
    return (datetime.now(tz=timezone.utc) - trial_start_time).total_seconds()


def _apply_agent_tool_statistics(
    result: TaskTrialResult, run_result: AgentRunResult
) -> TaskTrialResult:
    """Use agent-owned tool records when calls ran outside the scored runtime.

    Most agents execute tools directly in the canonical trial, whose submitted
    state already supplies ``tool_statistics``. AI Scientist instead uses
    isolated branch runtimes, so it returns their aggregate in run metadata.
    Applying that explicit aggregate here keeps reporting agent-agnostic while
    leaving the normal canonical-runtime path unchanged.
    """
    statistics = run_result.metadata.get("tool_statistics")
    if not isinstance(statistics, Mapping):
        return result
    result.tool_statistics = dict(statistics)
    if isinstance(result.state, dict):
        result.state["tool_statistics"] = dict(statistics)
    return result


def _finish_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    run_result: AgentRunResult,
    trial_start_time: datetime,
) -> TaskTrialResult:
    """Turn an :class:`AgentRunResult` into a scored :class:`TaskTrialResult`.

    Shared by the synchronous :func:`execute_single_trial` and the asynchronous
    :func:`aexecute_single_trial` so the surrender / infra-failure / submit
    state machine has exactly one implementation. Every branch performs blocking
    HTTP against the environment server (surrender/submit), so the async path
    runs this in a worker thread.
    """
    answer = run_result.answer
    messages = run_result.messages
    token_usage = run_result.token_usage

    if answer == SURRENDER_SENTINEL:
        try:
            result = interface.surrender_task(task_id)
            result.token_usage = token_usage
            result.messages = messages
            result.duration = _elapsed_seconds(trial_start_time)
            return _apply_agent_tool_statistics(result, run_result)
        except Exception as surrender_error:
            result = exception_trial_result(
                task_id=task_id,
                trial_index=trial_index,
                interface=interface,
                error=surrender_error,
                error_type="Surrender Error",
                token_usage=token_usage,
                surrendered=True,
                messages=messages,
            )
            result.duration = _elapsed_seconds(trial_start_time)
            return _apply_agent_tool_statistics(result, run_result)

    # An infrastructure failure (harness timeout, SDK crash, MCP transport
    # failure, iteration/budget exhaustion, ...) surfaces as a non-submit
    # status with an error string for `answer`. Record it as a trial error
    # rather than submitting the error string to the task scorer as if it
    # were a model answer. `"success"` and `"surrender"` are the only
    # submit-worthy statuses (surrender is already handled above).
    if run_result.status not in {"success", "surrender"}:
        result = exception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=RuntimeError(run_result.error_message or answer),
            error_type=f"Agent {run_result.status}",
            token_usage=token_usage,
            messages=messages,
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)

    try:
        result = interface.submit_answer(task_id, answer)
        result.token_usage = token_usage
        result.messages = messages
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)
    except Exception as submit_error:
        result = exception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=submit_error,
            error_type="Submission Error",
            token_usage=token_usage,
            messages=messages,
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)


async def _afinish_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    run_result: AgentRunResult,
    trial_start_time: datetime,
) -> TaskTrialResult:
    """Async twin of :func:`_finish_trial`.

    Drives the same surrender / infra-failure / submit state machine, but every
    terminal HTTP call (surrender, submit, and the error-path score lookup) goes
    through :func:`acall`, so an :class:`~corral.router.AsyncCorralRouter` is
    awaited directly on the loop and a synchronous router is offloaded per call.
    The synchronous :func:`_finish_trial` is left byte-for-byte for the serial
    path.
    """
    answer = run_result.answer
    messages = run_result.messages
    token_usage = run_result.token_usage

    if answer == SURRENDER_SENTINEL:
        try:
            result = await acall(interface.surrender_task, task_id)
            result.token_usage = token_usage
            result.messages = messages
            result.duration = _elapsed_seconds(trial_start_time)
            return _apply_agent_tool_statistics(result, run_result)
        except Exception as surrender_error:
            result = await aexception_trial_result(
                task_id=task_id,
                trial_index=trial_index,
                interface=interface,
                error=surrender_error,
                error_type="Surrender Error",
                token_usage=token_usage,
                surrendered=True,
                messages=messages,
            )
            result.duration = _elapsed_seconds(trial_start_time)
            return _apply_agent_tool_statistics(result, run_result)

    # A non-submit status is an infrastructure failure, recorded as a trial error
    # rather than submitted to the scorer (mirrors :func:`_finish_trial`).
    if run_result.status not in {"success", "surrender"}:
        result = await aexception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=RuntimeError(run_result.error_message or answer),
            error_type=f"Agent {run_result.status}",
            token_usage=token_usage,
            messages=messages,
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)

    try:
        result = await acall(interface.submit_answer, task_id, answer)
        result.token_usage = token_usage
        result.messages = messages
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)
    except Exception as submit_error:
        result = await aexception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=submit_error,
            error_type="Submission Error",
            token_usage=token_usage,
            messages=messages,
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return _apply_agent_tool_statistics(result, run_result)


def execute_single_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    agent: BaseAgent,
    verbose: bool = False,
    tool_verbosity: str | None = None,
    configure_timeout: float | None = None,
    enable_surrender: bool = False,
) -> TaskTrialResult:
    """Execute a single trial - pure function."""
    trial_start_time = datetime.now(tz=timezone.utc)

    try:
        status = interface.configure_additional_apps(task_id, timeout=configure_timeout)
        logger.info(f"Task {task_id} additional apps/services configured: {status}")

        # Agent always returns an AgentRunResult dataclass
        run_result = agent.run_agent(
            interface,
            task_id,
            verbose=verbose,
            tool_verbosity=tool_verbosity or "brief",
            enable_surrender=enable_surrender,
        )
        return _finish_trial(
            task_id, trial_index, interface, run_result, trial_start_time
        )
    except BudgetExhaustedError:
        # Re-raise to stop the benchmark immediately
        # When an error with the llm running out of credits occurs
        raise
    except Exception as agent_error:
        result = exception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=agent_error,
            error_type="Agent Error",
            token_usage=agent.get_total_token_usage(),
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return result


async def aexecute_single_trial(
    task_id: str,
    trial_index: int,
    interface: CorralRouter,
    agent: BaseAgent,
    verbose: bool = False,
    tool_verbosity: str | None = None,
    configure_timeout: float | None = None,
    enable_surrender: bool = False,
    configure_limiter: anyio.Semaphore | None = None,
) -> TaskTrialResult:
    """Async counterpart of :func:`execute_single_trial`.

    Drives the agent through :meth:`BaseAgent.arun_agent` so a natively-async
    agent (Claude Code) runs its harness coroutine directly on the scheduler
    loop instead of nesting an event loop inside a worker thread. The surrounding
    interface HTTP — configuring apps and the surrender/submit finish sequence —
    goes through :func:`acall` / :func:`_afinish_trial`, so an
    :class:`~corral.router.AsyncCorralRouter` is awaited directly on the loop and
    a synchronous router is offloaded to a worker thread. Either way the result
    is identical to the sync path.

    `configure_limiter` (from `ConcurrencyConfig.configure_apps`) is a
    benchmark-wide lease bounding how many trials run the app-configure step at
    once — the plan's fixed-port / expensive-setup lever. It is held **only**
    around `configure_additional_apps` (a short setup step), not the whole
    agent run, and released immediately after; `None` leaves the step
    unbounded (the historical behaviour). A trial waiting for a configure lease
    still holds its global concurrency slot — acceptable for a brief step, and
    the opposite of the per-task/per-model gates, which are deliberately
    acquired before the global limiter.
    """
    trial_start_time = datetime.now(tz=timezone.utc)

    try:
        configure_lease = (
            configure_limiter if configure_limiter is not None else nullcontext()
        )
        async with configure_lease:
            status = await acall(
                interface.configure_additional_apps,
                task_id,
                timeout=configure_timeout,
            )
        logger.info(f"Task {task_id} additional apps/services configured: {status}")

        run_result = await agent.arun_agent(
            interface,
            task_id,
            verbose=verbose,
            tool_verbosity=tool_verbosity or "brief",
            enable_surrender=enable_surrender,
        )
        return await _afinish_trial(
            task_id, trial_index, interface, run_result, trial_start_time
        )
    except BudgetExhaustedError:
        # Re-raise to stop the benchmark immediately (out of credits).
        raise
    except Exception as agent_error:
        result = await aexception_trial_result(
            task_id=task_id,
            trial_index=trial_index,
            interface=interface,
            error=agent_error,
            error_type="Agent Error",
            token_usage=agent.get_total_token_usage(),
        )
        result.duration = _elapsed_seconds(trial_start_time)
        return result


def run_independent_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults]], None],
) -> None:
    """Run trials independently for each task"""
    logger.info(
        f"Running independent trials for {len(task_ids)} tasks with {trials_per_task} trials each"
    )
    remaining_tasks = filter_incomplete_tasks(task_results, trials_per_task)

    for task_id in remaining_tasks:
        while len(task_results[task_id].trials) < trials_per_task:
            trial_index = len(task_results[task_id].trials)
            logger.info(f"Starting trial {trial_index + 1} for task {task_id}")

            result = trial_executor(task_id, trial_index)
            task_results[task_id].trials.append(result)

            logger.info(f"Trial {result.trial_id} completed. Score: {result.score:.3f}")

            # Save checkpoint after each trial
            checkpoint_saver(task_results)
            if not result.success:
                logger.error(f"Trial failed for task {task_id}")


def run_chained_trials(
    task_ids: list[str],
    trials_per_task: int,
    task_results: dict[str, TaskTrialResults],
    trial_executor: Callable[[str, int], TaskTrialResult],
    checkpoint_saver: Callable[[dict[str, TaskTrialResults], int], None],
    completed_rounds: int = 0,
    graph: Mapping[str, list[str]] | None = None,
) -> None:
    """Run trials in lockstep across all tasks.

    `task_ids` must already be in topological order (the runner enforces this)
    so a task's dependencies are always resolved by the time it runs. When
    `graph` is provided, a task whose upstream chain broke this round is
    recorded as *unreachable* without invoking the agent; this cascades for free
    to its own dependents, giving "reached step N of M" semantics.
    """
    graph = graph or {}
    for trial_round in range(completed_rounds, trials_per_task):
        logger.info(f"Starting trial round {trial_round + 1}/{trials_per_task}")

        success = True
        for task_id in task_ids:
            if len(task_results[task_id].trials) > trial_round:
                continue  # Already completed

            missing = _unsatisfied_dependency(task_id, graph, task_results, trial_round)
            if missing is not None:
                logger.warning(
                    f"Skipping task {task_id}: dependency {missing!r} did not "
                    f"produce a usable output this round (chain broken upstream)."
                )
                result = unreachable_trial_result(task_id, trial_round, missing)
                task_results[task_id].trials.append(result)
                success = False
                continue

            logger.info(f"Running trial {trial_round + 1} for task {task_id}")
            result = trial_executor(task_id, trial_round)
            task_results[task_id].trials.append(result)

            if not result.success:
                logger.error(
                    f"Trial failed for task {task_id}: {getattr(result, 'error_message', 'No error message')}"
                )
                logger.error(f"Full result: {result}")
                success = False
        # Save checkpoint after each round
        checkpoint_saver(task_results, trial_round + 1 if success else trial_round)


def create_wandb_config(
    agent: BaseAgent | None, session_id: str, **kwargs
) -> dict[str, Any]:
    """Create wandb configuration.

    `agent` may be `None` when the runner was configured with an
    `agent_factory` instead of a shared agent; in that case the agent-derived
    fields fall back to placeholders since no single agent represents the run.
    """
    return {
        "agent_type": agent.__class__.__name__
        if agent is not None
        else "agent_factory",
        "model": getattr(agent, "model", "unknown_model"),
        "session_id": session_id,
        "agent_max_iterations": getattr(agent, "max_iterations", None),
        "agent_temperature": getattr(agent, "temperature", None),
        **kwargs,
    }


class CorralRunner:
    """Simplified benchmark runner with functional approach."""

    def __init__(
        self,
        interface: CorralRouter,
        agent: BaseAgent | None = None,
        checkpoint_dir: str = "./benchmark_checkpoints",
        checkpoint_name: str | None = None,
        logger: CorralWandbLogger | None = None,
        enable_surrender: bool = False,
        metrics: list[Metric] | None = None,
        agent_factory: AgentFactory | None = None,
        concurrency: ConcurrencyConfig | None = None,
    ):
        """Initialize the runner.

        Exactly one agent source is required:

        * `agent`: a single shared agent. Fine for the serial `bench()`, but a
          shared agent accumulates per-run state and so **cannot** back
          concurrent trials — pass `agent_factory` for those.
        * `agent_factory`: a callable `(TrialContext) -> BaseAgent` that mints a
          fresh, isolated agent per trial. Required for concurrent benchmarking
          (`abench()` or `bench(max_concurrency>1)`).

        Both may be supplied; the factory then takes precedence when building a
        trial's agent, and the shared agent is used only for run metadata.

        `concurrency` is a default :class:`ConcurrencyConfig` for this runner —
        the target API's `CorralRunner(concurrency=ConcurrencyConfig(...))`. A
        config passed to `bench`/`abench` overrides it; if neither is given the
        run falls back to the scalar `max_concurrency`/`max_concurrency_per_task`
        arguments. Whichever config is in effect, its `tool_jobs_per_trial` is
        forwarded to each trial runtime over HTTP.
        """
        if agent is None and agent_factory is None:
            raise ValueError("CorralRunner requires either `agent` or `agent_factory`.")
        self.interface = interface
        self.agent = agent
        self.agent_factory = agent_factory
        self.concurrency = concurrency
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name or f"checkpoint_{self._agent_label()}"
        self.logger = logger
        self.enable_surrender = enable_surrender

        # Initialize metric registry
        self._metric_registry = MetricRegistry()
        self._default_k_values: list[int] = [5]  # Match BenchmarkResult default

        # Register initial metrics
        if metrics is not None:
            for metric in metrics:
                self._metric_registry.register(metric)
        # If no metrics provided, registry stays empty until bench() is called
        # This allows lazy initialization with proper k_values

    def _agent_label(self) -> str:
        """A stable name for the run's agent, for checkpoint/report filenames.

        Uses the shared agent's class name when present, else a generic label
        (the factory mints a different instance per trial, so no single class
        name represents the whole run).
        """
        if self.agent is not None:
            return self.agent.__class__.__name__
        return "AgentFactory"

    def _resolve_trial_agent(
        self,
        context: TrialContext,
        agent_factory: AgentFactory | None,
        hooks: AgentHooks | None,
        *,
        allow_shared: bool,
    ) -> BaseAgent:
        """Return the agent instance to use for one trial.

        Resolution order:

        1. an explicit `agent_factory` (per-call override or the instance one)
           produces a fresh, isolated agent — always safe, required for
           concurrency;
        2. otherwise, when `allow_shared` (serial execution only), reuse the
           shared `self.agent` — preserves the historical single-agent
           behaviour;
        3. otherwise, if the shared agent implements `spawn_for_trial(context)`,
           use that to mint an isolated agent;
        4. otherwise raise — we never silently share or deep-copy an arbitrary
           agent across concurrent trials.

        `hooks`, when provided, are attached to whichever agent is returned.
        """
        if agent_factory is not None:
            agent = agent_factory(context)
        elif allow_shared and self.agent is not None:
            agent = self.agent
        elif self.agent is not None and hasattr(self.agent, "spawn_for_trial"):
            agent = self.agent.spawn_for_trial(context)
        else:
            raise ValueError(
                "Concurrent benchmarking needs a fresh agent per trial. Pass "
                "`agent_factory=...` (to CorralRunner or abench/bench), or "
                "implement `agent.spawn_for_trial(context)`. Refusing to share a "
                "single mutable agent across concurrent trials."
            )
        if hooks is not None:
            agent.hooks = hooks
        return agent

    def _resolve_concurrency(
        self,
        concurrency: ConcurrencyConfig | None,
        max_concurrency: int,
        max_concurrency_per_task: int,
    ) -> ConcurrencyConfig:
        """Pick the effective :class:`ConcurrencyConfig` for a bench call.

        Precedence: an explicit `concurrency` passed to the call, else the
        runner-level default (`self.concurrency`), else a config built from the
        scalar `max_concurrency`/`max_concurrency_per_task` shortcuts. When a
        `ConcurrencyConfig` is in effect the scalar shortcuts are ignored — the
        config is the single source of truth (it carries `tool_jobs_per_trial`,
        `per_model`, and `configure_apps`, all enforced on the concurrent path).
        Callers with no config at all get the historical scalar behaviour,
        byte-for-byte.
        """
        effective = concurrency or self.concurrency
        if effective is not None:
            return effective
        return ConcurrencyConfig(
            global_trials=max_concurrency, per_task=max_concurrency_per_task
        )

    @property
    def metric_registry(self) -> MetricRegistry:
        """Access the internal metric registry.

        This allows advanced users to directly interact with the registry
        for operations like parallel calculation.

        Returns:
            The MetricRegistry instance used by this runner.
        """
        return self._metric_registry

    # Metric Introspection Methods
    def list_metrics(self, k_values: list[int] | None = None) -> list[dict[str, str]]:
        """List all metrics that will be used for benchmarking.

        Args:
            k_values: k values for pass@k metrics. Used only if no metrics
                      have been explicitly configured. If None, uses [1].

        Returns:
            List of dicts with 'name', 'display_name', 'description', and 'type'
        """
        # If registry is empty and no explicit metrics, show defaults
        if len(self._metric_registry.list_all()) == 0:
            metrics = get_default_metrics(k_values or [1])
        else:
            metrics = self._metric_registry.list_all()

        return [
            {
                "name": m.metadata.name,
                "display_name": m.metadata.display_name,
                "description": m.metadata.description,
                "type": "task" if isinstance(m, TaskMetric) else "overall",
            }
            for m in metrics
        ]

    def print_metrics(self, k_values: list[int] | None = None) -> None:
        """Print a formatted table of metrics that will be used.

        Args:
            k_values: k values for pass@k metrics. Used only if no metrics
                      have been explicitly configured. If None, uses [1].
        """
        metrics = self.list_metrics(k_values)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Configured Metrics ({len(metrics)} total)")
        logger.info(f"{'=' * 70}")
        for m in sorted(metrics, key=lambda x: (x["type"], x["name"])):
            logger.info(f"  [{m['type']:7}] {m['name']}: {m['description']}")
        logger.info(f"{'=' * 70}\n")

    # Metric Registration Methods
    def register_metric(self, metric: Metric) -> None:
        """Register a new metric.

        If no metrics have been explicitly registered yet, this will first
        populate the registry with default metrics before adding the new one.

        Args:
            metric: The metric instance to register.

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        # Initialize with defaults if registry is empty
        if len(self._metric_registry.list_all()) == 0:
            for default_metric in get_default_metrics(self._default_k_values):
                self._metric_registry.register(default_metric)

        self._metric_registry.register(metric)

    def unregister_metric(self, metric_name: str) -> Metric | None:
        """Unregister a metric by name.

        If no metrics have been explicitly registered yet, this will first
        populate the registry with default metrics before removing.

        Args:
            metric_name: The name of the metric to remove.

        Returns:
            The removed metric instance, or None if not found.
        """
        # Initialize with defaults if registry is empty
        if len(self._metric_registry.list_all()) == 0:
            for default_metric in get_default_metrics(self._default_k_values):
                self._metric_registry.register(default_metric)

        try:
            metric = self._metric_registry.get(metric_name)
            self._metric_registry.unregister(metric_name)
            return metric
        except KeyError:
            return None

    def clear_metrics(self) -> None:
        """Remove all registered metrics."""
        # Create a fresh empty registry
        self._metric_registry = MetricRegistry()

    def reset_metrics(self, k_values: list[int] | None = None) -> None:
        """Reset metrics to the default set.

        Args:
            k_values: k values for pass@k metrics. If None, uses [1].
        """
        k_vals = k_values or [1]
        self._default_k_values = k_vals
        self._metric_registry = MetricRegistry()
        for metric in get_default_metrics(k_vals):
            self._metric_registry.register(metric)

    def generate_latex_docs(
        self,
        task_ids: list[str],
        output_dir: str | None = None,
        level: int | str = 1,
        env_name: str | None = None,
        verbosity: str | None = None,
    ) -> None:
        """Generate LaTeX documentation for a list of tasks.

        The colorbox generation workflow:
        1. First task (main task, no dependencies in `input_map`): saves to cache only
        2. Subsequent tasks (subtasks, with `input_map` entries): add to cache and generate .tex
        3. After all tasks: clear the cache

        Args:
            task_ids: List of task IDs to generate LaTeX for.
            output_dir: Directory for output .tex files. Defaults to
                        "tex_files" in the current working directory.
            level: Task level identifier (e.g., 1, 2,...). Default: 1.
            env_name: Environment name (e.g., "afm", "catalyst").
            verbosity: Tool verbosity level used to filter tool descriptions and
                       return sections in the generated LaTeX. Accepts a
                       `ToolVerbosity` value string (e.g. "brief",
                       "detailed"). Defaults to "detailed" when not
                       provided.
        """
        output_dir = output_dir or str(Path.cwd() / "tex_files")
        logger.info(f"Generating LaTeX documentation for {len(task_ids)} tasks...")

        for task_id in task_ids:
            try:
                result = self.interface.generate_latex(
                    task_id=task_id,
                    output_dir=output_dir,
                    level=level,
                    env_name=env_name,
                    verbosity=verbosity,
                )
                logger.debug(
                    f"Generated LaTeX for task {task_id}: "
                    f"task={result.get('output_path')}, "
                    f"tools={result.get('tools_output_path')}, "
                    f"scoring={result.get('scoring_output_path')}"
                )
            except Exception as e:
                logger.warning(f"Failed to generate LaTeX for task {task_id}: {e}")

        logger.info("LaTeX documentation generation complete.")

    def _get_metrics_for_benchmark(self) -> list[Metric] | None:
        """Get the metrics list for creating a BenchmarkResult.

        If the registry is empty (no explicit configuration), returns None
        to let BenchmarkResult use its default behavior with k_values.
        Otherwise, returns the explicitly configured metrics.

        Returns:
            List of metrics if explicitly configured, or None for defaults.
        """
        registered = self._metric_registry.list_all()
        if len(registered) == 0:
            return None  # Let BenchmarkResult use defaults with k_values
        return registered

    def _make_trial_executor(
        self,
        *,
        session_id: str,
        benchmark_run_id: str,
        agent_factory: AgentFactory | None,
        hooks: AgentHooks | None,
        allow_shared: bool,
        verbose: bool,
        tool_verbosity: str | None,
        configure_timeout: float | None,
        use_runtimes: bool = False,
        episode_id_fn: Callable[[str, int], str] | None = None,
        close_episode_after_trial: bool = False,
        tool_jobs_per_trial: int | None = None,
        interface: CorralRouter | None = None,
    ) -> Callable[[str, int], TaskTrialResult]:
        """Build the `(task_id, trial_index) -> TaskTrialResult` trial runner.

        Each invocation resolves a fresh (or shared, when `allow_shared`) agent
        via :meth:`_resolve_trial_agent` and executes one trial. The returned
        callable is what both the serial and concurrent schedulers drive.

        `interface` overrides the router the synchronous executor drives (defaults
        to `self.interface`). The async benchmark passes a **synchronous view** of
        an :class:`~corral.router.AsyncCorralRouter` here, since this executor's
        blocking :func:`execute_single_trial` runs in a worker thread with no
        event loop; it is only ever used with `use_runtimes=False` in that case.

        When `use_runtimes` is set, the trial is routed through an isolated
        server-side *trial runtime*: a fresh runtime is created, the agent runs
        against a router scoped to it (`/trials/{id}/...`), and the runtime is
        closed afterwards. This is what makes concurrent repeated trials of the
        same task safe (`max_concurrency_per_task > 1`).

        `episode_id_fn(task_id, trial_index)` picks the *episode* a runtime joins
        (its shared dependency-output store). Independent trials use a unique id
        per trial (no sharing); a dependency chain gives every task in one trial
        round the same episode id so they see each other's outputs. Defaults to a
        per-`(task, trial)` id. `close_episode_after_trial` frees that store when
        the trial ends — correct only when the trial *owns* its episode (an
        independent trial); a chain's episode outlives its individual nodes, so
        its scheduler closes the episode once the whole round is done.

        `tool_jobs_per_trial` (from `ConcurrencyConfig.tool_jobs_per_trial`) is
        forwarded to `create_trial`, sizing each runtime's background-job pool;
        `None` leaves the server on its own default.
        """
        router = interface if interface is not None else self.interface
        if episode_id_fn is None:

            def episode_id_fn(task_id: str, trial_index: int) -> str:
                return f"{benchmark_run_id}:{task_id}:{trial_index}"

        def _execute(
            task_id: str, trial_index: int, interface, agent: BaseAgent
        ) -> TaskTrialResult:
            return execute_single_trial(
                task_id=task_id,
                trial_index=trial_index,
                interface=interface,
                agent=agent,
                verbose=verbose,
                tool_verbosity=tool_verbosity,
                configure_timeout=configure_timeout,
                enable_surrender=self.enable_surrender,
            )

        def trial_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            context = TrialContext(
                task_id=task_id,
                trial_index=trial_index,
                session_id=session_id,
                benchmark_run_id=benchmark_run_id,
            )
            agent = self._resolve_trial_agent(
                context, agent_factory, hooks, allow_shared=allow_shared
            )
            if not use_runtimes:
                return _execute(task_id, trial_index, router, agent)

            # Isolated runtime per trial: create -> run against scoped router ->
            # close. The scoped router rewrites every mutable call (and the MCP
            # URL) to this runtime, so same-task trials never share state. The
            # runtime joins the episode chosen by `episode_id_fn`, so chained
            # siblings in the same round read each other's outputs.
            episode_id = episode_id_fn(task_id, trial_index)
            created = router.create_trial(
                task_id,
                benchmark_run_id=benchmark_run_id,
                episode_id=episode_id,
                trial_index=trial_index,
                tool_jobs_per_trial=tool_jobs_per_trial,
            )
            trial_runtime_id = created["trial_runtime_id"]
            # Bind the run's verbosity into the trial router so it travels with
            # the trial rather than being read from the shared parent's mutable
            # `current_verbosity` (which every concurrent trial would otherwise
            # share); `None` falls back to the parent's, preserving behaviour.
            # Carry the runtime's server-side workspace so a sandbox-running
            # agent can write into the *scored* directory.
            trial_router = router.for_trial(
                trial_runtime_id,
                task_id,
                verbosity=tool_verbosity,
                workspace=created.get("workspace"),
            )
            try:
                return _execute(task_id, trial_index, trial_router, agent)
            finally:
                try:
                    router.close_trial(trial_runtime_id)
                except Exception as exc:
                    logger.warning(
                        f"Failed to close trial runtime {trial_runtime_id}: {exc}"
                    )
                if close_episode_after_trial:
                    self._close_episode_quietly(episode_id)

        return trial_executor

    def _make_atrial_executor(
        self,
        *,
        session_id: str,
        benchmark_run_id: str,
        agent_factory: AgentFactory | None,
        hooks: AgentHooks | None,
        allow_shared: bool,
        verbose: bool,
        tool_verbosity: str | None,
        configure_timeout: float | None,
        use_runtimes: bool = False,
        episode_id_fn: Callable[[str, int], str] | None = None,
        close_episode_after_trial: bool = False,
        tool_jobs_per_trial: int | None = None,
        configure_limiter: anyio.Semaphore | None = None,
    ) -> Callable[..., Awaitable[TaskTrialResult]]:
        """Async counterpart of :meth:`_make_trial_executor`.

        Returns an awaitable `(task_id, trial_index, *, agent=None) ->
        TaskTrialResult` that drives the trial through
        :func:`aexecute_single_trial`, so a natively-async agent runs its harness
        coroutine on the scheduler loop rather than in a nested event loop inside
        a worker thread. Runtime create/close and episode cleanup are blocking
        HTTP, so they are offloaded to worker threads; the semantics (isolated
        per-trial runtime, episode sharing, cleanup ownership) mirror the
        synchronous executor exactly.

        `agent` lets the scheduler pass an already-resolved agent. Per-model
        gating (`ConcurrencyConfig.per_model`) has to know a trial's model
        *before* it acquires the global limiter, so the scheduler resolves the
        agent up front, reads `agent.model`, and hands the same instance here
        rather than having the executor build a second one. When `None` the
        executor resolves its own agent as before (the no-`per_model` path is
        unchanged).

        `configure_limiter` (from `ConcurrencyConfig.configure_apps`) is the
        benchmark-wide app-configure lease, forwarded to
        :func:`aexecute_single_trial` and held only around the configure step.
        """
        if episode_id_fn is None:

            def episode_id_fn(task_id: str, trial_index: int) -> str:
                return f"{benchmark_run_id}:{task_id}:{trial_index}"

        async def _aexecute(
            task_id: str, trial_index: int, interface, agent: BaseAgent
        ) -> TaskTrialResult:
            return await aexecute_single_trial(
                task_id=task_id,
                trial_index=trial_index,
                interface=interface,
                agent=agent,
                verbose=verbose,
                tool_verbosity=tool_verbosity,
                configure_timeout=configure_timeout,
                enable_surrender=self.enable_surrender,
                configure_limiter=configure_limiter,
            )

        async def atrial_executor(
            task_id: str, trial_index: int, *, agent: BaseAgent | None = None
        ) -> TaskTrialResult:
            if agent is None:
                context = TrialContext(
                    task_id=task_id,
                    trial_index=trial_index,
                    session_id=session_id,
                    benchmark_run_id=benchmark_run_id,
                )
                agent = self._resolve_trial_agent(
                    context, agent_factory, hooks, allow_shared=allow_shared
                )
            if not use_runtimes:
                return await _aexecute(task_id, trial_index, self.interface, agent)

            # Isolated runtime per trial: create -> run against scoped router ->
            # close, mirroring the sync executor. Create/close HTTP goes through
            # `acall`, so an async router is awaited directly and a synchronous
            # one is offloaded so it does not stall the loop shared by siblings.
            episode_id = episode_id_fn(task_id, trial_index)
            created = await acall(
                self.interface.create_trial,
                task_id,
                benchmark_run_id=benchmark_run_id,
                episode_id=episode_id,
                trial_index=trial_index,
                tool_jobs_per_trial=tool_jobs_per_trial,
            )
            trial_runtime_id = created["trial_runtime_id"]
            # Bind the run's verbosity into the trial router (see the sync twin);
            # `None` falls back to the parent's current verbosity. For an async
            # interface this returns an AsyncTrialScopedRouter sharing its pool.
            # Carry the runtime's server-side workspace (see the sync twin).
            trial_router = self.interface.for_trial(
                trial_runtime_id,
                task_id,
                verbosity=tool_verbosity,
                workspace=created.get("workspace"),
            )
            try:
                return await _aexecute(task_id, trial_index, trial_router, agent)
            finally:
                try:
                    await acall(self.interface.close_trial, trial_runtime_id)
                except Exception as exc:
                    logger.warning(
                        f"Failed to close trial runtime {trial_runtime_id}: {exc}"
                    )
                if close_episode_after_trial:
                    await self._aclose_episode_quietly(episode_id)

        return atrial_executor

    @staticmethod
    def _chained_episode_id(benchmark_run_id: str, trial_round: int) -> str:
        """Episode id shared by every task in one trial round of a chain."""
        return f"{benchmark_run_id}:episode:{trial_round}"

    def _close_episode_quietly(self, episode_id: str) -> None:
        """Best-effort free of an episode's dependency store on the server.

        A router/server without episode support (or a stale/already-freed
        episode) simply leaves the store untouched; a failed cleanup must never
        fail the trial that produced real results.
        """
        close = getattr(self.interface, "close_episode", None)
        if not callable(close):
            return
        try:
            close(episode_id)
        except Exception as exc:
            logger.warning(f"Failed to close episode {episode_id}: {exc}")

    async def _aclose_episode_quietly(self, episode_id: str) -> None:
        """Async twin of :meth:`_close_episode_quietly`.

        Frees an episode's dependency store through :func:`acall`, so an async
        router is awaited and a synchronous one is offloaded; a router/server
        without episode support (or an already-freed episode) is a no-op and a
        failed cleanup never fails the trial that produced real results.
        """
        close = getattr(self.interface, "close_episode", None)
        if not callable(close):
            return
        try:
            await acall(close, episode_id)
        except Exception as exc:
            logger.warning(f"Failed to close episode {episode_id}: {exc}")

    # Run Benchmark Method
    def bench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
        max_concurrency: int = 1,
        max_concurrency_per_task: int = 1,
        agent_factory: AgentFactory | None = None,
        concurrency: ConcurrencyConfig | None = None,
    ) -> BenchmarkResult:
        """Run benchmark with functional approach.

        Args:
            task_ids: List of task IDs to benchmark. If None, uses all available tasks.
            trials_per_task: Number of trials to run per task.
            k_values: k values for pass@k metrics.
            verbose: Whether to enable verbose logging.
            session_id: Session identifier. Auto-generated if None.
            tool_verbosity: Verbosity level for tools.
            configure_timeout: Timeout for configuring additional apps.
            hooks: Agent hooks to inject into the agent before running.
            run_name: Name for the benchmark run (used in report filename).
                     If None, defaults to "unknown_env".
            max_concurrency: Maximum number of trials to run simultaneously. The
                default `1` preserves the original serial behaviour exactly. When
                `> 1`, work runs in parallel via :meth:`abench` (which requires an
                `agent_factory`, since a shared agent cannot back concurrent
                trials): independent tasks overlap directly, while a dependency
                chain runs as a concurrent *episode DAG* — sibling tasks in
                parallel, dependents after their parents succeed, and trial
                rounds pipelined as isolated episodes.
            max_concurrency_per_task: Maximum simultaneous trials of the *same*
                task. Above `1` repeated trials of one task overlap through
                isolated trial runtimes (independent tasks only).
            agent_factory: Per-call override of the runner's `agent_factory`.
            concurrency: A full :class:`ConcurrencyConfig`. When given (here or on
                the runner) it supersedes `max_concurrency`/
                `max_concurrency_per_task` and additionally carries
                `tool_jobs_per_trial` (forwarded to each trial runtime),
                `per_model` (caps simultaneous trials per agent model), and
                `configure_apps` (caps simultaneous app-configure steps).

        Returns:
            BenchmarkResult containing all trial results and metrics.

        Note:
            When the effective concurrency exceeds `1` this internally drives an
            event loop via `anyio.run`, so it must **not** be called from inside
            a running event loop — `await runner.abench(...)` there instead.
        """
        task_ids = task_ids or self.interface.get_available_tasks()
        config = self._resolve_concurrency(
            concurrency, max_concurrency, max_concurrency_per_task
        )

        if config.global_trials > 1 or config.per_task > 1:
            return anyio.run(
                partial(
                    self.abench,
                    task_ids=task_ids,
                    trials_per_task=trials_per_task,
                    k_values=k_values,
                    verbose=verbose,
                    session_id=session_id,
                    tool_verbosity=tool_verbosity,
                    configure_timeout=configure_timeout,
                    hooks=hooks,
                    run_name=run_name,
                    agent_factory=agent_factory,
                    concurrency=config,
                )
            )

        # Serial path: byte-for-byte the historical behaviour (shared agent OK).
        session_id = session_id or create_session_id()
        trial_executor = self._make_trial_executor(
            session_id=session_id,
            benchmark_run_id=f"run_{session_id}",
            agent_factory=agent_factory or self.agent_factory,
            hooks=hooks,
            allow_shared=True,
            verbose=verbose,
            tool_verbosity=tool_verbosity,
            configure_timeout=configure_timeout,
        )

        return self._run_benchmark(
            task_ids=task_ids,
            trial_executor=trial_executor,
            trials_per_task=trials_per_task,
            k_values=k_values,
            verbose=verbose,
            session_id=session_id,
            tool_verbosity=tool_verbosity,
            hooks=hooks,
            run_name=run_name,
        )

    async def abench(
        self,
        task_ids: list[str] | None = None,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
        max_concurrency: int = 1,
        max_concurrency_per_task: int = 1,
        agent_factory: AgentFactory | None = None,
        concurrency: ConcurrencyConfig | None = None,
    ) -> BenchmarkResult:
        """Async benchmark runner with bounded concurrency.

        This is the asynchronous counterpart of :meth:`bench`. Independent tasks
        run concurrently under a global concurrency limit, each trial in its own
        worker thread with its **own** agent instance. A dependency chain runs as
        a concurrent *episode DAG*: each trial round is an isolated episode whose
        sibling tasks run in parallel, dependents start once their parents
        succeed, broken chains propagate as *unreachable*, and rounds overlap up
        to the limit. A single collector applies results and writes checkpoints,
        so aggregation stays deterministic and lock-free. At `max_concurrency==1`
        a chain keeps the byte-for-byte serial path.

        Concurrent execution (`max_concurrency > 1`) requires an `agent_factory`
        (here or on the runner) because a single shared agent accumulates
        per-run state and cannot be used by two trials at once.

        Args mirror :meth:`bench`. Use this directly when you are already inside
        an event loop; use `bench(max_concurrency=...)` from synchronous code.

        Returns:
            BenchmarkResult containing all trial results and metrics.
        """
        task_ids = task_ids or await acall(self.interface.get_available_tasks)
        session_id = session_id or create_session_id()
        benchmark_run_id = f"run_{session_id}"
        config = self._resolve_concurrency(
            concurrency, max_concurrency, max_concurrency_per_task
        )

        # Resolve once, async-aware: an AsyncCorralRouter is awaited directly, a
        # synchronous router is offloaded. Threaded down to `_arun_benchmark` (and
        # into the wandb config) so it is never re-fetched on the loop.
        is_chained = await acall(self.interface.supports_dependency_chain)

        # Any overlap of trials — parallel independent tasks, repeated trials of
        # one task, or a chain running its rounds/siblings concurrently — forbids
        # a single shared agent (it accumulates per-run state).
        runs_concurrently = config.global_trials > 1 or config.per_task > 1

        # A dependency chain runs concurrently as an episode DAG: each node goes
        # through an episode-scoped trial runtime, so like `per_task > 1` it
        # needs server-side trial-runtime support. A single-concurrency chain
        # (`global_trials == 1`) keeps the byte-for-byte serial path.
        concurrent_chained = is_chained and config.global_trials > 1

        # Environment Concurrency Isolation, Phase 3: a `"process"` env (wetlab)
        # keeps process-global state, so its trials only overlap safely when each
        # runs in its own worker process. That routing only happens through a
        # trial runtime (`create_trial` → worker), so when the run overlaps
        # independent tasks and the server has a live worker pool backing some
        # selected `"process"` env, force trial-runtime routing even at the
        # default `per_task == 1` (chained runs already route via runtimes). The
        # matching serial-gate relaxation lives in `_arun_independent_trials`.
        worker_backend_active = await _worker_backend_active(self.interface)
        concurrency_modes = await _fetch_concurrency_modes(self.interface)
        process_needs_workers = (
            worker_backend_active
            and runs_concurrently
            and not is_chained
            and any(concurrency_modes.get(t) == "process" for t in task_ids)
        )
        use_runtimes = (
            (config.per_task > 1 and not is_chained)
            or concurrent_chained
            or process_needs_workers
        )
        if use_runtimes and not callable(getattr(self.interface, "create_trial", None)):
            raise ValueError(
                "Concurrent trial runtimes require a router/server that supports "
                "them (CorralRouter.create_trial). Update the server, or keep "
                "max_concurrency_per_task=1 and dependency chains serial."
            )

        effective_factory = agent_factory or self.agent_factory
        if (
            runs_concurrently
            and effective_factory is None
            and not (self.agent is not None and hasattr(self.agent, "spawn_for_trial"))
        ):
            # Fail fast with a clear message rather than letting each worker raise
            # (which anyio would surface wrapped in an exception group).
            raise ValueError(
                "Concurrent benchmarking (max_concurrency > 1) needs a fresh agent "
                "per trial. Pass `agent_factory=...` to abench()/bench() or the "
                "CorralRunner, or implement `agent.spawn_for_trial(context)`. A "
                "single shared agent cannot back concurrent trials."
            )

        # A chained node's episode is owned by the round's scheduler (siblings
        # share it), so the per-trial executor must not free it; an independent
        # trial owns its throwaway episode and frees it on completion.
        if concurrent_chained:

            def episode_id_fn(task_id: str, trial_round: int) -> str:  # noqa: ARG001
                # A chain's episode is keyed by round only; every task in one
                # round shares it, so the task id is intentionally ignored.
                return self._chained_episode_id(benchmark_run_id, trial_round)

            close_episode_after_trial = False
        else:
            episode_id_fn = None
            close_episode_after_trial = True

        # Two executors share the same wiring: the sync one backs the serial
        # fallback (a single-concurrency chain feeds the synchronous serial
        # machinery), while the async one backs every concurrent branch so agents
        # run natively on the loop instead of in a nested per-trial event loop.
        executor_kwargs: dict[str, Any] = {
            "session_id": session_id,
            "benchmark_run_id": benchmark_run_id,
            "agent_factory": effective_factory,
            "hooks": hooks,
            "allow_shared": not runs_concurrently,
            "verbose": verbose,
            "tool_verbosity": tool_verbosity,
            "configure_timeout": configure_timeout,
            "use_runtimes": use_runtimes,
            "episode_id_fn": episode_id_fn,
            "close_episode_after_trial": close_episode_after_trial,
            # Only meaningful when `use_runtimes` (create_trial is called);
            # forwarded to size each runtime's background-job pool.
            "tool_jobs_per_trial": config.tool_jobs_per_trial,
        }
        # The sync executor only ever feeds the serial fallback, which runs in a
        # worker thread with no event loop, so drive it against a synchronous view
        # of the interface (a no-op for an already-synchronous router). The async
        # executor keeps the real interface so it can await an AsyncCorralRouter.
        sync_interface = as_sync_interface(self.interface)
        trial_executor = self._make_trial_executor(
            interface=sync_interface, **executor_kwargs
        )
        # A benchmark-wide lease capping simultaneous app-configure steps
        # (`ConcurrencyConfig.configure_apps`); `None` leaves the step unbounded.
        # Built on the loop and shared by every concurrent trial (independent or
        # chained); the async executor holds it only around the configure call.
        configure_limiter = (
            anyio.Semaphore(config.configure_apps)
            if config.configure_apps is not None
            else None
        )
        atrial_executor = self._make_atrial_executor(
            configure_limiter=configure_limiter, **executor_kwargs
        )

        # Per-model gating needs a trial's model *before* it takes a global slot,
        # so the concurrent schedulers resolve the agent up front through this
        # seam (same context/factory/hooks the executor would use) and pass the
        # resolved instance back into the executor — only when `per_model` is set.
        def resolve_trial_agent(task_id: str, trial_index: int) -> BaseAgent:
            context = TrialContext(
                task_id=task_id,
                trial_index=trial_index,
                session_id=session_id,
                benchmark_run_id=benchmark_run_id,
            )
            return self._resolve_trial_agent(
                context,
                effective_factory,
                hooks,
                allow_shared=not runs_concurrently,
            )

        return await self._arun_benchmark(
            task_ids=task_ids,
            trial_executor=trial_executor,
            atrial_executor=atrial_executor,
            config=config,
            is_chained=is_chained,
            trials_per_task=trials_per_task,
            k_values=k_values,
            verbose=verbose,
            session_id=session_id,
            benchmark_run_id=benchmark_run_id,
            tool_verbosity=tool_verbosity,
            hooks=hooks,
            run_name=run_name,
            agent_resolver=resolve_trial_agent,
            worker_backend_active=worker_backend_active,
        )

    def bench_from_traces(
        self,
        traces: dict[str, list[LiteLLMMessage]],
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        configure_timeout: float | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark from previously saved conversation traces.

        Instead of building prompts from scratch, each task is initialised
        from the trace provided in `traces`.  A deep-copy of the prototype
        agent (`self.agent`) is created per task with its
        `_initial_messages` set to the corresponding trace so the agent
        continues from that conversation state.

        Args:
            traces: Mapping of task_id to the conversation trace (list of
                `LiteLLMMessage`) to replay from.
            trials_per_task: Number of trials to run per task.
            k_values: k values for pass@k metrics.
            verbose: Whether to enable verbose logging.
            session_id: Session identifier. Auto-generated if None.
            tool_verbosity: Verbosity level for tools.
            configure_timeout: Timeout for configuring additional apps.
            hooks: Agent hooks to inject into every agent clone.
            run_name: Name for the benchmark run (used in report filename).
                     If None, auto-generates one.

        Returns:
            BenchmarkResult containing all trial results and metrics.
        """
        task_ids = list(traces.keys())

        def trial_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            trace = traces[task_id]
            self.agent._initial_messages = list(trace)
            if hooks:
                self.agent.hooks = hooks
            try:
                return execute_single_trial(
                    task_id=task_id,
                    trial_index=trial_index,
                    interface=self.interface,
                    agent=self.agent,
                    verbose=verbose,
                    tool_verbosity=tool_verbosity,
                    configure_timeout=configure_timeout,
                    enable_surrender=self.enable_surrender,
                )
            finally:
                self.agent._initial_messages = None

        return self._run_benchmark(
            task_ids=task_ids,
            trial_executor=trial_executor,
            trials_per_task=trials_per_task,
            k_values=k_values,
            verbose=verbose,
            session_id=session_id,
            tool_verbosity=tool_verbosity,
            hooks=hooks,
            run_name=run_name,
            extra_wandb_config={"from_traces": True},
        )

    def _prepare_benchmark(
        self,
        *,
        task_ids: list[str],
        trials_per_task: int,
        k_values: int | list[int] | None,
        session_id: str | None,
        tool_verbosity: str | None,
        hooks: AgentHooks | None,
        extra_wandb_config: dict[str, Any] | None,
        dependency_chain: bool | None = None,
    ) -> tuple[str, list[int], dict[str, TaskTrialResults], Callable, datetime]:
        """Shared benchmark setup for the serial and concurrent paths.

        Sets verbosity, resolves the session id and k-values, loads/initialises
        results, wires the checkpoint saver, starts logging, and stamps the start
        time. Returns everything the execution phase needs.

        `dependency_chain` lets a caller pass in an already-resolved value for the
        wandb config; the async path supplies it (resolved async-aware in
        :meth:`abench`) so an :class:`~corral.router.AsyncCorralRouter` is never
        queried synchronously here. The serial path leaves it `None` and the
        (synchronous) router is queried directly, exactly as before.
        """
        if tool_verbosity:
            self.interface.set_verbosity(tool_verbosity)

        session_id = session_id or create_session_id()
        k_values = validate_k_values(k_values, trials_per_task)

        if trials_per_task == 0:
            raise ValueError("Number of trials per task must be greater than 0")

        task_results = self._load_or_initialize_results(task_ids, session_id)
        checkpoint_saver = partial(self._save_checkpoint, session_id)

        if self.logger:
            if dependency_chain is None:
                dependency_chain = self.interface.supports_dependency_chain()
            config = create_wandb_config(
                self.agent,
                session_id,
                trials_per_task=trials_per_task,
                k_values=k_values,
                tool_verbosity=self.interface.current_verbosity,
                task_ids=task_ids,
                dependency_chain=dependency_chain,
                enable_surrender=self.enable_surrender,
                hooks_enabled=hooks is not None,
                **(extra_wandb_config or {}),
            )
            self.logger.start_logging(config)

        start_time = datetime.now(tz=timezone.utc)
        return session_id, k_values, task_results, checkpoint_saver, start_time

    def _finalize_benchmark(
        self,
        *,
        task_results: dict[str, TaskTrialResults],
        session_id: str,
        k_values: list[int],
        verbose: bool,
        run_name: str | None,
        tool_verbosity: str | None,
        start_time: datetime,
    ) -> BenchmarkResult:
        """Shared benchmark teardown: build result, log, checkpoint, report."""
        end_time = datetime.now(tz=timezone.utc)
        total_duration = (end_time - start_time).total_seconds()

        result = BenchmarkResult(
            task_results=task_results,
            k=k_values,
            verbosity=self.interface.current_verbosity,
            verbose=verbose,
            total_duration=total_duration,
            metrics=self._get_metrics_for_benchmark(),
        )

        if self.logger:
            self.logger.log_final_results(result)

        # Save final checkpoint with finished suffix and remove original
        self._save_finished_checkpoint(session_id, task_results)
        self._remove_original_checkpoint(session_id)

        # Generate and save report
        self._save_benchmark_report(result, run_name, tool_verbosity)

        return result

    def _execute_serial(
        self,
        task_ids: list[str],
        trials_per_task: int,
        task_results: dict[str, TaskTrialResults],
        trial_executor: Callable[[str, int], TaskTrialResult],
        checkpoint_saver: Callable,
        session_id: str,
        interface: CorralRouter | None = None,
    ) -> None:
        """Serial trial orchestration (independent or chained), as before.

        `interface` overrides the router queried for the dependency graph
        (defaults to `self.interface`). The async single-concurrency fallback
        runs this whole method in a worker thread, so it passes a synchronous
        view of an :class:`~corral.router.AsyncCorralRouter` here; the serial
        `bench` path leaves it `None`, so the (synchronous) router is used
        directly, exactly as before.
        """
        router = interface if interface is not None else self.interface
        if router.supports_dependency_chain():
            checkpoint = self._load_checkpoint(session_id)
            completed_rounds = (
                checkpoint.get("completed_trials", 0) if checkpoint else 0
            )

            # Enforce the chained invariant up front: the selection must be
            # dependency-closed, and tasks must run in topological order so a
            # task's dependencies are always satisfied by the time it runs.
            graph = router.get_dependency_graph()
            assert_dependencies_selected(task_ids, graph)
            ordered_ids = order_selected(task_ids, graph)

            run_chained_trials(
                ordered_ids,
                trials_per_task,
                task_results,
                self._make_logging_trial_executor(trial_executor),
                self._make_chained_checkpoint_saver(checkpoint_saver),
                completed_rounds,
                graph=graph,
            )
        else:
            run_independent_trials(
                task_ids,
                trials_per_task,
                task_results,
                self._make_logging_trial_executor(trial_executor),
                checkpoint_saver,
            )

    def _run_benchmark(
        self,
        task_ids: list[str],
        trial_executor: Callable[[str, int], TaskTrialResult],
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        tool_verbosity: str | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
        extra_wandb_config: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Serial benchmark used by `bench` (max_concurrency=1) and `bench_from_traces`.

        Handles setup, logging, serial trial orchestration (independent or
        chained), result aggregation, checkpointing, and report generation.
        """
        session_id, k_values, task_results, checkpoint_saver, start_time = (
            self._prepare_benchmark(
                task_ids=task_ids,
                trials_per_task=trials_per_task,
                k_values=k_values,
                session_id=session_id,
                tool_verbosity=tool_verbosity,
                hooks=hooks,
                extra_wandb_config=extra_wandb_config,
            )
        )

        try:
            self._execute_serial(
                task_ids,
                trials_per_task,
                task_results,
                trial_executor,
                checkpoint_saver,
                session_id,
            )
            return self._finalize_benchmark(
                task_results=task_results,
                session_id=session_id,
                k_values=k_values,
                verbose=verbose,
                run_name=run_name,
                tool_verbosity=tool_verbosity,
                start_time=start_time,
            )
        finally:
            if self.logger:
                self.logger.finish()

    async def _arun_benchmark(
        self,
        task_ids: list[str],
        trial_executor: Callable[[str, int], TaskTrialResult],
        atrial_executor: Callable[..., Awaitable[TaskTrialResult]],
        config: ConcurrencyConfig,
        is_chained: bool,
        trials_per_task: int = 1,
        k_values: int | list[int] | None = None,
        verbose: bool = False,
        session_id: str | None = None,
        benchmark_run_id: str | None = None,
        tool_verbosity: str | None = None,
        hooks: AgentHooks | None = None,
        run_name: str | None = None,
        extra_wandb_config: dict[str, Any] | None = None,
        agent_resolver: Callable[[str, int], BaseAgent] | None = None,
        worker_backend_active: bool = False,
    ) -> BenchmarkResult:
        """Async counterpart of :meth:`_run_benchmark`.

        Independent tasks are scheduled concurrently (bounded by `config`).
        A dependency chain runs as a concurrent *episode DAG* when the global
        limit allows overlap; at `global_trials == 1` it falls back to the serial
        path executed in a worker thread so the event loop stays responsive.

        The concurrent branches drive the async `atrial_executor` (agents run
        natively on the loop); only the serial fallback uses the blocking
        `trial_executor`, since it feeds the synchronous serial machinery.

        `is_chained` is resolved once by :meth:`abench` (async-aware) and passed
        in, so an :class:`~corral.router.AsyncCorralRouter` is never queried
        synchronously here or when building the run's wandb config.

        `agent_resolver(task_id, trial_index) -> BaseAgent` lets the concurrent
        schedulers learn a trial's model before it takes a global slot, so
        `config.per_model` caps can gate on it (see :meth:`abench`); it is only
        consulted when `config.per_model` is set.

        `worker_backend_active` (resolved once by :meth:`abench`) tells the
        independent scheduler that the server has a live process-worker pool, so
        `"process"` trials are isolated per worker and may overlap instead of
        being serialised (Environment Concurrency Isolation, Phase 3).
        """
        session_id, k_values, task_results, checkpoint_saver, start_time = (
            self._prepare_benchmark(
                task_ids=task_ids,
                trials_per_task=trials_per_task,
                k_values=k_values,
                session_id=session_id,
                tool_verbosity=tool_verbosity,
                hooks=hooks,
                extra_wandb_config=extra_wandb_config,
                dependency_chain=is_chained,
            )
        )
        benchmark_run_id = benchmark_run_id or f"run_{session_id}"

        try:
            if is_chained:
                if config.global_trials > 1:
                    # Concurrent episode DAG: sibling nodes run in parallel and
                    # trial rounds overlap, each round an isolated episode.
                    await self._arun_chained_trials(
                        task_ids=task_ids,
                        trials_per_task=trials_per_task,
                        task_results=task_results,
                        config=config,
                        node_atrial_executor=atrial_executor,
                        checkpoint_saver=checkpoint_saver,
                        benchmark_run_id=benchmark_run_id,
                        agent_resolver=agent_resolver,
                    )
                else:
                    # Single-concurrency chain keeps the byte-for-byte serial
                    # path; run it in a worker thread to avoid blocking the loop.
                    # That thread has no event loop, so drive the serial machinery
                    # against a synchronous view of the interface (a no-op for an
                    # already-synchronous router); `trial_executor` was built the
                    # same way in `abench`.
                    await anyio.to_thread.run_sync(
                        partial(
                            self._execute_serial,
                            task_ids,
                            trials_per_task,
                            task_results,
                            trial_executor,
                            checkpoint_saver,
                            session_id,
                            interface=as_sync_interface(self.interface),
                        )
                    )
            else:
                await self._arun_independent_trials(
                    task_ids=task_ids,
                    trials_per_task=trials_per_task,
                    task_results=task_results,
                    config=config,
                    atrial_executor=atrial_executor,
                    checkpoint_saver=checkpoint_saver,
                    agent_resolver=agent_resolver,
                    worker_backend_active=worker_backend_active,
                )
            return self._finalize_benchmark(
                task_results=task_results,
                session_id=session_id,
                k_values=k_values,
                verbose=verbose,
                run_name=run_name,
                tool_verbosity=tool_verbosity,
                start_time=start_time,
            )
        finally:
            if self.logger:
                self.logger.finish()

    async def _arun_independent_trials(
        self,
        *,
        task_ids: list[str],
        trials_per_task: int,
        task_results: dict[str, TaskTrialResults],
        config: ConcurrencyConfig,
        atrial_executor: Callable[..., Awaitable[TaskTrialResult]],
        checkpoint_saver: Callable[[dict[str, TaskTrialResults]], None],
        agent_resolver: Callable[[str, int], BaseAgent] | None = None,
        worker_backend_active: bool = False,
    ) -> None:
        """Run independent task/trial pairs concurrently under a global limit.

        Each pending trial becomes a worker that (1) takes the task's lock so
        repeated trials of the same task stay serialised, (2) awaits the async
        `atrial_executor` under a global capacity limiter (agents run natively on
        this loop; a non-native agent's blocking `run()` is offloaded to a worker
        thread inside `arun_agent`), and (3) emits a :class:`TrialCompleted`
        event. A single collector coroutine applies those events — appending
        results, logging, and checkpointing — so all shared state has exactly one
        writer.

        When `config.per_model` caps a model, the worker resolves its agent up
        front (via `agent_resolver`), reads `agent.model`, and takes that model's
        gate *before* the per-task gate and global limiter — so a trial queued on
        its model cap holds neither a per-task nor a global slot — then hands the
        resolved agent to the executor so it is not built twice.
        """
        # Resume-aware: only schedule trials that have not been recorded yet.
        work_items = [
            TrialWorkItem(task_id, trial_index)
            for task_id in task_ids
            for trial_index in range(len(task_results[task_id].trials), trials_per_task)
        ]
        if not work_items:
            logger.info("All requested trials already completed; nothing to run.")
            return

        logger.info(
            f"Running {len(work_items)} trial(s) across {len(task_ids)} task(s) "
            f"with global concurrency {config.global_trials}"
        )

        # Make sure the default worker pool can service every in-flight trial's
        # thread offloads (non-native `run()` + surrender/submit HTTP).
        _ensure_thread_capacity(config.global_trials)

        # Bound how many trials run at once: at most `global_trials` are in
        # flight. A trial blocked on its per-task gate does not hold a slot.
        concurrency_limiter = anyio.CapacityLimiter(config.global_trials)
        # Bound simultaneous trials *of the same task* to `per_task`. At the
        # default (1) this serialises them exactly like a lock; above 1 it lets
        # that many overlap, which is safe only because each such trial runs in
        # its own isolated trial runtime (see `_make_atrial_executor`).
        per_task_gate: dict[str, anyio.Semaphore] = defaultdict(
            lambda: anyio.Semaphore(config.per_task)
        )
        # Optional per-model caps (`config.per_model`): one semaphore per capped
        # model, or `None` when unset. When set, the worker resolves its agent
        # early to learn `agent.model` and gate on it (see below).
        per_model_gate = _build_per_model_gates(config.per_model)
        # Envs that keep process-global state (`"process"`/`"serial"`) must never
        # overlap another globally-stateful trial. One shared `Semaphore(1)`
        # gates them so at most one such trial runs at a time; `"thread"` trials
        # ignore it and overlap freely up to `global_trials`. Empty set (a fake
        # interface or a server without `/concurrency`) means no clamping. When a
        # live worker pool backs this run (`worker_backend_active`), each
        # `"process"` trial is isolated in its own worker process, so it drops out
        # of the serial gate and overlaps too (Phase 3); only `"serial"` envs stay
        # gated. Independent trials each get a fresh worker, so this is safe here.
        serialised_ids = await _serialised_task_ids(
            self.interface, process_isolated=worker_backend_active
        )
        serial_gate = anyio.Semaphore(1)
        send_stream, receive_stream = anyio.create_memory_object_stream(math.inf)

        async def worker(item: TrialWorkItem, send) -> None:
            async with send:
                # When a per-model cap applies, learn this trial's model before
                # taking any slot by resolving the agent now, and reuse that same
                # instance in the executor (never build it twice). Absent a cap,
                # the executor resolves its own agent as before.
                agent: BaseAgent | None = None
                if per_model_gate is not None and agent_resolver is not None:
                    agent = agent_resolver(item.task_id, item.trial_index)
                model_gate = _model_gate(per_model_gate, getattr(agent, "model", None))
                serial_lease = _serial_lease(
                    serial_gate, item.task_id in serialised_ids
                )
                # Ordering: model gate, per-task gate, serial gate, then global
                # limiter — each acquired before the next so a trial queued on the
                # model, per-task, or serial cap holds no global concurrency slot;
                # the global limiter caps how many trials are actually in flight.
                async with (
                    model_gate,
                    per_task_gate[item.task_id],
                    serial_lease,
                    concurrency_limiter,
                ):
                    result = await atrial_executor(
                        item.task_id, item.trial_index, agent=agent
                    )
                await send.send(TrialCompleted(item.task_id, item.trial_index, result))

        async def collector(receive) -> None:
            async for event in receive:
                task_results[event.task_id].trials.append(event.result)
                logger.info(
                    f"Trial {event.result.trial_id} for task {event.task_id} "
                    f"completed. Score: {event.result.score:.3f}"
                )
                if self.logger:
                    self.logger.log_trial(event.result)
                checkpoint_saver(task_results)
                if not event.result.success:
                    logger.error(
                        f"Trial failed for task {event.task_id}: "
                        f"{event.result.error_message}"
                    )

        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(collector, receive_stream)
                # Closing the original send stream (after spawning every worker)
                # lets the collector's `async for` end once all worker clones are
                # closed, i.e. once every trial has been reported.
                async with send_stream:
                    for item in work_items:
                        task_group.start_soon(worker, item, send_stream.clone())
        except BaseException as exc:
            # A BudgetExhaustedError must stop the whole benchmark; anyio wraps
            # child failures in an exception group, so re-raise it bare when found.
            _reraise_budget_exhausted(exc)
            raise

    async def _arun_chained_trials(
        self,
        *,
        task_ids: list[str],
        trials_per_task: int,
        task_results: dict[str, TaskTrialResults],
        config: ConcurrencyConfig,
        node_atrial_executor: Callable[..., Awaitable[TaskTrialResult]],
        checkpoint_saver: Callable[..., None],
        benchmark_run_id: str,
        agent_resolver: Callable[[str, int], BaseAgent] | None = None,
    ) -> None:
        """Run a dependency chain concurrently, one *episode* per trial round.

        Each trial round is an isolated episode whose tasks run as a concurrent
        DAG: sibling nodes run in parallel, a dependent starts only after every
        parent has succeeded, and a broken parent marks its downstream cone
        *unreachable* without invoking the agent. Every node runs in its own
        trial runtime sharing the round's episode store, so nodes never share
        mutable state yet still see their dependencies' outputs. Episodes overlap
        up to the global limit, so even a linear chain pipelines across rounds.

        A single collector reassembles per-round results **in order** and
        checkpoints each fully-completed round, keeping `trials` round-ordered
        for pass@k and resume even though episodes finish out of order.

        When `config.per_model` caps a model, each node resolves its agent up
        front (via `agent_resolver`) and takes that model's gate *before* the
        global limiter, mirroring the independent path; the resolved agent is
        reused by the executor so it is never built twice.
        """
        graph = await acall(self.interface.get_dependency_graph)
        # Same invariants the serial chained path enforces: the selection must be
        # dependency-closed and run in topological order.
        assert_dependencies_selected(task_ids, graph)
        ordered_ids = order_selected(task_ids, graph)
        selected = set(task_ids)
        dep_graph = {
            tid: [dep for dep in graph.get(tid, []) if dep in selected]
            for tid in ordered_ids
        }

        # Resume-aware: a round is complete only when every task has a result for
        # it, so the first not-fully-recorded round is where we pick up.
        completed_rounds = min(
            (len(task_results[tid].trials) for tid in ordered_ids), default=0
        )
        pending_rounds = list(range(completed_rounds, trials_per_task))
        if not pending_rounds:
            logger.info("All requested trial rounds already completed; nothing to run.")
            return

        logger.info(
            f"Running {len(pending_rounds)} episode(s) over a "
            f"{len(ordered_ids)}-task chain with global concurrency "
            f"{config.global_trials}"
        )

        # Make sure the default worker pool can service every in-flight node's
        # thread offloads (non-native `run()` + runtime/submit HTTP).
        _ensure_thread_capacity(config.global_trials)

        # One capacity limiter across *all* episodes: at most `global_trials`
        # agent nodes actually execute at once. A node blocked waiting on its
        # parents holds no concurrency slot.
        concurrency_limiter = anyio.CapacityLimiter(config.global_trials)
        # Optional per-model caps, shared across every episode (see the
        # independent path); `None` when `config.per_model` is unset.
        per_model_gate = _build_per_model_gates(config.per_model)
        # Serialise nodes whose env keeps process-global state so they never
        # overlap another globally-stateful node — across sibling nodes *and*
        # across the overlapping episodes of a pipelined chain (see the
        # independent path). One shared `Semaphore(1)` for the whole run.
        # `process_isolated=False`: a chain pins all of an episode's nodes to a
        # single worker (to share the dependency-output store), so concurrent
        # `"process"` nodes there would share that worker's module globals — the
        # worker backend does *not* isolate them, so they must stay serialised
        # (unlike the independent path, where each trial gets its own worker).
        serialised_ids = await _serialised_task_ids(
            self.interface, process_isolated=False
        )
        serial_gate = anyio.Semaphore(1)
        send_stream, receive_stream = anyio.create_memory_object_stream(math.inf)

        async def node_executor(task_id: str, trial_round: int) -> TaskTrialResult:
            # Resolve the agent early only when a per-model cap applies, so the
            # node can gate on `agent.model` before taking a global slot; the
            # same instance is reused by the executor.
            agent: BaseAgent | None = None
            if per_model_gate is not None and agent_resolver is not None:
                agent = agent_resolver(task_id, trial_round)
            model_gate = _model_gate(per_model_gate, getattr(agent, "model", None))
            serial_lease = _serial_lease(serial_gate, task_id in serialised_ids)
            async with model_gate, serial_lease, concurrency_limiter:
                return await node_atrial_executor(task_id, trial_round, agent=agent)

        async def run_episode(trial_round: int, send) -> None:
            async with send:
                episode_id = self._chained_episode_id(benchmark_run_id, trial_round)
                try:
                    await run_episode_dag(
                        trial_round=trial_round,
                        ordered_ids=ordered_ids,
                        dep_graph=dep_graph,
                        node_executor=node_executor,
                        unreachable_result=unreachable_trial_result,
                        emit=send.send,
                    )
                finally:
                    # Free the round's dependency store now that every node is
                    # done; different rounds never share outputs.
                    await self._aclose_episode_quietly(episode_id)

        expected_per_round = len(ordered_ids)
        round_buffers: dict[int, dict[str, TaskTrialResult]] = defaultdict(dict)
        next_commit = completed_rounds

        async def collector(receive) -> None:
            nonlocal next_commit
            async for event in receive:
                round_buffers[event.trial_round][event.task_id] = event.result
                if self.logger:
                    self.logger.log_trial(event.result)
                if not event.result.success:
                    logger.error(
                        f"Trial failed for task {event.task_id} (round "
                        f"{event.trial_round + 1}): {event.result.error_message}"
                    )
                # Commit rounds strictly in order: a round is appended and
                # checkpointed only once it is complete *and* every earlier round
                # has committed, so out-of-order episode completion never
                # scrambles `trials`.
                while len(round_buffers.get(next_commit, {})) == expected_per_round:
                    finished = round_buffers.pop(next_commit)
                    for tid in ordered_ids:
                        result = finished[tid]
                        task_results[tid].trials.append(result)
                        logger.info(
                            f"Trial {result.trial_id} for task {tid} (round "
                            f"{next_commit + 1}) completed. Score: {result.score:.3f}"
                        )
                    checkpoint_saver(task_results, completed_trials=next_commit + 1)
                    next_commit += 1

        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(collector, receive_stream)
                # Closing the original send stream (after spawning every episode)
                # lets the collector's `async for` end once all episode clones are
                # closed, i.e. once every round has been fully reported.
                async with send_stream:
                    for trial_round in pending_rounds:
                        task_group.start_soon(
                            run_episode, trial_round, send_stream.clone()
                        )
        except BaseException as exc:
            _reraise_budget_exhausted(exc)
            raise

    @staticmethod
    def _load_traces_from_agent_logs(
        directory: Path, trial_index: int
    ) -> dict[str, list[LiteLLMMessage]]:
        """Extract traces from a directory of agent log JSON files."""
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {directory}")

        # Group files by task_id
        task_files: dict[str, list[Path]] = {}
        for file_path in json_files:
            try:
                with file_path.open() as f:
                    data = json.load(f)
                task_id = data.get("task_id")
                if task_id is None:
                    logger.warning(f"Skipping {file_path.name}: no 'task_id' field.")
                    continue
                task_files.setdefault(task_id, []).append(file_path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {file_path.name}: {e}")

        if not task_files:
            raise ValueError(
                f"No valid agent log files found in directory: {directory}"
            )

        traces: dict[str, list[LiteLLMMessage]] = {}

        for task_id, files in task_files.items():
            if trial_index >= len(files):
                logger.warning(
                    f"Task '{task_id}' has only {len(files)} log file(s), "
                    f"skipping (requested trial_index={trial_index})."
                )
                continue

            selected_file = files[trial_index]
            with selected_file.open() as f:
                data = json.load(f)

            messages = data.get("messages")
            if not messages:
                logger.warning(
                    f"Task '{task_id}' log {selected_file.name} has no messages."
                )
                continue

            traces[task_id] = [LiteLLMMessage(**msg) for msg in messages]

        if not traces:
            raise ValueError(
                f"No traces could be extracted from agent logs in: {directory}"
            )

        return traces

    def _load_or_initialize_results(
        self, task_ids: list[str], session_id: str
    ) -> dict[str, TaskTrialResults]:
        """Load existing results or initialize new ones"""
        checkpoint = self._load_checkpoint(session_id)

        if checkpoint and "task_results" in checkpoint:
            task_results = checkpoint["task_results"]
            logger.info(f"Loaded existing results for {len(task_results)} tasks")

            # Ensure all requested tasks have entries
            for task_id in task_ids:
                if task_id not in task_results:
                    task_results[task_id] = TaskTrialResults(task_id=task_id)
        else:
            task_results = initialize_task_results(task_ids)

        return task_results

    def _make_logging_trial_executor(self, trial_executor: Callable) -> Callable:
        """Wrap trial executor with logging"""

        def wrapped_executor(task_id: str, trial_index: int) -> TaskTrialResult:
            result = trial_executor(task_id, trial_index)
            if self.logger:
                self.logger.log_trial(result)
            return result

        return wrapped_executor

    def _make_chained_checkpoint_saver(self, checkpoint_saver: Callable) -> Callable:
        """Create checkpoint saver for chained execution"""

        def wrapped_saver(
            task_results: dict[str, TaskTrialResults], completed_trials: int
        ) -> None:
            checkpoint_saver(task_results, completed_trials=completed_trials)

        return wrapped_saver

    def _write_checkpoint_file(
        self,
        checkpoint_path: Path,
        checkpoint: dict,
        session_id: str,
        checkpoint_type: str = "checkpoint",
    ) -> None:
        """
        Write checkpoint data to file using an atomic write pattern.

        This method first writes the checkpoint to a temporary file and then renames it to the target path.
        This approach ensures that the checkpoint file is never left in a partially written or corrupted state,
        even if the process crashes or is interrupted during the write. The atomic rename operation guarantees
        that readers will either see the old file or the fully written new file, but never a half-written file.
        If an error occurs, the temporary file is cleaned up to avoid clutter.
        """
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            with temp_path.open("wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(checkpoint_path)
            logger.info(f"{checkpoint_type.title()} saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save {checkpoint_type}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _save_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults], **extra_data
    ) -> None:
        """
        Save checkpoint to file, including all relevant session and task state.

        This method centralizes the logic for checkpoint creation, ensuring that all necessary
        metadata (such as session ID and timestamp) is included. It delegates the actual file
        writing to an atomic method to guarantee data integrity. This design allows for robust
        recovery and resumption of long-running or multi-step processes.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **extra_data,
        }

        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            checkpoint_path, checkpoint, session_id, "checkpoint"
        )

    def _load_checkpoint(self, session_id: str) -> dict | None:
        """
        Load checkpoint from file, or recover the most recent one if not found.

        This method attempts to load a checkpoint for the given session. If the specific
        checkpoint file does not exist (e.g., due to interruption or cleanup), it searches
        for the most recent available checkpoint with the same naming pattern. This design
        increases robustness and allows for recovery from unexpected interruptions or missing files.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        if not checkpoint_path.exists():
            # Search for the most recent checkpoint with same checkpoint_name
            return self._find_most_recent_checkpoint()

        try:
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info("Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            return None

    def _find_most_recent_checkpoint(self) -> dict | None:
        """
        Find and load the most recent checkpoint file with the same checkpoint_name.

        This method scans the checkpoint directory for files matching the session pattern,
        excluding those marked as finished. It sorts the files by timestamp and loads the most
        recent one. This enables recovery from interruptions and ensures that progress is not lost
        if the latest checkpoint file is missing or incomplete. It is a fallback mechanism for robust
        checkpoint management.
        """

        # Pattern to match checkpoint files with same name but different session IDs
        # excluding "finished" files
        pattern = f"{self.checkpoint_name}_session_*.pkl"

        matching_files = []
        for file_path in self.checkpoint_dir.glob(pattern):
            file_name = file_path.name
            # Skip finished checkpoints
            if "finished" in file_name:
                continue

            # Extract session ID from filename
            match = re.search(r"session_(\d{8}_\d{6}_\d{6})", file_name)
            if match:
                session_timestamp = match.group(1)
                matching_files.append((file_path, session_timestamp))

        if not matching_files:
            logger.info("No existing checkpoints found")
            return None

        # Sort by session timestamp (most recent first)
        matching_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_file = matching_files[0][0]

        try:
            with most_recent_file.open("rb") as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded most recent checkpoint: {most_recent_file.name}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Error loading most recent checkpoint: {e}")
            return None

    def _save_finished_checkpoint(
        self, session_id: str, task_results: dict[str, TaskTrialResults]
    ) -> None:
        """
        Save the final checkpoint with a 'finished' suffix to mark completion.

        This method creates a checkpoint file that is clearly marked as finished, making it easy
        to distinguish between in-progress and completed runs. This helps prevent accidental
        resumption of already completed sessions and provides a clear audit trail for completed
        benchmarks. The atomic write pattern is used for reliability.
        """
        checkpoint = {
            "task_results": task_results,
            "session_id": session_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "finished",
        }

        finished_checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_finished_{session_id}.pkl"
        )

        self._write_checkpoint_file(
            finished_checkpoint_path, checkpoint, session_id, "final checkpoint"
        )

    def _remove_original_checkpoint(self, session_id: str) -> None:
        """
        Remove the original (unfinished) checkpoint file after completion.

        This method deletes the in-progress checkpoint file once a finished checkpoint has been
        written. This prevents confusion between incomplete and completed runs, and helps keep
        the checkpoint directory clean. The try/except block ensures that errors during removal
        do not interrupt the main workflow.
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.checkpoint_name}_{session_id}.pkl"
        )

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Original checkpoint removed for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to remove original checkpoint: {e}")

    def _save_benchmark_report(
        self, result: BenchmarkResult, run_name: str | None, tool_verbosity: str | None
    ) -> None:
        """
        Save benchmark report to a JSON file.

        If no run name is provided, generates a filename in the format:
        {model}-{agent}-{env}-{verbosity}-{timestamp}.json

        Args:
            result: The BenchmarkResult to save
            run_name: Name for the benchmark run. If None, uses "unknown_env"
            tool_verbosity: The verbosity level used (or None)
        """
        if run_name is not None:
            filename = run_name if run_name.endswith(".json") else f"{run_name}.json"
        else:
            # Extract components for filename (the shared agent may be absent
            # when the runner is factory-driven).
            model_name = getattr(self.agent, "model", None) or "unknown_model"
            # Sanitize model name for filesystem safety
            model_name = sanitize_model_name(model_name)

            agent_name = self._agent_label()

            verbosity_str = tool_verbosity if tool_verbosity else "None"

            # Add timestamp
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Construct filename
            filename = f"{model_name}-{agent_name}-{verbosity_str}-{timestamp}.json"

        # Save the report
        try:
            result.generate_report(filename)
            logger.info(f"Benchmark report saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")
