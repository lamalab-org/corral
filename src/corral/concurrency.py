from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.base_agent import BaseAgent
    from corral.report.results import TaskTrialResult

# Mirror of `corral.backend.executors.DEFAULT_JOB_CONCURRENCY` (the server's
# per-trial background-job default). Kept as a local literal so this deliberately
# light module stays free of backend imports; both are `4`, so an un-tuned run
# forwards exactly the value the server would have defaulted to and its
# behaviour is unchanged.
DEFAULT_TOOL_JOBS_PER_TRIAL = 4


@dataclass(frozen=True)
class TrialContext:
    """Immutable description of the trial an :class:`AgentFactory` is building for.

    Passed to the factory so it can mint an agent bound to this specific trial —
    e.g. select a per-trial MCP URL, workspace, or run/session tag — without any
    shared mutable state leaking between concurrent trials.

    Attributes:
        task_id: The task this trial belongs to.
        trial_index: Zero-based index of this trial within the task.
        session_id: The benchmark session id (checkpoint namespace).
        benchmark_run_id: Identifier for the whole benchmark invocation.
    """

    task_id: str
    trial_index: int
    session_id: str
    benchmark_run_id: str


@runtime_checkable
class AgentFactory(Protocol):
    """Builds a fresh :class:`~corral.agents.base_agent.BaseAgent` per trial.

    A single `BaseAgent` accumulates per-run state (messages, token usage,
    hooks, harness/SDK sessions), so it cannot be shared across trials that run
    at the same time. Concurrent benchmarking therefore takes a *factory* and
    calls it once per trial, giving every trial an isolated agent.
    """

    def __call__(self, context: TrialContext) -> BaseAgent: ...


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Bounds on how many trials — and how much per-trial work — run at once.

    `global_trials` and `per_task` are the two limits the async scheduler has
    always enforced directly; the remaining fields are the richer limits Section
    3 of `make_efficiency.md` calls for, modelled here so a run can express its
    full concurrency intent in one value object. All of them are now wired:

    * `tool_jobs_per_trial` is forwarded through `create_trial` over HTTP so
      each trial runtime caps its own background-job pool at this value.
    * `per_model` gates simultaneous trials by their agent's model: the
      concurrent scheduler resolves each trial's agent up front, reads its
      `model`, and acquires that model's semaphore *before* the global
      limiter (so a trial queued on its model cap holds no global slot). A model
      absent from the map is bounded only by `global_trials`.
    * `configure_apps` is a benchmark-wide lease held only around each trial's
      app-configure step, bounding how many trials configure additional
      apps/services at once.

    `per_model` and `configure_apps` are enforced on the **async** path
    (`abench`, and `bench` when the effective concurrency exceeds `1`); the
    byte-for-byte serial path runs one trial at a time and so needs neither.

    Attributes:
        global_trials: Maximum number of trials executing simultaneously across
            the whole benchmark.
        per_task: Maximum number of simultaneous trials **of the same task**.
            Values above `1` require the server to support *trial runtimes*
            (isolated per-trial environments); the runner then routes each
            trial through its own runtime so repeated trials of one task no
            longer share mutable server state. Bounded in practice by
            `global_trials`.
        per_model: Optional per-model cap on simultaneous trials, keyed by model
            name (e.g. `{"claude-opus": 8, "gpt-5.6": 16}`) — the plan's
            provider/rate-limit lever. Each cap must be `>= 1`. Enforced on the
            concurrent path by gating each trial on its agent's `model`.
        tool_jobs_per_trial: Maximum background jobs running at once **within a
            single trial runtime**. Forwarded through `create_trial` to the
            runtime's `JobManager` (via `Environment.for_trial`). Defaults to
            :data:`DEFAULT_TOOL_JOBS_PER_TRIAL`, matching the server's historical
            default so an un-tuned run is unchanged.
        configure_apps: Optional cap on how many trials may configure their
            additional apps/services simultaneously — the plan's fixed-port /
            expensive-setup lever. Must be `>= 1` when set. Enforced on the
            concurrent path as a benchmark-wide lease around the configure step.
    """

    global_trials: int = 1
    per_task: int = 1
    per_model: Mapping[str, int] | None = None
    tool_jobs_per_trial: int = DEFAULT_TOOL_JOBS_PER_TRIAL
    configure_apps: int | None = None

    def __post_init__(self) -> None:
        if self.global_trials < 1:
            raise ValueError(f"global_trials must be >= 1, got {self.global_trials}")
        if self.per_task < 1:
            raise ValueError(f"per_task must be >= 1, got {self.per_task}")
        if self.tool_jobs_per_trial < 1:
            raise ValueError(
                f"tool_jobs_per_trial must be >= 1, got {self.tool_jobs_per_trial}"
            )
        if self.configure_apps is not None and self.configure_apps < 1:
            raise ValueError(f"configure_apps must be >= 1, got {self.configure_apps}")
        if self.per_model is not None:
            for model, limit in self.per_model.items():
                if limit < 1:
                    raise ValueError(f"per_model[{model!r}] must be >= 1, got {limit}")


@dataclass(frozen=True)
class TrialWorkItem:
    """One unit of work for the scheduler: a single trial of a single task."""

    task_id: str
    trial_index: int


@dataclass(frozen=True)
class TrialCompleted:
    """Event emitted by a worker once a trial finishes.

    Workers never touch shared result/checkpoint state directly; they emit this
    event and a single collector coroutine applies it, keeping aggregation and
    checkpointing deterministic and lock-free.
    """

    task_id: str
    trial_index: int
    result: TaskTrialResult


@dataclass(frozen=True)
class EpisodeContext:
    """Immutable identity of one *episode*: a single trial round of a chain.

    Independent tasks make each trial its own episode; a dependency chain runs
    all of its tasks for trial round `trial_round` inside one episode. Every
    trial runtime opened with this episode's `episode_id` shares one
    server-side dependency-output store, so a downstream task reads its upstream
    siblings' outputs — while different rounds get different episode ids and so
    never share outputs.

    Attributes:
        episode_id: Stable id for the episode (the server's store key).
        trial_round: Zero-based trial round this episode represents.
        benchmark_run_id: Identifier for the whole benchmark invocation.
    """

    episode_id: str
    trial_round: int
    benchmark_run_id: str


@dataclass(frozen=True)
class EpisodeTaskCompleted:
    """Event emitted once one DAG node (task) of an episode finishes.

    Like :class:`TrialCompleted` but tagged with the `trial_round` so a single
    collector can reassemble per-round results in order even when episodes and
    their sibling nodes complete concurrently and out of order.
    """

    trial_round: int
    task_id: str
    result: TaskTrialResult
