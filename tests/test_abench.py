"""Tests for concurrent benchmarking (`abench` / `bench(max_concurrency>1)`).

These cover the first parallelism stage from `make_efficiency.md`:

* independent task/trial pairs run concurrently, bounded by a global limit;
* repeated trials of the *same* task stay serialised;
* every concurrent trial gets its **own** agent (via an `agent_factory`);
* a single collector still records every trial and checkpoints deterministically;
* resume-awareness skips already-recorded trials;
* concurrency without a factory fails fast with a clear error.

The whole flow is driven through the synchronous `bench(max_concurrency=...)`
entry point (which internally runs `abench` via `anyio.run`), so the tests
need no async test runner. Fakes stand in for the router and agents; the agents
run in worker threads, so a small `ConcurrencyTracker` can observe real
overlap.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial

import anyio
import pytest

from corral.agents.base_agent import BaseAgent
from corral.agents.schema import AgentRunResult
from corral.concurrency import ConcurrencyConfig, TrialContext
from corral.report import TaskTrialResult, TaskTrialResults
from corral.router import AsyncCorralRouter, CorralRouter
from corral.run import CorralRunner


class ConcurrencyTracker:
    """Records max simultaneous agents overall and per task (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self._per_task = defaultdict(int)
        self.max_per_task = 0
        # Peak simultaneous holders *per key* (unlike `max_per_task`, which is
        # the max across all keys). Used to check a per-model cap serialises one
        # model while another overlaps under the same global limit.
        self.max_by_key = defaultdict(int)

    @contextmanager
    def track(self, task_id: str):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self._per_task[task_id] += 1
            self.max_per_task = max(self.max_per_task, self._per_task[task_id])
            self.max_by_key[task_id] = max(
                self.max_by_key[task_id], self._per_task[task_id]
            )
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1
                self._per_task[task_id] -= 1


class TrackingAgent:
    """Minimal agent: sleeps (to expose overlap) then returns a fixed answer.

    The blocking `time.sleep` runs in a worker thread on every path — the sync
    scheduler calls `run_agent` in a thread, and the async scheduler awaits
    `arun_agent`, whose default here offloads `run_agent` to a thread (exactly as
    a non-native `BaseAgent` does). So real overlap is observable either way.
    """

    def __init__(self, tracker: ConcurrencyTracker, sleep: float = 0.02):
        self._tracker = tracker
        self._sleep = sleep

    def run_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        with self._tracker.track(task_id):
            time.sleep(self._sleep)
        return AgentRunResult(answer="42", status="success")

    async def arun_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        return await anyio.to_thread.run_sync(
            partial(self.run_agent, interface, task_id, **kwargs)
        )

    def get_total_token_usage(self) -> dict:
        return {}


class FakeRouter:
    """A stand-in CorralRouter recording configure/submit calls thread-safely."""

    def __init__(self, chained: bool = False):
        self.current_verbosity = "brief"
        self._chained = chained
        self._lock = threading.Lock()
        self.configure_calls: list[str] = []
        self.submit_calls: list[tuple[str, str]] = []
        # Trial-runtime bookkeeping (only exercised at per_task > 1).
        self.created_trials: list[str] = []
        self.created_trial_job_limits: list[int | None] = []
        self.closed_trials: list[str] = []
        self.trial_submit_calls: list[tuple[str, str, str]] = []
        # Verbosity each `for_trial` view was bound to (checks the runner binds
        # the run's verbosity into the trial router instead of a shared field).
        self.for_trial_verbosities: list[str | None] = []
        # Per-task env concurrency modes the scheduler reads to serialise
        # process-global envs; empty means every task is the default "thread".
        self.concurrency_modes: dict[str, str] = {}
        # Whether the server has a live process-worker pool (Phase 3). Off by
        # default, so "process" envs serialise exactly as in Phase 0; a test
        # flips it on to exercise the worker-isolated overlap path.
        self.worker_backend_active: bool = False

    def set_verbosity(self, verbosity: str) -> None:
        self.current_verbosity = verbosity

    def supports_dependency_chain(self) -> bool:
        return self._chained

    def get_available_tasks(self) -> list[str]:
        return []

    def get_concurrency_modes(self) -> dict[str, str]:
        return dict(self.concurrency_modes)

    def get_trial_worker_active(self) -> bool:
        return self.worker_backend_active

    def configure_additional_apps(self, task_id, timeout=None) -> str:
        with self._lock:
            self.configure_calls.append(task_id)
        return "configured"

    def get_task_status(self, task_id) -> dict:
        return {"score": 0.0}

    def submit_answer(self, task_id, answer) -> TaskTrialResult:
        with self._lock:
            self.submit_calls.append((task_id, answer))
        return TaskTrialResult(
            task_id=task_id,
            trial_id="attempt_1",
            score=1.0,
            state={},
            tool_statistics={},
        )

    def surrender_task(self, task_id) -> TaskTrialResult:
        return TaskTrialResult(
            task_id=task_id,
            trial_id="attempt_1",
            score=0.0,
            state={},
            tool_statistics={},
            surrendered=True,
        )

    def create_trial(
        self,
        task_id,
        benchmark_run_id=None,
        episode_id=None,
        trial_index=None,
        tool_jobs_per_trial=None,
    ) -> dict:
        with self._lock:
            self.created_trials.append(task_id)
            self.created_trial_job_limits.append(tool_jobs_per_trial)
            rid = f"tr_{task_id}_{len(self.created_trials)}"
        return {
            "trial_runtime_id": rid,
            "task_id": task_id,
            "workspace": None,
            "mcp_url": f"/trials/{rid}/mcp",
        }

    def for_trial(
        self, trial_runtime_id, task_id, *, verbosity=None, workspace=None
    ) -> "_FakeTrialRouter":
        with self._lock:
            self.for_trial_verbosities.append(verbosity)
        return _FakeTrialRouter(self, trial_runtime_id, task_id, verbosity=verbosity)

    def close_trial(self, trial_runtime_id) -> None:
        with self._lock:
            self.closed_trials.append(trial_runtime_id)


class _FakeTrialRouter:
    """Stand-in for TrialScopedRouter: records which runtime each call hit."""

    def __init__(
        self, parent: FakeRouter, trial_runtime_id: str, task_id: str, *, verbosity=None
    ):
        self._parent = parent
        self.trial_runtime_id = trial_runtime_id
        self.task_id = task_id
        # Mirror the real router: a bound verbosity overrides the parent's.
        self.current_verbosity = verbosity or parent.current_verbosity

    def set_verbosity(self, verbosity: str) -> None:
        self.current_verbosity = verbosity

    def configure_additional_apps(self, task_id=None, timeout=None) -> str:
        with self._parent._lock:
            self._parent.configure_calls.append(self.task_id)
        return "configured"

    def get_task_status(self, task_id=None) -> dict:
        return {"score": 0.0}

    def submit_answer(self, task_id, answer) -> TaskTrialResult:
        with self._parent._lock:
            self._parent.trial_submit_calls.append(
                (self.task_id, answer, self.trial_runtime_id)
            )
        return TaskTrialResult(
            task_id=self.task_id,
            trial_id=self.trial_runtime_id,
            score=1.0,
            state={},
            tool_statistics={},
        )

    def surrender_task(self, task_id=None) -> TaskTrialResult:
        return TaskTrialResult(
            task_id=self.task_id,
            trial_id=self.trial_runtime_id,
            score=0.0,
            state={},
            tool_statistics={},
            surrendered=True,
        )


def _make_runner(tmp_path, tracker, *, sleep=0.02, record_agents=None):
    interface = FakeRouter()

    def factory(context: TrialContext) -> TrackingAgent:
        agent = TrackingAgent(tracker, sleep=sleep)
        if record_agents is not None:
            # Record the agent object itself, not `id(agent)`: a trial's agent
            # becomes unreferenced once the trial ends, so CPython can recycle its
            # `id()` for a later agent and make distinct instances look identical.
            # Holding the reference keeps every id live and the identity check exact.
            record_agents.append((context.task_id, context.trial_index, agent))
        return agent

    runner = CorralRunner(
        interface, agent_factory=factory, checkpoint_dir=str(tmp_path)
    )
    return runner, interface


def test_concurrency_config_defaults_are_serial():
    config = ConcurrencyConfig()
    assert config.global_trials == 1
    assert config.per_task == 1


def test_concurrency_config_rejects_zero_global():
    with pytest.raises(ValueError, match="global_trials"):
        ConcurrencyConfig(global_trials=0)


def test_concurrency_config_allows_per_task_gt_one():
    # Repeated trials of one task now overlap through isolated trial runtimes.
    config = ConcurrencyConfig(global_trials=4, per_task=2)
    assert config.per_task == 2


def test_concurrency_config_rejects_zero_per_task():
    with pytest.raises(ValueError, match="per_task"):
        ConcurrencyConfig(per_task=0)


def test_concurrency_config_models_richer_limits():
    # Section 3 fields are modelled on the value object and all enforced:
    # tool_jobs_per_trial over create_trial, per_model + configure_apps on the
    # concurrent scheduler.
    config = ConcurrencyConfig(
        global_trials=16,
        per_task=4,
        per_model={"claude-opus": 8, "gpt-5.6": 16},
        tool_jobs_per_trial=6,
        configure_apps=2,
    )
    assert config.tool_jobs_per_trial == 6
    assert config.configure_apps == 2
    assert config.per_model == {"claude-opus": 8, "gpt-5.6": 16}


def test_concurrency_config_tool_jobs_defaults_to_server_default():
    from corral.concurrency import DEFAULT_TOOL_JOBS_PER_TRIAL

    # Left un-tuned, the field carries the server's own historical default, so an
    # existing run forwards exactly what the server would have used anyway.
    assert ConcurrencyConfig().tool_jobs_per_trial == DEFAULT_TOOL_JOBS_PER_TRIAL


def test_concurrency_config_rejects_zero_tool_jobs():
    with pytest.raises(ValueError, match="tool_jobs_per_trial"):
        ConcurrencyConfig(tool_jobs_per_trial=0)


def test_concurrency_config_rejects_zero_configure_apps():
    with pytest.raises(ValueError, match="configure_apps"):
        ConcurrencyConfig(configure_apps=0)


def test_concurrency_config_rejects_zero_per_model_limit():
    with pytest.raises(ValueError, match="per_model"):
        ConcurrencyConfig(per_model={"claude-opus": 0})


def test_bench_concurrent_runs_every_trial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker)

    result = runner.bench(
        task_ids=["a", "b", "c"],
        trials_per_task=2,
        max_concurrency=4,
        run_name="report",
    )

    total_trials = sum(len(r.trials) for r in result.task_results.values())
    assert total_trials == 6
    assert len(interface.submit_calls) == 6
    assert all(t.score == 1.0 for r in result.task_results.values() for t in r.trials)


def test_bench_concurrency_is_bounded_and_parallel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, _ = _make_runner(tmp_path, tracker, sleep=0.05)

    runner.bench(
        task_ids=[f"task_{i}" for i in range(8)],
        trials_per_task=1,
        max_concurrency=3,
        run_name="report",
    )

    # Never exceeds the global limit, but genuinely overlaps (proves parallelism).
    assert tracker.max_active <= 3
    assert tracker.max_active >= 2


def test_same_task_trials_are_serialised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.03)

    result = runner.bench(
        task_ids=["solo"],
        trials_per_task=3,
        max_concurrency=4,
        run_name="report",
    )

    # Even with slack in the global limit, one task never runs two trials at once.
    assert tracker.max_per_task == 1
    assert len(result.task_results["solo"].trials) == 3
    assert len(interface.submit_calls) == 3


def test_per_task_trials_overlap_through_runtimes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.05)

    result = runner.bench(
        task_ids=["solo"],
        trials_per_task=3,
        max_concurrency=3,
        max_concurrency_per_task=2,
        run_name="report",
    )

    # per_task=2 lets two trials of the same task overlap (but never three).
    assert tracker.max_per_task == 2
    # Each trial ran in its *own* isolated runtime, created and then closed.
    assert len(interface.created_trials) == 3
    assert len(interface.closed_trials) == 3
    # Every trial submitted against a distinct runtime (no shared server state).
    assert len({rid for _, _, rid in interface.trial_submit_calls}) == 3
    assert len(result.task_results["solo"].trials) == 3


@pytest.mark.parametrize("mode", ["process", "serial"])
def test_serialised_env_trials_never_overlap(tmp_path, monkeypatch, mode):
    """A "process"/"serial" env is serialised even with global + per_task slack.

    Phase 0 guardrail: an env that keeps process-global state runs one trial at
    a time, overriding the requested `per_task=2`, so it can never be corrupted
    by an overlapping trial until the worker backend lands.
    """
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.03)
    interface.concurrency_modes = {"wet": mode}

    result = runner.bench(
        task_ids=["wet"],
        trials_per_task=3,
        max_concurrency=3,
        max_concurrency_per_task=2,
        run_name="report",
    )

    assert tracker.max_per_task == 1
    assert tracker.max_active == 1
    assert len(result.task_results["wet"].trials) == 3


def test_serialised_envs_do_not_overlap_across_tasks(tmp_path, monkeypatch):
    """Different process-stateful tasks share module globals, so *no* two of
    their trials may run at once — the exact wetlab cross-task corruption."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.03)
    interface.concurrency_modes = {"wet_a": "process", "wet_b": "process"}

    runner.bench(
        task_ids=["wet_a", "wet_b"],
        trials_per_task=2,
        max_concurrency=4,
        run_name="report",
    )

    # One shared serial gate spans every serialised task, so even wet_a vs wet_b
    # never overlap despite plenty of global concurrency.
    assert tracker.max_active == 1


def test_serial_gate_leaves_thread_envs_parallel(tmp_path, monkeypatch):
    """The guardrail only clamps serialised trials against each other; "thread"
    envs (absent from the map) still overlap up to the global limit."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.05)
    interface.concurrency_modes = {"wet": "process"}  # only "wet" is serialised

    runner.bench(
        task_ids=["wet", "safe_1", "safe_2", "safe_3"],
        trials_per_task=1,
        max_concurrency=3,
        run_name="report",
    )

    # The three "thread" tasks genuinely overlap even though "wet" is serialised.
    assert tracker.max_active >= 2


def test_process_envs_overlap_when_worker_pool_active(tmp_path, monkeypatch):
    """With a live worker pool, independent "process" trials run in their own
    worker processes, so the scheduler lets them overlap instead of serialising.

    Two different process-stateful tasks that the Phase-0 guardrail keeps strictly
    serialised (`test_serialised_envs_do_not_overlap_across_tasks`) now overlap
    once the server advertises a worker backend.
    """
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.05)
    interface.concurrency_modes = {"wet_a": "process", "wet_b": "process"}
    interface.worker_backend_active = True

    runner.bench(
        task_ids=["wet_a", "wet_b"],
        trials_per_task=1,
        max_concurrency=2,
        run_name="report",
    )

    # They overlap (no serial gate), and each was routed through a trial runtime
    # so the server could hand it to a worker process (use_runtimes was forced).
    assert tracker.max_active == 2
    assert sorted(interface.created_trials) == ["wet_a", "wet_b"]


def test_process_envs_serialised_without_worker_pool(tmp_path, monkeypatch):
    """No worker pool (default) -> a "process" env is still serialised and runs
    in-process (no trial runtimes), i.e. the Phase-0 fallback is untouched."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.03)
    interface.concurrency_modes = {"wet_a": "process", "wet_b": "process"}
    # worker_backend_active stays False.

    runner.bench(
        task_ids=["wet_a", "wet_b"],
        trials_per_task=1,
        max_concurrency=2,
        run_name="report",
    )

    assert tracker.max_active == 1
    # Serial in-process path: trials hit /tasks/{id}, not per-trial runtimes.
    assert interface.created_trials == []


def test_serial_envs_stay_serialised_even_with_worker_pool(tmp_path, monkeypatch):
    """A "serial" env never overlaps anything, even when a worker pool is live —
    it is the explicit escape hatch, distinct from worker-isolatable "process"."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.05)
    interface.concurrency_modes = {"ser_a": "serial", "ser_b": "serial"}
    interface.worker_backend_active = True

    runner.bench(
        task_ids=["ser_a", "ser_b"],
        trials_per_task=1,
        max_concurrency=2,
        run_name="report",
    )

    assert tracker.max_active == 1


def test_tool_jobs_per_trial_forwarded_to_create_trial(tmp_path, monkeypatch):
    """A ConcurrencyConfig.tool_jobs_per_trial reaches each create_trial call.

    Section 3 wiring: the runner forwards the per-trial background-job limit over
    HTTP so the server sizes each runtime's JobManager, instead of the fixed
    server-side default.
    """
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.0)

    runner.bench(
        task_ids=["solo"],
        trials_per_task=2,
        concurrency=ConcurrencyConfig(
            global_trials=2, per_task=2, tool_jobs_per_trial=7
        ),
        run_name="report",
    )

    assert len(interface.created_trials) == 2
    assert interface.created_trial_job_limits == [7, 7]


def test_runner_level_concurrency_config_is_used(tmp_path, monkeypatch):
    """A config on the CorralRunner drives concurrency without call-site scalars."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    interface = FakeRouter()

    def factory(context: TrialContext) -> TrackingAgent:
        return TrackingAgent(tracker, sleep=0.0)

    runner = CorralRunner(
        interface,
        agent_factory=factory,
        checkpoint_dir=str(tmp_path),
        concurrency=ConcurrencyConfig(
            global_trials=3, per_task=2, tool_jobs_per_trial=5
        ),
    )

    # No max_concurrency* on the call — the runner's config alone selects the
    # concurrent path and its tool_jobs_per_trial is forwarded.
    runner.bench(task_ids=["solo"], trials_per_task=2, run_name="report")

    assert len(interface.created_trials) == 2
    assert interface.created_trial_job_limits == [5, 5]


class ModeledAgent(TrackingAgent):
    """A TrackingAgent carrying a `model` and tracking concurrency by model.

    The concurrent scheduler reads `agent.model` to apply a per-model cap;
    tracking by model (not task) lets a test see one model serialised while
    another overlaps under the same global limit.
    """

    def __init__(self, tracker, model: str, sleep: float = 0.05):
        super().__init__(tracker, sleep=sleep)
        self.model = model

    def run_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        with self._tracker.track(self.model):
            time.sleep(self._sleep)
        return AgentRunResult(answer="42", status="success")


def test_per_model_caps_simultaneous_trials_by_model(tmp_path, monkeypatch):
    """`per_model` serialises a capped model while others overlap freely.

    Two tasks use the capped model, two use an uncapped one, all four under a
    global limit of 4. The cap (1) must keep the capped model to one trial at a
    time; the uncapped model, gated only by the global limiter, overlaps — proof
    the gate is model-specific, not just a global throttle.
    """
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    interface = FakeRouter()
    models = {"a": "capped", "b": "capped", "c": "free", "d": "free"}

    def factory(context: TrialContext) -> ModeledAgent:
        return ModeledAgent(tracker, models[context.task_id], sleep=0.05)

    runner = CorralRunner(
        interface, agent_factory=factory, checkpoint_dir=str(tmp_path)
    )
    result = runner.bench(
        task_ids=["a", "b", "c", "d"],
        trials_per_task=1,
        concurrency=ConcurrencyConfig(global_trials=4, per_model={"capped": 1}),
        run_name="report",
    )

    # The capped model never runs two trials at once...
    assert tracker.max_by_key["capped"] == 1
    # ...while the uncapped model genuinely overlaps under the same global limit.
    assert tracker.max_by_key["free"] == 2
    # All four trials still completed and submitted.
    total = sum(len(r.trials) for r in result.task_results.values())
    assert total == 4
    assert len(interface.submit_calls) == 4


def test_per_model_absent_model_bounded_only_by_global_limit(tmp_path, monkeypatch):
    """A model missing from `per_model` gets no extra cap (only the global one)."""
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    interface = FakeRouter()

    def factory(context: TrialContext) -> ModeledAgent:
        # Every trial uses "other", which is *not* in the per_model map.
        return ModeledAgent(tracker, "other", sleep=0.05)

    runner = CorralRunner(
        interface, agent_factory=factory, checkpoint_dir=str(tmp_path)
    )
    runner.bench(
        task_ids=[f"t{i}" for i in range(4)],
        trials_per_task=1,
        concurrency=ConcurrencyConfig(global_trials=3, per_model={"capped": 1}),
        run_name="report",
    )

    # "other" is uncapped, so it overlaps right up to the global limit of 3.
    assert tracker.max_by_key["other"] == 3


class ConfigureCountingRouter(FakeRouter):
    """FakeRouter whose configure step sleeps and records its own concurrency."""

    def __init__(self, configure_tracker: ConcurrencyTracker, configure_sleep=0.05):
        super().__init__()
        self._configure_tracker = configure_tracker
        self._configure_sleep = configure_sleep

    def configure_additional_apps(self, task_id, timeout=None) -> str:
        with self._configure_tracker.track("configure"):
            time.sleep(self._configure_sleep)
        with self._lock:
            self.configure_calls.append(task_id)
        return "configured"


def test_configure_apps_caps_concurrent_configure_steps(tmp_path, monkeypatch):
    """`configure_apps` bounds how many trials configure apps at once.

    Six trials run under a global limit of 6, so without the lease all six would
    configure simultaneously. With `configure_apps=2` the configure step never
    has more than two trials in it, yet all six still complete.
    """
    monkeypatch.chdir(tmp_path)
    agent_tracker = ConcurrencyTracker()
    configure_tracker = ConcurrencyTracker()
    interface = ConfigureCountingRouter(configure_tracker, configure_sleep=0.05)

    def factory(context: TrialContext) -> TrackingAgent:
        return TrackingAgent(agent_tracker, sleep=0.005)

    runner = CorralRunner(
        interface, agent_factory=factory, checkpoint_dir=str(tmp_path)
    )
    result = runner.bench(
        task_ids=[f"t{i}" for i in range(6)],
        trials_per_task=1,
        concurrency=ConcurrencyConfig(global_trials=6, configure_apps=2),
        run_name="report",
    )

    # The lease holds: never more than two configure steps in flight...
    assert configure_tracker.max_active <= 2
    # ...but it genuinely allows the two it permits to overlap.
    assert configure_tracker.max_active == 2
    total = sum(len(r.trials) for r in result.task_results.values())
    assert total == 6


def test_configure_apps_unset_leaves_configure_unbounded(tmp_path, monkeypatch):
    """Without `configure_apps` the configure step overlaps up to the global limit."""
    monkeypatch.chdir(tmp_path)
    agent_tracker = ConcurrencyTracker()
    configure_tracker = ConcurrencyTracker()
    interface = ConfigureCountingRouter(configure_tracker, configure_sleep=0.05)

    def factory(context: TrialContext) -> TrackingAgent:
        return TrackingAgent(agent_tracker, sleep=0.005)

    runner = CorralRunner(
        interface, agent_factory=factory, checkpoint_dir=str(tmp_path)
    )
    runner.bench(
        task_ids=[f"t{i}" for i in range(4)],
        trials_per_task=1,
        concurrency=ConcurrencyConfig(global_trials=4),
        run_name="report",
    )

    # No lease: all four configure steps overlap under the global limit of 4.
    assert configure_tracker.max_active == 4


def test_trial_verbosity_is_bound_into_the_runtime_router(tmp_path, monkeypatch):
    """The runner binds the run's verbosity into each per-trial router.

    Section 10 of the plan: a shared router's mutable `current_verbosity` must
    not carry a trial's verbosity. Instead the runner passes `verbosity=` to
    `for_trial` so it travels with the isolated trial router.
    """
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker, sleep=0.0)

    runner.bench(
        task_ids=["solo"],
        trials_per_task=2,
        max_concurrency=2,
        max_concurrency_per_task=2,
        tool_verbosity="detailed",
        run_name="report",
    )

    # Every trial router was constructed bound to the run's verbosity, not left
    # to read it from the (shared) parent's field.
    assert interface.for_trial_verbosities == ["detailed", "detailed"]


def test_per_task_gt_one_requires_runtime_support(tmp_path):
    class _NoRuntimeRouter(FakeRouter):
        create_trial = None  # type: ignore[assignment]

    runner = CorralRunner(
        _NoRuntimeRouter(),
        agent_factory=lambda ctx: TrackingAgent(ConcurrencyTracker()),
        checkpoint_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="trial runtimes"):
        runner.bench(
            task_ids=["a"],
            trials_per_task=2,
            max_concurrency=2,
            max_concurrency_per_task=2,
        )


def test_each_trial_gets_a_fresh_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    created: list[tuple[str, int, int]] = []
    runner, _ = _make_runner(tmp_path, tracker, record_agents=created)

    runner.bench(
        task_ids=["a", "b"],
        trials_per_task=2,
        max_concurrency=4,
        run_name="report",
    )

    # One distinct agent instance per (task, trial): no sharing across trials.
    assert len(created) == 4
    assert len({id(agent) for _, _, agent in created}) == 4


def test_resume_skips_already_recorded_trials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ConcurrencyTracker()
    runner, interface = _make_runner(tmp_path, tracker)

    existing = TaskTrialResult(
        task_id="a", trial_id="attempt_1", score=1.0, state={}, tool_statistics={}
    )
    monkeypatch.setattr(
        runner,
        "_load_or_initialize_results",
        lambda task_ids, session_id: {
            "a": TaskTrialResults(task_id="a", trials=[existing])
        },
    )

    result = runner.bench(
        task_ids=["a"],
        trials_per_task=2,
        max_concurrency=2,
        run_name="report",
    )

    # One trial pre-existed; only the missing one is scheduled/submitted.
    assert len(result.task_results["a"].trials) == 2
    assert len(interface.submit_calls) == 1


def test_concurrent_without_factory_fails_fast(tmp_path):
    shared_agent = TrackingAgent(ConcurrencyTracker())
    runner = CorralRunner(
        FakeRouter(), agent=shared_agent, checkpoint_dir=str(tmp_path)
    )

    with pytest.raises(ValueError, match="fresh agent per trial"):
        runner.bench(task_ids=["a"], trials_per_task=1, max_concurrency=2)


def test_runner_requires_an_agent_source(tmp_path):
    with pytest.raises(ValueError, match="agent.*agent_factory"):
        CorralRunner(FakeRouter(), checkpoint_dir=str(tmp_path))


class _ThreadRecordingAgent(BaseAgent):
    """Minimal non-native agent: records the thread its blocking `run` ran on."""

    def __init__(self):
        super().__init__(user_prompt="tool_calling/user_prompt")
        self.run_thread_name = None

    @property
    def requires_answer_extraction(self) -> bool:
        # Skip the extractor so the run needs no LLM call.
        return False

    def run(
        self,
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
        **kwargs,
    ) -> str:
        self.run_thread_name = threading.current_thread().name
        return "ok"


def test_arun_agent_offloads_sync_run_to_worker_thread():
    agent = _ThreadRecordingAgent()
    main_thread = threading.current_thread().name

    # The default async seam offloads the blocking `run()` to a worker thread so
    # the event loop is never blocked by a non-native (synchronous) agent.
    result = anyio.run(agent.arun_agent, "iface", "task-1")

    assert result.answer == "ok"
    assert agent.run_thread_name is not None
    assert agent.run_thread_name != main_thread


class _NativeAsyncAgent(BaseAgent):
    """Native async agent: overrides `arun`, records the loop thread it ran on."""

    def __init__(self):
        super().__init__(user_prompt="tool_calling/user_prompt")
        self.arun_thread_name = None

    @property
    def requires_answer_extraction(self) -> bool:
        return False

    def run(
        self,
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
        **kwargs,
    ) -> str:  # pragma: no cover - the native path must not fall back to this
        raise AssertionError("native agent must not use the sync run() path")

    async def arun(
        self,
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
        **kwargs,
    ) -> str:
        self.arun_thread_name = threading.current_thread().name
        return "native-ok"


def test_native_arun_runs_on_the_event_loop_thread():
    agent = _NativeAsyncAgent()
    main_thread = threading.current_thread().name

    # A native agent's coroutine runs directly on the loop (no worker thread),
    # and `run()` is never touched.
    result = anyio.run(agent.arun_agent, "iface", "task-1")

    assert result.answer == "native-ok"
    assert agent.arun_thread_name == main_thread


class _InterfaceRecordingAgent(BaseAgent):
    """Non-native agent that records the interface its blocking `run` received."""

    def __init__(self):
        super().__init__(user_prompt="tool_calling/user_prompt")
        self.seen_interface = None

    @property
    def requires_answer_extraction(self) -> bool:
        return False

    def run(
        self,
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
        **kwargs,
    ) -> str:
        self.seen_interface = interface
        return "ok"


def test_default_arun_hands_run_a_synchronous_view_of_an_async_router():
    """A non-native agent's offloaded `run()` never sees the async router.

    The worker thread has no event loop, so the async router's coroutine-
    returning methods would be unusable there. The default `arun` converts it to
    an equivalent synchronous `CorralRouter` for `run()`; a synchronous interface
    passes straight through.
    """
    async_router = AsyncCorralRouter("http://x", default_verbosity="detailed")
    agent = _InterfaceRecordingAgent()

    result = anyio.run(agent.arun_agent, async_router, "task-1")

    assert result.answer == "ok"
    assert isinstance(agent.seen_interface, CorralRouter)
    assert not isinstance(agent.seen_interface, AsyncCorralRouter)
    # The synchronous view targets the same server and carries the verbosity.
    assert agent.seen_interface.base_url == "http://x"
    assert agent.seen_interface.current_verbosity == "detailed"


def test_default_arun_passes_a_synchronous_interface_through_unchanged():
    sync_router = CorralRouter("http://y")
    agent = _InterfaceRecordingAgent()

    anyio.run(agent.arun_agent, sync_router, "task-1")

    # An already-synchronous interface is handed to `run()` unchanged (identity).
    assert agent.seen_interface is sync_router
