"""Tests for dependency-aware episodes (PR 3 of `make_efficiency.md`).

Two layers:

* the reusable :func:`corral.episode.run_episode_dag` primitive — sibling nodes
  run concurrently, a dependent waits for its parents to *succeed*, and a broken
  parent propagates *unreachable* through its whole downstream cone without
  invoking the agent;
* the runner wired end-to-end through `bench(max_concurrency>1)` on a chained
  fake router that models per-episode dependency stores — each round is an
  isolated episode, rounds and siblings overlap, results stay round-ordered, and
  every episode's store is created and freed.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial

import anyio
import pytest

from corral.agents.schema import AgentRunResult
from corral.concurrency import ConcurrencyConfig, EpisodeTaskCompleted
from corral.episode import run_episode_dag
from corral.report import TaskTrialResult
from corral.run import CorralRunner, unreachable_trial_result


def _result(task_id: str, score: float) -> TaskTrialResult:
    return TaskTrialResult(
        task_id=task_id,
        trial_id=f"attempt_{task_id}",
        score=score,
        state={},
        tool_statistics={},
    )


async def _drive_dag(ordered_ids, dep_graph, scores, *, sleep=0.01):
    """Run one DAG, returning (executed_order, max_concurrency, events)."""
    executed: list[str] = []
    events: list[EpisodeTaskCompleted] = []
    state = {"active": 0, "max": 0}

    async def node_executor(task_id: str, trial_round: int) -> TaskTrialResult:
        executed.append(task_id)
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        await anyio.sleep(sleep)  # hold the slot so real overlap is observable
        state["active"] -= 1
        return _result(task_id, scores.get(task_id, 1.0))

    async def emit(event: EpisodeTaskCompleted) -> None:
        events.append(event)

    await run_episode_dag(
        trial_round=0,
        ordered_ids=ordered_ids,
        dep_graph=dep_graph,
        node_executor=node_executor,
        unreachable_result=unreachable_trial_result,
        emit=emit,
    )
    return executed, state["max"], events


def test_independent_siblings_run_concurrently():
    executed, max_active, events = anyio.run(
        _drive_dag, ["a", "b", "c"], {"a": [], "b": [], "c": []}, {}
    )
    assert sorted(executed) == ["a", "b", "c"]
    assert max_active == 3  # all three overlapped
    assert {e.task_id for e in events} == {"a", "b", "c"}


def test_dependent_waits_for_parents():
    # Diamond: a -> {b, c} -> d. b and c overlap; a before them; d after both.
    dep_graph = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    executed, max_active, events = anyio.run(
        _drive_dag, ["a", "b", "c", "d"], dep_graph, {}
    )

    order = {tid: i for i, tid in enumerate(executed)}
    assert order["a"] < order["b"]
    assert order["a"] < order["c"]
    assert order["d"] > order["b"]
    assert order["d"] > order["c"]
    # b and c are the only pair with no edge between them, so peak overlap is 2.
    assert max_active == 2
    assert len(events) == 4


def test_broken_parent_propagates_unreachable():
    # a fails; its cone (c depends on a, e depends on c) is unreachable.
    # b is independent and still runs.
    dep_graph = {"a": [], "b": [], "c": ["a"], "e": ["c"]}
    executed, _max_active, events = anyio.run(
        _drive_dag, ["a", "b", "c", "e"], dep_graph, {"a": 0.0}
    )

    # Only reachable nodes actually invoked the executor.
    assert sorted(executed) == ["a", "b"]

    by_task = {e.task_id: e.result for e in events}
    assert by_task["a"].success is False
    assert by_task["b"].success is True
    # c and e are unreachable, scored 0, and record the broken dependency.
    for tid, missing in (("c", "a"), ("e", "c")):
        assert by_task[tid].success is False
        assert by_task[tid].state["unreachable"] is True
        assert by_task[tid].state["missing_dependency"] == missing


class _ConcurrencyTracker:
    """Records the max number of agents running at once (thread-safe).

    `track(key)` also records a per-key peak, so a test can check a per-model
    cap serialises one model while the rest of the DAG overlaps.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self._per_key = defaultdict(int)
        self.max_by_key = defaultdict(int)

    @contextmanager
    def track(self, key: str = "_all"):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self._per_key[key] += 1
            self.max_by_key[key] = max(self.max_by_key[key], self._per_key[key])
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1
                self._per_key[key] -= 1


class _ChainAgent:
    """Agent that sleeps (to expose overlap) then answers with its task id."""

    def __init__(self, tracker: _ConcurrencyTracker, sleep: float):
        self._tracker = tracker
        self._sleep = sleep

    def run_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        with self._tracker.track():
            time.sleep(self._sleep)
        return AgentRunResult(answer=f"answer_{task_id}", status="success")

    async def arun_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        # Non-native agent: the async scheduler offloads the blocking run to a
        # worker thread (exactly as BaseAgent's default seam does), so the
        # `time.sleep` overlap stays observable off the event loop.
        return await anyio.to_thread.run_sync(
            partial(self.run_agent, interface, task_id, **kwargs)
        )

    def get_total_token_usage(self) -> dict:
        return {}


class _ChainedRouter:
    """Fake router modelling a dependency chain with per-episode stores.

    Only the surface the chained scheduler and `execute_single_trial` touch is
    implemented. `create_trial` records the episode a task joins; a per-episode
    store records each task's submission; `submit_answer` fails tasks named in
    `failing` (score 0) so the scheduler can propagate unreachability.
    """

    def __init__(self, graph: dict[str, list[str]], failing: set[str] | None = None):
        self.current_verbosity = "brief"
        self._graph = graph
        self._failing = set(failing or ())
        self._lock = threading.Lock()
        self.created: list[tuple[str, str]] = []  # (task_id, episode_id)
        self.closed_trials: list[str] = []
        self.closed_episodes: list[str] = []
        self.episode_stores: dict[str, dict[str, str]] = defaultdict(dict)
        self._rid_episode: dict[str, str] = {}
        self._rid_task: dict[str, str] = {}
        self._counter = 0

    def set_verbosity(self, verbosity: str) -> None:
        self.current_verbosity = verbosity

    def supports_dependency_chain(self) -> bool:
        return True

    def get_available_tasks(self) -> list[str]:
        return list(self._graph)

    def get_dependency_graph(self) -> dict[str, list[str]]:
        return {tid: list(deps) for tid, deps in self._graph.items()}

    def configure_additional_apps(self, task_id, timeout=None) -> str:
        return "configured"

    def get_task_status(self, task_id) -> dict:
        return {"score": 0.0}

    def submit_answer(self, task_id, answer) -> TaskTrialResult:
        score = 0.0 if task_id in self._failing else 1.0
        return TaskTrialResult(
            task_id=task_id,
            trial_id=f"attempt_{task_id}",
            score=score,
            state={},
            tool_statistics={},
        )

    def surrender_task(self, task_id) -> TaskTrialResult:
        return TaskTrialResult(
            task_id=task_id,
            trial_id=f"attempt_{task_id}",
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
            self._counter += 1
            rid = f"tr_{task_id}_{self._counter}"
            self.created.append((task_id, episode_id))
            self._rid_episode[rid] = episode_id
            self._rid_task[rid] = task_id
            # Touch the store so it exists even before any submission.
            _ = self.episode_stores[episode_id]
        return {
            "trial_runtime_id": rid,
            "task_id": task_id,
            "workspace": None,
            "mcp_url": f"/trials/{rid}/mcp",
        }

    def for_trial(
        self, trial_runtime_id, task_id, *, verbosity=None, workspace=None
    ) -> "_ChainedTrialRouter":
        return _ChainedTrialRouter(self, trial_runtime_id, task_id, verbosity=verbosity)

    def close_trial(self, trial_runtime_id) -> None:
        with self._lock:
            self.closed_trials.append(trial_runtime_id)

    def close_episode(self, episode_id) -> None:
        with self._lock:
            self.closed_episodes.append(episode_id)
            self.episode_stores.pop(episode_id, None)


class _ChainedTrialRouter:
    """Runtime-scoped view: submissions land in the bound episode's store."""

    def __init__(
        self, parent: _ChainedRouter, rid: str, task_id: str, *, verbosity=None
    ):
        self._parent = parent
        self.trial_runtime_id = rid
        self.task_id = task_id
        self.current_verbosity = verbosity or parent.current_verbosity
        self._episode_id = parent._rid_episode[rid]

    def set_verbosity(self, verbosity: str) -> None:
        self.current_verbosity = verbosity

    def configure_additional_apps(self, task_id=None, timeout=None) -> str:
        return "configured"

    def get_task_status(self, task_id=None) -> dict:
        return {"score": 0.0}

    def submit_answer(self, task_id, answer) -> TaskTrialResult:
        with self._parent._lock:
            self._parent.episode_stores[self._episode_id][self.task_id] = answer
        score = 0.0 if self.task_id in self._parent._failing else 1.0
        return TaskTrialResult(
            task_id=self.task_id,
            trial_id=self.trial_runtime_id,
            score=score,
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


def _chained_runner(tmp_path, graph, tracker, *, sleep=0.02, failing=None):
    interface = _ChainedRouter(graph, failing=failing)
    runner = CorralRunner(
        interface,
        agent_factory=lambda ctx: _ChainAgent(tracker, sleep),
        checkpoint_dir=str(tmp_path),
    )
    return runner, interface


def test_chained_bench_runs_all_rounds_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = _ConcurrencyTracker()
    graph = {"a": [], "b": ["a"]}
    runner, interface = _chained_runner(tmp_path, graph, tracker)

    result = runner.bench(
        task_ids=["a", "b"],
        trials_per_task=3,
        max_concurrency=4,
        run_name="report",
    )

    # Every task/round recorded.
    assert len(result.task_results["a"].trials) == 3
    assert len(result.task_results["b"].trials) == 3
    assert all(t.success for t in result.task_results["a"].trials)
    assert all(t.success for t in result.task_results["b"].trials)
    # Three isolated episodes, each created and then freed.
    episode_ids = {ep for _, ep in interface.created}
    assert len(episode_ids) == 3
    assert sorted(interface.closed_episodes) == sorted(episode_ids)
    # a and b of one round shared their episode.
    per_episode = defaultdict(set)
    for task_id, ep in interface.created:
        per_episode[ep].add(task_id)
    assert all(members == {"a", "b"} for members in per_episode.values())


def test_chained_siblings_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = _ConcurrencyTracker()
    # Diamond so b and c are genuine siblings.
    graph = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    runner, _ = _chained_runner(tmp_path, graph, tracker, sleep=0.05)

    runner.bench(
        task_ids=["a", "b", "c", "d"],
        trials_per_task=1,
        max_concurrency=4,
        run_name="report",
    )

    # b and c (at least) ran at the same time.
    assert tracker.max_active >= 2


def test_chained_dependency_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = _ConcurrencyTracker()
    graph = {"a": [], "b": ["a"]}
    runner, interface = _chained_runner(tmp_path, graph, tracker, failing={"a"})

    result = runner.bench(
        task_ids=["a", "b"],
        trials_per_task=2,
        max_concurrency=4,
        run_name="report",
    )

    # a failed every round; b was never executed (no runtime created for it).
    assert all(not t.success for t in result.task_results["a"].trials)
    created_tasks = [task_id for task_id, _ in interface.created]
    assert created_tasks.count("a") == 2
    assert "b" not in created_tasks
    # b is recorded as unreachable, pointing at the broken dependency.
    for trial in result.task_results["b"].trials:
        assert trial.state["unreachable"] is True
        assert trial.state["missing_dependency"] == "a"


def test_chained_concurrent_without_factory_fails_fast(tmp_path):
    interface = _ChainedRouter({"a": [], "b": ["a"]})
    runner = CorralRunner(
        interface,
        agent=_ChainAgent(_ConcurrencyTracker(), 0.0),  # shared agent, not a factory
        checkpoint_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="fresh agent per trial"):
        runner.bench(task_ids=["a", "b"], trials_per_task=1, max_concurrency=2)


def test_chained_concurrent_requires_runtime_support(tmp_path):
    class _NoRuntimeChain(_ChainedRouter):
        create_trial = None  # type: ignore[assignment]

    runner = CorralRunner(
        _NoRuntimeChain({"a": [], "b": ["a"]}),
        agent_factory=lambda ctx: _ChainAgent(_ConcurrencyTracker(), 0.0),
        checkpoint_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="trial runtimes"):
        runner.bench(task_ids=["a", "b"], trials_per_task=1, max_concurrency=2)


def test_chained_serial_still_works_without_factory(tmp_path, monkeypatch):
    """A single-concurrency chain keeps the shared-agent serial path."""
    monkeypatch.chdir(tmp_path)
    tracker = _ConcurrencyTracker()
    graph = {"a": [], "b": ["a"]}
    interface = _ChainedRouter(graph)
    runner = CorralRunner(
        interface,
        agent=_ChainAgent(tracker, 0.0),  # shared agent, allowed when serial
        checkpoint_dir=str(tmp_path),
    )

    result = runner.bench(task_ids=["a", "b"], trials_per_task=1, run_name="report")

    assert len(result.task_results["a"].trials) == 1
    assert len(result.task_results["b"].trials) == 1
    # Serial path does not open trial runtimes.
    assert interface.created == []


class _ModeledChainAgent(_ChainAgent):
    """A chain agent carrying a `model` and tracking concurrency by it."""

    def __init__(self, tracker: _ConcurrencyTracker, sleep: float, model: str):
        super().__init__(tracker, sleep)
        self.model = model

    def run_agent(self, interface, task_id, **kwargs) -> AgentRunResult:
        with self._tracker.track(self.model):
            time.sleep(self._sleep)
        return AgentRunResult(answer=f"answer_{task_id}", status="success")


def test_chained_per_model_caps_sibling_overlap(tmp_path, monkeypatch):
    """`per_model` serialises capped siblings that would otherwise overlap.

    In this diamond `b` and `c` are genuine siblings (see
    :func:`test_chained_siblings_overlap`, where they overlap). Giving both the
    same capped model with a cap of 1 must keep them from running at once, while
    the chain still completes — the concurrent DAG path honours `per_model`.
    """
    monkeypatch.chdir(tmp_path)
    tracker = _ConcurrencyTracker()
    graph = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    models = {"a": "root", "b": "capped", "c": "capped", "d": "leaf"}
    interface = _ChainedRouter(graph)
    runner = CorralRunner(
        interface,
        agent_factory=lambda ctx: _ModeledChainAgent(
            tracker, 0.05, models[ctx.task_id]
        ),
        checkpoint_dir=str(tmp_path),
    )

    result = runner.bench(
        task_ids=["a", "b", "c", "d"],
        trials_per_task=1,
        concurrency=ConcurrencyConfig(global_trials=4, per_model={"capped": 1}),
        run_name="report",
    )

    # b and c share the capped model, so even as siblings they never overlap.
    assert tracker.max_by_key["capped"] == 1
    # The whole chain still ran and succeeded.
    for tid in ("a", "b", "c", "d"):
        assert len(result.task_results[tid].trials) == 1
        assert all(t.success for t in result.task_results[tid].trials)
