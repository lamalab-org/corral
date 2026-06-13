"""Tests for dependency-graph closure and topological ordering.

Covers the invariant from ``better_execution_order.md``: a chained run executes
tasks in dependency order, and an incomplete selection is rejected up front
rather than surfacing as ``NOT YET AVAILABLE`` text inside a prompt.
"""

import pytest

from corral.backend.task import (
    assert_dependencies_selected,
    order_selected,
    topological_order,
)

# A small diamond: analyze depends on retrieve; report depends on both.
GRAPH = {
    "retrieve": [],
    "clean": ["retrieve"],
    "analyze": ["retrieve"],
    "report": ["clean", "analyze"],
}


class TestAssertDependenciesSelected:
    def test_full_set_passes(self):
        assert_dependencies_selected(list(GRAPH), GRAPH)  # no raise

    def test_independent_only_passes(self):
        assert_dependencies_selected(["retrieve"], GRAPH)  # no raise

    def test_subset_missing_dependency_raises_with_pair(self):
        with pytest.raises(ValueError) as exc:
            assert_dependencies_selected(["analyze"], GRAPH)
        message = str(exc.value)
        assert "analyze needs" in message
        assert "retrieve" in message
        assert "dependency-closed" in message

    def test_reports_every_missing_dependency(self):
        with pytest.raises(ValueError) as exc:
            assert_dependencies_selected(["report"], GRAPH)
        message = str(exc.value)
        assert "report needs" in message
        assert "clean" in message
        assert "analyze" in message


class TestOrderSelected:
    def test_dependencies_precede_dependents(self):
        ordered = order_selected(list(GRAPH), GRAPH)
        assert set(ordered) == set(GRAPH)
        for task, deps in GRAPH.items():
            for dep in deps:
                assert ordered.index(dep) < ordered.index(task)

    def test_reverse_authoring_order_still_deps_first(self):
        # Authored dependents-first; ordering must still put deps first.
        reversed_ids = list(reversed(list(GRAPH)))
        ordered = order_selected(reversed_ids, GRAPH)
        assert ordered.index("retrieve") < ordered.index("analyze")
        assert ordered.index("analyze") < ordered.index("report")

    def test_ignores_edges_to_non_selected_tasks(self):
        # Selecting only {retrieve, analyze} must not pull in clean/report and
        # must still order retrieve before analyze.
        ordered = order_selected(["analyze", "retrieve"], GRAPH)
        assert ordered == ["retrieve", "analyze"]

    def test_cycle_raises(self):
        cyclic = {"a": ["b"], "b": ["a"]}
        with pytest.raises(ValueError, match="Cycle detected"):
            order_selected(list(cyclic), cyclic)


class TestTopologicalOrderDelegates:
    def test_matches_graph_ordering(self):
        from corral.backend.task import InputRef, TaskDefinition

        def task(name, deps):
            return TaskDefinition(
                name=name,
                description="",
                tools=[],
                scoring_fn=lambda answer: 0.0,
                submission_format={},
                input_map={d: InputRef(d) for d in deps},
            )

        tasks = {tid: task(tid, deps) for tid, deps in GRAPH.items()}
        ordered = topological_order(tasks)
        for tid, deps in GRAPH.items():
            for dep in deps:
                assert ordered.index(dep) < ordered.index(tid)


class TestChainedShortCircuit:
    """Broken-chain handling in ``run_chained_trials`` (no agent, no raise)."""

    @staticmethod
    def _result(task_id, score):
        from corral.report.results import TaskTrialResult

        return TaskTrialResult(
            task_id=task_id,
            trial_id="attempt_1",
            score=score,
            state={},
            tool_statistics={},
        )

    def _run(self, ordered_ids, graph, scores):
        """Run one round with a fake executor; return (results, executed_ids)."""
        from corral.report.results import TaskTrialResults
        from corral.run import run_chained_trials

        task_results = {t: TaskTrialResults(task_id=t) for t in ordered_ids}
        executed: list[str] = []

        def fake_executor(task_id, trial_index):
            executed.append(task_id)
            return self._result(task_id, scores[task_id])

        run_chained_trials(
            ordered_ids,
            trials_per_task=1,
            task_results=task_results,
            trial_executor=fake_executor,
            checkpoint_saver=lambda results, completed: None,
            graph=graph,
        )
        return task_results, executed

    def test_broken_chain_cascades_without_agent(self):
        # A -> B -> C; A fails (score 0), so B and C are unreachable.
        graph = {"A": [], "B": ["A"], "C": ["B"]}
        scores = {"A": 0.0, "B": 1.0, "C": 1.0}

        task_results, executed = self._run(["A", "B", "C"], graph, scores)

        # Only A actually ran; B and C were short-circuited.
        assert executed == ["A"]

        b = task_results["B"].trials[0]
        c = task_results["C"].trials[0]
        for result, missing in ((b, "A"), (c, "B")):
            assert result.score == 0.0
            assert result.success is False
            assert result.error_message is None  # stays out of error-rate
            assert result.token_usage == {}  # 0 tokens: a non-attempt
            assert result.state["unreachable"] is True
            assert result.state["missing_dependency"] == missing

    def test_healthy_chain_runs_every_task(self):
        graph = {"A": [], "B": ["A"], "C": ["B"]}
        scores = {"A": 1.0, "B": 1.0, "C": 1.0}

        _, executed = self._run(["A", "B", "C"], graph, scores)
        assert executed == ["A", "B", "C"]
