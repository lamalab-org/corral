"""Tests for dependency-graph closure and topological ordering.

Covers the invariant from `better_execution_order.md`: a chained run executes
tasks in dependency order, and an incomplete selection is rejected up front
rather than surfacing as `NOT YET AVAILABLE` text inside a prompt.
"""

import pytest

from corral.core.task import (
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
        from corral.core.task import InputRef, TaskDefinition

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
