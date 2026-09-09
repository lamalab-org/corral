from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeType,
    PlannedAction,
    ResearchStage,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools import (
    BranchSessionHandle,
    ExecutionPool,
    ReplayDiverged,
)
from corral.core.action import Action


@dataclass
class Response:
    success: bool
    result: str


class StatefulSessions:
    def __init__(self):
        self.states = {}
        self.closed = []

    def _workspace(self, execution_id):
        return f"/fake/{execution_id}"

    def create_branch(self):
        execution_id = f"execution-{len(self.states) + 1}"
        self.states[execution_id] = 0
        return BranchSessionHandle(
            execution_id=execution_id,
            workspace=self._workspace(execution_id),
            execute=partial(self.execute_action, execution_id),
        )

    def close_branch(self, execution_id):
        self.closed.append(execution_id)

    def execute_action(self, execution_id, action: Action):
        if action.arguments["operation"] == "set":
            self.states[execution_id] = action.arguments["value"]
        else:
            self.states[execution_id] += action.arguments["value"]
        return Response(success=True, result=str(self.states[execution_id]))


class CloneableSessions(StatefulSessions):
    def clone_branch(self, source_execution_id):
        session = self.create_branch()
        self.states[session.execution_id] = self.states[source_execution_id]
        return session


class ReadySessions(StatefulSessions):
    def __init__(self):
        super().__init__()
        self.ready = set()

    def create_branch(self):
        session = super().create_branch()
        self.ready.add(session.execution_id)
        return session

    def execute_action(self, execution_id, action: Action):
        assert execution_id in self.ready
        return super().execute_action(execution_id, action)


TOOLS = {
    "tools": [
        {
            "name": "mutate",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation": {"enum": ["set", "add"]},
                    "value": {"type": "integer"},
                },
                "required": ["operation", "value"],
                "additionalProperties": False,
            },
        }
    ]
}


def action(operation, value):
    return PlannedAction(
        purpose=f"{operation} state",
        tool_name="mutate",
        arguments={"operation": operation, "value": value},
        expected_information="updated state",
    )


def node(node_id, branch, plan, parent_id=None):
    return ExperimentNode(
        id=node_id,
        parent_id=parent_id,
        branch_id=branch.branch_id,
        execution_id=branch.execution_id,
        branch_workspace=branch.workspace,
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis=node_id,
        rationale="exercise branch state",
        plan=plan,
    )


def execute_and_commit(pool, branch, experiment_node):
    start = len(branch.action_history)
    experiment_node.observations = branch.executor.execute_plan(experiment_node.plan)
    pool.commit(branch, experiment_node, start)


def test_execution_pool_uses_a_ready_action_native_branch_session():
    sessions = ReadySessions()
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=1,
    )

    branch = pool.create()
    experiment_node = node("root", branch, [action("set", 1)])
    execute_and_commit(pool, branch, experiment_node)

    assert branch.execution_id in sessions.ready
    assert experiment_node.observations[0].success is True
    assert pool.tool_statistics()["total_calls"] == 1
    assert pool.tool_statistics()["successful_calls"] == 1


def test_execution_pool_reuses_first_child_and_replays_parent_for_sibling():
    sessions = StatefulSessions()
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=4,
    )
    tree = ExperimentTree()

    root_branch = pool.acquire(None, tree, prefer_existing=True)
    root = node("root", root_branch, [action("set", 1)])
    execute_and_commit(pool, root_branch, root)
    tree.add(root)

    child_branch = pool.acquire(root, tree, prefer_existing=True)
    assert child_branch is root_branch
    child = node("child", child_branch, [action("add", 2)], parent_id=root.id)
    execute_and_commit(pool, child_branch, child)
    tree.add(child)

    sibling_branch = pool.acquire(root, tree, prefer_existing=True)
    assert sibling_branch is not root_branch
    sibling = node("sibling", sibling_branch, [action("add", 5)], parent_id=root.id)
    execute_and_commit(pool, sibling_branch, sibling)
    tree.add(sibling)

    assert sessions.states[root_branch.execution_id] == 3
    assert sessions.states[sibling_branch.execution_id] == 6
    assert pool.budget.scientific_calls == 3
    assert pool.budget.replay_calls == 1
    assert pool.tool_statistics()["total_calls"] == 4
    assert pool.replay_results[0].exact is True
    assert [item.action for item in tree.executed_trajectory("sibling")] == [
        action("set", 1),
        action("add", 5),
    ]

    pool.close_all()
    assert sessions.closed == ["execution-1", "execution-2"]


def test_execution_pool_prefers_environment_clone_without_replay_cost():
    sessions = CloneableSessions()
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=2,
    )
    tree = ExperimentTree()
    root_branch = pool.acquire(None, tree, prefer_existing=True)
    root = node("root", root_branch, [action("set", 2)])
    execute_and_commit(pool, root_branch, root)
    tree.add(root)

    clone = pool.acquire(
        root,
        tree,
        prefer_existing=False,
        prefer_clone=True,
    )
    child = node("child", clone, [action("add", 3)], parent_id=root.id)
    execute_and_commit(pool, clone, child)

    assert clone.inheritance_method == "clone"
    assert sessions.states[clone.execution_id] == 5
    assert pool.executions_cloned == 1
    assert pool.budget.scientific_calls == 2
    assert pool.budget.replay_calls == 0


class DriftingSessions(StatefulSessions):
    def execute_action(self, execution_id, action: Action):
        response = super().execute_action(execution_id, action)
        if execution_id != "execution-1":
            response.result = str(float(response.result) + 0.1)
        return response


def _one_action_parent(pool, tree):
    branch = pool.acquire(None, tree, prefer_existing=True)
    parent = node("root", branch, [action("set", 1)])
    execute_and_commit(pool, branch, parent)
    tree.add(parent)
    return branch, parent


def test_successful_but_different_replay_is_fatal_by_default():
    sessions = DriftingSessions()
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=2,
    )
    tree = ExperimentTree()
    original_branch, parent = _one_action_parent(pool, tree)

    with pytest.raises(ReplayDiverged, match="physical state is unsafe"):
        pool.acquire(parent, tree, prefer_existing=False)

    replay = pool.replay_results[0]
    assert replay.exact is False
    assert replay.equivalent is False
    assert sessions.closed == ["execution-2"]
    assert list(pool.active) == [original_branch.branch_id]


def test_environment_comparator_can_accept_semantically_equivalent_replay():
    sessions = DriftingSessions()
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=2,
        replay_equivalence=lambda original, replayed: abs(
            float(original.result) - float(replayed.result)
        )
        < 0.2,
    )
    tree = ExperimentTree()
    _, parent = _one_action_parent(pool, tree)

    fork = pool.acquire(parent, tree, prefer_existing=False)

    assert fork.head_node_id == parent.id
    assert pool.replay_results[0].exact is False
    assert pool.replay_results[0].equivalent is True


class LocalWorkspaceSessions(StatefulSessions):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def _workspace(self, execution_id):
        workspace = self.root / execution_id
        workspace.mkdir()
        return str(workspace)


def test_winning_branch_artifacts_are_promoted_before_cleanup(tmp_path):
    sessions = LocalWorkspaceSessions(tmp_path)
    pool = ExecutionPool(
        sessions=sessions,
        tools=TOOLS,
        max_tool_calls=1,
    )
    tree = ExperimentTree()
    branch = pool.acquire(None, tree, prefer_existing=True)
    root = node("root", branch, [])
    tree.add(root)
    artifact = Path(branch.workspace) / "results" / "model.json"
    artifact.parent.mkdir()
    artifact.write_text('{"answer": 42}', encoding="utf-8")
    destination = tmp_path / "canonical"

    promotion = pool.promote_artifacts(
        [root],
        tree,
        destination_workspace=str(destination),
    )

    assert promotion is not None
    assert promotion.files == ("results/model.json",)
    assert (destination / "results" / "model.json").read_text() == '{"answer": 42}'
