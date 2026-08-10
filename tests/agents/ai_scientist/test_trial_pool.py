from dataclasses import dataclass
from pathlib import Path

import pytest

from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeType,
    PlannedAction,
    ResearchStage,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools import ReplayDiverged, TrialPool


@dataclass
class Response:
    success: bool
    result: str


class StatefulRouter:
    def __init__(self):
        self.states = {}
        self.closed = []

    def create_trial(self, task_id):
        trial_id = f"trial-{len(self.states) + 1}"
        self.states[trial_id] = 0
        return {"trial_runtime_id": trial_id, "workspace": f"/fake/{trial_id}"}

    def for_trial(self, trial_runtime_id, task_id, *, verbosity=None, workspace=None):
        return StatefulTrial(self, trial_runtime_id)

    def close_trial(self, trial_runtime_id):
        self.closed.append(trial_runtime_id)


class StatefulTrial:
    def __init__(self, parent, trial_runtime_id):
        self.parent = parent
        self.trial_runtime_id = trial_runtime_id

    def execute_tool(self, task_id, tool_name, arguments):
        if arguments["operation"] == "set":
            self.parent.states[self.trial_runtime_id] = arguments["value"]
        else:
            self.parent.states[self.trial_runtime_id] += arguments["value"]
        return Response(
            success=True, result=str(self.parent.states[self.trial_runtime_id])
        )


class CloneableRouter(StatefulRouter):
    def clone_trial(self, source_trial_runtime_id, task_id):
        descriptor = self.create_trial(task_id)
        self.states[descriptor["trial_runtime_id"]] = self.states[
            source_trial_runtime_id
        ]
        return descriptor


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
        trial_runtime_id=branch.trial_runtime_id,
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


def test_trial_pool_reuses_first_child_and_replays_parent_for_sibling():
    interface = StatefulRouter()
    pool = TrialPool(
        interface=interface,
        task_id="task",
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

    assert interface.states[root_branch.trial_runtime_id] == 3
    assert interface.states[sibling_branch.trial_runtime_id] == 6
    assert pool.budget.scientific_calls == 3
    assert pool.budget.replay_calls == 1
    assert pool.replay_results[0].exact is True
    assert [item.action for item in tree.executed_trajectory("sibling")] == [
        action("set", 1),
        action("add", 5),
    ]

    pool.close_all()
    assert interface.closed == ["trial-1", "trial-2"]


def test_trial_pool_prefers_environment_clone_without_replay_cost():
    interface = CloneableRouter()
    pool = TrialPool(
        interface=interface,
        task_id="task",
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
    assert interface.states[clone.trial_runtime_id] == 5
    assert pool.trials_cloned == 1
    assert pool.budget.scientific_calls == 2
    assert pool.budget.replay_calls == 0


class DriftingTrial(StatefulTrial):
    def execute_tool(self, task_id, tool_name, arguments):
        response = super().execute_tool(task_id, tool_name, arguments)
        if self.trial_runtime_id != "trial-1":
            response.result = str(float(response.result) + 0.1)
        return response


class DriftingRouter(StatefulRouter):
    def for_trial(self, trial_runtime_id, task_id, *, verbosity=None, workspace=None):
        return DriftingTrial(self, trial_runtime_id)


def _one_action_parent(pool, tree):
    branch = pool.acquire(None, tree, prefer_existing=True)
    parent = node("root", branch, [action("set", 1)])
    execute_and_commit(pool, branch, parent)
    tree.add(parent)
    return branch, parent


def test_successful_but_different_replay_is_fatal_by_default():
    interface = DriftingRouter()
    pool = TrialPool(
        interface=interface,
        task_id="task",
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
    assert interface.closed == ["trial-2"]
    assert list(pool.active) == [original_branch.branch_id]


def test_environment_comparator_can_accept_semantically_equivalent_replay():
    interface = DriftingRouter()
    pool = TrialPool(
        interface=interface,
        task_id="task",
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


class LocalWorkspaceRouter(StatefulRouter):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def create_trial(self, task_id):
        descriptor = super().create_trial(task_id)
        workspace = self.root / descriptor["trial_runtime_id"]
        workspace.mkdir()
        descriptor["workspace"] = str(workspace)
        return descriptor


def test_winning_branch_artifacts_are_promoted_before_cleanup(tmp_path):
    interface = LocalWorkspaceRouter(tmp_path)
    pool = TrialPool(
        interface=interface,
        task_id="task",
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
