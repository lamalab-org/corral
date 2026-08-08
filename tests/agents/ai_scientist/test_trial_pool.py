from dataclasses import dataclass

from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeType,
    PlannedAction,
    ResearchStage,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools import TrialPool


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
