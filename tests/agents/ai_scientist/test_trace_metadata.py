from corral.agents import AIScientistAgent
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeStatus,
    NodeType,
    ResearchStage,
)
from corral.agents.ai_scientist.state import (
    ScientistState,
    StageProgress,
    TaskFormulation,
)


def scientist_with_tree() -> tuple[AIScientistAgent, ExperimentNode, ExperimentNode]:
    agent = AIScientistAgent(model="test-model")
    state = ScientistState(
        task_prompt="Find the answer",
        tools=[],
        formulation=TaskFormulation(objective="Find it", required_answer="One value"),
    )
    root = ExperimentNode(
        id="node_0001",
        stage=ResearchStage.PRELIMINARY,
        node_type=NodeType.DRAFT,
        status=NodeStatus.SUCCESSFUL,
        hypothesis="Measure the primary signal before choosing a candidate.",
        rationale="High information gain",
    )
    child = ExperimentNode(
        id="node_0002",
        parent_id=root.id,
        stage=ResearchStage.PRELIMINARY,
        node_type=NodeType.DEBUG,
        status=NodeStatus.FAILED,
        hypothesis="Repair the failed measurement path.",
        rationale="The first path failed",
        depth=1,
    )
    state.tree.add(root)
    state.tree.add(child)
    state.current_stage = ResearchStage.PRELIMINARY
    state.stages[ResearchStage.PRELIMINARY] = StageProgress(
        stage=ResearchStage.PRELIMINARY,
        best_node_id=root.id,
    )
    agent.last_state = state
    agent.messages = [
        {
            "role": "system",
            "content": "Evaluate scientific evidence",
            "name": "evaluate_node_0002",
        },
        {
            "role": "user",
            "content": "Evaluate the experiment",
            "name": "evaluate_node_0002",
        },
        {
            "role": "assistant",
            "content": "{}",
            "name": "evaluate_node_0002",
            "id": "response-1",
        },
    ]
    return agent, root, child


def test_scientist_trace_metadata_labels_tree_without_mutating_messages():
    agent, root, child = scientist_with_tree()
    original_messages = [message.copy() for message in agent.messages]

    metadata = agent._trace_metadata()

    assert metadata["schema"] == "corral.ai_scientist.graph"
    assert metadata["root_node_ids"] == [root.id]
    assert metadata["stage_winner_node_ids"] == {"preliminary": root.id}
    assert metadata["edges"] == [
        {"source": root.id, "target": child.id, "kind": "parent"}
    ]
    assert metadata["nodes"][0]["label"].startswith(
        "node_0001 [preliminary/draft; successful]"
    )
    assert metadata["nodes"][1]["label"].startswith(
        "node_0002 [preliminary/debug; failed]"
    )
    assert metadata["message_links"] == [
        {
            "message_index": 0,
            "message_name": "evaluate_node_0002",
            "node_ids": [child.id],
        },
        {
            "message_index": 1,
            "message_name": "evaluate_node_0002",
            "node_ids": [child.id],
        },
        {
            "message_index": 2,
            "message_name": "evaluate_node_0002",
            "node_ids": [child.id],
        },
    ]
    assert agent.messages == original_messages
    assert all(
        set(message) <= {"role", "content", "name", "id"} for message in agent.messages
    )


def test_verbose_save_puts_graph_beside_messages(monkeypatch):
    agent, root, child = scientist_with_tree()
    original_messages = [message.copy() for message in agent.messages]
    saved = {}

    def capture_save(**kwargs):
        saved.update(kwargs)
        return "unused.json"

    monkeypatch.setattr("corral.agents.base_agent.save_agent_messages", capture_save)

    agent._save_run_messages("task-1", "brief")

    assert saved["messages"] == original_messages
    assert saved["trace_metadata"]["root_node_ids"] == [root.id]
    assert saved["trace_metadata"]["edges"] == [
        {"source": root.id, "target": child.id, "kind": "parent"}
    ]
    assert agent.messages == original_messages
