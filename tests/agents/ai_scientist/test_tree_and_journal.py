from corral.agents.ai_scientist.journal import ResearchJournal
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeEvaluation,
    NodeStatus,
    NodeType,
    Observation,
    Recommendation,
    ResearchStage,
)
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.search.tree import ExperimentTree


def evaluated_node(
    node_id,
    *,
    status=NodeStatus.SUCCESSFUL,
    parent_id=None,
    debug_depth=0,
    conclusion="claim",
    recommendation=Recommendation.CONTINUE,
):
    return ExperimentNode(
        id=node_id,
        parent_id=parent_id,
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis=f"hypothesis-{node_id}",
        rationale="rationale",
        status=status,
        debug_depth=debug_depth,
        observations=[
            Observation(
                action_index=0,
                purpose="measure",
                tool_name="measure",
                success=status == NodeStatus.SUCCESSFUL,
                result="42" if status == NodeStatus.SUCCESSFUL else None,
                error="boom" if status != NodeStatus.SUCCESSFUL else None,
            )
        ],
        evaluation=NodeEvaluation(
            validity=0.8,
            task_progress=0.7,
            evidence_strength=0.75,
            information_gain=0.6,
            consistency=0.9,
            recommendation=recommendation,
            reason="useful",
            conclusions=[conclusion],
        ),
    )


def test_journal_keeps_reliable_evidence_from_sibling_branches():
    journal = ResearchJournal()
    left = evaluated_node("left", conclusion="fact X")
    right = evaluated_node("right", conclusion="fact Y")

    journal.integrate(left)
    journal.integrate(right)

    assert [node_id for node_id, _ in journal.observations] == ["left", "right"]
    assert {claim.text for claim in journal.claims} == {"fact X", "fact Y"}


def test_selector_can_debug_failure_but_respects_max_debug_depth():
    tree = ExperimentTree()
    successful = evaluated_node("node_0001")
    failed = evaluated_node("node_0002", status=NodeStatus.FAILED)
    tree.add(successful)
    tree.add(failed)

    selector = TreeSelector(debug_probability=1.0, max_debug_depth=1, random_seed=0)
    assert selector.select(tree).id == "node_0002"

    capped = ExperimentTree()
    capped.add(evaluated_node("node_0001"))
    capped.add(evaluated_node("node_0002", status=NodeStatus.FAILED, debug_depth=1))
    assert selector.select(capped).id == "node_0001"


def test_selector_falls_back_to_successful_ancestor_after_capped_debug_leaf():
    tree = ExperimentTree()
    tree.add(evaluated_node("root"))
    tree.add(
        evaluated_node(
            "capped-debug",
            status=NodeStatus.FAILED,
            parent_id="root",
            debug_depth=2,
        )
    )

    selector = TreeSelector(max_debug_depth=2)

    assert [node.id for node in tree.leaves()] == ["capped-debug"]
    assert selector.select(tree).id == "root"


def test_selector_and_best_nodes_exclude_abandoned_branches():
    tree = ExperimentTree()
    tree.add(
        evaluated_node(
            "abandoned",
            recommendation=Recommendation.ABANDON,
        )
    )
    tree.add(evaluated_node("continue"))

    selector = TreeSelector()

    assert selector.select(tree).id == "continue"
    assert [node.id for node in tree.best()] == ["continue"]
