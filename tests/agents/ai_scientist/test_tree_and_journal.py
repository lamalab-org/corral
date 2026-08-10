from corral.agents.ai_scientist.journal import ResearchJournal
from corral.agents.ai_scientist.search.evaluator import extract_measured_outcome
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    MeasuredMetric,
    MeasuredOutcome,
    NodeEvaluation,
    NodeStatus,
    NodeType,
    Observation,
    Recommendation,
    ResearchStage,
)
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.state import TaskFormulation


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


def test_journal_does_not_promote_partial_node_conclusions_to_claims():
    journal = ResearchJournal()
    partial = evaluated_node("partial")
    partial.status = NodeStatus.PARTIAL

    journal.integrate(partial)

    assert [node_id for node_id, _ in journal.observations] == ["partial"]
    assert journal.claims == []


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


def test_selector_can_reexpand_a_strong_internal_checkpoint():
    tree = ExperimentTree()
    root = evaluated_node("root")
    root.evaluation.task_progress = 1.0
    root.evaluation.evidence_strength = 1.0
    child = evaluated_node("child", parent_id="root")
    child.depth = 1
    child.evaluation.task_progress = 0.1
    child.evaluation.evidence_strength = 0.1
    tree.add(root)
    tree.add(child)

    selector = TreeSelector(max_children_per_node=3)

    assert root not in tree.leaves()
    assert selector.select(tree).id == "root"
    assert selector.remaining_child_slots(tree, root) == 2


def test_selector_advances_past_a_partial_checkpoint_after_continuation():
    tree = ExperimentTree()
    partial = evaluated_node("partial", status=NodeStatus.PARTIAL)
    continuation = evaluated_node("continuation", parent_id=partial.id)
    continuation.node_type = NodeType.CONTINUE
    continuation.depth = 1
    tree.add(partial)
    tree.add(continuation)

    assert TreeSelector().select(tree).id == continuation.id


def test_selector_prefers_breadth_over_an_equivalent_new_leaf():
    tree = ExperimentTree()
    root = evaluated_node("root")
    child = evaluated_node("child", parent_id="root")
    child.depth = 1
    tree.add(root)
    tree.add(child)

    selector = TreeSelector(max_children_per_node=3, exploration_weight=0.2)

    assert selector.select(tree).id == "root"


def test_selector_stops_reexpanding_a_checkpoint_at_its_child_cap():
    tree = ExperimentTree()
    root = evaluated_node("root")
    child = evaluated_node("child", parent_id="root")
    child.depth = 1
    tree.add(root)
    tree.add(child)

    selector = TreeSelector(max_children_per_node=1)

    assert selector.select(tree).id == "child"
    assert selector.remaining_child_slots(tree, root) == 0


def test_exploration_bonus_prefers_an_equivalent_underexpanded_root():
    tree = ExperimentTree()
    expanded = evaluated_node("expanded")
    unexplored = evaluated_node("unexplored")
    child = evaluated_node("child", parent_id="expanded")
    child.depth = 1
    tree.add(expanded)
    tree.add(unexplored)
    tree.add(child)

    selector = TreeSelector(max_children_per_node=3, exploration_weight=0.2)

    assert selector.select(tree).id == "unexplored"


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


def test_later_stage_selector_is_scoped_to_its_explicit_seed_and_nodes():
    tree = ExperimentTree()
    unrelated = evaluated_node("unrelated")
    unrelated.stage = ResearchStage.PRELIMINARY
    unrelated.evaluation.task_progress = 1.0
    unrelated.evaluation.evidence_strength = 1.0
    seed = evaluated_node("seed")
    seed.stage = ResearchStage.PRELIMINARY
    child = evaluated_node("research-child", parent_id=seed.id)
    child.stage = ResearchStage.RESEARCH
    child.depth = 1
    tree.add(unrelated)
    tree.add(seed)
    tree.add(child)

    selected = TreeSelector().select(
        tree,
        stage=ResearchStage.RESEARCH,
        seed_node_id=seed.id,
    )

    assert selected is not None
    assert selected.id in {seed.id, child.id}
    assert selected.id != unrelated.id


def test_declared_metric_is_parsed_from_tool_evidence_with_provenance():
    experiment = evaluated_node("measured")
    experiment.observations[0].result = '{"metrics": {"validation_loss": 0.125}}'
    formulation = TaskFormulation(
        objective="minimize validation loss",
        required_answer="best setup",
        measured_metric=MeasuredMetric(
            name="validation_loss",
            maximize=False,
        ),
    )

    outcome = extract_measured_outcome(experiment, formulation)

    assert outcome is not None
    assert outcome.name == "validation_loss"
    assert outcome.value == 0.125
    assert outcome.maximize is False
    assert outcome.provenance == (
        "tool=measure;observation=0;path=result.metrics.validation_loss"
    )


def test_arbitrary_numbers_are_not_promoted_without_a_declared_metric():
    experiment = evaluated_node("unanchored")
    experiment.observations[0].result = (
        '{"measured_outcome": {"name": "score", "value": 42, ' '"maximize": true}}'
    )

    outcome = extract_measured_outcome(
        experiment,
        TaskFormulation(objective="answer", required_answer="value"),
    )

    assert outcome is None


def test_explicit_outcome_must_match_the_declared_metric_and_direction():
    experiment = evaluated_node("mismatched")
    experiment.observations[0].result = (
        '{"measured_outcome": {"name": "accuracy", "value": 0.99, '
        '"maximize": true}, "validation_loss": 0.125}'
    )
    formulation = TaskFormulation(
        objective="minimize validation loss",
        required_answer="best setup",
        measured_metric=MeasuredMetric(
            name="validation_loss",
            maximize=False,
        ),
    )

    outcome = extract_measured_outcome(experiment, formulation)

    assert outcome is not None
    assert outcome.name == "validation_loss"
    assert outcome.value == 0.125
    assert outcome.maximize is False
    assert outcome.provenance.endswith("path=result.validation_loss")


def test_measured_outcome_anchors_tree_ranking_ahead_of_critic_scores():
    tree = ExperimentTree()
    objective_winner = evaluated_node("objective-winner")
    objective_winner.evaluation.task_progress = 0.1
    objective_winner.evaluation.evidence_strength = 0.1
    objective_winner.measured_outcome = MeasuredOutcome(
        name="loss",
        value=0.1,
        maximize=False,
        provenance="tool=measure;observation=0;path=result.loss",
    )
    critic_winner = evaluated_node("critic-winner")
    critic_winner.evaluation.task_progress = 1.0
    critic_winner.evaluation.evidence_strength = 1.0
    critic_winner.measured_outcome = MeasuredOutcome(
        name="loss",
        value=0.9,
        maximize=False,
        provenance="tool=measure;observation=0;path=result.loss",
    )
    tree.add(objective_winner)
    tree.add(critic_winner)

    assert tree.best(limit=1) == [objective_winner]
    assert TreeSelector(exploration_weight=0).select(tree) == objective_winner
