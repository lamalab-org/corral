"""Best-first selection with bounded probabilistic debugging."""

import math
import random

from corral.agents.ai_scientist.search.evaluator import (
    EvaluationWeights,
    evaluation_priority,
)
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeStatus,
    NodeType,
    Recommendation,
    ResearchStage,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree


class TreeSelector:
    def __init__(
        self,
        *,
        debug_probability: float = 0.2,
        max_debug_depth: int = 2,
        max_children_per_node: int = 3,
        exploration_weight: float = 0.1,
        random_seed: int = 0,
        weights: EvaluationWeights | None = None,
    ) -> None:
        self.debug_probability = debug_probability
        self.max_debug_depth = max_debug_depth
        self.max_children_per_node = max_children_per_node
        self.exploration_weight = exploration_weight
        self._random = random.Random(random_seed)
        self.weights = weights or EvaluationWeights()

    def select(
        self,
        tree: ExperimentTree,
        *,
        allow_failures: bool = True,
        allow_partial: bool = True,
        stage: ResearchStage | None = None,
        seed_node_id: str | None = None,
    ) -> ExperimentNode | None:
        """Select an expandable checkpoint within one stage search scope.

        A later stage sees only its explicitly inherited seed and nodes created
        in that stage. This prevents a global critic score from silently
        jumping back to an unrelated branch from an earlier stage.
        """
        candidates = [
            node
            for node in tree.nodes
            if self._in_scope(node, stage=stage, seed_node_id=seed_node_id)
            and self._child_count(tree, node, stage) < self.max_children_per_node
        ]
        return self._select_from(
            tree,
            candidates,
            allow_failures=allow_failures,
            allow_partial=allow_partial,
            stage=stage,
        )

    def _select_from(
        self,
        tree: ExperimentTree,
        candidates: list[ExperimentNode],
        *,
        allow_failures: bool = True,
        allow_partial: bool = True,
        stage: ResearchStage | None = None,
    ) -> ExperimentNode | None:
        failed = [
            node
            for node in candidates
            if allow_failures
            and node.status in {NodeStatus.FAILED, NodeStatus.INVALID}
            and node.debug_depth < self.max_debug_depth
            and not self._abandoned(node)
        ]
        successful = [
            node
            for node in candidates
            if node.status == NodeStatus.SUCCESSFUL and not self._abandoned(node)
        ]
        partial = [
            node
            for node in candidates
            if allow_partial
            and node.status == NodeStatus.PARTIAL
            and not self._abandoned(node)
            and not any(
                child.node_type == NodeType.CONTINUE
                for child in tree.children(node.id)
                if stage is None
                or (child.stage == stage and not child.boundary_validation)
            )
        ]

        # A partial node is a valid physical checkpoint, but not yet scientific
        # evidence. Resume the strongest unfinished workflow before opening a
        # different hypothesis or debugging an unrelated failure.
        if partial:
            return max(partial, key=lambda node: self._priority(tree, node, stage))

        choose_failure = bool(failed) and (
            not successful or self._random.random() < self.debug_probability
        )
        pool = failed if choose_failure else successful
        if not pool:
            pool = failed
        if not pool:
            return None
        return max(pool, key=lambda node: self._priority(tree, node, stage))

    def remaining_child_slots(
        self,
        tree: ExperimentTree,
        node: ExperimentNode,
        *,
        stage: ResearchStage | None = None,
    ) -> int:
        """Return how many more alternative children ``node`` may spawn."""
        return max(
            0,
            self.max_children_per_node - self._child_count(tree, node, stage),
        )

    def _priority(
        self,
        tree: ExperimentTree,
        node: ExperimentNode,
        stage: ResearchStage | None = None,
    ) -> float:
        """Combine scientific quality with an under-expansion bonus."""
        scope = [
            item
            for item in tree.nodes
            if stage is None
            or (item.stage == stage and not item.boundary_validation)
            or item.id == node.id
        ]
        total_expansions = sum(self._child_count(tree, item, stage) for item in scope)
        # Prefer under-expanded, shallower checkpoints when scientific quality
        # is comparable. Dividing by depth is important: otherwise every new
        # leaf receives the largest exploration bonus and the search collapses
        # back into chains despite internal nodes remaining technically valid.
        exploration = (
            self.exploration_weight
            * math.sqrt(
                math.log(total_expansions + 2)
                / (self._child_count(tree, node, stage) + 1)
            )
            / (node.depth + 1)
        )
        return evaluation_priority(node, self.weights) + exploration

    @staticmethod
    def _in_scope(
        node: ExperimentNode,
        *,
        stage: ResearchStage | None,
        seed_node_id: str | None,
    ) -> bool:
        if stage is None:
            return True
        return node.id == seed_node_id or (
            node.stage == stage and not node.boundary_validation
        )

    @staticmethod
    def _child_count(
        tree: ExperimentTree,
        node: ExperimentNode,
        stage: ResearchStage | None,
    ) -> int:
        if stage is None:
            return tree.child_count(node.id)
        return sum(
            child.stage == stage and not child.boundary_validation
            for child in tree.children(node.id)
        )

    @staticmethod
    def _abandoned(node: ExperimentNode) -> bool:
        return (
            node.evaluation is not None
            and node.evaluation.recommendation == Recommendation.ABANDON
        )
