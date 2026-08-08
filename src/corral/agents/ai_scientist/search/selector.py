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
    ) -> ExperimentNode | None:
        """Select the best expandable checkpoint, including internal nodes."""
        candidates = [
            node
            for node in tree.nodes
            if tree.child_count(node.id) < self.max_children_per_node
        ]
        return self._select_from(
            tree,
            candidates,
            allow_failures=allow_failures,
            allow_partial=allow_partial,
        )

    def _select_from(
        self,
        tree: ExperimentTree,
        candidates: list[ExperimentNode],
        *,
        allow_failures: bool = True,
        allow_partial: bool = True,
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
                child.node_type == NodeType.CONTINUE for child in tree.children(node.id)
            )
        ]

        # A partial node is a valid physical checkpoint, but not yet scientific
        # evidence. Resume the strongest unfinished workflow before opening a
        # different hypothesis or debugging an unrelated failure.
        if partial:
            return max(partial, key=lambda node: self._priority(tree, node))

        choose_failure = bool(failed) and (
            not successful or self._random.random() < self.debug_probability
        )
        pool = failed if choose_failure else successful
        if not pool:
            pool = failed
        if not pool:
            return None
        return max(pool, key=lambda node: self._priority(tree, node))

    def remaining_child_slots(self, tree: ExperimentTree, node: ExperimentNode) -> int:
        """Return how many more alternative children ``node`` may spawn."""
        return max(0, self.max_children_per_node - tree.child_count(node.id))

    def _priority(self, tree: ExperimentTree, node: ExperimentNode) -> float:
        """Combine scientific quality with an under-expansion bonus."""
        total_expansions = sum(tree.child_count(item.id) for item in tree.nodes)
        # Prefer under-expanded, shallower checkpoints when scientific quality
        # is comparable. Dividing by depth is important: otherwise every new
        # leaf receives the largest exploration bonus and the search collapses
        # back into chains despite internal nodes remaining technically valid.
        exploration = (
            self.exploration_weight
            * math.sqrt(
                math.log(total_expansions + 2) / (tree.child_count(node.id) + 1)
            )
            / (node.depth + 1)
        )
        return evaluation_priority(node, self.weights) + exploration

    @staticmethod
    def _abandoned(node: ExperimentNode) -> bool:
        return (
            node.evaluation is not None
            and node.evaluation.recommendation == Recommendation.ABANDON
        )
