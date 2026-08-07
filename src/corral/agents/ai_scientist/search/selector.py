"""Best-first selection with bounded probabilistic debugging."""

import random

from corral.agents.ai_scientist.search.evaluator import (
    EvaluationWeights,
    evaluation_priority,
)
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeStatus,
    Recommendation,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree


class TreeSelector:
    def __init__(
        self,
        *,
        debug_probability: float = 0.2,
        max_debug_depth: int = 2,
        random_seed: int = 0,
        weights: EvaluationWeights | None = None,
    ) -> None:
        self.debug_probability = debug_probability
        self.max_debug_depth = max_debug_depth
        self._random = random.Random(random_seed)
        self.weights = weights or EvaluationWeights()

    def select(self, tree: ExperimentTree) -> ExperimentNode | None:
        """Select a promising leaf, occasionally choosing a debuggable failure."""
        candidates = tree.leaves()
        selected = self._select_from(candidates)
        if selected is not None:
            return selected

        # A capped failed leaf must not make the whole search tree a dead end.
        # Fall back to any earlier successful checkpoint so the manager can try
        # a genuinely different child branch.
        return self._select_from(tree.nodes, allow_failures=False)

    def _select_from(
        self,
        candidates: list[ExperimentNode],
        *,
        allow_failures: bool = True,
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

        choose_failure = bool(failed) and (
            not successful or self._random.random() < self.debug_probability
        )
        pool = failed if choose_failure else successful
        if not pool:
            pool = failed
        if not pool:
            return None
        return max(pool, key=lambda node: evaluation_priority(node, self.weights))

    @staticmethod
    def _abandoned(node: ExperimentNode) -> bool:
        return (
            node.evaluation is not None
            and node.evaluation.recommendation == Recommendation.ABANDON
        )
