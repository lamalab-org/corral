"""External experiment-tree state owned by the manager."""

from corral.agents.ai_scientist.search.evaluator import node_ranking_key
from corral.agents.ai_scientist.search.nodes import (
    ExecutedAction,
    ExperimentNode,
    NodeStatus,
    Recommendation,
    ResearchStage,
)


class ExperimentTree:
    """A deterministic insertion-ordered tree of experiment nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, ExperimentNode] = {}
        self._children: dict[str | None, list[str]] = {}
        self._next_number = 1

    def next_id(self) -> str:
        node_id = f"node_{self._next_number:04d}"
        self._next_number += 1
        return node_id

    def add(self, node: ExperimentNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Duplicate experiment node id: {node.id}")
        if node.parent_id is not None and node.parent_id not in self._nodes:
            raise ValueError(f"Unknown parent node: {node.parent_id}")
        self._nodes[node.id] = node
        self._children.setdefault(node.parent_id, []).append(node.id)

    def get(self, node_id: str) -> ExperimentNode:
        return self._nodes[node_id]

    def children(self, node_id: str | None) -> list[ExperimentNode]:
        return [self._nodes[item] for item in self._children.get(node_id, [])]

    def child_count(self, node_id: str) -> int:
        """Return the number of alternative continuations from a checkpoint."""
        return len(self._children.get(node_id, []))

    @property
    def nodes(self) -> list[ExperimentNode]:
        return list(self._nodes.values())

    def by_stage(
        self, stage: ResearchStage, *, include_boundary: bool = True
    ) -> list[ExperimentNode]:
        return [
            node
            for node in self._nodes.values()
            if node.stage == stage
            and (include_boundary or not node.boundary_validation)
        ]

    def leaves(self) -> list[ExperimentNode]:
        return [node for node in self._nodes.values() if not self.children(node.id)]

    def trajectory(self, node_id: str) -> list[ExperimentNode]:
        """Return the root-to-node logical trajectory, inclusive."""
        path: list[ExperimentNode] = []
        node = self.get(node_id)
        while True:
            path.append(node)
            if node.parent_id is None:
                break
            node = self.get(node.parent_id)
        return list(reversed(path))

    def executed_trajectory(self, node_id: str) -> list[ExecutedAction]:
        """Return the physical actions needed to reconstruct ``node_id``."""
        return [
            executed
            for node in self.trajectory(node_id)
            for executed in node.executed_actions
        ]

    def best(self, limit: int = 3) -> list[ExperimentNode]:
        evaluated = [
            node
            for node in self._nodes.values()
            if node.evaluation is not None
            and node.status == NodeStatus.SUCCESSFUL
            and node.evaluation.recommendation != Recommendation.ABANDON
        ]
        return sorted(evaluated, key=node_ranking_key, reverse=True)[:limit]

    def model_dump(self) -> dict[str, object]:
        return {"nodes": [node.model_dump(mode="json") for node in self.nodes]}

    def __len__(self) -> int:
        return len(self._nodes)
