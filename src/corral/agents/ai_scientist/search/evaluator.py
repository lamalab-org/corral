"""Deterministic conversion of model evaluation dimensions to tree priority."""

from dataclasses import dataclass

from corral.agents.ai_scientist.search.nodes import ExperimentNode


@dataclass(frozen=True)
class EvaluationWeights:
    validity: float = 0.20
    task_progress: float = 0.25
    evidence_strength: float = 0.25
    information_gain: float = 0.20
    consistency: float = 0.10
    depth_penalty: float = 0.01
    failed_node_penalty: float = 0.15


def evaluation_priority(
    node: ExperimentNode, weights: EvaluationWeights | None = None
) -> float:
    """Score a node without consulting the benchmark scorer."""
    weights = weights or EvaluationWeights()
    if node.evaluation is None:
        return float("-inf")
    evaluation = node.evaluation
    score = (
        weights.validity * evaluation.validity
        + weights.task_progress * evaluation.task_progress
        + weights.evidence_strength * evaluation.evidence_strength
        + weights.information_gain * evaluation.information_gain
        + weights.consistency * evaluation.consistency
        - weights.depth_penalty * node.depth
    )
    if node.status.value in {"failed", "invalid"}:
        score -= weights.failed_node_penalty
    return score
