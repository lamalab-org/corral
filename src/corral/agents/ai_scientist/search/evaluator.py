"""Deterministic objective extraction and experiment-tree ranking."""

import ast
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    MeasuredMetric,
    MeasuredOutcome,
    Observation,
)

if TYPE_CHECKING:
    from corral.agents.ai_scientist.state import TaskFormulation


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


def node_ranking_key(
    node: ExperimentNode,
    *,
    critic_bonus: float = 0.0,
    weights: EvaluationWeights | None = None,
) -> tuple[int, float, float]:
    """Rank measured outcomes first, using the critic as support and fallback.

    Objective values are intentionally not mixed into a weighted LLM score:
    their arbitrary units would make such a sum meaningless. Within a search
    whose nodes expose the same declared metric, the signed observed value is
    therefore the primary key. The critic remains the tie-breaker and the sole
    key for environments without a sensible scalar.
    """
    critic = evaluation_priority(node, weights) + critic_bonus
    if node.measured_outcome is None:
        return (0, critic, critic)
    return (1, node.measured_outcome.directional_value, critic)


def measured_improvement(
    candidate: ExperimentNode,
    baseline: ExperimentNode,
) -> float | None:
    """Return directional objective improvement for comparable observations."""
    candidate_outcome = candidate.measured_outcome
    baseline_outcome = baseline.measured_outcome
    if candidate_outcome is None or baseline_outcome is None:
        return None
    if (
        candidate_outcome.name.casefold() != baseline_outcome.name.casefold()
        or candidate_outcome.maximize != baseline_outcome.maximize
    ):
        return None
    return candidate_outcome.directional_value - baseline_outcome.directional_value


def extract_measured_outcome(
    node: ExperimentNode,
    formulation: "TaskFormulation",
) -> MeasuredOutcome | None:
    """Parse a declared metric from successful physical tool observations.

    Extraction requires the task-wide metric established during formulation.
    An explicit ``measured_outcome`` object is accepted only when its name and
    direction match that declaration; otherwise a tool could make an unrelated
    scalar outrank the research objective. Matching JSON/dict keys and results
    consisting solely of one number are also accepted. This deliberately avoids
    mining arbitrary numbers from prose and can never consult Corral scoring.
    """
    metric = formulation.measured_metric
    if metric is None:
        return None
    for observation in reversed(node.observations):
        if not observation.success or observation.result is None:
            continue
        payload = _decode_observation(observation.result)
        explicit = _explicit_outcome(payload, observation, metric)
        if explicit is not None:
            return explicit
        value, path = _declared_metric_value(payload, metric)
        if value is None:
            continue
        return MeasuredOutcome(
            name=metric.name,
            value=value,
            maximize=metric.maximize,
            unit=metric.unit,
            provenance=_provenance(observation, path),
        )
    return None


def _decode_observation(result: str) -> Any:
    text = result.strip()
    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(text)
        except (ValueError, SyntaxError, TypeError):
            pass
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return float(text)
    return text


def _explicit_outcome(
    payload: Any,
    observation: Observation,
    metric: MeasuredMetric,
) -> MeasuredOutcome | None:
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("measured_outcome")
    path = "measured_outcome"
    if candidate is None and isinstance(payload.get("metric"), dict):
        candidate = payload["metric"]
        path = "metric"
    if not isinstance(candidate, dict):
        return None
    try:
        outcome = MeasuredOutcome.model_validate(
            {
                **candidate,
                "provenance": _provenance(observation, path),
            }
        )
    except (TypeError, ValueError):
        return None
    if (
        _normalise_metric_name(outcome.name) != _normalise_metric_name(metric.name)
        or outcome.maximize != metric.maximize
    ):
        return None
    return outcome.model_copy(
        update={"name": metric.name, "unit": metric.unit, "maximize": metric.maximize}
    )


def _declared_metric_value(
    payload: Any, metric: MeasuredMetric
) -> tuple[float | None, str]:
    if _is_number(payload):
        return float(payload), "value"
    wanted = _normalise_metric_name(metric.name)
    stack: list[tuple[str, Any]] = [("result", payload)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                child_path = f"{path}.{key}"
                if _normalise_metric_name(str(key)) == wanted and _is_number(value):
                    return float(value), child_path
                stack.append((child_path, value))
        elif isinstance(item, list):
            stack.extend(
                (f"{path}[{index}]", value) for index, value in enumerate(item)
            )
    return None, ""


def _provenance(observation: Observation, path: str) -> str:
    return (
        f"tool={observation.tool_name};observation={observation.action_index};"
        f"path={path}"
    )


def _normalise_metric_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
