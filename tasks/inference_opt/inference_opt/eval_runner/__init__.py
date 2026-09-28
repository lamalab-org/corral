"""Run inference-opt policies."""

from __future__ import annotations

from inference_opt.eval_runner.evaluator import PolicyEvaluator, evaluate_in_process
from inference_opt.eval_runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator", "RunSpec", "RunSummary", "run_in_process"]


def run_in_process(spec: RunSpec) -> RunSummary:
    """Evaluate in this process, for tests and local development."""
    return evaluate_in_process(spec)
