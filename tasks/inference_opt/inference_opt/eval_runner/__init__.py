"""Run inference-opt policies."""

from __future__ import annotations

from inference_opt.eval_runner.evaluator import PolicyEvaluator
from inference_opt.eval_runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator", "RunSpec", "RunSummary", "run_in_process"]


def run_in_process(spec: RunSpec) -> RunSummary:
    """Compatibility helper for tests and local development."""
    return PolicyEvaluator().run(spec)
