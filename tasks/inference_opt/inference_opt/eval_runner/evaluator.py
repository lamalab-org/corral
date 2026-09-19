"""Evaluate one inference-opt policy run."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from inference_opt.eval_runner.__main__ import run
from inference_opt.eval_runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator"]


@dataclass(frozen=True, slots=True)
class PolicyEvaluator:
    """Run policy code in the current task process."""

    def run(self, spec: RunSpec) -> RunSummary:
        # Inspect otherwise writes a process-global trace file under the
        # user's application-data directory. In-process evaluations must keep
        # that artifact task-local and must not collide with another run.
        trace_file = Path(spec.out_dir) / "inspect-trace.log"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        with _environment("INSPECT_TRACE_FILE", str(trace_file)):
            return run(spec)


@contextmanager
def _environment(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
