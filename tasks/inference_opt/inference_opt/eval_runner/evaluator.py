"""Evaluate one inference-opt policy run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from inference_opt.eval_runner.__main__ import run, scrubbed_environment
from inference_opt.eval_runner.spec import RunSpec, RunSummary

__all__ = ["PolicyEvaluator"]


@dataclass(frozen=True, slots=True)
class PolicyEvaluator:
    """Run policy code in the current task process."""

    def run(
        self, spec: RunSpec, targets: dict[str, str] | None = None
    ) -> RunSummary:
        # Inspect otherwise writes a process-global trace file under the
        # user's application-data directory. In-process evaluations must keep
        # that artifact task-local and must not collide with another run.
        trace_file = Path(spec.out_dir) / "inspect-trace.log"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        api_key = spec.api_key or os.environ.get("VLLM_API_KEY")
        safe_spec = replace(spec, api_key=api_key)
        with (
            _environment("INSPECT_TRACE_FILE", str(trace_file)),
            scrubbed_environment(),
        ):
            return run(safe_spec, targets=targets)

    def run_many(
        self, jobs: Sequence[tuple[RunSpec, dict[str, str] | None]]
    ) -> list[RunSummary]:
        """Evaluate several specs at once, e.g. one per student model.

        A single job runs in-process. Several jobs each get their own process,
        because an in-process run swaps process-global environment variables and
        Inspect state that concurrent runs would clobber.
        """
        if len(jobs) <= 1:
            return [self.run(spec, targets=targets) for spec, targets in jobs]
        private = Path(tempfile.mkdtemp(prefix="inference-opt-specs-"))
        try:
            processes = [
                _spawn(spec, targets, private / f"spec-{index}.json")
                for index, (spec, targets) in enumerate(jobs)
            ]
            for process in processes:
                process.wait()
        finally:
            shutil.rmtree(private, ignore_errors=True)
        return [_read_summary(spec, process) for (spec, _), process in zip(jobs, processes)]


def _spawn(
    spec: RunSpec, targets: dict[str, str] | None, spec_path: Path
) -> subprocess.Popen[bytes]:
    trace_file = Path(spec.out_dir) / "inspect-trace.log"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    # A stale summary would pass off a crashed child as a finished run.
    Path(spec.summary_path).unlink(missing_ok=True)
    env = {**os.environ, "INSPECT_TRACE_FILE": str(trace_file)}
    if spec.api_key:
        env["VLLM_API_KEY"] = spec.api_key
    # The key travels in the environment, not in a file beside the questions.
    replace(spec, api_key=None).write(spec_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "inference_opt.eval_runner", str(spec_path), "--targets-stdin"],
        stdin=subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(targets or {}).encode("utf-8"))
    process.stdin.close()
    return process


def _read_summary(spec: RunSpec, process: subprocess.Popen[bytes]) -> RunSummary:
    try:
        return RunSummary.read(spec.summary_path)
    except (OSError, ValueError):
        return RunSummary(
            run_id=spec.run_id,
            ok=False,
            error=f"evaluator process exited with code {process.returncode} "
            "without writing a summary",
        )


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
