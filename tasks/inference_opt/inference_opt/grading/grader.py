"""Grade runs, each in its own process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from inference_opt.grading.spec import GradeSpec, GradeSummary

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Grader"]


@dataclass(frozen=True, slots=True)
class Grader:
    """Inspect allows one evaluation per process, so each grade gets its own."""

    def grade(self, spec: GradeSpec, targets: dict[str, str]) -> GradeSummary:
        return self.grade_many([(spec, targets)])[0]

    def grade_many(
        self, jobs: Sequence[tuple[GradeSpec, dict[str, str]]]
    ) -> list[GradeSummary]:
        if not jobs:
            return []
        private = Path(tempfile.mkdtemp(prefix="inference-opt-grade-"))
        try:
            processes = [
                _spawn(spec, targets, private / f"spec-{index}.json")
                for index, (spec, targets) in enumerate(jobs)
            ]
            for process in processes:
                process.wait()
        finally:
            shutil.rmtree(private, ignore_errors=True)
        return [_read(spec, process) for (spec, _), process in zip(jobs, processes, strict=True)]


def _spawn(
    spec: GradeSpec, targets: dict[str, str], spec_path: Path
) -> subprocess.Popen[bytes]:
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(spec.summary_path).unlink(missing_ok=True)
    spec.write(spec_path)
    # Inspect otherwise writes a process-global trace file under the user's
    # application-data directory; keep it beside this run's private log.
    env = {**os.environ, "INSPECT_TRACE_FILE": str(out / "inspect-trace.log")}
    process = subprocess.Popen(
        [sys.executable, "-m", "inference_opt.grading", str(spec_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        env=env,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(targets).encode("utf-8"))
    process.stdin.close()
    return process


def _read(spec: GradeSpec, process: subprocess.Popen[bytes]) -> GradeSummary:
    try:
        return GradeSummary.read(spec.summary_path)
    except (OSError, ValueError):
        return GradeSummary(
            run_id=spec.run_id,
            error=f"grader exited with code {process.returncode} without a summary",
        )
