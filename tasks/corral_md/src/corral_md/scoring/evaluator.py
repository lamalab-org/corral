"""Construct or run a Corral MD numerical evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

from corral.evaluation import SubmissionScore

from .common import EvaluationError, Evidence, Rubric
from .final_quantities import FinalQuantityScorer
from .level2_quantities import Level2QuantityScorer
from .task_2 import Task2Scorer
from .task_3_temperature import Task3TemperatureScorer
from .task_6 import Task6Scorer


class WorkflowScorer:
    """Score one Corral MD workflow from its retained evidence."""

    def __init__(
        self,
        task_number: int,
        *,
        level: int = 2,
        verifier=None,
    ):
        if type(task_number) is not int or task_number not in range(1, 11):
            raise ValueError("task_number must be an integer from 1 to 10")
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        if task_number not in (1, 4, 8) and (task_number, level) not in (
            (3, 1),
            (5, 2),
        ):
            raise ValueError(
                f"Task {task_number} Level {level} uses a dedicated scorer; "
                f"use check_level{level}_workflow({task_number})"
            )
        self.task_number = task_number
        self.level = level
        self.verifier = verifier

    @property
    def module(self):
        return import_module(f"corral_md.scoring.task_{self.task_number}")

    def __call__(self, submission) -> float:
        """Return a completed scalar or raise an explicit evaluation error."""
        return self.score_submission(submission)

    def score_submission(self, submission) -> float:
        return self.evaluate_submission(submission).score

    def evaluate_submission(self, submission) -> SubmissionScore:
        """Retain check diagnostics when invoked by the benchmark reporting adapter."""
        report = self.evaluate(submission)
        failures = [
            f"{check['name']}: {check['detail'] or 'Required evidence check failed'}"
            for check in report["checks"]
            if check["status"] == "failed"
        ]
        return SubmissionScore(
            score=float(report["score"]),
            feedback="; ".join(failures) or None,
            metadata={"workflow_evaluation": report},
        )

    def evaluate(self, submission) -> dict:
        """Calculate a score directly from the retained evidence."""
        if self.task_number in (3, 4, 5, 8):
            return self.module.evaluate_submission(
                submission, level=self.level, verifier=self.verifier
            )
        rubric = Rubric(self.task_number)
        try:
            evidence = Evidence(
                submission, task_number=self.task_number, level=self.level
            )
        except Exception as exc:
            rubric.error("readable_manifest", 100, exc)
            report = rubric.as_dict()
            report["level"] = self.level
            return report
        if self.task_number == 1:
            rubric.fail_fast = True
            diagnostics = {}
            try:
                diagnostics = self.module.evaluate(
                    evidence,
                    rubric,
                    level=self.level,
                )
            except (ValueError, KeyError, IndexError, OSError, TypeError) as exc:
                remaining = 100 - sum(c["points"] for c in rubric.checks)
                rubric.error("usable_recorded_evidence", remaining, exc)
            fingerprint = evidence.fingerprint()
            return {
                **rubric.as_dict(),
                "mode": "final_temperature" if self.level == 1 else "final_quantities",
                "scoring_policy": (
                    self.module.TEMPERATURE_POLICY
                    if self.level == 1
                    else self.module.DIFFUSION_POLICY
                ),
                "level": self.level,
                "evidence_sha256": fingerprint,
                "calculated": diagnostics,
            }

        raise ValueError("Task uses a dedicated scorer")


def check_level2_workflow(
    task_number: int,
    verification_backend: str | None = None,
    assets_dir: str | None = None,
) -> WorkflowScorer | Task2Scorer | Task3TemperatureScorer | Level2QuantityScorer:
    """Construct the default Level 2 numerical scoring policy."""
    if type(task_number) is not int or task_number not in range(1, 11):
        raise ValueError("task_number must be an integer from 1 to 10")
    if task_number == 1:
        return WorkflowScorer(1, level=2)
    if task_number == 2:
        return Task2Scorer(level=2)
    if task_number == 3:
        return Task3TemperatureScorer()
    if task_number in (6, 7, 9, 10):
        return Level2QuantityScorer(task_number)
    backend = verification_backend or os.getenv(
        "CORRAL_MD_VERIFICATION",
        "modal",
    )
    if backend == "offline":
        return WorkflowScorer(task_number, level=2)
    if backend == "local":
        from corral_md.scoring.local_verifier import LocalVerifier

        verifier = LocalVerifier(assets_dir)
        return WorkflowScorer(task_number, level=2, verifier=verifier)
    if backend != "modal":
        raise ValueError("verification_backend must be 'offline', 'local', or 'modal'")
    from corral_md.scoring.verification import ModalVerifier

    return WorkflowScorer(task_number, level=2, verifier=ModalVerifier())


def check_level1_workflow(
    task_number: int,
    verification_backend: str | None = None,
    assets_dir: str | None = None,
) -> WorkflowScorer | Task2Scorer | Task6Scorer | FinalQuantityScorer:
    """Construct the default Level 1 numerical scoring policy."""
    if type(task_number) is not int or task_number not in range(1, 11):
        raise ValueError("task_number must be an integer from 1 to 10")
    if task_number == 2:
        return Task2Scorer(level=1)
    if task_number == 6:
        return Task6Scorer()
    if task_number in (7, 9, 10):
        return FinalQuantityScorer(task_number)
    if task_number == 1:
        return WorkflowScorer(1, level=1)
    backend = verification_backend or "modal"
    if task_number == 5:
        verifier = None
        if backend == "local":
            from .local_verifier import Level1LocalVerifier

            verifier = Level1LocalVerifier(assets_dir)
        elif backend == "modal":
            from .level1_trusted import Level1ModalVerifier

            verifier = Level1ModalVerifier()
        elif backend != "offline":
            raise ValueError(
                "verification_backend must be 'offline', 'local', or 'modal'"
            )
        return FinalQuantityScorer(task_number, verifier=verifier)
    if backend == "offline":
        return WorkflowScorer(task_number, level=1)
    if backend == "local":
        from corral_md.scoring.local_verifier import (
            Level1LocalVerifier,
        )

        return WorkflowScorer(
            task_number,
            level=1,
            verifier=Level1LocalVerifier(assets_dir),
        )
    if backend != "modal":
        raise ValueError("verification_backend must be 'offline', 'local', or 'modal'")
    from corral_md.scoring.level1_trusted import Level1ModalVerifier

    return WorkflowScorer(task_number, level=1, verifier=Level1ModalVerifier())


__all__ = [
    "EvaluationError",
    "WorkflowScorer",
    "check_level1_workflow",
    "check_level2_workflow",
    "main",
]


def main() -> int:
    """Run the evaluator CLI with offline, local, or Modal verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_number", type=int, choices=range(1, 11))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, help="Write the evaluation report here")
    parser.add_argument(
        "--verification",
        choices=("offline", "local", "modal"),
        default="local",
        help="Independent calculation backend (default: local)",
    )
    parser.add_argument("--assets-dir", help="Prepared catalogs for local verification")
    args = parser.parse_args()
    factory = check_level1_workflow if args.level == 1 else check_level2_workflow
    scorer = factory(
        args.task_number,
        verification_backend=args.verification,
        assets_dir=args.assets_dir,
    )
    try:
        report = scorer.evaluate(args.manifest)
    except EvaluationError as exc:
        report = {**exc.report, "task_number": args.task_number, "level": args.level}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 2 if report["status"] == "evaluation_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
