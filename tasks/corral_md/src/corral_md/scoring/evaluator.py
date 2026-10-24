"""Construct, run, or finish a Corral MD workflow evaluation.

No review file is discovered in submitted artifacts. Passing ``--review`` is an
explicit trusted evaluator action; review identity is an audit label, not an
authentication mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

from corral.evaluation import SubmissionScore

from .common import Evidence, Rubric
from .task_2 import Task2Scorer
from .task_6 import Task6Scorer
from .task_7 import Task7Scorer


class PendingReviewError(RuntimeError):
    """No task score exists yet because evidence needs independent review."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            "Workflow evidence requires review before scoring: "
            + ", ".join(report["pending_checks"])
        )


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
        if task_number in (2, 6, 7):
            raise ValueError(f"Task {task_number} uses Task{task_number}Scorer")
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.task_number = task_number
        self.level = level
        self.verifier = verifier

    @property
    def module(self):
        return import_module(f"corral_md.scoring.task_{self.task_number}")

    def __call__(self, submission) -> float:
        """Return a completed scalar, never a zero standing in for pending review."""
        return self.score_submission(submission)

    def score_submission(self, submission, *, review: dict | None = None) -> float:
        return self.evaluate_submission(submission, review=review).score

    def evaluate_submission(
        self, submission, *, review: dict | None = None
    ) -> SubmissionScore:
        """Retain check diagnostics when invoked by the benchmark reporting adapter."""
        report = self.evaluate(submission, review=review)
        if report["score"] is None:
            raise PendingReviewError(report)
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

    def evaluate(self, submission, *, review: dict | None = None) -> dict:
        """Inspect evidence; optional review is trusted evaluator input only."""
        if self.task_number in (3, 4, 5, 8, 9, 10):
            return self.module.evaluate_submission(
                submission, level=self.level, verifier=self.verifier, review=review
            )
        rubric = Rubric(self.task_number, binary=True)
        try:
            evidence = Evidence(
                submission, task_number=self.task_number, level=self.level
            )
        except Exception as exc:
            rubric.error("readable_manifest", 100, exc)
            if review is not None:
                raise ValueError(
                    "An unreadable submission cannot receive review credit"
                ) from exc
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
            if review is not None:
                rubric.apply_review(review, fingerprint)
            return {
                **rubric.as_dict(),
                "level": self.level,
                "evidence_sha256": fingerprint,
                "calculated": diagnostics,
            }

        raise ValueError("Task uses a dedicated scorer")


def _task6_scorer(level, verification_backend, assets_dir):
    backend = verification_backend or os.getenv("CORRAL_MD_VERIFICATION", "modal")
    if backend == "offline":
        return Task6Scorer(level=level)
    if backend == "local":
        from .local_verifier import LocalVerifier

        return Task6Scorer(level=level, verifier=LocalVerifier(assets_dir))
    if backend == "modal":
        from .verification import ModalVerifier

        return Task6Scorer(level=level, verifier=ModalVerifier())
    raise ValueError("verification_backend must be 'offline', 'local', or 'modal'")


def check_level2_workflow(
    task_number: int,
    verification_backend: str | None = None,
    assets_dir: str | None = None,
) -> WorkflowScorer | Task2Scorer | Task6Scorer | Task7Scorer:
    """Use recorded silicon evidence and independent verification for silver."""
    if type(task_number) is not int or task_number not in range(1, 11):
        raise ValueError("task_number must be an integer from 1 to 10")
    if task_number == 1:
        return WorkflowScorer(1, level=2)
    if task_number == 2:
        return Task2Scorer(level=2)
    if task_number == 6:
        return _task6_scorer(2, verification_backend, assets_dir)
    if task_number == 7:
        return Task7Scorer(level=2)
    backend = verification_backend or os.getenv(
        "CORRAL_MD_VERIFICATION",
        "modal" if task_number in (3, 4, 5, 8, 9, 10) else "offline",
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
) -> WorkflowScorer | Task2Scorer | Task6Scorer | Task7Scorer:
    """Construct the Level 1 scorer with independent MACE checks by default."""
    if type(task_number) is not int or task_number not in range(1, 11):
        raise ValueError("task_number must be an integer from 1 to 10")
    if task_number == 2:
        return Task2Scorer(level=1)
    if task_number == 6:
        return _task6_scorer(1, verification_backend, assets_dir)
    if task_number == 7:
        return Task7Scorer(level=1)
    if task_number == 1:
        return WorkflowScorer(1, level=1)
    backend = verification_backend or ("modal" if task_number >= 3 else "offline")
    if backend == "offline":
        return WorkflowScorer(task_number, level=1)
    if backend == "local":
        from corral_md.scoring.local_verifier import (
            Level1LocalVerifier,
        )

        return WorkflowScorer(
            task_number,
            level=1,
            verifier=Level1LocalVerifier(assets_dir) if task_number >= 3 else None,
        )
    if backend != "modal":
        raise ValueError("verification_backend must be 'offline', 'local', or 'modal'")
    if task_number < 3:
        raise ValueError("Level 1 Modal verification applies to tasks 3-10")
    from corral_md.scoring.level1_trusted import Level1ModalVerifier

    return WorkflowScorer(task_number, level=1, verifier=Level1ModalVerifier())


__all__ = [
    "PendingReviewError",
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
    parser.add_argument("--review", type=Path, help="Trusted evaluator review JSON")
    parser.add_argument("--output", type=Path, help="Write the evaluation report here")
    parser.add_argument(
        "--verification",
        choices=("offline", "local", "modal"),
        default="local",
        help="Independent calculation backend (default: local)",
    )
    parser.add_argument("--assets-dir", help="Prepared catalogs for local verification")
    args = parser.parse_args()
    review = json.loads(args.review.read_text()) if args.review else None
    factory = check_level1_workflow if args.level == 1 else check_level2_workflow
    scorer = factory(
        args.task_number,
        verification_backend=args.verification,
        assets_dir=args.assets_dir,
    )
    if isinstance(scorer, Task2Scorer | Task6Scorer | Task7Scorer):
        if review is not None:
            parser.error(
                f"Task {scorer.task_number} is scored directly from logs and does not use reviews"
            )
        report = scorer.evaluate(args.manifest)
    else:
        report = scorer.evaluate(args.manifest, review=review)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 2 if report["status"] == "pending_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
