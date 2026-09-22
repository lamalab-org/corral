"""Construct, run, or finish a Corral MD workflow evaluation.

No review file is discovered in submitted artifacts. Passing ``--review`` is an
explicit trusted evaluator action; review identity is an audit label, not an
authentication mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib import import_module
from pathlib import Path

from corral_md.workflow_scoring.common import (
    Evidence,
    Rubric,
    UnsupportedEvidence,
    level1_reproducibility,
    reproducibility,
)


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

    def __init__(self, task_number: int, *, level: int = 2, verifier=None):
        if type(task_number) is not int or task_number not in range(1, 11):
            raise ValueError("task_number must be an integer from 1 to 10")
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.task_number = task_number
        self.level = level
        self.verifier = verifier

    @property
    def module(self):
        return import_module(f"corral_md.workflow_scoring.task_{self.task_number}")

    def __call__(self, submission) -> float:
        """Return a completed scalar, never a zero standing in for pending review."""
        return self.score_submission(submission)

    def score_submission(self, submission, *, review: dict | None = None) -> float:
        report = self.evaluate(submission, review=review)
        if report["score"] is None:
            raise PendingReviewError(report)
        return float(report["score"])

    def evaluate(self, submission, *, review: dict | None = None) -> dict:
        """Inspect evidence; optional review is trusted evaluator input only."""
        rubric = Rubric(self.task_number, binary=True)
        try:
            evidence = Evidence(submission)
        except Exception as exc:
            rubric.check(
                "readable_manifest", 100, False, f"{type(exc).__name__}: {exc}"
            )
            if review is not None:
                raise ValueError(
                    "An unreadable submission cannot receive review credit"
                ) from exc
            report = rubric.as_dict()
            report["level"] = self.level
            return report
        initial_fingerprint = None
        if self.level == 1:
            level1_reproducibility(evidence, rubric)
        else:
            reproducibility(evidence, rubric)
        if rubric.failed:
            remaining = max(0, 100 - sum(c["points"] for c in rubric.checks))
            rubric.check(
                "remaining_evidence",
                remaining,
                False,
                "Task checks were not evaluated after a reproducibility check failed.",
            )
        else:
            if self.verifier is not None:
                initial_fingerprint = evidence.fingerprint()
            rubric.fail_fast = True
            try:
                if self.level == 1:
                    from corral_md.workflow_scoring.level1 import (
                        evaluate as evaluate_level1,
                    )

                    evaluate_level1(evidence, rubric, self.task_number)
                else:
                    self.module.evaluate(evidence, rubric)
            except UnsupportedEvidence as exc:
                remaining = max(0, 100 - sum(c["points"] for c in rubric.checks))
                rubric.check("remaining_evidence", remaining, None, str(exc))
            except Exception as exc:
                remaining = max(0, 100 - sum(c["points"] for c in rubric.checks))
                rubric.check(
                    "remaining_evidence",
                    remaining,
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            remaining = max(0, 100 - sum(c["points"] for c in rubric.checks))
            if rubric.failed and remaining:
                rubric.check(
                    "remaining_evidence",
                    remaining,
                    False,
                    "Remaining task checks were not evaluated after a failure.",
                )
        if not any(check["name"] == "execution_provenance" for check in rubric.checks):
            rubric.unverified(
                "execution_provenance",
                "Only submitted artifacts were inspected. Actual simulator/model calls and absence of undisclosed resets or test-data use are not independently attested.",
            )
        verification = None
        if self.verifier is not None and not rubric.failed:
            from corral_md.workflow_scoring.verification import (
                apply_verification,
            )

            verification = self.verifier.evaluate(evidence, self.task_number)
            if verification["evidence_sha256"] != initial_fingerprint:
                raise RuntimeError(
                    "Submission changed during grading; retry with frozen evidence"
                )
            apply_verification(rubric, verification)
        evidence_sha256 = None
        if rubric.pending_checks or review is not None:
            backend_release = (
                verification.get("backend", {}).get("release_id")
                if verification
                else None
            )
            evidence_sha256 = hashlib.sha256(
                (
                    f"corral_md.workflow.level_{self.level}.task_{self.task_number}"
                    + (".modal" if self.verifier is not None else ".offline")
                    + "\0"
                    + evidence.fingerprint()
                    + "\0"
                    + json.dumps(backend_release)
                ).encode()
            ).hexdigest()
        if review is not None:
            rubric.apply_review(review, evidence_sha256)
        report = rubric.as_dict()
        report["level"] = self.level
        report["evidence_sha256"] = evidence_sha256
        if verification is not None:
            report["mode"] = "artifact_and_modal"
            report["verification"] = verification
        return report


def check_level2_workflow(
    task_number: int, verification_backend: str | None = None
) -> WorkflowScorer:
    """Construct a workflow scorer with optional independent verification."""
    backend = verification_backend or os.getenv("CORRAL_MD_VERIFICATION", "offline")
    if backend == "offline":
        return WorkflowScorer(task_number, level=2)
    if backend != "modal":
        raise ValueError("verification_backend must be 'offline' or 'modal'")

    from corral_md.workflow_scoring.verification import ModalVerifier

    return WorkflowScorer(
        task_number, level=2, verifier=ModalVerifier(require_provenance=True)
    )


def check_level1_workflow(
    task_number: int, verification_backend: str | None = None
) -> WorkflowScorer:
    """Construct the artifact-only scorer for one preparatory workflow."""
    # The current Modal verification plans cover the complete Level 2 workflow.
    # Do not let a process-wide Level 2 deployment setting make Level 1 tasks
    # un-loadable or accidentally verify later-stage artifacts.
    backend = verification_backend or "offline"
    if backend != "offline":
        if backend == "modal":
            raise ValueError(
                "Level 1 scoring is artifact-only; verification_backend must be 'offline'"
            )
        raise ValueError("verification_backend must be 'offline'")
    return WorkflowScorer(task_number, level=1)


__all__ = [
    "PendingReviewError",
    "WorkflowScorer",
    "check_level1_workflow",
    "check_level2_workflow",
    "main",
]


def main() -> int:
    """Run the evaluator CLI, optionally with independent Modal verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_number", type=int, choices=range(1, 11))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    parser.add_argument("--review", type=Path, help="Trusted evaluator review JSON")
    parser.add_argument("--output", type=Path, help="Write the evaluation report here")
    parser.add_argument(
        "--verification",
        choices=("offline", "modal"),
        help="Independent Modal calculator and provenance checks",
    )
    parser.add_argument(
        "--release-id", help="Evaluator-selected pinned verifier release"
    )
    parser.add_argument("--run-id", help="Bind provenance to the evaluated execution")
    parser.add_argument(
        "--action-id", help="Bind provenance to its controlled MD action"
    )
    args = parser.parse_args()
    review = json.loads(args.review.read_text()) if args.review else None
    factory = check_level1_workflow if args.level == 1 else check_level2_workflow
    scorer = factory(args.task_number, verification_backend=args.verification)
    if args.release_id or args.run_id or args.action_id:
        if scorer.verifier is None:
            parser.error("--release-id/--run-id/--action-id require Modal verification")
        scorer.verifier.release_id = args.release_id
        scorer.verifier.run_id = args.run_id
        scorer.verifier.action_id = args.action_id
    report = scorer.evaluate(args.manifest, review=review)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 2 if report["status"] == "pending_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
