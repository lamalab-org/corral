"""Independent MACE checks for the preparatory Level 1 workflows.

Only evaluator-selected geometry and calculator parameters go to the trusted
Modal worker. Submitted scripts and claimed answers remain local. The worker
uses the release-pinned teacher checkpoint, never a path supplied by an agent.
"""

from __future__ import annotations

import secrets

from .verification import ModalVerifier, Plan

# This is the immutable teacher asset in modal_app/assets.json. Keep the value
# independent of submission metadata: a submitted digest is only a claim.
PINNED_TEACHER_SHA256 = (
    "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845"
)


def build_level1_plan(e, task_number: int, challenge: str) -> Plan:
    """Build the remaining shared Level 1 phonon verification plan."""
    if task_number != 4:
        raise ValueError("This task uses its own independent calculation plan")
    from corral_md.scoring.task_4 import build_plan

    return build_plan(e, challenge)


class Level1ModalVerifier(ModalVerifier):
    """Use the existing isolated worker with Level 1 evidence contracts."""

    def evaluate(self, evidence, task_number):
        fingerprint = evidence.fingerprint()
        challenge = secrets.token_hex(32)
        plan = build_level1_plan(evidence, task_number, challenge)
        checks, remotes, aborted = self._evaluate_plan(plan, fingerprint)
        if evidence.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during independent verification; "
                "retry with frozen evidence"
            )
        return {
            "mode": "modal",
            "evidence_sha256": fingerprint,
            "challenge": challenge,
            "checks": checks,
            "aborted": aborted,
            "max_parallel_calculations": self.max_parallel_calculations,
            "backend": self._backend(remotes),
        }
