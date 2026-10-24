"""Focused checks for Level 1's evaluator-owned MACE verification plan."""

import pytest
from corral_md.scoring.common import Rubric
from corral_md.scoring.verification import apply_verification

MODEL_TARGET = "independent_model_calculation"


def test_required_independent_calculation_keeps_binary_score_pending_until_passed():
    def rubric():
        result = Rubric(3, binary=True)
        result.check("saved_evidence", 99, True)
        result.unverified(MODEL_TARGET, "Awaiting trusted calculation", points=1)
        return result

    pending = rubric()
    apply_verification(pending, None)
    assert pending.score is None

    passed = rubric()
    apply_verification(
        passed,
        {"checks": [{"id": "spot", "status": "passed", "targets": [MODEL_TARGET]}]},
    )
    assert passed.score == 1.0

    failed = rubric()
    apply_verification(
        failed,
        {"checks": [{"id": "spot", "status": "failed", "targets": [MODEL_TARGET]}]},
    )
    assert failed.score == 0.0

    unavailable = rubric()
    apply_verification(
        unavailable,
        {"checks": [{"id": "spot", "status": "unverified", "targets": [MODEL_TARGET]}]},
    )
    assert unavailable.score is None
    with pytest.raises(ValueError, match="trusted verification"):
        unavailable.apply_review(
            {
                "evidence_sha256": "frozen-evidence",
                "decisions": {
                    MODEL_TARGET: {
                        "passed": True,
                        "reviewer": "evaluator",
                        "reason": "No remote calculation was available",
                    }
                },
            },
            "frozen-evidence",
        )
