"""Focused checks for Level 1's evaluator-owned MACE verification plan."""

import pytest
from corral_md.scoring.common import EvaluationError, Rubric
from corral_md.scoring.verification import apply_verification

MODEL_TARGET = "independent_model_calculation"


def test_required_independent_calculation_errors_unless_verified():
    def rubric():
        result = Rubric(3)
        result.check("saved_evidence", 99, True)
        result.unverified(MODEL_TARGET, "Awaiting trusted calculation", points=1)
        return result

    pending = rubric()
    apply_verification(pending, None)
    with pytest.raises(EvaluationError, match=MODEL_TARGET):
        _ = pending.score

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
    with pytest.raises(EvaluationError, match=MODEL_TARGET):
        _ = unavailable.score
    assert not hasattr(unavailable, "apply_review")
