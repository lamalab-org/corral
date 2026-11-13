"""Conservative reduction of independent behavioral labels."""

import pytest

from corral.epistemic_analysis.decision_units import aggregate_or


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["present", "uncertain", "absent"], "present"),
        (["absent", "uncertain"], "uncertain"),
        (["absent", "error"], "uncertain"),
        (["not_applicable", "not_applicable"], "not_applicable"),
        (["absent", "absent"], "absent"),
        ([], "uncertain"),
    ],
)
def test_node_or_preserves_unknowns(statuses, expected):
    assert aggregate_or([{"status": s} for s in statuses]) == expected
