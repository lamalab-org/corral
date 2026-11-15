"""Decisions wire contract and independent motif reporting; no live API calls."""

import ast
from pathlib import Path

import pytest

from corral.epistemic_analysis.motifs import (
    MOTIFS,
    STATUS_DESCRIPTIONS,
)
from corral.epistemic_analysis.reports import (
    EpistemicReport,
    MotifAssessment,
)


def test_motif_registry_preserves_original_pattern_names_and_kinds():
    # Read literals without importing the old analysis script or loading .env.
    source = Path(__file__).parents[2] / "reasoning_reports" / "analyze.py"
    definitions = {
        node.target.id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"SUBGRAPH_DESCRIPTIONS", "ANTIPATTERN_DESCRIPTIONS"}
    }
    # Definitions may become sharper without breaking stored pattern identities.
    assert {motif.name: motif.kind for motif in MOTIFS} == {
        **dict.fromkeys(definitions["SUBGRAPH_DESCRIPTIONS"], "productive"),
        **dict.fromkeys(definitions["ANTIPATTERN_DESCRIPTIONS"], "antipattern"),
    }
    assert len(MOTIFS) == 18


@pytest.mark.parametrize("status", STATUS_DESCRIPTIONS)
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("evidence_ids", [(), (1, 2)])
def test_historical_selection_does_not_override_current_status(
    status, selected, evidence_ids
):
    assessment = MotifAssessment(
        name="unsupported_judgment",
        kind="antipattern",
        family="evidence_handling",
        status=status,
        confidence=0.8,
        probabilities={status: 1.0},
        selected=selected,
        explanation_supported=True,
        explanation=(
            "The current proposal still claims five DBE against the formula."
            if status == "active"
            else "The earlier proposal claimed five DBE; it was corrected to four."
        ),
        evidence_message_ids=evidence_ids,
    )
    report = EpistemicReport(
        model="mock",
        assessed_at="",
        message_count=2,
        trajectory_sha256="",
        assessments=(assessment,),
        usage={},
        raw_response={},
    )
    # Even a supported explanation of an earlier error cannot make a nonactive
    # assessment an outstanding problem. Active findings still reach the actor.
    actionable = status == "active" and selected
    assert report.present_antipatterns == ((assessment,) if actionable else ())
    assert (assessment.explanation in report.format_for_agent()) == actionable
    assert ("unsupported_judgment" in report.format_for_agent()) == actionable
    assert report.to_dict()["assessments"][0]["name"] == "unsupported_judgment"
    assert report.to_dict()["assessments"][0]["explanation_supported"] is True


def test_descriptive_fixed_belief_does_not_hide_a_real_active_problem():
    def assessment(name):
        return MotifAssessment(
            name=name,
            kind="antipattern",
            family="experimental_strategy",
            status="active",
            confidence=0.9,
            probabilities={"active": 1.0},
            selected=True,
            explanation_supported=True,
            explanation="The observations support this classification.",
            evidence_message_ids=(1,),
        )

    fixed = assessment("fixed_belief_trace")
    contradiction = assessment("contradiction_without_repair")
    report = EpistemicReport(
        model="mock",
        assessed_at="",
        message_count=1,
        trajectory_sha256="",
        assessments=(fixed, contradiction),
        usage={},
        raw_response={},
    )
    assert report.present_antipatterns == (contradiction,)
    assert "fixed_belief_trace" not in report.format_for_agent()
    assert "contradiction_without_repair" in report.format_for_agent()
    assert report.format_for_agent() == (
        "Graph-matched unconstructive epistemic patterns (check against the observations):\n"
        f"- contradiction_without_repair: {contradiction.explanation}"
    )
    assert report.to_dict()["assessments"][0]["name"] == "fixed_belief_trace"
    assert report.to_dict()["assessments"][1]["name"] == "contradiction_without_repair"
